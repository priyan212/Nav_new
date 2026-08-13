"""HTTP-client replacement for NavDPStandalone.sample_pointgoal -- talks to
tryout/navdp_s2diff_server.py instead of running guidance in-process.

This is a THIRD, independent way to get S2Diff-guided NavDP into the rover
pipeline (the others: nav_pipeline/s2diff_navdp.py, in-process, used by
LAUNCH/launch_rover_s2diff.sh; and tryout/navdp_s2diff_server.py standalone
with no client at all). This module is the missing client for that server.
It changes nothing else -- same class-level monkeypatch pattern as
s2diff_navdp.py, so DINO/SAM/CLIP/depth estimation/obstacle-guard/GUI in
pipeline.py and isaac_gui.py all run completely unchanged; only how NavDP
trajectories are obtained differs.

Per tick: takes the current (last) memory frame pipeline.py already built,
re-encodes it as PNG, derives a caller-supplied obstacle-pixel list from the
same frame (a flat height-band filter -- simpler than obstacle_guard.py's
local-slope ground filter, since that filter isn't built to emit pixel
coordinates; see _obstacle_pixels_from_depth's docstring), and POSTs it all
to the server's /pointgoal_step. The server's own selection stays out of the
equation -- this returns ALL candidate trajectories + its per-candidate
`all_values` (higher = better, same sign convention as NavDPStandalone's own
critic) as (trajs, critic), so pipeline.py's existing _select_trajectory /
swept_clearance / forward_guard veto still makes the final call exactly like
it does for the in-process path.
"""

from __future__ import annotations

import io
import json

import numpy as np
import requests
import torch
from PIL import Image

from .goal_utils import intrinsics_from_fov
from .obstacle_guard import GuardConfig

_PINK = "\033[95m"
_RESET = "\033[0m"


def _print_guidance_debug(payload: dict) -> None:
    """How much this tick's S2Diff particle-guidance (CBF-style barrier +
    circulation energy layered on the DDPM score, see tube_planner/
    s2diff_guidance.py) changed the outcome vs. plain critic-argmax NavDP --
    i.e. what candidate a "pure-navdp" run (no guidance) would have picked.
    Both all_trajectory/all_values (pre-guidance-selection candidates+critic)
    and the s2diff diagnostics are already in every /pointgoal_step response
    (navdp_s2diff_server.py) -- this was simply never read/printed client-side."""

    s2 = payload.get("s2diff")
    if s2 is None:
        return
    all_values = np.asarray(payload["all_values"], dtype=np.float32)[0]
    all_traj = np.asarray(payload["all_trajectory"], dtype=np.float32)[0]
    critic_best_idx = int(np.argmax(all_values))
    guided_idx = int(s2["selected_index"][0])

    if guided_idx < 0 or guided_idx >= all_traj.shape[0]:
        endpoint_delta = float("nan")
    else:
        endpoint_delta = float(
            np.linalg.norm(all_traj[guided_idx, -1, :2] - all_traj[critic_best_idx, -1, :2])
        )

    print(
        f"{_PINK}[s2diff-guidance] picked #{guided_idx} vs critic-best #{critic_best_idx} "
        f"(same={guided_idx == critic_best_idx}) | endpoint delta={endpoint_delta:.3f}m | "
        f"noise_correction mean={s2['mean_guidance_noise_correction'][0]:.3f} "
        f"final={s2['final_guidance_noise_correction'][0]:.3f} "
        f"max={s2['maximum_guidance_noise_correction'][0]:.3f} | "
        f"barrier_energy={s2['selected_barrier_energy'][0]:.3f} "
        f"circulation_energy={s2['selected_circulation_energy'][0]:.3f} "
        f"clearance={s2['selected_minimum_clearance'][0]:.3f} | "
        f"valid_obstacle_pts={s2['valid_obstacle_points'][0]} | "
        f"fallback_stop={s2['fallback_stop'][0]} escape_turn={s2['escape_turn'][0]}{_RESET}"
    )


def _obstacle_pixels_from_depth(
    depth_hw: np.ndarray, fx: float, fy: float, cx: float, cy: float, guard: GuardConfig
) -> list[list[int]]:
    """Flat height-band obstacle-pixel picker (simpler than obstacle_guard.py's
    local-slope ground filter, which isn't built to return pixel coordinates).
    Good enough for the server's soft in-loop nudge -- pipeline.py's own,
    better-filtered depth_to_obstacle_points()/forward_guard()/
    swept_clearance() are unaffected by this and remain the real safety net.
    """

    h, w = depth_hw.shape
    stride = max(guard.stride, 1)
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    d = depth_hw[::stride, ::stride]

    valid = np.isfinite(d) & (d >= 0.15) & (d <= guard.max_range)
    y_cam = (vv.astype(np.float32) - cy) / fy * d
    z_up = guard.cam_height - y_cam
    obstacle = valid & (z_up > guard.ground_band) & (z_up < guard.overhead)

    us = uu[obstacle].astype(np.int64)
    vs = vv[obstacle].astype(np.int64)
    pixels = np.stack([us, vs], axis=1)
    max_pixels = 400
    if pixels.shape[0] > max_pixels:
        idx = np.linspace(0, pixels.shape[0] - 1, max_pixels).astype(np.int64)
        pixels = pixels[idx]
    return pixels.tolist()


def make_http_sample_pointgoal(
    server_url: str = "http://127.0.0.1:8888",
    fov_deg: float = 90.0,
    stop_threshold: float = 0.3,
    guard: GuardConfig | None = None,
    timeout_s: float = 5.0,
):
    """Build a sample_pointgoal(self, goal_point, images, depths, sample_num=32)
    replacement that calls tryout/navdp_s2diff_server.py over HTTP."""

    guard = guard or GuardConfig()
    state = {"reset_done": False}

    def _ensure_reset(h: int, w: int) -> None:
        if state["reset_done"]:
            return
        fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_deg)
        intrinsic = [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]
        resp = requests.post(
            f"{server_url}/navigator_reset",
            json={"intrinsic": intrinsic, "batch_size": 1, "stop_threshold": stop_threshold},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        state["reset_done"] = True
        print(f"[s2diff-http] navigator_reset ok: {resp.json()} (server={server_url})")

    @torch.no_grad()
    def sample_pointgoal(self, goal_point, images, depths, sample_num=32):
        img = images[0, -1].detach().cpu().float().numpy()  # (H,W,3) in [0,1], current frame
        dep = depths[0, -1, ..., 0].detach().cpu().float().numpy()  # (H,W) meters, current frame
        h, w = dep.shape
        _ensure_reset(h, w)

        fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_deg)
        obstacle_pixels = _obstacle_pixels_from_depth(dep, fx, fy, cx, cy, guard)
        # Named-obstacle pixels for this tick, if pipeline.py's step() got an
        # avoid_text: stashed as a plain attribute on this NavDPStandalone
        # instance (self here) right before it called sample_pointgoal --
        # see pipeline.py's _step_inner. Merged in raw; the server's
        # PixelObstacleConfig still filters by depth validity/range.
        avoid_pixels = getattr(self, "_pending_avoid_pixels", None)
        if avoid_pixels is not None and len(avoid_pixels):
            obstacle_pixels = obstacle_pixels + np.asarray(avoid_pixels, dtype=np.int64).tolist()

        goal = np.asarray(goal_point, dtype=np.float32).reshape(1, 3)
        # goal_point here is checkpoint-native [x fwd, y RIGHT, z]
        # (pipeline.py flips before calling sample_pointgoal); the server's
        # goal_x/goal_y are [x fwd, y LEFT] (S2DIFF_GUIDANCE.md's convention,
        # matching obstacle_pixels' frame) -- flip back.
        goal_left = goal.copy()
        goal_left[:, 1] = -goal_left[:, 1]

        rgb_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        depth_scaled = np.clip(dep * 10000.0, 0, np.iinfo(np.int32).max).astype(np.int32)

        img_buf = io.BytesIO()
        Image.fromarray(rgb_u8, mode="RGB").save(img_buf, format="PNG")
        img_buf.seek(0)
        dep_buf = io.BytesIO()
        Image.fromarray(depth_scaled, mode="I").save(dep_buf, format="PNG")
        dep_buf.seek(0)

        goal_data = {
            "goal_x": goal_left[:, 0].tolist(),
            "goal_y": goal_left[:, 1].tolist(),
            "obstacle_pixels": [obstacle_pixels],
        }
        files = {
            "image": ("frame.png", img_buf, "image/png"),
            "depth": ("depth.png", dep_buf, "image/png"),
        }
        resp = requests.post(
            f"{server_url}/pointgoal_step",
            data={"goal_data": json.dumps(goal_data)},
            files=files,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        _print_guidance_debug(payload)
        trajs = torch.as_tensor(np.asarray(payload["all_trajectory"], dtype=np.float32)[0])
        critic = torch.as_tensor(np.asarray(payload["all_values"], dtype=np.float32)[0])
        return trajs, critic

    return sample_pointgoal


def patch_navdp_standalone_http(
    server_url: str = "http://127.0.0.1:8888",
    fov_deg: float = 90.0,
    stop_threshold: float = 0.3,
    guard: GuardConfig | None = None,
) -> None:
    """Install the HTTP-client sampler onto the NavDPStandalone class.

    Same contract as s2diff_navdp.patch_navdp_standalone(): must run before
    any DinoNavDPPipeline is constructed, and edits no file on disk.
    """

    from . import navdp_net

    navdp_net.NavDPStandalone.sample_pointgoal = make_http_sample_pointgoal(
        server_url=server_url, fov_deg=fov_deg, stop_threshold=stop_threshold, guard=guard
    )
