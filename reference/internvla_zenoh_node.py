#!/usr/bin/env python3
"""
InternVLA-N1 Zenoh Inference Node  (camera-only, RGB, language-goal)
===================================================================
Phase-2 sibling of ``omnivla_zenoh_node.py``.  Runs on the GPU machine and
speaks the SAME Zenoh contract (camera-in -> cmd_vel-out), so the Pi / Isaac
side is unchanged.  ONLY the inference core differs: instead of OmniVLA it runs
InternVLA-N1's System-2 VLM (Qwen2.5-VL) to turn an RGB frame + language
instruction into a navigation command.

This node is ADDITIVE and non-destructive: it does not import or modify the
OmniVLA pipeline.  Pick the model at launch time with the ``--nav-model`` flag
in ``nav_model_launch.sh`` (default = omnivla, i.e. today's behaviour).

Env:  runs under the `internnav` conda env with transformers 4.51.0 shadowed:
    PYTHONPATH=/home/i3d/internnav_n1_tf451 \
    /mnt/bigdisk/conda_envs/internnav/bin/python inference/internvla_zenoh_node.py

Subscribes (Zenoh, CDR ROS 2 msgs):
  image_raw            – sensor_msgs/Image     (webcam / Isaac camera)
  omnivla/goal_text    – std_msgs/String       (change instruction; shared topic)
Publishes (Zenoh, CDR):
  cmd_vel              – geometry_msgs/Twist
  omnivla/explanation  – std_msgs/String

Safety: CAMERA-ONLY.  No LiDAR e-stop (per project requirement).  The model's
own STOP action ends navigation; a Grounding-DINO vision-stop layer is added in
Phase 3 for the real rover.

Two inference modes, selected by CFG.use_navdp / --use-navdp (default OFF):

  System-2 only (default, RGB, NO depth / NO NavDP):
    agent.step_s2() -> System-2 VLM emits EITHER
      * a discrete action  {STOP, forward, left, right, look-down}, OR
      * a pixel goal, normalized to Qwen-VL's standard 0-1000 grounding scale.
    We map discrete actions directly to cmd_vel, and convert a pixel goal into a
    reactive visual-servo steer (horizontal offset -> angular vel). This path
    never touches System-1/NavDP, so no depth sensor or depth checkpoint is
    required -- matches the project's original RGB-only constraint exactly.

  NavDP depth-conditioned obstacle avoidance (opt-in, --use-navdp):
    Uses a DIFFERENT checkpoint (InternVLA-N1-w-NavDP, ~16.8GB) with
    system1="navdp_async", which builds the model's REAL trajectory-diffusion
    module. Since there's still no depth sensor, depth is estimated monocularly
    (DepthAnythingV2-metric) from the same RGB frame each tick and fed through
    agent.step()'s full System-2+System-1 pipeline. Validated 2026-07-14: real
    trained weights (no random-init), ~17GB VRAM, ~0.3s/tick steady-state
    (after a one-time ~10s CUDA warmup absorbed at startup). The diffusion
    model's 32 sampled trajectories are NOT critic-filtered by the model
    itself (no invocable critic method exists for this checkpoint's class --
    see memory notes), so `traj_to_actions` is monkeypatched to majority-
    cluster the 32 samples (by position at navdp_lookahead_idx) and average
    only the largest agreeing cluster, instead of blindly averaging all 32
    (which washes multimodal choices like "go left around" vs "go right
    around" into a mushy, non-committal path). A separate independent
    depth-based safety layer (hard stop + reactive steer-around) provides the
    actual collision-avoidance guarantee -- see _apply_depth_safety /
    _apply_reactive_avoid. See internvla-n1-phase1 memory notes for the full
    history of fixes and why each exists.
"""

import sys
import os
import math
import struct
import time
import argparse
from threading import Lock, Thread
from typing import Optional

import numpy as np
from PIL import Image
import torch

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh Python library not found (pip install eclipse-zenoh).")
    sys.exit(1)


# ================================================================
#  CDR Helpers  (Common Data Representation — DDS wire format)
#  Copied from omnivla_zenoh_node.py so this node stays independent
#  (same wire format; model-agnostic).
# ================================================================
class CDRReader:
    def __init__(self, data: bytes):
        self.data = data
        self.le = data[1] in (0x01, 0x11)
        self.end = "<" if self.le else ">"
        self.offset = 4
        self.base = 4

    def _align(self, n: int):
        rem = (self.offset - self.base) % n
        if rem:
            self.offset += n - rem

    def read_uint8(self) -> int:
        v = self.data[self.offset]; self.offset += 1
        return v

    def read_int32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "i", self.data, self.offset)
        self.offset += 4
        return v

    def read_uint32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "I", self.data, self.offset)
        self.offset += 4
        return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset : self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data = self.data[self.offset : self.offset + count]
        self.offset += count
        return data


class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n: int):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_uint32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_float64(self, v: float):
        self._align(8)
        self.buf += struct.pack("<d", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self.buf += encoded

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/msg/Image CDR -> numpy RGB array (H, W, 3)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()          # header
    height = r.read_uint32(); width = r.read_uint32()
    encoding = r.read_string()
    r.read_uint8(); r._align(4); r.read_uint32()              # is_bigendian, step
    pixel_data = r.read_sequence_uint8()

    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None
    enc = encoding.lower()
    if enc == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    if img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def parse_string(cdr_data: bytes) -> str:
    return CDRReader(cdr_data).read_string()


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    """geometry_msgs/msg/Twist -> CDR bytes."""
    w = CDRWriter()
    w.write_float64(linear_x)   # linear.x
    w.write_float64(0.0)        # linear.y
    w.write_float64(0.0)        # linear.z
    w.write_float64(0.0)        # angular.x
    w.write_float64(0.0)        # angular.y
    w.write_float64(angular_z)  # angular.z
    return w.to_bytes()


def serialize_string(text: str) -> bytes:
    w = CDRWriter()
    w.write_string(text)
    return w.to_bytes()


# ================================================================
#  Configuration
# ================================================================
class NodeConfig:
    # Paths — kept entirely under OmniVLA_safe. The original InternNav checkout at
    # motion_planning/Navigation/InternNav was deleted externally mid-integration
    # (not by this project); the `internnav` package code was re-cloned here.
    # Checkpoint weights survived in the shared HF cache (not deleted), so we load
    # directly from there instead of a local checkpoints/ copy.
    internnav_repo: str = "/mnt/bigdisk/Priyan/OmniVLA_safe/third_party/InternNav"
    model_path: str = (
        "/mnt/bigdisk/hf_cache/transformers/models--InternRobotics--InternVLA-N1-DualVLN/"
        "snapshots/a698a9e898b4001621a319e1bc89f02ec715cc86"
    )
    device: str = "cuda:0"

    # --- Depth-conditioned obstacle avoidance (opt-in, --use-navdp) -----------
    # Default OFF: today's System-2-only (RGB, no depth) behaviour is unchanged
    # unless explicitly requested. When ON, uses a DIFFERENT checkpoint
    # (InternVLA-N1-w-NavDP, ~16.8GB) with system1="navdp_async", which builds
    # the REAL depth-conditioned trajectory-diffusion module. Since there's no
    # depth sensor (project constraint), depth is estimated monocularly
    # (DepthAnythingV2-metric) from the same RGB frame each tick.
    use_navdp: bool = False
    navdp_model_path: str = (
        "/mnt/bigdisk/Priyan/OmniVLA_safe/third_party/InternNav/checkpoints/InternVLA-N1-w-NavDP"
    )
    depth_encoder_ckpt: str = (
        "/mnt/bigdisk/Priyan/OmniVLA_safe/third_party/InternNav/checkpoints/depth_anything_v2_vits.pth"
    )
    depth_metric_ckpt: str = (
        "/mnt/bigdisk/Priyan/OmniVLA_safe/third_party/InternNav/checkpoints/"
        "depth_anything_v2_metric_hypersim_vits.pth"
    )
    # Index 8/32 (the immediate next micro-step) under-represented the model's
    # actual overall direction toward the identified target, and is where
    # diffusion-averaging noise is worst (smallest displacement -> atan2 most
    # noise-sensitive). Moved further along the trajectory, closer to where it
    # actually points, at the cost of being a longer-horizon (less immediately
    # reactive) prediction -- reasonable trade for "aim at the real target."
    navdp_lookahead_idx: int = 24   # index into the 32-step predicted trajectory to steer toward
    navdp_window: int = 4           # average +/- this many points around navdp_lookahead_idx (noise reduction)
    navdp_turn_gain: float = 1.2    # proportional heading-error -> angular-velocity gain
    navdp_damping: float = 0.5      # cross-tick smoothing on (lin,ang): 0=none, closer to 1=more damped

    # --- Majority-cluster trajectory aggregation (replaces blind averaging) ---
    # agent.step()'s own traj_to_actions() does plain np.mean() over the 32
    # diffusion-sampled trajectories. In an ambiguous/multimodal scene (e.g.
    # some samples go left around an obstacle, some go right), that average
    # blends two valid-but-opposite paths into a mushy, non-committal direction
    # that follows neither -- the root cause of poor target-centering. We
    # monkeypatch traj_to_actions (in the agent module's OWN namespace, since
    # `from ... import traj_to_actions` already bound the name there at import
    # time -- patching vln_utils.traj_to_actions would not affect that call)
    # to cluster the 32 samples by their position at navdp_lookahead_idx and
    # average only the LARGEST cluster instead of all 32.
    navdp_cluster_radius_m: float = 0.4      # samples within this distance of each other = "same cluster"
    navdp_cluster_min_frac: float = 0.25     # if the largest cluster is smaller than this fraction, fall back to plain mean (not enough agreement to trust a subset)

    # --- Independent depth-based safety net (hard override, model-agnostic) ---
    # agent.step()'s own NavDP path averages 32 sampled trajectories with NO
    # critic-based safety filtering (verified 2026-07-14: traj_to_actions does
    # plain np.mean, and the trained critic_head that WOULD filter unsafe paths
    # has no callable method in the class this checkpoint actually uses -- see
    # phase-1 memory notes). So there is no guaranteed collision avoidance from
    # the model itself. This adds a simple, deterministic, model-independent
    # check on the SAME depth map already computed each tick (zero extra cost):
    # force zero velocity if anything is too close in the forward cone,
    # regardless of what the model decided.
    #
    # No InternNav-published stop-distance applies here: their own collision
    # mechanisms use different sensing entirely -- ground-truth occupancy grids
    # for A* sim planners (continuous_planner.py/discrete_planner.py), or a
    # top-down depth camera with height-based floor/obstacle classification for
    # their Isaac Sim robot controller (vln_move_by_flash_with_collision_
    # controller.py) -- neither translates to a forward-facing monocular-camera
    # distance threshold. Per user decision (2026-07-14): plain hard stop at
    # 30cm, no gradual slowdown zone.
    navdp_estop_distance_m: float = 0.30     # hard stop below this -- ONLY threshold, no slowdown zone
    navdp_cone_frac_w: float = 0.5           # center fraction of image width treated as "ahead"
    navdp_cone_frac_h: float = 0.5           # bottom fraction of image height treated as "ahead" (ground-level bias)
    navdp_cone_percentile: float = 5.0       # low percentile (not bare min) of cone depths -- robust to noisy outlier pixels

    # --- Reactive steer-around (separate from the hard stop above) ---
    # The hard stop above can only ever brake -- it has no ability to route
    # AROUND an obstacle. This adds classic reactive-avoidance steering, using
    # the SAME depth map (still zero extra model cost): compare left-half vs
    # right-half depth in a wider "caution zone" ahead, and bias angular
    # velocity away from whichever side is closer, scaled by how close/
    # asymmetric it is. Purely geometric, no model involvement -- deliberately
    # separate from the hard stop so tuning one doesn't require touching the
    # other.
    navdp_avoid_distance_m: float = 0.9      # start biasing steering when closer than this
    navdp_avoid_gain: float = 1.5            # how strongly to steer away from the closer side
    # Confirmed live on a real rover (2026-07-15): a glossy/reflective floor
    # made monocular depth read near-identical left/right values (differing
    # by only ~0.01-0.02m, pure noise) almost everywhere, since the "obstacle"
    # the depth model saw was actually a floor reflection, not real geometry.
    # Trusting min(left,right) vs which-is-smaller under that noise flips the
    # steer-away SIGN essentially at random tick to tick -- a real, observed
    # cause of "moving randomly" that has nothing to do with the model's own
    # (fairly stable, in the same log) trajectory heading. Require a minimum
    # difference before trusting a directional steer; below it, still ease
    # forward speed (there IS something close ahead) but don't guess a turn.
    navdp_avoid_min_diff_m: float = 0.15
    navdp_avoid_ang_damping: float = 0.6     # cross-tick smoothing on the avoid contribution itself

    # --- Stall / physically-blocked recovery (model- AND depth-independent) ---
    # Observed live (2026-07-15, Isaac Sim): against a large, flat, low-texture
    # obstacle (a blank white surface filling most of the frame), the
    # monocular depth estimate never read "too close" -- the model kept
    # commanding a left turn for 10+ consecutive ticks while the camera view
    # barely changed (mean grayscale delta ~0.7/255 over 8s / ~10 ticks --
    # confirmed via screenshot diff, i.e. the rover was physically wedged /
    # not making progress), and neither the 30cm hard-stop nor the 0.9m
    # reactive-avoid ever engaged. This is a known weakness of monocular
    # depth on large, texture-poor surfaces (no parallax/edge cues to anchor
    # scale) -- and it applies even in System-2-only mode, which has no depth
    # at all. This layer needs neither: it detects "commanding real motion but
    # the RGB frame isn't changing" directly, and forces a brief reverse+turn
    # recovery -- a final, independent safety net that catches exactly the
    # case the depth-based layers can miss.
    stall_frame_diff_thresh: float = 1.5     # mean abs grayscale delta (0-255, 32x24 downsample) below this = "no visible motion"
    stall_ticks_confirm: int = 4             # consecutive stalled ticks (while commanding real motion) before triggering recovery
    stall_recover_ticks: int = 3             # ticks to hold the reverse+turn recovery maneuver
    stall_backup_lin: float = 0.12           # reverse speed during recovery
    stall_backup_ang: float = 0.4            # turn speed during recovery (continues the model's last turn direction)

    # --- Final motion smoothing (uniform across ALL modes/kinds) --------------
    # Real per-tick latency varies enormously in practice (150ms-1.7s measured
    # live on the real rover -- generation length differs per decision, and a
    # look-down retry roughly doubles that tick's compute), and until now only
    # the NavDP trajectory path had any cross-tick damping -- discrete actions
    # and the plain pixel-servo path snapped straight to their target lin/ang
    # every tick with zero memory. That produced exactly the reported
    # "turns, THEN moves forward" behaviour: a discrete LEFT (full angular,
    # partial linear) followed by a discrete FWD (zero angular, full linear)
    # arrives as two separate instantaneous commands rather than one smooth
    # blend. This final stage rate-limits (lin, ang) together, regardless of
    # which path produced them, scaled by ACTUAL elapsed wall-clock time since
    # the last tick (not a fixed nominal period) -- so it inherently adapts to
    # the irregular update rate instead of assuming a steady predict_hz.
    # Expressed as a ramp TIME (seconds to go from 0 to the current
    # max_linear/max_angular) rather than a fixed accel constant, so it
    # auto-scales whether CFG.max_linear/max_angular are Isaac's fast values or
    # the real rover's slow ones. Skipped entirely for stop/hard-stop, which
    # must remain instantaneous.
    #
    # Set well ABOVE the typical observed tick interval (real measured range
    # 150ms-1.7s) deliberately: if the ramp time were shorter than a typical
    # tick, a single tick's elapsed dt would already permit the full swing and
    # this would rarely engage. At 1.0s, a typical ~0.3-0.7s tick only permits
    # 30-70% of the full range, so a genuine multi-tick blend is visible.
    motion_ramp_time_s: float = 1.0
    # VLM input
    resize_w: int = 384
    resize_h: int = 384
    num_history: int = 8
    plan_step_gap: int = 1
    # Navigation velocities (real physical units, m/s and rad/s) — kept low
    max_linear: float = 0.15
    max_angular: float = 0.25
    predict_hz: float = 2.0
    instruction: str = "go straight down the hallway and stop at the door"
    # Pixel-goal visual-servo steering
    servo_forward_frac: float = 0.8   # forward speed fraction at zero offset (was 0.6 -- felt sluggish)
    center_deadband: float = 0.10     # |offset| below this = go straight
    discrete_turn_frac: float = 0.5   # forward-speed fraction used for in-place left/right (was 0.3)
    # Discrete ACT_FWD carries zero heading feedback of its own (always
    # ang=0.0) -- observed live (2026-07-16, real rover): ~85% of ticks in a
    # run were plain ACT_FWD, and with no correction between the rare
    # look-down/trajectory ticks, the rover visibly veered off-target instead
    # of centering on it. _decayed_heading_bias() carries a decaying fraction
    # of the last real corrective heading (from a trajectory or pixel-servo
    # tick) into subsequent blind-FWD ticks instead of assuming ang=0.
    discrete_fwd_bias_decay_s: float = 3.0
    # Goal-reached: N STOP actions in a row confirms arrival
    stop_confirm_count: int = 2


CFG = NodeConfig()

# InternVLA-N1 discrete action indices (from the agent's actions2idx)
ACT_STOP, ACT_FWD, ACT_LEFT, ACT_RIGHT, ACT_LOOKDOWN = 0, 1, 2, 3, 5


# ================================================================
#  N1 model loading + compat shims  (Phase-1 findings, made explicit)
# ================================================================
AGENT = {"agent": None, "intrinsic": None}


def _apply_n1_compat_shims():
    """Load-time fixes required to run these checkpoints under transformers 4.51.0.

    1. `config.text_config` — InternNav's internvla_n1.py:48 reads it, but 4.51.0's
       Qwen2.5-VL config is FLAT. get_text_config() returns self for flat configs,
       so expose text_config -> self (return self DIRECTLY; going through
       get_text_config() recurses via its hasattr check).
    2. `system1` — controls which System-1 submodule (if any) gets built:
       - CFG.use_navdp=False (default): this node runs System-2 ONLY (RGB,
         discrete-action / pixel-goal); System-1 is never invoked. We FORCE
         system1="navdp" (NOT "*_async") so the arch builds ONLY `latent_queries`
         (needed because forward() reads it unconditionally) and skips every
         other System-1 submodule. Safe because n_traj_tokens==0 in this mode,
         so latent_queries' value is never actually used.
       - CFG.use_navdp=True: forces system1="navdp_async" to build the REAL
         depth-conditioned NavDP trajectory-diffusion module (validated in the
         2026-07-14 spike: genuine trained weights, no random-init, sane output).
    3. `DAT_RGBD_Patch_Backbone`'s default checkpoint path (only relevant when
       use_navdp=True) — it hardcodes a RELATIVE path "checkpoints/depth_anything_
       v2_vits.pth" that only resolves if CWD happens to be the InternNav repo
       root. Patch the default to our absolute path instead.
    """
    from transformers import Qwen2_5_VLConfig
    if not hasattr(Qwen2_5_VLConfig, "text_config"):
        Qwen2_5_VLConfig.text_config = property(lambda self: self)

    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig
    target_system1 = "navdp_async" if CFG.use_navdp else "navdp"
    if getattr(InternVLAN1ModelConfig, "_n1_system1_patched_value", None) != target_system1:
        _orig = InternVLAN1ModelConfig.__init__

        def _init(self, **kwargs):
            _orig(self, **kwargs)
            self.system1 = target_system1

        InternVLAN1ModelConfig.__init__ = _init
        InternVLAN1ModelConfig._n1_system1_patched_value = target_system1

    if CFG.use_navdp:
        import internnav.model.encoder.navdp_backbone as navdp_backbone
        if not getattr(navdp_backbone.DAT_RGBD_Patch_Backbone, "_ckpt_path_patched", False):
            _orig_backbone_init = navdp_backbone.DAT_RGBD_Patch_Backbone.__init__

            def _patched_backbone_init(self, *args, checkpoint=CFG.depth_encoder_ckpt, **kwargs):
                _orig_backbone_init(self, *args, checkpoint=checkpoint, **kwargs)

            navdp_backbone.DAT_RGBD_Patch_Backbone.__init__ = _patched_backbone_init
            navdp_backbone.DAT_RGBD_Patch_Backbone._ckpt_path_patched = True

        _install_clustered_traj_to_actions()


def _cluster_by_radius(points: np.ndarray, radius_m: float) -> list:
    """Simple, dependency-free radius-based clustering (union-find over a
    distance graph). Returns a list of index-lists, one per cluster. N=32
    points, so the O(N^2) pairwise distance check is trivial."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < radius_m:
                union(i, j)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _clustered_trajectory_reduce(all_traj_xy: np.ndarray) -> np.ndarray:
    """Replace agent.step()'s blind np.mean(all_trajectory, axis=0) with
    majority-cluster aggregation: cluster the N sampled trajectories by their
    position at navdp_lookahead_idx, average only the LARGEST cluster's full
    trajectories. Falls back to the plain mean if no cluster reaches
    navdp_cluster_min_frac of the samples (not enough agreement to trust a
    subset over the full population). See NodeConfig's comment above
    navdp_cluster_radius_m for the full rationale."""
    n = all_traj_xy.shape[0]
    feat_idx = min(CFG.navdp_lookahead_idx, all_traj_xy.shape[1] - 1)
    features = all_traj_xy[:, feat_idx, :]

    clusters = _cluster_by_radius(features, CFG.navdp_cluster_radius_m)
    largest = max(clusters, key=len)
    AGENT["navdp_cluster_info"] = f"{len(largest)}/{n}"

    if len(largest) / n < CFG.navdp_cluster_min_frac:
        return np.mean(all_traj_xy, axis=0)
    return np.mean(all_traj_xy[largest], axis=0)


def _install_clustered_traj_to_actions():
    """Monkeypatch traj_to_actions in internvla_n1_agent_realworld's OWN
    namespace -- `from ...vln_utils import traj_to_actions` already bound the
    name there at import time, so patching vln_utils.traj_to_actions directly
    would NOT affect agent.step()'s call (Python resolves globals via the
    DEFINING module's namespace, not the original source module's)."""
    import internnav.agent.internvla_n1_agent_realworld as agent_module
    if getattr(agent_module, "_traj_to_actions_clustered_patched", False):
        return
    _orig_traj_to_actions = agent_module.traj_to_actions

    def _clustered_traj_to_actions(dp_actions, use_discrate_action=True):
        if use_discrate_action:
            # Not our code path (agent.step() always calls with False), but
            # preserve original behaviour for defensive completeness.
            return _orig_traj_to_actions(dp_actions, use_discrate_action)
        dp_actions[:, :, :2] /= 4.0  # unnormalize, matches original exactly
        start_xy = np.zeros((dp_actions.shape[0], 2))
        delta = dp_actions.float().cpu().numpy()
        cumsum_xy = np.cumsum(delta[:, :, :2], axis=1)
        B, T = delta.shape[0], delta.shape[1]
        all_traj = np.zeros((B, T + 1, 2))
        all_traj[:, 0] = start_xy
        all_traj[:, 1:] = start_xy[:, None, :] + cumsum_xy
        return _clustered_trajectory_reduce(all_traj)

    agent_module.traj_to_actions = _clustered_traj_to_actions
    agent_module._traj_to_actions_clustered_patched = True


def load_model():
    print(f"\n{'=' * 60}")
    print(f"  InternVLA-N1 Zenoh Node — loading model on {CFG.device}")
    print(f"  Mode: {'NavDP depth-conditioned avoidance (opt-in)' if CFG.use_navdp else 'System-2 only (default)'}")
    print(f"{'=' * 60}")

    sys.path.insert(0, CFG.internnav_repo)
    if CFG.use_navdp:
        # navdp.py imports diffusion_policy, an InternNav submodule not on the
        # default Python path (see internvla-n1-phase1 memory notes).
        sys.path.insert(0, f"{CFG.internnav_repo}/third_party/diffusion-policy")
    _apply_n1_compat_shims()
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    model_path = CFG.navdp_model_path if CFG.use_navdp else CFG.model_path
    args = argparse.Namespace(
        device=CFG.device, model_path=model_path,
        resize_w=CFG.resize_w, resize_h=CFG.resize_h,
        num_history=CFG.num_history, plan_step_gap=CFG.plan_step_gap,
    )
    agent = InternVLAN1AsyncAgent(args)
    # Debug frames off the (full) bigdisk.
    agent.save_dir = "/tmp/internvla_n1_runs/" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(agent.save_dir, exist_ok=True)

    if not CFG.use_navdp:
        # step_s2() unconditionally runs a SECOND forward pass (generate_latents)
        # whenever the VLM emits a pixel-goal, to produce a System-1/NavDP trajectory
        # latent. In System-2-only mode we never use that latent (_pixel_to_cmd only
        # reads the raw pixel coordinate) -- it was pure wasted compute, ~1s per
        # pixel-goal tick measured in the Isaac Sim test on 2026-07-14. Stub it out.
        # NOT applied in NavDP mode: there, the real latent is required by step_s1().
        def _skip_generate_latents(output_ids, pixel_values, image_grid_thw):
            return None
        agent.model.generate_latents = _skip_generate_latents

    # Camera intrinsics (unused by either inference path -- neither does true
    # camera-frame back-projection -- but the agent signature requires it).
    intrinsic = np.array([[386.5, 0.0, 328.9, 0.0],
                          [0.0, 386.5, 244.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
    AGENT["agent"] = agent
    AGENT["intrinsic"] = intrinsic

    if CFG.use_navdp:
        _load_depth_estimator()
        _warmup_navdp()

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Model ready. VRAM allocated: {vram:.2f} GB\n")


def _load_depth_estimator():
    """Monocular metric depth (DepthAnythingV2-metric-hypersim, vits) -- there is
    no depth sensor (project constraint: RGB camera only), so NavDP's real depth
    input is estimated fresh from each RGB frame. Validated in the 2026-07-14
    spike: sane 0.6-11m range on the test frame, no NaN."""
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    print("  Loading monocular depth estimator (DepthAnythingV2-metric)...")
    depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
    state = torch.load(CFG.depth_metric_ckpt, map_location="cpu")
    depth_model.load_state_dict(state, strict=False)
    AGENT["depth_model"] = depth_model.to(CFG.device).eval()


def _warmup_navdp():
    """First-ever forward pass through a freshly loaded CUDA model pays a one-time
    kernel-compilation cost. Absorb that here at startup instead of on the first
    real navigation tick, which would otherwise cause a jarring stall.

    Two SEPARATE warmups are needed: step_s1's rgbs/depths tensors are always
    224x224 (that resize happens in infer_cmd, matching what step_s1 expects),
    but the depth ESTIMATOR (infer_image) consumes the RAW camera frame BEFORE
    that resize -- warming it up with a 224x224 square dummy (as an earlier
    version of this function did) produces a different internal resize geometry
    than a real ~4:3 camera frame, so cudnn recompiles anyway on the first real
    tick (measured: 10s instead of the expected ~0.3s). Use a realistic
    non-square shape here instead."""
    print("  Warming up NavDP CUDA kernels (one-time, ~10s)...")
    agent = AGENT["agent"]
    intrinsic = AGENT["intrinsic"]
    dummy_camera_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)  # ~ typical camera frame shape
    dummy_rgb_224 = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_depth_224 = np.ones((224, 224), dtype=np.float32) * 2.0
    n_query, hidden = agent.model.config.n_query, agent.model.config.hidden_size
    dummy_latent = torch.randn(1, n_query, hidden, device=CFG.device, dtype=torch.bfloat16)
    rgbs = torch.stack([torch.from_numpy(dummy_rgb_224 / 255.0)] * 2).unsqueeze(0).to(CFG.device)
    depths = torch.stack([torch.from_numpy(dummy_depth_224)] * 2).unsqueeze(0).unsqueeze(-1).to(CFG.device)
    t0 = time.time()
    agent.step_s1(dummy_latent, rgbs, depths)
    AGENT["depth_model"].infer_image(dummy_camera_rgb[:, :, ::-1])
    # The residual first-tick cost isn't step_s1/depth (both warmed above) --
    # it's step_s2's own language-model generate() call, which agent.step()
    # runs FIRST and which neither of the above touches. Warm it up too, then
    # reset() to discard the dummy conversation history it leaves behind.
    agent.step_s2(dummy_camera_rgb, dummy_depth_224, np.eye(4), "go forward", intrinsic, look_down=False)
    agent.reset()  # discard dummy conversation history
    agent.save_dir = "/tmp/internvla_n1_runs/" + time.strftime("%Y%m%d_%H%M%S")  # reset() reverts this; redo (bigdisk is full)
    os.makedirs(agent.save_dir, exist_ok=True)
    print(f"  Warmup done in {time.time()-t0:.1f}s")


# ================================================================
#  Inference: RGB frame + instruction -> (lin, ang, meta)
# ================================================================
def infer_cmd(rgb: np.ndarray, instruction: str) -> tuple:
    """Dispatch to the System-2-only path (default) or the depth-conditioned
    NavDP path (--use-navdp), per CFG.use_navdp. See module docstring.

    Then run the stall-recovery check on TOP of whichever path ran -- it is
    model- and depth-independent (RGB motion only), so it applies uniformly
    to both modes. See NodeConfig's comment above stall_frame_diff_thresh
    for why this exists."""
    if CFG.use_navdp:
        lin, ang, kind, detail = _infer_cmd_navdp(rgb, instruction)
    else:
        lin, ang, kind, detail = _infer_cmd_system2_only(rgb, instruction)
    if kind in ("trajectory", "pixel"):
        # Remember the last real heading correction so a later blind ACT_FWD
        # tick (which has none of its own) can carry a decaying fraction of
        # it instead of assuming ang=0 -- see _decayed_heading_bias().
        AGENT["last_heading_bias_ang"] = ang
        AGENT["last_heading_bias_ts"] = time.time()
    lin, ang, kind, detail = _apply_motion_smoothing(lin, ang, kind, detail)
    return _apply_stall_recovery(rgb, lin, ang, kind, detail)


def _apply_motion_smoothing(lin: float, ang: float, kind: str, detail: str) -> tuple:
    """Final, universal rate-limiter on (lin, ang) -- applies identically
    whether the raw command came from a discrete action, the pixel-servo, or
    a NavDP trajectory, so switching between them (or firing the same
    discrete action repeatedly) ramps smoothly instead of snapping. See
    NodeConfig's motion_ramp_time_s comment for the full rationale.

    Runs BEFORE stall-recovery (which still overrides decisively, unsmoothed,
    when it triggers) and is skipped entirely for stop/hard-stop -- those must
    remain instantaneous, not ramped down."""
    if kind in ("stop", "obstacle-stop"):
        AGENT.pop("smooth_prev_lin", None)
        AGENT.pop("smooth_prev_ang", None)
        AGENT.pop("smooth_prev_t", None)
        return lin, ang, kind, detail

    now = time.time()
    prev_lin = AGENT.get("smooth_prev_lin")
    prev_ang = AGENT.get("smooth_prev_ang")
    prev_t = AGENT.get("smooth_prev_t")

    if prev_lin is None or prev_t is None:
        AGENT["smooth_prev_lin"] = lin
        AGENT["smooth_prev_ang"] = ang
        AGENT["smooth_prev_t"] = now
        return lin, ang, kind, detail

    dt = max(now - prev_t, 1e-3)
    max_dlin = (CFG.max_linear / CFG.motion_ramp_time_s) * dt
    max_dang = (CFG.max_angular / CFG.motion_ramp_time_s) * dt

    smoothed_lin = prev_lin + float(np.clip(lin - prev_lin, -max_dlin, max_dlin))
    smoothed_ang = prev_ang + float(np.clip(ang - prev_ang, -max_dang, max_dang))

    AGENT["smooth_prev_lin"] = smoothed_lin
    AGENT["smooth_prev_ang"] = smoothed_ang
    AGENT["smooth_prev_t"] = now

    return smoothed_lin, smoothed_ang, kind, (
        f"{detail} | ramp(target lin={lin:+.3f} ang={ang:+.3f})"
    )


def _apply_stall_recovery(rgb: np.ndarray, lin: float, ang: float, kind: str, detail: str) -> tuple:
    """Final, model/depth-independent safety net: if we're commanding real
    motion but the camera view isn't changing tick-over-tick, we're
    physically wedged against something the model/depth-based layers missed.
    Detected purely from downsampled grayscale RGB deltas -- no model or
    depth map needed, so it protects System-2-only mode too. See
    NodeConfig's stall_* comment for the live observation that motivated
    this."""
    if kind == "stop":
        AGENT.pop("stall_prev_frame", None)
        AGENT["stall_count"] = 0
        AGENT["stall_recovering"] = 0
        return lin, ang, kind, detail

    recovering = AGENT.get("stall_recovering", 0)
    if recovering > 0:
        AGENT["stall_recovering"] = recovering - 1
        recover_ang = AGENT.get("stall_recover_ang", CFG.stall_backup_ang)
        return -CFG.stall_backup_lin, recover_ang, "stall-recover", (
            f"{detail} | STALL recovery ({recovering} ticks left)"
        )

    small = np.asarray(Image.fromarray(rgb).convert("L").resize((32, 24)), dtype=np.float32)
    prev = AGENT.get("stall_prev_frame")
    AGENT["stall_prev_frame"] = small
    commanding_motion = abs(lin) > 0.02 or abs(ang) > 0.05

    if prev is None or not commanding_motion:
        AGENT["stall_count"] = 0
        return lin, ang, kind, detail

    frame_diff = float(np.abs(small - prev).mean())
    if frame_diff > CFG.stall_frame_diff_thresh:
        AGENT["stall_count"] = 0
        return lin, ang, kind, detail

    AGENT["stall_count"] = AGENT.get("stall_count", 0) + 1
    if AGENT["stall_count"] < CFG.stall_ticks_confirm:
        return lin, ang, kind, detail

    AGENT["stall_count"] = 0
    AGENT["stall_recovering"] = CFG.stall_recover_ticks
    recover_ang = CFG.stall_backup_ang if ang >= 0 else -CFG.stall_backup_ang
    AGENT["stall_recover_ang"] = recover_ang
    return -CFG.stall_backup_lin, recover_ang, "stall-recover", (
        f"{detail} | STALL: no visual motion for {CFG.stall_ticks_confirm} ticks -> backing up"
    )


def _infer_cmd_navdp(rgb: np.ndarray, instruction: str) -> tuple:
    """Full agent.step(): System-2 language reasoning + (when it emits a
    pixel-goal) System-1/NavDP depth-conditioned trajectory diffusion.

    Depth is estimated monocularly from the SAME rgb frame each tick (no depth
    sensor exists). Reuses InternNav's own agent.step() branching logic (not
    reimplemented here) so this matches upstream's tested dual-system behaviour
    exactly -- see the 2026-07-14 spike notes for why this is safe/validated.

    Returns kind in {"stop","discrete","trajectory","obstacle-stop"}. The last
    is an independent depth-based safety override -- see NodeConfig's
    navdp_estop_distance_m block for why this exists.
    """
    agent = AGENT["agent"]
    intrinsic = AGENT["intrinsic"]
    depth_map = AGENT["depth_model"].infer_image(rgb[:, :, ::-1])  # infer_image expects BGR

    out = agent.step(rgb, depth_map, np.eye(4), instruction, intrinsic, look_down=False)
    if out.output_action is not None and len(out.output_action) > 0 and int(out.output_action[0]) == ACT_LOOKDOWN:
        # Rover camera can't tilt; re-run the model's look-down refinement
        # on the same frame (matches http_internvla_server.py's eval_dual()).
        out = agent.step(rgb, depth_map, np.eye(4), instruction, intrinsic, look_down=True)

    if out.output_action is not None and len(out.output_action) > 0:
        lin, ang, kind, detail = _discrete_action_to_cmd(int(out.output_action[0]))
    elif out.output_trajectory is not None:
        lin, ang, kind, detail = _trajectory_to_cmd(np.asarray(out.output_trajectory))
    else:
        lin, ang, kind, detail = 0.0, 0.0, "discrete", "no-output"

    return _apply_depth_safety(lin, ang, kind, detail, depth_map)


def _forward_cone_min_depth(depth_map: np.ndarray) -> float:
    """Robust 'closest obstacle ahead' distance (meters) from the estimated
    depth map: a low percentile (not the bare min) of a center/bottom-biased
    crop, since single noisy outlier pixels near object edges are common in
    monocular depth estimation and would otherwise trigger false-positive stops."""
    h, w = depth_map.shape
    y0 = int(h * (1.0 - CFG.navdp_cone_frac_h))
    x0 = int(w * (1.0 - CFG.navdp_cone_frac_w) / 2.0)
    x1 = int(w * (1.0 + CFG.navdp_cone_frac_w) / 2.0)
    cone = depth_map[y0:h, x0:x1]
    return float(np.percentile(cone, CFG.navdp_cone_percentile))


def _apply_depth_safety(lin: float, ang: float, kind: str, detail: str, depth_map: np.ndarray) -> tuple:
    """Independent, model-agnostic safety override. Runs AFTER the model has
    already decided what it wants to do. Does not touch goal-reached ("stop")
    decisions, since those are the model's own arrival judgment, not a
    collision risk.

    Plain binary hard stop at navdp_estop_distance_m (30cm) -- no gradual
    slowdown zone. No InternNav-published stop-distance applies to our
    forward-facing monocular-camera setup (see NodeConfig's comment above
    navdp_estop_distance_m for why); this is a deliberately simple fallback."""
    if kind == "stop":
        return lin, ang, kind, detail

    obstacle_dist = _forward_cone_min_depth(depth_map)
    if obstacle_dist < CFG.navdp_estop_distance_m:
        return 0.0, 0.0, "obstacle-stop", f"blocked @ {obstacle_dist:.2f}m (model wanted {kind}:{detail})"

    return _apply_reactive_avoid(lin, ang, kind, detail, depth_map)


def _apply_reactive_avoid(lin: float, ang: float, kind: str, detail: str, depth_map: np.ndarray) -> tuple:
    """Classic reactive obstacle-avoidance steering: compare left-half vs
    right-half depth in the same forward cone used for the hard stop, and bias
    angular velocity away from whichever side is closer. Purely geometric (no
    model), separate from and complementary to the hard stop -- that one can
    only brake, this one can actually route around something before it gets
    that close.

    Two noise-mitigations added after a live real-rover test (2026-07-15)
    showed "random" left/right steering on a glossy floor: monocular depth
    read near-identical left/right values there (differing by only
    ~0.01-0.02m -- a floor reflection, not real asymmetric geometry), so
    always trusting whichever side read marginally smaller flipped the steer
    direction on pure noise, tick to tick.
    1. Minimum-difference gate (navdp_avoid_min_diff_m): below this, treat
       left/right as indistinguishable -- ease speed (there IS something
       close ahead) but do NOT guess a turn direction.
    2. Cross-tick damping on the avoid contribution itself, same technique
       already used for trajectory steering -- smooths any residual noise
       even when a real asymmetry legitimately triggers a turn.
    """
    h, w = depth_map.shape
    y0 = int(h * (1.0 - CFG.navdp_cone_frac_h))
    x0 = int(w * (1.0 - CFG.navdp_cone_frac_w) / 2.0)
    x1 = int(w * (1.0 + CFG.navdp_cone_frac_w) / 2.0)
    cone = depth_map[y0:h, x0:x1]
    mid = cone.shape[1] // 2
    left_dist = float(np.percentile(cone[:, :mid], CFG.navdp_cone_percentile))
    right_dist = float(np.percentile(cone[:, mid:], CFG.navdp_cone_percentile))
    closer = min(left_dist, right_dist)

    if closer >= CFG.navdp_avoid_distance_m:
        AGENT["navdp_prev_avoid_ang"] = 0.0
        return lin, ang, kind, detail

    urgency = float(np.clip(1.0 - closer / CFG.navdp_avoid_distance_m, 0.0, 1.0))
    diff = abs(left_dist - right_dist)

    if diff < CFG.navdp_avoid_min_diff_m:
        AGENT["navdp_prev_avoid_ang"] = 0.0
        lin = lin * (1.0 - 0.5 * urgency)
        return lin, ang, kind, (
            f"{detail} | caution L={left_dist:.2f} R={right_dist:.2f} "
            f"(diff<{CFG.navdp_avoid_min_diff_m:.2f}, no turn)"
        )

    # closer==left -> steer right (negative); closer==right -> steer left (positive)
    side_sign = -1.0 if left_dist < right_dist else 1.0
    raw_avoid_ang = side_sign * urgency * CFG.navdp_avoid_gain * CFG.max_angular
    prev_avoid_ang = AGENT.get("navdp_prev_avoid_ang", raw_avoid_ang)
    avoid_ang = (CFG.navdp_avoid_ang_damping * prev_avoid_ang
                 + (1.0 - CFG.navdp_avoid_ang_damping) * raw_avoid_ang)
    AGENT["navdp_prev_avoid_ang"] = avoid_ang

    ang = float(np.clip(ang + avoid_ang, -CFG.max_angular, CFG.max_angular))
    lin = lin * (1.0 - 0.5 * urgency)  # ease off forward speed while dodging
    return lin, ang, kind, f"{detail} | avoid L={left_dist:.2f} R={right_dist:.2f} urgency={urgency:.2f}"


def _trajectory_to_cmd(traj_xy: np.ndarray) -> tuple:
    """Convert a NavDP local-frame xy trajectory (meters, start at origin) into
    a reactive steer: heading toward a lookahead waypoint, forward speed tapers
    with heading error. Mirrors _pixel_to_cmd's style but depth-aware.

    Two noise-mitigations added after observing erratic/wobbly steering in live
    testing (2026-07-14):
    1. Average a WINDOW of trajectory points around the lookahead index rather
       than reading a single point. `traj_to_actions` is monkeypatched (see
       _install_clustered_traj_to_actions) to majority-cluster the 32
       diffusion-SAMPLED trajectories rather than blindly averaging all of
       them, but even within the winning cluster individual samples still
       vary -- and near the trajectory's start dx,dy are still small in
       magnitude, so atan2 amplifies small per-sample noise into large heading
       swings. A window is far less sensitive to this.
    2. Cross-tick exponential damping: every tick re-samples 32 fresh diffusion
       trajectories from scratch (no fixed seed), so even the WINDOWED estimate
       can disagree somewhat tick-to-tick for ambiguous/multimodal scenes. Blend
       with the previous command -- same technique OmniVLA's own controller
       already uses (angular_damping) -- to turn tick-to-tick disagreement into
       smooth correction instead of visible wobble.
    """
    lo = max(0, CFG.navdp_lookahead_idx - CFG.navdp_window)
    hi = min(len(traj_xy) - 1, CFG.navdp_lookahead_idx + CFG.navdp_window)
    window = traj_xy[lo:hi + 1]
    dx, dy = float(np.mean(window[:, 0])), float(np.mean(window[:, 1]))
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return 0.0, 0.0, "trajectory", "no-motion"

    heading = math.atan2(dy, dx)
    raw_ang = float(np.clip(heading * CFG.navdp_turn_gain, -1.0, 1.0)) * CFG.max_angular
    raw_lin = CFG.max_linear * max(0.0, 1.0 - abs(heading) / (math.pi / 2.0))

    prev_ang = AGENT.get("navdp_prev_ang", raw_ang)
    prev_lin = AGENT.get("navdp_prev_lin", raw_lin)
    ang = CFG.navdp_damping * prev_ang + (1.0 - CFG.navdp_damping) * raw_ang
    lin = CFG.navdp_damping * prev_lin + (1.0 - CFG.navdp_damping) * raw_lin
    AGENT["navdp_prev_ang"] = ang
    AGENT["navdp_prev_lin"] = lin

    cluster_info = AGENT.get("navdp_cluster_info", "?")
    return float(lin), float(ang), "trajectory", (
        f"window=({dx:.2f},{dy:.2f})m heading={math.degrees(heading):+.0f}deg cluster={cluster_info}"
    )


def _decayed_heading_bias() -> tuple:
    """Discrete ACT_FWD has no heading feedback of its own. Carry a decaying
    fraction of the last real corrective heading (recorded in infer_cmd()
    whenever a trajectory/pixel tick actually ran) into subsequent blind-FWD
    ticks, rather than assuming perfectly-straight is always correct. See
    NodeConfig's discrete_fwd_bias_decay_s comment for the live observation
    that motivated this."""
    ts = AGENT.get("last_heading_bias_ts")
    if ts is None:
        return 0.0, ""
    elapsed = time.time() - ts
    if elapsed > CFG.discrete_fwd_bias_decay_s * 3:   # fully decayed, don't bother
        return 0.0, ""
    ang = AGENT.get("last_heading_bias_ang", 0.0) * math.exp(-elapsed / CFG.discrete_fwd_bias_decay_s)
    ang = float(np.clip(ang, -CFG.max_angular, CFG.max_angular))
    return ang, f"+bias({ang:+.3f}@{elapsed:.1f}s)"


def _discrete_action_to_cmd(a: int) -> tuple:
    """Shared discrete-action -> cmd_vel mapping, used by both inference paths."""
    if a == ACT_STOP:
        return 0.0, 0.0, "stop", "STOP"
    if a == ACT_FWD:
        bias_ang, bias_note = _decayed_heading_bias()
        return CFG.max_linear, bias_ang, "discrete", f"forward{bias_note}"
    if a == ACT_LEFT:
        return CFG.max_linear * CFG.discrete_turn_frac, +CFG.max_angular, "discrete", "left"
    if a == ACT_RIGHT:
        return CFG.max_linear * CFG.discrete_turn_frac, -CFG.max_angular, "discrete", "right"
    return 0.0, 0.0, "discrete", f"unknown({a})"


def _infer_cmd_system2_only(rgb: np.ndarray, instruction: str) -> tuple:
    """Run System-2 once and turn its output into (lin, ang, kind, detail).

    Returns kind in {"stop","discrete","pixel"} for logging/goal logic.
    """
    agent = AGENT["agent"]
    intrinsic = AGENT["intrinsic"]
    H, W = rgb.shape[:2]
    dummy_depth = np.zeros((H, W), dtype=np.float32)   # never read by step_s2

    action_seq, latent, pixel = agent.step_s2(
        rgb, dummy_depth, np.eye(4), instruction, intrinsic, look_down=False
    )

    # ---- Discrete action branch (pure RGB) ----
    if action_seq is not None and len(action_seq) > 0:
        a = int(action_seq[0])
        if a == ACT_LOOKDOWN:
            # Rover camera can't tilt; re-run the model's look-down refinement
            # on the same frame (matches http_internvla_server behaviour).
            a2, _, px2 = agent.step_s2(rgb, dummy_depth, np.eye(4),
                                       instruction, intrinsic, look_down=True)
            if a2 is not None and len(a2) > 0:
                a = int(a2[0])
            elif px2 is not None:
                return _pixel_to_cmd(px2, W)
        return _discrete_action_to_cmd(a)

    # ---- Pixel-goal branch: reactive visual servo (no depth) ----
    if pixel is not None:
        return _pixel_to_cmd(pixel, W)

    # Nothing usable -> hold.
    return 0.0, 0.0, "discrete", "no-output"


def _pixel_to_cmd(pixel, img_w: int) -> tuple:
    """Convert a VLM pixel goal into a reactive steer.

    We use the horizontal (x/column) component: offset left/right of centre
    -> turn, forward speed tapers with |offset|. Depth-free.

    Axis: step_s2 builds pixel_goal = [coord[1], coord[0]]. InternNav's own
    dialog_agent.py treats pixel_goal AS (x, y) — `x, y, r = pixel_goal[0],
    pixel_goal[1], 2` before a PIL ellipse draw — so pixel_goal[0] is the
    horizontal/column value, NOT index 1.

    Scale: empirically (Isaac Sim test, 2026-07-14) observed pixel_goal values
    ranged ~102-481, exceeding both resize_w=384 and Qwen's own internal
    ~392px smart-resize bound -- so these are NOT raw pixel coordinates in
    either of those spaces. This matches Qwen-VL's standard 0-1000 normalized
    grounding-coordinate convention (the model is Qwen2.5-VL-based), which we
    use here instead.
    """
    x = float(pixel[0])
    ref = 1000.0
    offset = (x - ref / 2.0) / (ref / 2.0)            # -1 (left) .. +1 (right)
    offset = float(np.clip(offset, -1.0, 1.0))
    if abs(offset) < CFG.center_deadband:
        return CFG.max_linear, 0.0, "pixel", f"straight(x={x:.0f})"
    ang = -offset * CFG.max_angular                    # right offset -> turn right (neg)
    lin = CFG.max_linear * CFG.servo_forward_frac * (1.0 - abs(offset))
    return float(lin), float(ang), "pixel", f"servo(x={x:.0f}, off={offset:+.2f})"


# ================================================================
#  Main Zenoh Node
# ================================================================
class InternVLAN1ZenohNode:
    # Real rover firmware zeroes velocity on its own if no /cmd_vel arrives
    # within CMD_VEL_TIMEOUT_MS=500ms -- a real inference tick can take up to
    # ~1.5-1.7s (variable generation length, look-down retries), so publishing
    # only once per tick left gaps past that deadline: the firmware force-
    # stopped the rover mid-tick, then it resumed once the next result
    # published -- observed live as "moves ~2s, stops, moves again". Fixed by
    # a background heartbeat that re-sends the last known command at a fast
    # fixed rate, decoupled from the (slow, variable) inference rate.
    HEARTBEAT_PERIOD_S = 0.15  # well under the firmware's 500ms cmd_vel timeout

    def __init__(self, session: zenoh.Session):
        self.session = session
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.instruction = CFG.instruction
        self._running = True
        self._frame_count = 0
        self._infer_count = 0
        self._stop_streak = 0
        self._goal_reached = False
        self._last_cmd = (0.0, 0.0)

        self.pub_cmd = session.declare_publisher("cmd_vel")
        self.pub_explain = session.declare_publisher("omnivla/explanation")
        self.sub_image = session.declare_subscriber("image_raw", self._on_image)
        self.sub_goal = session.declare_subscriber("omnivla/goal_text", self._on_goal)

        print("[INFO] Zenoh subs: image_raw, omnivla/goal_text  (camera-only, no LiDAR)")
        print("[INFO] Zenoh pubs: cmd_vel, omnivla/explanation")

        self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(self.HEARTBEAT_PERIOD_S)
            with self.lock:
                lin, ang = self._last_cmd
            self.pub_cmd.put(serialize_twist(lin, ang))

    def _on_image(self, sample: zenoh.Sample):
        try:
            rgb = parse_image(bytes(sample.payload))
            if rgb is not None:
                with self.lock:
                    self.latest_rgb = rgb
                    self._frame_count += 1
        except Exception as e:
            print(f"[WARN] image parse failed: {e}")

    def _on_goal(self, sample: zenoh.Sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if text and text != self.instruction:
                print(f"[INFO] New instruction: '{text}'")
                self.instruction = text
                self._goal_reached = False
                self._stop_streak = 0
                AGENT.pop("navdp_prev_ang", None)   # don't carry stale steering into a new goal
                AGENT.pop("navdp_prev_lin", None)
                AGENT.pop("navdp_prev_avoid_ang", None)
                AGENT.pop("stall_prev_frame", None)
                AGENT["stall_count"] = 0
                AGENT["stall_recovering"] = 0
                # smooth_prev_lin/ang/t deliberately NOT reset here: they
                # track the rover's actual physical velocity, which doesn't
                # teleport to zero just because the instruction changed --
                # letting the ramp continue from wherever it is keeps the
                # transition to a new goal smooth instead of snapping.
                if AGENT["agent"] is not None:
                    AGENT["agent"].reset()   # clear history for the new episode
                    AGENT["agent"].save_dir = "/tmp/internvla_n1_runs/" + time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs(AGENT["agent"].save_dir, exist_ok=True)
        except Exception:
            pass

    def publish_cmd(self, lin: float, ang: float):
        with self.lock:
            self._last_cmd = (lin, ang)
        self.pub_cmd.put(serialize_twist(lin, ang))

    def publish_explanation(self, text: str):
        self.pub_explain.put(serialize_string(text))

    def spin(self):
        period = 1.0 / CFG.predict_hz
        last_status = time.time()
        print(f"[INFO] Inference loop at {CFG.predict_hz} Hz")
        print(f"[INFO] Instruction: '{self.instruction}'")
        print("[INFO] Waiting for camera frames via Zenoh...")
        try:
            while self._running:
                t0 = time.time()
                self._tick()
                if time.time() - last_status > 10.0:
                    print(f"[STATUS] frames_rx={self._frame_count} "
                          f"inferences={self._infer_count} "
                          f"goal={'REACHED' if self._goal_reached else 'tracking'}")
                    last_status = time.time()
                dt = period - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
        finally:
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.publish_cmd(0.0, 0.0)
            print("[INFO] Sent zero velocity. Goodbye.")

    def _tick(self):
        with self.lock:
            rgb = self.latest_rgb
        if rgb is None:
            return
        if self._goal_reached:
            self.publish_cmd(0.0, 0.0)
            return

        t0 = time.time()
        lin, ang, kind, detail = infer_cmd(rgb, self.instruction)
        infer_ms = (time.time() - t0) * 1000.0

        # Goal-reached: consecutive STOPs
        if kind == "stop":
            self._stop_streak += 1
            if self._stop_streak >= CFG.stop_confirm_count:
                print(f"\n{'=' * 50}\n  GOAL REACHED: '{self.instruction}'\n{'=' * 50}\n")
                self._goal_reached = True
                self.publish_cmd(0.0, 0.0)
                self.publish_explanation(f"GOAL REACHED: '{self.instruction}'. Stopping.")
                return
        else:
            self._stop_streak = 0

        self.publish_cmd(lin, ang)
        self.publish_explanation(
            f"InternVLA-N1 [{kind}:{detail}] -> lin={lin:.3f} ang={ang:.3f} "
            f"| instruction='{self.instruction}'"
        )
        self._infer_count += 1
        if self._infer_count <= 5 or self._infer_count % 20 == 0:
            print(f"[PRED #{self._infer_count}] {kind}:{detail} "
                  f"lin={lin:.3f} ang={ang:.3f} ({infer_ms:.0f}ms)")


# ================================================================
#  CLI & entry point
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="InternVLA-N1 Zenoh inference node (GPU side, camera-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pi-ip", type=str, default=None,
                   help="Pi/Isaac IP for explicit Zenoh peer; omit for multicast scouting")
    p.add_argument("--instruction", type=str, default=CFG.instruction)
    p.add_argument("--predict-hz", type=float, default=CFG.predict_hz)
    p.add_argument("--max-linear", type=float, default=CFG.max_linear)
    p.add_argument("--max-angular", type=float, default=CFG.max_angular)
    p.add_argument("--model-path", type=str, default=CFG.model_path)
    p.add_argument("--internnav-repo", type=str, default=CFG.internnav_repo)
    p.add_argument("--use-navdp", action="store_true",
                   help="Opt-in depth-conditioned obstacle avoidance (InternVLA-N1-w-NavDP + "
                        "monocular depth estimation). Default OFF: System-2-only, unchanged behaviour.")
    p.add_argument("--navdp-model-path", type=str, default=CFG.navdp_model_path)
    return p.parse_args()


def main():
    args = parse_args()
    CFG.instruction = args.instruction
    CFG.predict_hz = args.predict_hz
    CFG.max_linear = args.max_linear
    CFG.max_angular = args.max_angular
    CFG.model_path = args.model_path
    CFG.internnav_repo = args.internnav_repo
    CFG.use_navdp = args.use_navdp
    CFG.navdp_model_path = args.navdp_model_path

    load_model()

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting (auto-discover)")
    session = zenoh.open(config)
    print("[INFO] Zenoh session opened.")

    node = InternVLAN1ZenohNode(session)
    node.spin()
    session.close()


if __name__ == "__main__":
    main()
