#!/usr/bin/env python3
"""DINO + NavDP Mars GUI variant — detect every frame, but the GOAL is locked
to the belief, not to whatever DINO's best box is this tick.

Reuses mars_gui.py's SharedState, tkinter App, Zenoh wiring, and inference
loop unchanged (same layout, same contract, same buttons/presets) but swaps
in BeliefOnlyPipeline: Grounding DINO + SAM still run every frame (unlike the
original detect-once design this replaced), but a fresh detection is only
allowed to move the goal if it lands within ``distractor_gate_m`` of where the
locked-on target is currently believed to be (its ego-motion-propagated
position — see mars_gui.pose_odom_delta, the same SE(2) propagation math
verified to agree with mars_gui.robot_frame_goal to ~1e-6). A detection
outside that radius is a different object that happens to match the same
text query (e.g. another stone) — it is NOT fed into the belief; that tick
instead falls back to pure ego-motion propagation, exactly as if the target
had been briefly occluded.

This means the belief (navdp.extensions.SubgoalBeliefBank), not the raw
per-frame detection, is the single source of truth for the goal once a
target has been acquired: a nearer/more-confident distractor of the same
class can never hijack it, and continuous re-detection of the SAME object
still keeps refreshing the belief's confidence/precision every frame instead
of relying solely on propagation.

Before any target has ever been acquired, every frame's detection is a
candidate first fix (there is nothing yet to gate against), so the rover
just rotates in place (SEARCH) until DINO finds something.

Run (from Nav_new root, internnav conda env, habitat_sim_node running):
    python MARS/scripts/mars_belief_only_gui.py [--target "big stone"] [--distractor-gate 1.5]
"""

import argparse
import signal
import time
from threading import Thread
from typing import Optional

import numpy as np
import torch

import tkinter as tk

from mars_gui import (  # noqa: E402
    App,
    GuardConfig,
    MarsPipeline,
    PipelineConfig,
    SharedState,
    StepResult,
    apply_avoid_cooldown,
    bearing_to_angular,
    depth_to_obstacle_points,
    forward_guard,
    goal_point_from_detection,
    heartbeat_loop,
    inference_loop,
    intrinsics_from_fov,
    mask_centroid,
    mask_median_depth,
    pixel_depth_to_point,
    pose_odom_delta,
    preprocess_depth,
    preprocess_rgb,
    serialize_twist,
    swept_clearance,
    zenoh_setup,
)
from navdp.extensions.belief_bank import ego_motion_update  # noqa: E402

import zenoh  # noqa: E402

DISTRACTOR_GATE_M = 1.5   # detections farther than this from the belief are ignored as distractors


# ------------------------------------------------------------------------- #
class BeliefOnlyPipeline(MarsPipeline):
    """MarsPipeline, but the goal is always the belief's mean, and a fresh
    detection only ever gets to update that mean if it's within
    ``distractor_gate_m`` of where the tracked target is currently predicted
    to be. See module docstring for the full rationale.
    """

    def __init__(self, cfg: PipelineConfig = PipelineConfig(), use_depth_estimator: bool = True,
                 distractor_gate_m: float = DISTRACTOR_GATE_M):
        # belief_confidence_min is unused by this pipeline's _step_inner
        # (gating, not confidence, decides whether to trust a detection) --
        # passed through only because MarsPipeline.__init__ requires it.
        super().__init__(cfg, use_depth_estimator=use_depth_estimator, belief_confidence_min=0.0)
        self.distractor_gate_m = float(distractor_gate_m)

    def _step_inner(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None,
                     pose: Optional[tuple] = None) -> StepResult:
        # `pose` is accepted only for signature parity with
        # DinoNavDPPipeline.step()/_step_inner(); this subclass sources pose
        # from self._pose_for_tick, set via set_pose() before each step().
        cfg = self.cfg
        res = StepResult()
        H, W = rgb.shape[:2]
        timing = {}

        t0 = time.time()
        det = self.detector.detect_best(rgb, target_text)
        timing["dino"] = time.time() - t0

        t0 = time.time()
        if depth is None:
            if self.depther is None:
                raise ValueError("no depth given and depth estimator disabled")
            depth = self.depther.estimate(rgb)
        timing["depth"] = time.time() - t0

        rgb_p, dep_p = preprocess_rgb(rgb), preprocess_depth(depth)
        self._memory.append(rgb_p)
        self._memory_d.append(dep_p)
        self._memory = self._memory[-self._memory_size:]
        self._memory_d = self._memory_d[-self._memory_size:]

        fx, fy, cx, cy = intrinsics_from_fov(W, H, cfg.horizontal_fov_deg)

        detected_goal = None
        close_by_size = False
        if det is not None:
            res.detection = det
            if self.segmenter is not None:
                t1 = time.time()
                mask = self.segmenter.segment_box(rgb, det.box)
                timing["sam"] = time.time() - t1
                if mask is not None:
                    res.mask = mask
                    d = mask_median_depth(depth, mask)
                    if d is not None:
                        u, v = mask_centroid(mask)
                        detected_goal = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
                    close_by_size = mask.mean() > cfg.mask_stop_frac
            if detected_goal is None:  # SAM disabled, empty mask, or no valid mask depth
                detected_goal = goal_point_from_detection(det.box, depth, fx, fy, cx, cy)
                close_by_size = (det.box[3] - det.box[1]) / H > cfg.bbox_stop_frac

        # --- odometry since the last tick (needed every tick, for both the
        # gate prediction below and belief propagation) --------------------- #
        pose = self._pose_for_tick
        odom = [0.0, 0.0, 0.0]
        if pose is not None and self._belief_pose is not None:
            odom = pose_odom_delta(self._belief_pose, pose)
        if pose is not None:
            self._belief_pose = pose

        slot = self._belief.get("target")
        # where the locked-on target should be THIS tick if it's still the
        # same object -- must propagate before gating, or a detection would
        # be compared against a stale, previous-tick-frame estimate
        predicted_mu = ego_motion_update(slot.mu, odom) if slot.initialized else slot.mu

        accept = False
        if detected_goal is not None:
            if not slot.initialized:
                accept = True  # nothing locked on yet -- this detection IS the goal now
            else:
                gate_dist = float(np.linalg.norm(
                    np.asarray(detected_goal[:2], dtype=np.float32) - predicted_mu[:2]
                ))
                accept = gate_dist <= self.distractor_gate_m

        if accept:
            self._belief.update(
                {"target": {"visible": True, "position": detected_goal[:2], "confidence": float(det.score)}},
                odom_delta=odom, step=self._belief_tick,
            )
        else:
            # nothing detected, or what was detected is a distractor (outside
            # the gate) -- propagate the locked-on target by odometry only,
            # exactly as if it had been occluded this tick
            self._belief.update({"target": {"visible": False}}, odom_delta=odom, step=self._belief_tick)
        self._belief_tick += 1
        slot = self._belief.get("target")
        res.belief_confidence = slot.confidence
        res.belief_used = (not accept) and slot.initialized

        if not slot.initialized:
            # never yet acquired a target -- nothing to steer toward
            res.state = "SEARCH"
            res.angular = cfg.search_angular
            res.timing = timing
            return res

        goal = np.array([slot.mu[0], slot.mu[1], 0.0], dtype=np.float32)
        if np.linalg.norm(goal[:2]) < cfg.stop_distance or (accept and close_by_size):
            res.state = "STOP"
            res.goal_point = goal
            res.timing = timing
            return res

        res.goal_point = goal
        res.state = "TRACK"

        # --- obstacle guard (model-agnostic, same as MarsPipeline) ------- #
        obstacle_pts = None
        if cfg.avoid_enabled:
            t0 = time.time()
            obstacle_pts = depth_to_obstacle_points(depth, fx, fy, cx, cy, cfg.guard, exclude_mask=res.mask)
            res.obstacle_points = obstacle_pts
            min_fwd, escape = forward_guard(obstacle_pts, cfg.guard)
            res.min_forward = min_fwd
            timing["guard"] = time.time() - t0
            if min_fwd < cfg.guard.hard_stop_dist:
                self._avoid_streak += 1
                if self._avoid_streak >= cfg.avoid_confirm_ticks:
                    res.state = "AVOID"
                    res.linear = -0.5 * cfg.max_linear if min_fwd < cfg.guard.reverse_dist else 0.0
                    res.angular = escape * cfg.max_angular
                    self._avoid_side = escape
                    self._avoid_cooldown = cfg.avoid_cooldown_ticks
                    res.timing = timing
                    return res
            else:
                self._avoid_streak = 0

        # --- NavDP -------------------------------------------------------- #
        t0 = time.time()
        M = self._memory_size
        frames = np.stack(self._memory)
        if frames.shape[0] < M:
            pad = np.zeros((M - frames.shape[0],) + frames.shape[1:], dtype=frames.dtype)
            frames = np.concatenate([pad, frames], axis=0)
        images = torch.from_numpy(frames).unsqueeze(0)
        if cfg.policy_type == "crossmodal":
            depths = torch.from_numpy(dep_p).unsqueeze(0).unsqueeze(0)
        else:
            dframes = np.stack(self._memory_d)
            if dframes.shape[0] < M:
                pad = np.zeros((M - dframes.shape[0],) + dframes.shape[1:], dtype=dframes.dtype)
                dframes = np.concatenate([pad, dframes], axis=0)
            depths = torch.from_numpy(dframes).unsqueeze(0)
        goal_native = np.array([goal[0], self._goal_y_sign * goal[1], goal[2]], dtype=np.float32)
        trajs, critic = self.policy.sample_pointgoal(
            goal_native.reshape(1, 3), images, depths, sample_num=cfg.sample_num
        )
        trajs = trajs.cpu().numpy()
        critic = critic.cpu().numpy()
        timing["navdp"] = time.time() - t0

        clearances = swept_clearance(trajs, obstacle_pts) if obstacle_pts is not None else None
        idx = self._select_trajectory(trajs, critic, goal, clearances)
        chosen = trajs[idx]
        res.trajectory = chosen
        res.all_trajectories = trajs
        res.critic = critic

        if res.min_forward < cfg.guard.slow_dist:
            res.linear, res.angular = self._command_from_trajectory(chosen, res.min_forward)
            res.linear *= max(0.25, res.min_forward / cfg.guard.slow_dist)
        else:
            bearing = float(np.arctan2(goal[1], goal[0]))
            res.angular = bearing_to_angular(
                bearing, cfg.max_angular, cfg.ang_min_cmd,
                cfg.servo_deadband, np.radians(cfg.servo_ramp_deg),
            )
            res.linear = cfg.max_linear * max(0.2, 1.0 - 0.8 * abs(res.angular) / cfg.max_angular)
        res.angular, self._avoid_cooldown = apply_avoid_cooldown(
            res.angular, res.state, self._avoid_side, self._avoid_cooldown,
            cfg.avoid_bias_gain, cfg.max_angular,
        )
        res.timing = timing
        return res


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Nav_new Mars GUI (Habitat) — goal locked to belief, distractors gated out"
    )
    ap.add_argument("--target", default="big stone")
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5)
    ap.add_argument("--max-angular", type=float, default=0.4)
    ap.add_argument("--invert-angular", action="store_true")
    ap.add_argument("--distractor-gate", type=float, default=DISTRACTOR_GATE_M,
                     help="meters: detections farther than this from the belief's predicted "
                          "target position are treated as distractors and ignored")
    ap.add_argument("--max-climb-deg", type=float, default=GuardConfig().max_climb_deg,
                     help="terrain rising up to this many degrees is treated as driveable ground, "
                          "not an obstacle -- raise this if the rover balks at climbing real slopes/hills, "
                          "lower it if it's driving over things it shouldn't")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = BeliefOnlyPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(0.15, args.max_angular),
        invert_angular=args.invert_angular,
        guard=GuardConfig(max_climb_deg=args.max_climb_deg),
    ), distractor_gate_m=args.distractor_gate)

    session = zenoh.open(zenoh.Config())
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    _subs, pubs = zenoh_setup(session, st)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz), daemon=True).start()

    root = tk.Tk()
    App(root, st, pubs)
    root.title(f"Nav_new — Mars DINO+NavDP (belief-locked goal, gate {args.distractor_gate:.1f}m)")

    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, root.destroy))

    try:
        root.mainloop()
    finally:
        running["on"] = False
        time.sleep(0.2)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        time.sleep(0.1)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        session.close()
        print("[INFO] zero velocity sent, session closed")


if __name__ == "__main__":
    main()
