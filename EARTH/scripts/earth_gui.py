#!/usr/bin/env python3
"""DINO + NavDP Earth GUI — control panel for the Habitat Indian Bend & Pima sim.

Same layout and Zenoh contract as nav_pipeline/isaac_gui.py and
MARS/scripts/mars_gui.py (which it is based on), with commands tailored to
this real-world scene (a Sketchfab photogrammetry scan of a road/roundabout,
parking lots, a stucco building, a Target pylon sign, and an active dirt/
construction lot — see EARTH/scripts/survey.py for how these were found):

- Earth object presets ("yellow building", "target sign", ...) -> DINO text goals
- "Random goal": picks a random world point 4-8 m ahead (within +/-60 deg)
  and drives NavDP point-goal directly — no detection involved. Uses the
  ground-truth pose the habitat node publishes on earth/pose.
- "Go home": point-goal back to the spawn position.
- "Reset rover": teleports the sim rover to its start pose (earth/reset).

Run (from Nav_new root, internnav conda env, habitat_sim_node running):
    python EARTH/scripts/earth_gui.py [--target "yellow building"]
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from threading import Lock, Thread
from typing import Optional

import numpy as np

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

import torch

NAV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, NAV_ROOT)
# Reuses the NavDP policy + belief-bank code cloned for MARS rather than a
# second copy — it's generic, not Mars-specific.
NAVDP_ROOT = os.path.abspath(os.path.join(NAV_ROOT, "MARS", "mars-habitatsim", "navdp"))
sys.path.insert(0, NAVDP_ROOT)

from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig, StepResult  # noqa: E402
from nav_pipeline.goal_utils import (  # noqa: E402
    goal_point_from_detection,
    intrinsics_from_fov,
    pixel_depth_to_point,
    preprocess_depth,
    preprocess_rgb,
)
from nav_pipeline.obstacle_guard import (  # noqa: E402
    GuardConfig,
    apply_avoid_cooldown,
    depth_to_obstacle_points,
    forward_guard,
    swept_clearance,
)
from nav_pipeline.sam_segmenter import mask_centroid, mask_median_depth  # noqa: E402
from nav_pipeline.zenoh_node import (  # noqa: E402
    CAMERA_COMPRESSED_KEYS,
    CAMERA_KEYS,
    DEPTH_KEYS,
    parse_compressed_image,
    parse_image,
    parse_string,
    serialize_path,
    serialize_string,
    serialize_twist,
)
from navdp.extensions.belief_bank import SubgoalBeliefBank  # noqa: E402

# Picked from a manual survey of the scan (EARTH/scripts/survey.py output):
# a tan stucco building, a red "Target" pylon sign, cars along the road/lots,
# construction dirt mounds, a parked excavator, and landscaping shrubs.
PRESETS = ["yellow building", "target sign", "parked car", "sand mound", "excavator", "bush"]
HEARTBEAT_PERIOD_S = 0.15
DEPTH_STALE_S = 1.0
BELIEF_CONFIDENCE_MIN = 0.15    # target-out-of-view: below this, belief is too stale to steer on -> SEARCH
POSE_STALE_S = 2.0
POINT_GOAL_REACHED_M = 0.7      # ground-truth distance at which a point goal counts as reached
RANDOM_BEARING_DEG = 60.0
RANDOM_DIST_RANGE = (4.0, 8.0)
# Surveyed drivable footprint (see habitat_sim_node.py) -- asymmetric, unlike
# Marsyard's origin-centered square yard, so goals are clamped per-axis.
WORLD_X_LIMIT = (-47.5, 90.5)
WORLD_Z_LIMIT = (-152.8, 161.1)


# ------------------------------------------------------------------------- #
class EarthPipeline(DinoNavDPPipeline):
    """Adds a detection-free point-goal step (goal given in the robot frame,
    x forward / y left, meters) reusing the parent's guard + NavDP + servo.

    Also replaces the base pipeline's DINO-target goal memory (a single frozen
    robot-frame vector, held for a fixed frame count once the target drops out
    of view) with a persistent belief (navdp.extensions.SubgoalBeliefBank):
    while unseen, the last observed target position is propagated by the
    rover's own ego-motion (from earth/pose) and its confidence decays smoothly,
    instead of the estimate silently going stale as the rover keeps moving.
    Recovery falls back to SEARCH once that confidence decays below
    ``belief_confidence_min``, rather than after a fixed number of frames.
    """

    def __init__(self, cfg: PipelineConfig = PipelineConfig(), use_depth_estimator: bool = True,
                 belief_confidence_min: float = BELIEF_CONFIDENCE_MIN):
        super().__init__(cfg, use_depth_estimator=use_depth_estimator)
        self.belief_confidence_min = float(belief_confidence_min)
        self._belief = SubgoalBeliefBank(["target"], dim=2, sigma_visible=0.05, odom_noise=0.02)
        self._belief_tick = 0
        self._belief_pose: Optional[dict] = None   # last earth/pose seen by the belief update
        self._pose_for_tick: Optional[dict] = None  # set by set_pose() before each step()

    def set_pose(self, pose: Optional[dict]) -> None:
        """Feed the current earth/pose (ground-truth rover pose) for the next
        step()'s belief propagation. Pass None if pose is missing/stale --
        the belief update then skips ego-motion (falls back to the base
        pipeline's frozen-vector behavior for that tick)."""
        self._pose_for_tick = pose

    def reset(self):
        super().reset()
        self._belief.reset()
        self._belief_tick = 0
        self._belief_pose = None

    def step_point(self, rgb: np.ndarray, goal_xy, depth: Optional[np.ndarray] = None) -> StepResult:
        res = self._step_point_inner(rgb, goal_xy, depth)
        # same command shaping as DinoNavDPPipeline.step()
        if res.state in ("TRACK", "AVOID", "SEARCH") and abs(res.angular) > 0.01:
            boosted = abs(res.angular) + self.cfg.ang_boost
            boosted = max(boosted, self.cfg.ang_min_cmd)
            res.angular = float(np.clip(np.copysign(boosted, res.angular),
                                        -self.cfg.max_angular, self.cfg.max_angular))
        if self.cfg.invert_angular:
            res.angular = -res.angular
        a = self.cfg.smoothing
        if a > 0 and res.state != "STOP":
            pl, pa = self._prev_cmd
            res.linear = (1 - a) * res.linear + a * pl
            res.angular = (1 - a) * res.angular + a * pa
        self._prev_cmd = (res.linear, res.angular)
        return res

    def _step_point_inner(self, rgb, goal_xy, depth) -> StepResult:
        cfg = self.cfg
        res = StepResult()
        H, W = rgb.shape[:2]
        timing = {}

        t0 = time.time()
        if depth is None:
            depth = self.depther.estimate(rgb)
        timing["depth"] = time.time() - t0

        rgb_p, dep_p = preprocess_rgb(rgb), preprocess_depth(depth)
        self._memory.append(rgb_p)
        self._memory_d.append(dep_p)
        self._memory = self._memory[-self._memory_size:]
        self._memory_d = self._memory_d[-self._memory_size:]

        goal = np.array([goal_xy[0], goal_xy[1], 0.0], dtype=np.float32)
        res.goal_point = goal
        res.state = "TRACK"
        if np.linalg.norm(goal[:2]) < cfg.stop_distance:
            res.state = "STOP"
            res.timing = timing
            return res

        fx, fy, cx, cy = intrinsics_from_fov(W, H, cfg.horizontal_fov_deg)

        obstacle_pts = None
        if cfg.avoid_enabled:
            t0 = time.time()
            obstacle_pts = depth_to_obstacle_points(depth, fx, fy, cx, cy, cfg.guard)
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
            if abs(bearing) < cfg.servo_deadband:
                res.angular = 0.0
                res.linear = cfg.max_linear
            else:
                res.angular = float(np.clip(cfg.kp_angular * bearing,
                                            -cfg.max_angular, cfg.max_angular))
                res.linear = cfg.max_linear * max(0.2, 1.0 - 0.8 * abs(res.angular) / cfg.max_angular)
        res.angular, self._avoid_cooldown = apply_avoid_cooldown(
            res.angular, res.state, self._avoid_side, self._avoid_cooldown,
            cfg.avoid_bias_gain, cfg.max_angular,
        )
        res.timing = timing
        return res

    def _step_inner(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None) -> StepResult:
        """Same as DinoNavDPPipeline._step_inner, except the DINO/SAM goal
        point feeds a persistent SubgoalBeliefBank instead of a frozen
        last-seen vector: while the target is out of view its estimated
        position is propagated by the rover's own ego-motion (earth/pose) and
        its confidence decays, so SEARCH only kicks in once that confidence
        drops below belief_confidence_min (see class docstring)."""
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
        goal = None
        close_by_size = False
        if det is not None:
            res.detection = det
            if self.segmenter is not None:
                t0 = time.time()
                mask = self.segmenter.segment_box(rgb, det.box)
                timing["sam"] = time.time() - t0
                if mask is not None:
                    res.mask = mask
                    d = mask_median_depth(depth, mask)
                    if d is not None:
                        u, v = mask_centroid(mask)
                        goal = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
                    close_by_size = mask.mean() > cfg.mask_stop_frac
            if goal is None:  # SAM disabled, empty mask, or no valid mask depth
                goal = goal_point_from_detection(det.box, depth, fx, fy, cx, cy)
                close_by_size = (det.box[3] - det.box[1]) / H > cfg.bbox_stop_frac

        # --- belief update: ego-motion-propagated goal memory --------------- #
        pose = self._pose_for_tick
        odom = [0.0, 0.0, 0.0]
        if pose is not None and self._belief_pose is not None:
            odom = pose_odom_delta(self._belief_pose, pose)
        if pose is not None:
            self._belief_pose = pose
        if goal is not None:
            obs = {"target": {"visible": True, "position": goal[:2], "confidence": float(det.score)}}
        else:
            obs = {"target": {"visible": False}}
        self._belief.update(obs, odom_delta=odom, step=self._belief_tick)
        self._belief_tick += 1
        slot = self._belief.get("target")
        res.belief_confidence = slot.confidence
        res.belief_used = False

        if goal is not None:
            self._last_goal, self._lost_count = goal, 0
            if np.linalg.norm(goal[:2]) < cfg.stop_distance or close_by_size:
                res.state = "STOP"
                res.goal_point = goal
                res.timing = timing
                return res
        else:
            self._lost_count += 1
            if pose is None:
                # no odometry to propagate with this tick -- degrade to the
                # base pipeline's frozen-vector behavior exactly
                if self._last_goal is not None and self._lost_count <= cfg.lost_patience:
                    goal = self._last_goal
            elif slot.initialized and slot.confidence >= self.belief_confidence_min:
                # target out of view, but the belief is still confident enough
                # to steer on -- use the ego-motion-propagated estimate
                goal = np.array([slot.mu[0], slot.mu[1], 0.0], dtype=np.float32)
                res.belief_used = True
            if goal is None:
                # search: rotate in place toward the last known side
                res.state = "SEARCH"
                side = 1.0
                if self._last_goal is not None and self._last_goal[1] < 0:
                    side = -1.0
                res.angular = side * cfg.search_angular
                res.timing = timing
                return res

        res.goal_point = goal
        res.state = "TRACK"

        # --- obstacle guard --------------------------------------------- #
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

        # --- NavDP ------------------------------------------------------ #
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
            if abs(bearing) < cfg.servo_deadband:
                res.angular = 0.0
                res.linear = cfg.max_linear
            else:
                res.angular = float(np.clip(cfg.kp_angular * bearing,
                                            -cfg.max_angular, cfg.max_angular))
                res.linear = cfg.max_linear * max(0.2, 1.0 - 0.8 * abs(res.angular) / cfg.max_angular)
        res.angular, self._avoid_cooldown = apply_avoid_cooldown(
            res.angular, res.state, self._avoid_side, self._avoid_cooldown,
            cfg.avoid_bias_gain, cfg.max_angular,
        )
        res.timing = timing
        return res


# ------------------------------------------------------------------------- #
class SharedState:
    def __init__(self, target: str):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.frame_count = 0
        self.mode = "text"                      # "text" | "point"
        self.target = target
        self.world_goal = None                  # (gx, gz) habitat world, point mode
        self.pose = None                        # {"x","z","yaw","t"} from earth/pose
        self.pose_t = 0.0
        self.home = None                        # first pose seen = spawn
        self.stopped = True                     # start idle until a command is given
        self.goal_reached = False
        self.reset_pipeline = False
        self.last_cmd = (0.0, 0.0)
        # display
        self.display_rgb: Optional[np.ndarray] = None
        self.detection = None
        self.mask: Optional[np.ndarray] = None
        self.state_text = "waiting for camera"
        self.vel_text = "lin 0.000  ang +0.000"
        self.lat_text = ""
        self.trajs = None
        self.chosen = None
        self.goal_pt = None
        self.obstacles = None
        self.min_forward = float("inf")
        self.infer_count = 0


def _forward_left(yaw: float):
    """Habitat world (x, z) axes of this rover's forward/left at a given yaw
    (yaw=0 faces -Z, +yaw turns left -- see habitat_sim_node.py)."""
    return (-math.sin(yaw), -math.cos(yaw)), (-math.cos(yaw), math.sin(yaw))


def robot_frame_goal(pose: dict, world_goal) -> np.ndarray:
    """Habitat world (x, z, yaw) + world goal -> robot frame [x fwd, y left]."""
    dx = world_goal[0] - pose["x"]
    dz = world_goal[1] - pose["z"]
    fwd, left = _forward_left(pose["yaw"])
    return np.array([dx * fwd[0] + dz * fwd[1], dx * left[0] + dz * left[1]], dtype=np.float32)


def pose_odom_delta(prev_pose: dict, cur_pose: dict) -> list:
    """Body-frame ego-motion [dx fwd, dy left, dtheta] between two earth/pose
    readings, in the same (fwd, left, yaw) convention as robot_frame_goal --
    the frame a SubgoalBeliefBank target estimate is expressed in. Feeds
    SubgoalBeliefBank.update(odom_delta=...) so an out-of-view target's
    stored position keeps tracking the world as the rover moves."""
    dx = cur_pose["x"] - prev_pose["x"]
    dz = cur_pose["z"] - prev_pose["z"]
    fwd, left = _forward_left(prev_pose["yaw"])
    dbx = dx * fwd[0] + dz * fwd[1]
    dby = dx * left[0] + dz * left[1]
    dtheta = (cur_pose["yaw"] - prev_pose["yaw"] + math.pi) % (2 * math.pi) - math.pi
    return [dbx, dby, dtheta]


def zenoh_setup(session: zenoh.Session, st: SharedState):
    def on_image(sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 3:
            with st.lock:
                st.latest_rgb = img
                st.frame_count += 1

    def on_compressed(sample):
        img = parse_compressed_image(bytes(sample.payload))
        if img is not None:
            with st.lock:
                st.latest_rgb = img
                st.frame_count += 1

    def on_depth(sample):
        d = parse_image(bytes(sample.payload))
        if d is not None and d.ndim == 2:
            with st.lock:
                st.latest_depth = d
                st.latest_depth_t = time.time()

    def on_pose(sample):
        try:
            p = json.loads(parse_string(bytes(sample.payload)))
            with st.lock:
                st.pose = p
                st.pose_t = time.time()
                if st.home is None:
                    st.home = (p["x"], p["z"])
        except Exception:
            pass

    subs = (
        [session.declare_subscriber(k, on_image) for k in CAMERA_KEYS]
        + [session.declare_subscriber(k, on_compressed) for k in CAMERA_COMPRESSED_KEYS]
        + [session.declare_subscriber(k, on_depth) for k in DEPTH_KEYS]
        + [session.declare_subscriber("earth/pose", on_pose)]
    )
    pubs = {
        "cmd": session.declare_publisher("cmd_vel"),
        "explain": session.declare_publisher("omnivla/explanation"),
        "path": session.declare_publisher("omnivla/waypoints"),
        "reset": session.declare_publisher("earth/reset"),
    }
    return subs, pubs


def heartbeat_loop(st: SharedState, pubs, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = st.last_cmd
        pubs["cmd"].put(serialize_twist(lin, ang))


def inference_loop(pipe: EarthPipeline, st: SharedState, pubs, running,
                   predict_hz: float, stop_confirm: int = 3):
    period = 1.0 / predict_hz
    stop_streak = 0
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            mode, target, world_goal = st.mode, st.target, st.world_goal
            pose, pose_age = st.pose, time.time() - st.pose_t
            paused = st.stopped or st.goal_reached
            do_reset = st.reset_pipeline
            st.reset_pipeline = False
        if do_reset:
            pipe.reset()
            stop_streak = 0
        if rgb is None:
            time.sleep(0.1)
            continue
        if paused:
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.1)
            continue
        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None

        goal_label = f"'{target}'"
        try:
            if mode == "point":
                if world_goal is None or pose is None or pose_age > POSE_STALE_S:
                    with st.lock:
                        st.state_text = "POINT: waiting for earth/pose"
                        st.last_cmd = (0.0, 0.0)
                    time.sleep(0.2)
                    continue
                goal_xy = robot_frame_goal(pose, world_goal)
                goal_label = f"point ({world_goal[0]:.1f}, {world_goal[1]:.1f})"
                res = pipe.step_point(rgb, goal_xy, depth=depth)
                # ground truth beats pixel geometry for point goals
                if float(np.linalg.norm(goal_xy)) < POINT_GOAL_REACHED_M:
                    res.state = "STOP"
            else:
                pipe.set_pose(pose if pose is not None and pose_age <= POSE_STALE_S else None)
                res = pipe.step(rgb, target, depth=depth)
        except Exception as e:
            print(f"[ERROR] pipeline step: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
            time.sleep(0.5)
            continue

        if res.state == "STOP":
            stop_streak += 1
        else:
            stop_streak = 0
        reached = stop_streak >= (1 if mode == "point" else stop_confirm)

        with st.lock:
            st.display_rgb = rgb
            st.detection = res.detection
            st.mask = res.mask
            st.trajs = res.all_trajectories
            st.chosen = res.trajectory
            st.goal_pt = res.goal_point
            st.obstacles = res.obstacle_points
            st.min_forward = res.min_forward
            st.infer_count += 1
            if reached:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.state_text = f"GOAL REACHED: {goal_label}"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                if mode == "text":
                    st.state_text = res.state
                    if getattr(res, "belief_used", False):
                        st.state_text += f" (belief mem, conf {res.belief_confidence:.2f})"
                else:
                    st.state_text = f"POINT {res.state}"
                st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v*1000:.0f}ms" for k, v in res.timing.items())

        if reached:
            pubs["explain"].put(serialize_string(f"GOAL REACHED: {goal_label}. Stopping."))
        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        score = f"{res.detection.score:.2f}" if res.detection else "-"
        pubs["explain"].put(serialize_string(
            f"EARTH DINO+NavDP [{res.state}] det={score} -> lin={res.linear:.3f} "
            f"ang={res.angular:.3f} | goal={goal_label}"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448
    PLOT_SIZE = 448
    PLOT_RANGE = 3.5

    def __init__(self, root: tk.Tk, st: SharedState, pubs):
        self.root = root
        self.st = st
        self.pubs = pubs
        root.title("Nav_new — Earth DINO + NavDP (Habitat)")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        # Seed a blank image so the label already occupies its final size —
        # otherwise the column is 0-width until the first camera frame
        # arrives, and the window jumps when it does.
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)
        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="Target:").pack(side="left")
        self.entry = ttk.Entry(bar, width=30)
        self.entry.insert(0, st.target)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", lambda e: self.send_target())
        ttk.Button(bar, text="Send", command=self.send_target).pack(side="left", padx=2)
        ttk.Button(bar, text="Random goal", command=self.random_goal).pack(side="left", padx=8)
        ttk.Button(bar, text="Go home", command=self.go_home).pack(side="left", padx=2)
        ttk.Button(bar, text="Reset rover", command=self.reset_rover).pack(side="left", padx=8)
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        presets = ttk.Frame(main)
        presets.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(presets, text="Go to:").pack(side="left")
        for p in PRESETS:
            ttk.Button(presets, text=p, command=lambda t=p: self.send_target(t)).pack(side="left", padx=2)

        # Fixed character width on the two dynamic-text rows: their content
        # (state name, counters, latency numbers) changes length on every
        # refresh tick, and an unconstrained Label makes the whole window
        # resize to match on every tick.
        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=110, anchor="w")
        self.status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=110, anchor="w")
        self.info.grid(row=4, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(66, self.refresh)

    # ---------------- commands ---------------- #
    def send_target(self, text: Optional[str] = None):
        t = text if text is not None else self.entry.get().strip()
        if not t:
            return
        if text is not None:
            self.entry.delete(0, "end")
            self.entry.insert(0, t)
        with self.st.lock:
            self.st.mode = "text"
            self.st.target = t
            self.st.world_goal = None
            self.st.stopped = False
            self.st.goal_reached = False
            self.st.reset_pipeline = True

    def random_goal(self):
        with self.st.lock:
            pose = self.st.pose
        if pose is None:
            self.status.configure(text="no earth/pose yet — is habitat_sim_node running?")
            return
        rel = math.radians(random.uniform(-RANDOM_BEARING_DEG, RANDOM_BEARING_DEG))
        dist = random.uniform(*RANDOM_DIST_RANGE)
        theta = pose["yaw"] + rel
        gx = float(np.clip(pose["x"] - dist * math.sin(theta), *WORLD_X_LIMIT))
        gz = float(np.clip(pose["z"] - dist * math.cos(theta), *WORLD_Z_LIMIT))
        with self.st.lock:
            self.st.mode = "point"
            self.st.world_goal = (gx, gz)
            self.st.stopped = False
            self.st.goal_reached = False
            self.st.reset_pipeline = True

    def go_home(self):
        with self.st.lock:
            home = self.st.home
        if home is None:
            self.status.configure(text="no earth/pose yet — is habitat_sim_node running?")
            return
        with self.st.lock:
            self.st.mode = "point"
            self.st.world_goal = home
            self.st.stopped = False
            self.st.goal_reached = False
            self.st.reset_pipeline = True

    def reset_rover(self):
        self.pubs["reset"].put(serialize_string(""))
        with self.st.lock:
            self.st.stopped = True
            self.st.world_goal = None
            self.st.last_cmd = (0.0, 0.0)

    def stop(self):
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    # ------------------------------------------------------------------ #
    def refresh(self):
        if self.closed:
            return
        with self.st.lock:
            rgb = self.st.display_rgb if self.st.display_rgb is not None else self.st.latest_rgb
            det = self.st.detection
            mask = self.st.mask
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers = self.st.frame_count, self.st.infer_count
            mode, target, world_goal = self.st.mode, self.st.target, self.st.world_goal
            pose = self.st.pose
            stopped = self.st.stopped

        if rgb is not None:
            frame = rgb
            if mask is not None and mask.shape[:2] == rgb.shape[:2]:
                frame = rgb.copy()
                frame[mask] = (0.55 * frame[mask] + 0.45 * np.array([0, 255, 60])).astype(np.uint8)
            img = Image.fromarray(frame).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            if det is not None:
                d = ImageDraw.Draw(img)
                x0, y0, x1, y1 = det.box
                d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(0, 255, 60), width=3)
                d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), f"{det.label} {det.score:.2f}", fill=(0, 255, 60))
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

        def to_px(x, y):  # robot frame (x fwd, y left) -> canvas
            return S / 2 - (y / R) * (S / 2), S - (x / R) * S * 0.92 - 20

        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")
        if obstacles is not None and len(obstacles):
            for ox, oy in obstacles[:: max(1, len(obstacles) // 400)]:
                px, py = to_px(ox, oy)
                self.plot.create_rectangle(px - 1, py - 1, px + 1, py + 1, fill="#8a8a8a", outline="")
        if trajs is not None:
            for t in trajs:
                pts = [to_px(p[0], p[1]) for p in t[::2]]
                self.plot.create_line(*[c for xy in pts for c in xy], fill="#cccccc")
        if chosen is not None:
            pts = [to_px(p[0], p[1]) for p in chosen]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)
        if goal is not None:
            gx, gy = to_px(goal[0], goal[1])
            self.plot.create_text(gx, gy, text="★", fill="#d4a017", font=("TkDefaultFont", 22))

        mode_txt = "STOPPED" if stopped else state_text
        if mode == "point" and world_goal is not None and pose is not None:
            dist = math.hypot(world_goal[0] - pose["x"], world_goal[1] - pose["z"])
            goal_txt = f"point ({world_goal[0]:.1f}, {world_goal[1]:.1f})  dist {dist:.2f}m"
        else:
            goal_txt = f"'{target}'"
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  goal: {goal_txt}   {vel_text}{fwd}")
        pose_txt = f"pose ({pose['x']:.1f}, {pose['z']:.1f}, {pose['yaw']:.2f})" if pose else "pose: -"
        self.info.configure(text=f"{pose_txt}   frames {frames}   inferences {infers}   {lat}")
        self.root.after(66, self.refresh)


def main():
    ap = argparse.ArgumentParser(description="Nav_new Earth DINO+NavDP GUI (Habitat)")
    ap.add_argument("--target", default="yellow building")
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--invert-angular", action="store_true")
    ap.add_argument("--belief-confidence-min", type=float, default=BELIEF_CONFIDENCE_MIN,
                     help="target-out-of-view belief confidence floor below which the rover "
                          "gives up on the propagated goal memory and switches to SEARCH")
    ap.add_argument("--max-climb-deg", type=float, default=GuardConfig().max_climb_deg,
                     help="terrain rising up to this many degrees is treated as driveable ground, "
                          "not an obstacle -- raise this if the rover balks at climbing curbs/mounds, "
                          "lower it if it's driving over things it shouldn't")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = EarthPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(0.3, args.max_angular),
        invert_angular=args.invert_angular,
        guard=GuardConfig(max_climb_deg=args.max_climb_deg),
    ), belief_confidence_min=args.belief_confidence_min)

    session = zenoh.open(zenoh.Config())
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    _subs, pubs = zenoh_setup(session, st)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz), daemon=True).start()

    root = tk.Tk()
    App(root, st, pubs)

    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, root.destroy))

    def _tick():
        root.after(200, _tick)

    _tick()
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
