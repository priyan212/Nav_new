"""DINO -> NavDP navigation pipeline for the 6WD rover.

Per frame:
  1. Grounding DINO detects the text target -> bbox.
  2. Depth: external (Isaac Sim / sensor) or Depth Anything V2 metric (RGB-only).
  3. bbox + depth + intrinsics -> 3D point goal in the robot frame.
  4. NavDP samples N candidate trajectories conditioned on the point goal
     (checkpoint-native convention: y right-positive — empirically verified,
     see scripts/diag_goal_conditioning.py).
  5. Trajectory selection: combined score of goal progress (endpoint heading /
     distance toward goal) and the NavDP critic (collision safety). This keeps
     the rover goal-directed even though the extracted checkpoint's learned
     point conditioning is weak.
  6. Stop when the goal is within stop_distance or the bbox fills the view.

Returns a (linear, angular) velocity command plus rich debug info.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .dino_detector import GroundingDinoDetector
from .goal_utils import (
    goal_point_from_detection,
    intrinsics_from_fov,
    preprocess_depth,
    preprocess_rgb,
)
from .navdp_net import NavDPStandalone
from .obstacle_guard import (
    GuardConfig,
    apply_avoid_cooldown,
    depth_to_obstacle_points,
    forward_guard,
    swept_clearance,
)


@dataclass
class PipelineConfig:
    device: str = "cuda:0"
    horizontal_fov_deg: float = 90.0
    sample_num: int = 32
    # policy backend:
    #   "crossmodal" - official standalone NavDP (navdp-cross-modal.ckpt),
    #                  strong trained point-goal conditioning, y-left convention
    #   "extracted"  - NavDP weights extracted from InternVLA-N1-w-NavDP,
    #                  weak conditioning, y-right convention (sign-flipped here)
    policy_type: str = "crossmodal"
    # trajectory selection
    critic_weight: float = 1.0       # safety weight vs goal progress
    critic_floor: float = -10.0      # samples below this critic value are discarded
    #   (for "extracted" use 0.16 — its critic lives in a narrow 0.17 band)
    critic_keep_frac: float = 0.5    # among clearance-safe samples, keep top fraction by critic
    # depth-based obstacle guard (hard, model-agnostic)
    avoid_enabled: bool = True
    guard: GuardConfig = field(default_factory=GuardConfig)
    # command shaping (rover caps, matching the old OmniVLA node limits)
    invert_angular: bool = False     # flip outgoing turn direction (real-rover wiring)
    kp: float = 1.0
    kp_angular: float = 2.2          # heading-error -> angular gain (saturates at max_angular)
    urgency_gain: float = 2.0        # extra angular gain as obstacles close in (< slow_dist)
    max_linear: float = 0.15
    max_angular: float = 0.25
    waypoint_index: int = 8          # look-ahead waypoint on the chosen trajectory
    # ── proven real-rover mechanics (ported from the OmniVLA node) ──
    # Steering in open space is a DETERMINISTIC visual servo on the detection
    # bearing (stable tick to tick), not the resampled diffusion heading.
    servo_deadband: float = 0.05     # |bearing| below this -> drive straight (OmniVLA value)
    ang_boost: float = 0.05          # constant added in the turn direction (OmniVLA value)
    ang_min_cmd: float = 0.12        # stiction floor: minimum |angular| outside the deadband —
    #                                  the 6WD skid-steer won't yaw at all on smaller commands
    smoothing: float = 0.5           # cross-tick EMA on (lin, ang); 0 = off
    avoid_confirm_ticks: int = 2     # consecutive guard hits before AVOID engages
    avoid_cooldown_ticks: int = 8    # keep biasing steering away from the escape side for this many
    #                                  more ticks after AVOID releases, so the rover actually clears the
    #                                  obstacle's lateral footprint before goal-bearing servo resumes --
    #                                  otherwise it snaps straight back at the just-avoided obstacle and
    #                                  oscillates AVOID/TRACK in place (see apply_avoid_cooldown)
    avoid_bias_gain: float = 0.15    # rad/s added toward the escape side during the cooldown window
    # SAM 2.1 segmentation layer (DINO bbox -> instance mask -> goal)
    use_sam: bool = True
    mask_stop_frac: float = 0.30     # mask area fraction of image -> stop
    # goal / stopping
    stop_distance: float = 0.8       # meters from object at which to stop
    bbox_stop_frac: float = 0.55     # bbox height fraction of image -> stop (no-SAM fallback)
    detect_score_min: float = 0.3
    # target loss behavior
    search_angular: float = 0.2      # spin to re-acquire when target not seen
    lost_patience: int = 5           # frames to keep last goal before searching


@dataclass
class StepResult:
    linear: float = 0.0
    angular: float = 0.0
    state: str = "SEARCH"            # SEARCH | TRACK | AVOID | STOP
    detection: Optional[object] = None
    mask: Optional[np.ndarray] = None
    goal_point: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    all_trajectories: Optional[np.ndarray] = None
    critic: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    min_forward: float = float("inf")
    timing: dict = field(default_factory=dict)


class DinoNavDPPipeline:
    def __init__(self, cfg: PipelineConfig = PipelineConfig(), use_depth_estimator: bool = True):
        self.cfg = cfg
        t0 = time.time()
        self.detector = GroundingDinoDetector(device=cfg.device, box_threshold=cfg.detect_score_min)
        if cfg.policy_type == "crossmodal":
            from .navdp_crossmodal import NavDPCrossModal

            self.policy = NavDPCrossModal.load(device=cfg.device)
            self._goal_y_sign = 1.0    # checkpoint uses ROS y-left directly
        else:
            self.policy = NavDPStandalone.load(device=cfg.device)
            self._goal_y_sign = -1.0   # embedded checkpoint uses y-right
        self._memory_size = self.policy.memory_size
        self.segmenter = None
        if cfg.use_sam:
            from .sam_segmenter import Sam2Segmenter

            self.segmenter = Sam2Segmenter(device=cfg.device)
        self.depther = None
        if use_depth_estimator:
            from .depth_estimator import MetricDepthEstimator

            self.depther = MetricDepthEstimator(device=cfg.device)
        print(f"[pipeline] models loaded in {time.time() - t0:.1f}s")

        self._memory: list = []      # last processed RGB frames (max 2)
        self._memory_d: list = []
        self._lost_count = 0
        self._last_goal: Optional[np.ndarray] = None
        self._avoid_streak = 0
        self._avoid_side = 0.0
        self._avoid_cooldown = 0
        self._prev_cmd = (0.0, 0.0)

    def reset(self):
        self._memory, self._memory_d = [], []
        self._lost_count = 0
        self._last_goal = None
        self._avoid_streak = 0
        self._avoid_side = 0.0
        self._avoid_cooldown = 0
        self._prev_cmd = (0.0, 0.0)

    # ------------------------------------------------------------------ #
    def _select_trajectory(self, trajs: np.ndarray, critic: np.ndarray, goal: np.ndarray,
                           clearances: Optional[np.ndarray] = None):
        """Score = goal progress − collision risk. goal is [x fwd, y left].

        Hard vetoes first: trajectories whose clearance to depth obstacle
        points is below guard.clearance are discarded, then the bottom
        (1 - critic_keep_frac) by critic among survivors. Goal progress only
        chooses among what survived. If nothing is safe, the max-clearance
        sample wins outright.
        """
        endpoints = trajs[:, -1, :2]                          # (N, 2) x fwd, y left
        goal_xy = goal[:2]
        goal_dist = np.linalg.norm(goal_xy) + 1e-6
        # progress: how much closer the endpoint gets to the goal, normalized
        progress = (goal_dist - np.linalg.norm(endpoints - goal_xy, axis=1)) / goal_dist
        # heading alignment of the early trajectory segment toward the goal
        early = trajs[:, min(4, trajs.shape[1] - 1), :2]
        early_norm = np.linalg.norm(early, axis=1) + 1e-6
        align = (early @ goal_xy) / (early_norm * goal_dist)
        score = progress + 0.5 * align + self.cfg.critic_weight * critic

        if clearances is not None:
            safe = clearances >= self.cfg.guard.margin
            if not safe.any():
                return int(np.argmax(clearances))            # least-bad escape
            score[~safe] = -np.inf
            # among safe samples, drop the weakest critic fraction
            safe_idx = np.where(safe)[0]
            if len(safe_idx) > 2:
                thr = np.quantile(critic[safe_idx], 1.0 - self.cfg.critic_keep_frac)
                weak = safe & (critic < thr)
                if (safe & ~weak).any():
                    score[weak] = -np.inf
        else:
            unsafe = critic < self.cfg.critic_floor
            if not unsafe.all():
                score[unsafe] = -np.inf
        return int(np.argmax(score))

    def _command_from_trajectory(self, traj: np.ndarray, min_forward: float = float("inf")):
        """Look-ahead waypoint -> (v, w), matching the rover's velocity caps.

        Angular authority scales with obstacle urgency: near obstacles the
        gain is boosted (saturating at max_angular) and the look-ahead uses
        the widest heading over the horizon so a curving escape trajectory
        commands an immediate, committed turn instead of a lazy drift.
        """
        wp = traj[min(self.cfg.waypoint_index, len(traj) - 1)]
        dist = float(np.linalg.norm(wp[:2]))
        heading = float(np.arctan2(wp[1], wp[0]))

        urgent = min_forward < self.cfg.guard.slow_dist
        if urgent:
            # widest waypoint heading over the horizon = the turn the policy
            # actually intends; act on it NOW rather than easing into it
            xs, ys = traj[1:, 0], traj[1:, 1]
            headings = np.arctan2(ys, np.maximum(xs, 1e-3))
            heading = float(headings[np.argmax(np.abs(headings))])

        gain = self.cfg.kp_angular
        if urgent:
            gain *= 1.0 + self.cfg.urgency_gain * (1.0 - min_forward / self.cfg.guard.slow_dist)
        angular = np.clip(gain * heading, -self.cfg.max_angular, self.cfg.max_angular)

        linear = np.clip(self.cfg.kp * dist, 0.0, self.cfg.max_linear)
        # slow down while turning hard (fraction of angular authority in use)
        linear *= max(0.15, 1.0 - 0.8 * abs(angular) / self.cfg.max_angular)
        return float(linear), float(angular)

    # ------------------------------------------------------------------ #
    def step(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None) -> StepResult:
        res = self._step_inner(rgb, target_text, depth)
        # angular boost + stiction floor: guarantee real yaw on the skid-steer
        # chassis (proven necessary on this rover — small commands do not rotate it)
        if res.state in ("TRACK", "AVOID", "SEARCH") and abs(res.angular) > 0.01:
            boosted = abs(res.angular) + self.cfg.ang_boost
            boosted = max(boosted, self.cfg.ang_min_cmd)
            res.angular = float(np.clip(np.copysign(boosted, res.angular),
                                        -self.cfg.max_angular, self.cfg.max_angular))
        if self.cfg.invert_angular:
            res.angular = -res.angular
        # cross-tick EMA smoothing (ported from the OmniVLA node's damping)
        a = self.cfg.smoothing
        if a > 0 and res.state != "STOP":
            pl, pa = self._prev_cmd
            res.linear = (1 - a) * res.linear + a * pl
            res.angular = (1 - a) * res.angular + a * pa
        self._prev_cmd = (res.linear, res.angular)
        return res

    def _step_inner(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None) -> StepResult:
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

        # update observation memory (up to policy.memory_size frames)
        rgb_p, dep_p = preprocess_rgb(rgb), preprocess_depth(depth)
        self._memory.append(rgb_p)
        self._memory_d.append(dep_p)
        self._memory = self._memory[-self._memory_size:]
        self._memory_d = self._memory_d[-self._memory_size:]

        # --- goal ------------------------------------------------------ #
        fx, fy, cx, cy = intrinsics_from_fov(W, H, self.cfg.horizontal_fov_deg)
        goal = None
        if det is not None:
            res.detection = det
            close_by_size = False
            if self.segmenter is not None:
                t0 = time.time()
                mask = self.segmenter.segment_box(rgb, det.box)
                timing["sam"] = time.time() - t0
                if mask is not None:
                    from .sam_segmenter import mask_centroid, mask_median_depth

                    res.mask = mask
                    d = mask_median_depth(depth, mask)
                    if d is not None:
                        u, v = mask_centroid(mask)
                        from .goal_utils import pixel_depth_to_point

                        goal = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
                    close_by_size = mask.mean() > self.cfg.mask_stop_frac
            if goal is None:  # SAM disabled, empty mask, or no valid mask depth
                goal = goal_point_from_detection(det.box, depth, fx, fy, cx, cy)
                close_by_size = (det.box[3] - det.box[1]) / H > self.cfg.bbox_stop_frac
            if goal is not None:
                self._last_goal, self._lost_count = goal, 0
                if np.linalg.norm(goal[:2]) < self.cfg.stop_distance or close_by_size:
                    res.state = "STOP"
                    res.goal_point = goal
                    res.timing = timing
                    return res
        if goal is None:
            self._lost_count += 1
            if self._last_goal is not None and self._lost_count <= self.cfg.lost_patience:
                goal = self._last_goal
            else:
                # search: rotate in place toward the last known side
                res.state = "SEARCH"
                side = 1.0
                if self._last_goal is not None and self._last_goal[1] < 0:
                    side = -1.0
                res.angular = side * self.cfg.search_angular
                res.timing = timing
                return res

        res.goal_point = goal
        res.state = "TRACK"

        # --- obstacle guard --------------------------------------------- #
        obstacle_pts = None
        if self.cfg.avoid_enabled:
            t0 = time.time()
            obstacle_pts = depth_to_obstacle_points(
                depth, fx, fy, cx, cy, self.cfg.guard, exclude_mask=res.mask
            )
            res.obstacle_points = obstacle_pts
            min_fwd, escape = forward_guard(obstacle_pts, self.cfg.guard)
            res.min_forward = min_fwd
            timing["guard"] = time.time() - t0
            if min_fwd < self.cfg.guard.hard_stop_dist:
                # hysteresis: monocular depth is noisy — require consecutive
                # confirmations before engaging AVOID (prevents state flapping
                # that looks like random movement)
                self._avoid_streak += 1
                if self._avoid_streak >= self.cfg.avoid_confirm_ticks:
                    res.state = "AVOID"
                    res.linear = -0.5 * self.cfg.max_linear if min_fwd < self.cfg.guard.reverse_dist else 0.0
                    res.angular = escape * self.cfg.max_angular  # full turn authority
                    # latch the escape side + re-arm the cooldown (every
                    # trigger resets it to the full value, so a persistent
                    # obstacle keeps the post-escape bias alive throughout)
                    self._avoid_side = escape
                    self._avoid_cooldown = self.cfg.avoid_cooldown_ticks
                    res.timing = timing
                    return res
            else:
                self._avoid_streak = 0

        # --- NavDP ------------------------------------------------------ #
        t0 = time.time()
        M = self._memory_size
        frames = np.stack(self._memory)
        if frames.shape[0] < M:  # front-pad with zeros (matches official agent at episode start)
            pad = np.zeros((M - frames.shape[0],) + frames.shape[1:], dtype=frames.dtype)
            frames = np.concatenate([pad, frames], axis=0)
        images = torch.from_numpy(frames).unsqueeze(0)
        if self.cfg.policy_type == "crossmodal":
            depths = torch.from_numpy(dep_p).unsqueeze(0).unsqueeze(0)  # current depth only (1,1,H,W,1)
        else:
            dframes = np.stack(self._memory_d)
            if dframes.shape[0] < M:
                pad = np.zeros((M - dframes.shape[0],) + dframes.shape[1:], dtype=dframes.dtype)
                dframes = np.concatenate([pad, dframes], axis=0)
            depths = torch.from_numpy(dframes).unsqueeze(0)
        goal_native = np.array([goal[0], self._goal_y_sign * goal[1], goal[2]], dtype=np.float32)
        trajs, critic = self.policy.sample_pointgoal(
            goal_native.reshape(1, 3), images, depths, sample_num=self.cfg.sample_num
        )
        trajs = trajs.cpu().numpy()
        critic = critic.cpu().numpy()
        timing["navdp"] = time.time() - t0

        clearances = None
        if obstacle_pts is not None:
            clearances = swept_clearance(trajs, obstacle_pts)
        idx = self._select_trajectory(trajs, critic, goal, clearances)
        chosen = trajs[idx]
        res.trajectory = chosen
        res.all_trajectories = trajs
        res.critic = critic

        if res.min_forward < self.cfg.guard.slow_dist:
            # obstacle zone: follow the clearance-vetoed NavDP trajectory
            res.linear, res.angular = self._command_from_trajectory(chosen, res.min_forward)
            res.linear *= max(0.25, res.min_forward / self.cfg.guard.slow_dist)
        else:
            # open space: deterministic visual servo on the goal bearing —
            # stable tick to tick (the diffusion heading resamples every tick,
            # which read as random wandering on the real rover)
            bearing = float(np.arctan2(goal[1], goal[0]))  # +left, ROS convention
            if abs(bearing) < self.cfg.servo_deadband:
                res.angular = 0.0
                res.linear = self.cfg.max_linear
            else:
                res.angular = float(np.clip(self.cfg.kp_angular * bearing,
                                            -self.cfg.max_angular, self.cfg.max_angular))
                res.linear = self.cfg.max_linear * max(
                    0.2, 1.0 - 0.8 * abs(res.angular) / self.cfg.max_angular
                )
        res.angular, self._avoid_cooldown = apply_avoid_cooldown(
            res.angular, res.state, self._avoid_side, self._avoid_cooldown,
            self.cfg.avoid_bias_gain, self.cfg.max_angular,
        )
        res.timing = timing
        return res
