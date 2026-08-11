#!/usr/bin/env python3
"""
InternVLA-N1 DualVLN Zenoh Inference Node  (camera-only, RGB, language-goal)
=============================================================================
Sibling of ``internvla_zenoh_node.py``, which deliberately bypasses this
checkpoint's own System-1 (it forces `system1` to either nothing or a
swapped-in NavDP checkpoint). THIS node runs the checkpoint natively, exactly
as released and described in "Ground Slow, Move Fast: A Dual-System
Foundation Model for Generalizable Vision-and-Language Navigation" (arXiv
2512.08186, model name DualVLN / InternVLA-N1): System-2 is a QwenVL-2.5-7B
VLM that "grounds slowly" (~2Hz) by predicting a pixel goal + a compact
latent goal from the instruction + RGB history; System-1 is a lightweight
Diffusion Transformer that "moves fast" by turning that goal into a smooth
32-waypoint local trajectory every tick. Both are trained jointly and shipped
in the SAME checkpoint (InternVLA-N1-DualVLN) -- no separate System-1 weights
to load. This makes it well suited to long, compound, multi-landmark
instructions (its own training data is R2R/RxR-style: "walk through the
opening between the kitchen and the dining room, turn right, go through the
doorway and stop next to the closet...").

Runs on the GPU machine and speaks the SAME Zenoh contract (camera-in ->
cmd_vel-out) as every other launcher in this repo, so the Pi/rover side is
unchanged.

This node is ADDITIVE and non-destructive: it does not import or modify
`internvla_zenoh_node.py`, any `nav_pipeline/` module, or any other launcher.

Env:  runs under the `internnav` conda env with transformers 4.51.0 shadowed:
    PYTHONPATH=/home/i3d/internnav_n1_tf451 \
    /mnt/bigdisk/conda_envs/internnav/bin/python reference/internvla_dualvln_zenoh_node.py

Subscribes (Zenoh, CDR ROS 2 msgs):
  image_raw            – sensor_msgs/Image     (webcam / Isaac camera)
  omnivla/goal_text    – std_msgs/String       (change instruction; shared topic)
Publishes (Zenoh, CDR):
  cmd_vel              – geometry_msgs/Twist
  omnivla/explanation  – std_msgs/String

Safety: the DualVLN model itself is RGB-only (its native `nextdit_async`
System-1 branch accepts a depth argument but never reads it -- confirmed in
`internvla_n1_arch.py`). There is no LiDAR. As an INDEPENDENT safety net
(model-agnostic, same code `internvla_zenoh_node.py`'s `--use-navdp` mode
already validated live), a monocular depth estimate (DepthAnythingV2-metric)
is computed from the same RGB frame each tick purely to feed a hard stop
(_apply_depth_safety, 30cm) and reactive steer-around (_apply_reactive_avoid,
0.9m) -- this exists regardless of what the navigation model decided.

Single inference mode (no --use-navdp / System-2-only split, unlike the
sibling node): every tick runs the checkpoint's own native dual-system.
`agent.step_s2()` emits EITHER a discrete action {STOP, forward, left, right,
look-down} OR a pixel goal + latent goal; on a pixel goal, `agent.step_s1()`
runs the REAL trained Diffusion Transformer (`generate_traj`) to produce 32
sampled local trajectories. Those samples are NOT critic-filtered by the
model itself, so `traj_to_actions` is monkeypatched (unchanged from the
sibling node) to majority-cluster the 32 samples and average only the
largest agreeing cluster, instead of blindly averaging all 32 (which washes
multimodal choices like "go left around" vs "go right around" into a mushy,
non-committal path). See internvla-n1-phase1 memory notes for the full
history of the shared fixes this node inherits.
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
    import cv2
except ImportError:
    cv2 = None

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh Python library not found (pip install eclipse-zenoh).")
    sys.exit(1)

# repo root (parent of this file's dir) -- `python reference/foo.py` puts only
# reference/ on sys.path[0], not the repo root, so nav_pipeline (a sibling
# top-level package) isn't importable without this. Inserted, not appended,
# to match this file's other sys.path.insert(0, ...) (see load_model()).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.dino_detector import GroundingDinoDetector  # noqa: E402


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

    def read_float32(self) -> float:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "f", self.data, self.offset)
        self.offset += 4
        return v


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


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/msg/CompressedImage CDR -> numpy RGB array (H, W, 3).

    The Hiwonder backend's Pi-side bridge (landerpi/bridge.py) publishes
    ONLY this compressed topic (image_raw/compressed, JPEG) -- it never
    publishes plain image_raw at all. Copied from nav_pipeline/zenoh_node.py
    so this node stays independent (same wire format; model-agnostic)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    r.read_string()  # format
    jpeg_bytes = r.read_sequence_uint8()
    try:
        if cv2 is not None:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        import io
        return np.array(Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def parse_string(cdr_data: bytes) -> str:
    return CDRReader(cdr_data).read_string()


def parse_float32_multiarray(cdr_data: bytes) -> list:
    """std_msgs/Float32MultiArray CDR -> list[float] (the .data field).
    Copied from nav_pipeline/zenoh_node.py (same wire format) -- feeds
    rover/rpm's [left_rpm, right_rpm, imu_heading_deg, imu_calib,
    lateral_m_s] into OdometryLogger, see landerpi/bridge.py's docstring."""
    r = CDRReader(cdr_data)
    dim_count = r.read_uint32()
    for _ in range(dim_count):
        r.read_string()   # label
        r.read_uint32()    # size
        r.read_uint32()    # stride
    r.read_uint32()        # data_offset
    n = r.read_uint32()
    return [r.read_float32() for _ in range(n)]


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


def serialize_trajectory(traj_xy: np.ndarray) -> bytes:
    """Local-frame (x fwd, y left, meters) waypoint list -> CDR bytes, for the
    standalone GUI (internvla_dualvln_gui.py) to draw the predicted path.
    Deliberately a minimal custom format (count + flat float64 pairs), not a
    real ROS2 message type -- only this node and that GUI need to speak it."""
    w = CDRWriter()
    w.write_uint32(len(traj_xy))
    for x, y in traj_xy:
        w.write_float64(float(x))
        w.write_float64(float(y))
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

    # --- Independent monocular-depth safety net (model never sees this) -------
    # The checkpoint's native System-1 (nextdit_async) is RGB-only and never
    # reads a depth argument. There is no depth sensor (project constraint),
    # so this estimate exists SOLELY to feed the hard-stop/reactive-avoid
    # safety layer below -- same DepthAnythingV2-metric model already
    # validated live by internvla_zenoh_node.py's --use-navdp mode.
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
    # Originally 1.0s (see reference/internvla_zenoh_node.py, where it was
    # tuned against a MOSTLY-CONSISTENT trajectory-following signal). Reduced
    # to 0.35s here (2026-08-10) after live testing on the Hiwonder bot with
    # this checkpoint's own discrete-action System-2: it re-decides fresh
    # every tick (~0.5-0.8s observed), and when it flips direction for a
    # single tick before reverting, the OLD 1.0s ramp needed 2-3 ticks to
    # actually reach the new target -- by then the model had already reverted
    # to its previous decision, so genuine corrections never took physical
    # effect and the bot's net motion stayed dominated by whichever direction
    # it had been repeating. At 0.35s, a typical ~0.5-0.8s tick now covers the
    # full swing (still damped for faster ticks, e.g. a 150ms tick only
    # covers ~43% of it) -- still smooths the original "turns, THEN moves"
    # snap this was added for, just fast enough that a real direction change
    # can actually land. NOT yet re-validated against a long live run --
    # change this constant alone if it needs further tuning, per this
    # project's live-rover-tuning practice of one constant at a time.
    motion_ramp_time_s: float = 0.35
    # VLM input
    resize_w: int = 384
    resize_h: int = 384
    num_history: int = 8
    plan_step_gap: int = 1
    # Navigation velocities (real physical units, m/s and rad/s) — kept low.
    # Lowered 2026-08-10 (0.15/0.25 -> 0.06/0.10) to reduce perceived lag/
    # jitter. Raised partway back up 2026-08-11 after live evidence the
    # 0.06/0.10 pair was too low to reliably move the rover AT ALL: logged
    # legs commanding ang=0.100 repeatedly landed at dist<0.02m/dtheta<2deg
    # (encoder-confirmed near-zero actual motion -- not a sensor artifact),
    # tripping the stall-recovery backup (stall_backup_lin=0.12,
    # stall_backup_ang=0.4) -- and EVERY leg immediately following one of
    # those backups then moved normally (0.08-0.21m), i.e. the stronger
    # recovery kick was what broke static friction, not the model's own
    # (too-weak) discrete-turn command. 0.10 rad/s was apparently below this
    # rover's stiction threshold on today's floor -- what looked like "the
    # bot keeps moving back for no reason" was this safety net correctly
    # reacting to a command that was too weak to move the wheels at all.
    # Override with --max-linear/--max-angular for a one-off run without
    # changing this default.
    max_linear: float = 0.08
    max_angular: float = 0.15
    predict_hz: float = 2.0
    instruction: str = "go straight down the hallway and stop at the door"
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

    # --- Stop-decide-move-stop control cadence (2026-08-11, requested change) --
    # Previously: continuous closed loop -- infer_cmd() ran every ~0.5s
    # (predict_hz) WHILE the rover kept moving (heartbeat thread re-sending
    # the last command between ticks), so a new decision could land mid-
    # motion and blend in via motion smoothing. That's what let a frozen/
    # stale camera frame (see the 2026-08-10 Pi-network-dropout incident)
    # silently keep driving the rover on old data -- nothing forced a stop
    # to happen before trusting the next frame as current.
    # Now: each "leg" is one full stop -> settle -> ONE model decision ->
    # execute that single decision for a short, bounded window -> full stop
    # -> repeat. This bounds how far any single (possibly wrong) decision
    # can carry the rover before a fresh reassessment, and guarantees the
    # frame used for each decision was captured while the rover was
    # genuinely stationary (no motion blur, no stale-frame-while-moving
    # risk). Independent safety checks (depth hard-stop) still run
    # continuously during the execute window -- only the heavy System-2/
    # System-1 model call is now once-per-leg rather than continuous.
    settle_time_s: float = 0.4     # after commanding stop, wait this long before trusting the next frame as "current" (lets motion blur/latency clear)
    move_duration_s: float = 3.0   # trajectory-mode leg cap: System-1 ran, 30+/32 samples agreed -- a real, confident path direction, worth committing to
    # Live-observed (2026-08-11, "move forward and stop next to the white
    # air cooler", cooler directly in view): the rover swung right ~78deg
    # over 6 legs, then left ~220deg over 11 legs (tripping the repeat-
    # breaker), then right ~170deg again -- textbook indecisive oscillation,
    # net progress near zero, matching the user's "clockwise and anti-
    # clockwise movements" report exactly. Root cause, cross-checked against
    # the paper's own real-world section (2512.08186 Sec 5.2): their
    # discrete actions are "view adjustment actions... transformed into
    # world coordinates via odometry and tracked with an MPC controller" --
    # i.e. a SMALL, BOUNDED reorientation, immediately re-observed -- not a
    # velocity held for seconds. Holding every discrete LEFT/RIGHT at full
    # power for the full move_duration_s (originally shared with trajectory
    # legs) turned each of the model's tentative, low-conviction glances
    # into a large committed swing, so two flip-flopping discrete decisions
    # in a row cost a big, visible reversal instead of a small correction.
    # A discrete leg now gets its own much shorter cap -- long enough to
    # depart the ramp-up and register real angular motion for the odometry-
    # based stall/logging check, short enough that indecision costs a small
    # nudge, not a lurch -- while trajectory-mode legs (System-1 actually
    # ran, model showed real path consensus) keep the full move_duration_s.
    discrete_move_duration_s: float = 1.0
    leg_safety_poll_s: float = 0.15  # depth-safety re-check cadence during the execute window (cheap, no model call)

    # --- Grounding verification (2026-08-11) -------------------------------
    # System-2 already does GROUNDING internally -- its pixel-goal IS a
    # grounded pointer into the image -- but it's a black box: nothing
    # outside the model confirms the named landmark is actually there.
    # Concretely bitten by this twice tonight: the model latched "GOAL
    # REACHED" while its own saved debug frame showed a person's legs, and
    # separately a lab shelving unit -- neither anywhere near the
    # instructed target. Grounding DINO (already used and validated
    # elsewhere in this project, see nav_pipeline/dino_detector.py) is
    # reused UNCHANGED here as an independent, external check: before
    # latching goal-reached, verify the instruction's actual target phrase
    # (see _extract_target_phrase) is open-vocabulary-detected in the
    # CURRENT frame with reasonable confidence. Does NOT fix the separate,
    # already-identified bugs from tonight (greedy-decoding repeat-spam,
    # physical wheel stalls) -- this only closes the "hallucinated arrival"
    # failure mode specifically.
    grounding_model_id: str = "IDEA-Research/grounding-dino-base"
    grounding_box_threshold: float = 0.30
    # If grounding keeps disagreeing with the model's own STOP this many
    # consecutive times, trust the model anyway rather than loop forever --
    # covers cases where the target phrase extraction is poor, the object is
    # technically in frame but DINO's vocabulary doesn't match the exact
    # wording, or the camera framing genuinely can't see it (see the
    # 2026-08-11 camera-tilt finding). Logged loudly either way.
    grounding_max_reject_before_trust: int = 4

    # --- Discrete-repeat circuit breaker (model-decoding-level failure, NOT
    # a physical obstruction -- see stall_* above for that case) -----------
    # Observed live (2026-08-10, real Hiwonder hardware, instruction "walk
    # straight,turn right at the black chair, take left and stop at the
    # door"): System-2 emitted ACT_RIGHT on 36/36 consecutive ticks (the full
    # captured run) at full angular deflection, with ZERO variation --
    # never once STOP/FWD/a pixel-goal. At that rate the rover rotated over
    # 480 degrees (more than a full circle) continuously with the decision
    # never changing. This is NOT the stall/wedged case (the camera view was
    # genuinely changing throughout -- real rotation was happening); it's a
    # decoding-level degeneracy: the checkpoint's own step_s2() calls
    # `model.generate(..., do_sample=False, ...)` (confirmed in
    # internvla_n1_agent_realworld.py and in this run's own startup warning
    # log: `top_k=1`) -- fully greedy/deterministic, so once it locks onto an
    # answer for a given (current-frame, history-frames) conditioning, there
    # is no mechanism to escape it; a slowly-evolving real camera feed can
    # keep re-producing the same argmax pick indefinitely. Rather than
    # touching the vendored decoding call (would affect the model's tuned,
    # validated behavior repo-wide), this reacts purely from the outside: if
    # the SAME discrete turn direction accumulates more rotation than any
    # single real indoor turn plausibly needs (a full U-turn is ~180 deg;
    # this gives ~1.1x that margin), force agent.reset() -- which clears
    # episode_idx/rgb_list/conversation_history, so the model's NEXT call
    # samples history-observation frames from scratch (for the very next
    # call, episode_idx==0 means NO history images at all, i.e. genuinely
    # different conditioning, not just "the same slowly-drifting window") --
    # and hold one paused tick so the reset is visible before resuming.
    discrete_repeat_break_rotation_rad: float = 3.49   # ~200 degrees of accumulated same-direction discrete turning


CFG = NodeConfig()

# InternVLA-N1 discrete action indices (from the agent's actions2idx)
ACT_STOP, ACT_FWD, ACT_LEFT, ACT_RIGHT, ACT_LOOKDOWN = 0, 1, 2, 3, 5


# ================================================================
#  N1 model loading + compat shims  (Phase-1 findings, made explicit)
# ================================================================
AGENT = {"agent": None, "intrinsic": None}


def _apply_n1_compat_shims():
    """Load-time fixes required to run this checkpoint under transformers 4.51.0.

    1. `config.text_config` — InternNav's internvla_n1.py:48 reads it, but 4.51.0's
       Qwen2.5-VL config is FLAT. get_text_config() returns self for flat configs,
       so expose text_config -> self (return self DIRECTLY; going through
       get_text_config() recurses via its hasattr check).

    Unlike internvla_zenoh_node.py, this node does NOT monkeypatch `system1`:
    the checkpoint's own config.json already says "nextdit_async" (confirmed
    via HF cache inspection), which is exactly the paper's real trained
    Diffusion-Transformer System-1 — that's the whole point of this node, so
    it's left to load natively.
    """
    from transformers import Qwen2_5_VLConfig
    if not hasattr(Qwen2_5_VLConfig, "text_config"):
        Qwen2_5_VLConfig.text_config = property(lambda self: self)

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
    print(f"  InternVLA-N1 DualVLN Zenoh Node — loading model on {CFG.device}")
    print(f"  Mode: native dual-system (System-2 QwenVL-7B + System-1 DiT, arXiv 2512.08186)")
    print(f"{'=' * 60}")

    sys.path.insert(0, CFG.internnav_repo)
    _apply_n1_compat_shims()
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    args = argparse.Namespace(
        device=CFG.device, model_path=CFG.model_path,
        resize_w=CFG.resize_w, resize_h=CFG.resize_h,
        num_history=CFG.num_history, plan_step_gap=CFG.plan_step_gap,
    )
    agent = InternVLAN1AsyncAgent(args)
    # Debug frames off the (full) bigdisk.
    agent.save_dir = "/tmp/internvla_n1_runs/" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(agent.save_dir, exist_ok=True)

    # generate_latents() is left REAL (not stubbed): the native System-1 (DiT)
    # needs the real latent goal from step_s2()'s second forward pass whenever
    # it emits a pixel-goal -- unlike internvla_zenoh_node.py's System-2-only
    # mode, which stubs this out because it never runs System-1 at all.

    # Camera intrinsics (unused by the model -- neither System-2's pixel-goal
    # grounding nor System-1's egocentric DiT trajectory does true camera-frame
    # back-projection -- but the agent signature requires it).
    intrinsic = np.array([[386.5, 0.0, 328.9, 0.0],
                          [0.0, 386.5, 244.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
    AGENT["agent"] = agent
    AGENT["intrinsic"] = intrinsic

    _load_depth_estimator()
    _load_grounding_detector()
    _warmup_model()

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"  Model ready. VRAM allocated: {vram:.2f} GB\n")


def _load_grounding_detector():
    """Independent, external verification for STOP/goal-reached decisions --
    see NodeConfig's grounding_* comment for why this exists. Reused
    unchanged from nav_pipeline/dino_detector.py (already validated
    elsewhere in this project)."""
    print("  Loading Grounding DINO (goal-reached verification)...")
    AGENT["dino"] = GroundingDinoDetector(
        model_id=CFG.grounding_model_id, device=CFG.device,
        box_threshold=CFG.grounding_box_threshold,
    )


import re as _re  # noqa: E402


def _extract_target_phrase(instruction: str) -> Optional[str]:
    """Heuristic target-landmark extraction for the Grounding DINO prompt:
    grab the phrase after the LAST 'stop' in the instruction -- every
    instruction tried on this project so far ends '...stop at/next to/near
    the X', so this reliably isolates the FINAL landmark (not an earlier
    waypoint like 'turn right at the black chair'). Returns None if the
    instruction has no 'stop' clause -- callers should skip verification
    (trust the model) rather than ground against a meaningless phrase."""
    matches = list(_re.finditer(r"\bstop\b", instruction, _re.IGNORECASE))
    if not matches:
        return None
    tail = instruction[matches[-1].end():]
    tail = _re.sub(r"^[\s,]*\b(at|near|next to|by|beside)\b", "", tail, flags=_re.IGNORECASE)
    tail = tail.strip(" .,!\n")
    tail = _re.sub(r"^(the|a|an)\s+", "", tail, flags=_re.IGNORECASE)
    return tail or None


def _load_depth_estimator():
    """Monocular metric depth (DepthAnythingV2-metric-hypersim, vits) -- there is
    no depth sensor (project constraint: RGB camera only) and the model itself
    never reads this (see module docstring), so this exists SOLELY to feed the
    independent safety net (_apply_depth_safety/_apply_reactive_avoid). Same
    checkpoint/loading code already validated live by internvla_zenoh_node.py's
    --use-navdp mode: sane 0.6-11m range, no NaN."""
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    print("  Loading monocular depth estimator (DepthAnythingV2-metric)...")
    depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
    state = torch.load(CFG.depth_metric_ckpt, map_location="cpu")
    depth_model.load_state_dict(state, strict=False)
    AGENT["depth_model"] = depth_model.to(CFG.device).eval()


def _warmup_model():
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
    print("  Warming up CUDA kernels (one-time, ~10s)...")
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
    # inference_mode wraps step_s1/step_s2 for a real reason, not caution for
    # its own sake: generate_traj()'s nextdit/DiT branch (internnav's
    # internvla_n1.py) runs its whole multi-step flow-matching denoising loop
    # WITHOUT torch.no_grad() -- only the RGB-feature sub-block above it is
    # wrapped. Left unwrapped here, PyTorch retains a full autograd graph
    # across all denoising steps x the 32-sample x2-CFG batch, for a graph
    # nothing will ever backward() through -- confirmed live: this alone was
    # enough to OOM a 24GB GPU on the very first step_s1 call, on top of the
    # ~16.7GB checkpoint already resident. inference_mode() is a strict outer
    # restriction (safe to nest around HF's own already-no_grad generate()
    # calls too), so it's applied around every agent.step*() call in this
    # file rather than patched into the vendored third_party/InternNav code.
    with torch.inference_mode():
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
    """Run the native dual-system inference path, then the stall-recovery
    check on top -- it is model- and depth-independent (RGB motion only). See
    NodeConfig's comment above stall_frame_diff_thresh for why this exists."""
    lin, ang, kind, detail = _infer_cmd_dualvln(rgb, instruction)
    if kind in ("trajectory", "pixel"):
        # Remember the last real heading correction so a later blind ACT_FWD
        # tick (which has none of its own) can carry a decaying fraction of
        # it instead of assuming ang=0 -- see _decayed_heading_bias().
        AGENT["last_heading_bias_ang"] = ang
        AGENT["last_heading_bias_ts"] = time.time()
    # Runs on the RAW kind/detail from _infer_cmd_dualvln (before motion
    # smoothing rewrites detail with a "| ramp(...)" suffix) so its
    # left/right/forward label check stays exact.
    lin, ang, kind, detail = _apply_discrete_repeat_breaker(lin, ang, kind, detail)
    lin, ang, kind, detail = _apply_motion_smoothing(lin, ang, kind, detail)
    return _apply_stall_recovery(rgb, lin, ang, kind, detail)


def _apply_discrete_repeat_breaker(lin: float, ang: float, kind: str, detail: str) -> tuple:
    """Model-decoding-level circuit breaker -- see NodeConfig's
    discrete_repeat_break_rotation_rad comment for the live 2026-08-10
    observation that motivated this (36/36 ticks of unbroken ACT_RIGHT,
    >480 degrees of continuous rotation, greedy/do_sample=False decoding).
    Unlike _apply_stall_recovery (frame ISN'T changing -> physically
    wedged), this catches the opposite signature: the frame IS genuinely
    changing (real rotation is happening, motion is NOT wedged) but the
    DECISION never does. Tracks accumulated |ang|*(time this decision will
    actually be applied for) while the same left/right label repeats
    back-to-back; once it exceeds the configured rotation budget, forces the
    underlying agent to forget its rolling observation history (so its next
    decision is conditioned on genuinely different input).

    Under the stop-decide-move-stop cadence (2026-08-11), infer_cmd() -- and
    so this function -- is called once per LEG, not continuously, and each
    decision is then executed for a bounded window by _execute_leg() --
    CFG.discrete_move_duration_s for discrete actions (the only kind this
    function ever tracks; see that constant's comment), CFG.move_duration_s
    for trajectory legs. Using wall-clock time.time() gaps between calls
    would therefore include the settle window and the full previous leg's
    execution time, wildly overcounting actual applied rotation. Use the
    actual per-leg cap as the per-call contribution instead -- the correct
    physical measure of how long THIS decision's ang will actually be
    commanded for."""
    base_label = detail.split(" | ", 1)[0]
    if kind != "discrete" or base_label not in ("left", "right"):
        AGENT["repeat_dir"] = None
        AGENT["repeat_accum_rad"] = 0.0
        return lin, ang, kind, detail

    prev_dir = AGENT.get("repeat_dir")
    if prev_dir != base_label:
        AGENT["repeat_dir"] = base_label
        AGENT["repeat_accum_rad"] = abs(ang) * CFG.discrete_move_duration_s
        return lin, ang, kind, detail

    accum = AGENT.get("repeat_accum_rad", 0.0) + abs(ang) * CFG.discrete_move_duration_s
    AGENT["repeat_accum_rad"] = accum

    if accum < CFG.discrete_repeat_break_rotation_rad:
        return lin, ang, kind, detail

    AGENT["repeat_dir"] = None
    AGENT["repeat_accum_rad"] = 0.0
    agent = AGENT.get("agent")
    if agent is not None:
        agent.reset()
        agent.save_dir = "/tmp/internvla_n1_runs/" + time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(agent.save_dir, exist_ok=True)
    return 0.0, 0.0, "discrete-repeat-break", (
        f"{base_label} repeated for {math.degrees(accum):.0f}deg with no "
        f"FWD/STOP/trajectory -- resetting model history, pausing"
    )


def _apply_motion_smoothing(lin: float, ang: float, kind: str, detail: str) -> tuple:
    """Final, universal rate-limiter on (lin, ang) -- applies identically
    whether the raw command came from a discrete action or a DiT trajectory,
    so switching between them (or firing the same discrete action repeatedly)
    ramps smoothly instead of snapping. See NodeConfig's motion_ramp_time_s
    comment for the full rationale.

    Runs BEFORE stall-recovery (which still overrides decisively, unsmoothed,
    when it triggers) and is skipped entirely for stop/hard-stop -- those must
    remain instantaneous, not ramped down."""
    if kind in ("stop", "obstacle-stop", "discrete-repeat-break"):
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


def _infer_cmd_dualvln(rgb: np.ndarray, instruction: str) -> tuple:
    """Full agent.step(): System-2 language reasoning + (when it emits a
    pixel-goal) System-1's real trained Diffusion Transformer trajectory
    generation -- the checkpoint's native dual-system, unmodified.

    A monocular depth estimate is computed from the SAME rgb frame each tick,
    purely for the independent safety net below (the model itself is RGB-only
    -- see module docstring). Reuses InternNav's own agent.step() branching
    logic unmodified, so this matches upstream's own dual-system behaviour
    exactly.

    Returns kind in {"stop","discrete","trajectory","obstacle-stop"}. The last
    is an independent depth-based safety override -- see NodeConfig's
    navdp_estop_distance_m block for why this exists.
    """
    agent = AGENT["agent"]
    intrinsic = AGENT["intrinsic"]
    depth_map = AGENT["depth_model"].infer_image(rgb[:, :, ::-1])  # infer_image expects BGR

    # inference_mode: see _warmup_model's comment -- generate_traj()'s DiT
    # denoising loop isn't wrapped in no_grad upstream, so every real tick
    # needs this too, not just warmup.
    with torch.inference_mode():
        out = agent.step(rgb, depth_map, np.eye(4), instruction, intrinsic, look_down=False)
        if out.output_action is not None and len(out.output_action) > 0 and int(out.output_action[0]) == ACT_LOOKDOWN:
            # Rover camera can't tilt; re-run the model's look-down refinement
            # on the same frame (matches http_internvla_server.py's eval_dual()).
            out = agent.step(rgb, depth_map, np.eye(4), instruction, intrinsic, look_down=True)

    if out.output_action is not None and len(out.output_action) > 0:
        lin, ang, kind, detail = _discrete_action_to_cmd(int(out.output_action[0]))
        AGENT["last_trajectory_xy"] = None
    elif out.output_trajectory is not None:
        traj_xy = np.asarray(out.output_trajectory)
        # Stashed for the GUI (see InternVLADualVLNZenohNode._tick) -- this is
        # the already majority-cluster-reduced path (_clustered_trajectory_reduce),
        # i.e. exactly what _trajectory_to_cmd below steers toward, not the raw
        # 32 diffusion samples.
        AGENT["last_trajectory_xy"] = traj_xy
        lin, ang, kind, detail = _trajectory_to_cmd(traj_xy)
    else:
        lin, ang, kind, detail = 0.0, 0.0, "discrete", "no-output"
        AGENT["last_trajectory_xy"] = None

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
    """Convert the DiT's egocentric local-frame xy trajectory (meters, start
    at origin, re-anchored to the live camera frame on every call) into a
    reactive steer: heading toward a lookahead waypoint, forward speed tapers
    with heading error.

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


# ================================================================
#  Main Zenoh Node
# ================================================================
class InternVLADualVLNZenohNode:
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
        self._stop_grounding_rejects = 0
        self._goal_reached = False
        self._last_cmd = (0.0, 0.0)

        # Real per-wheel encoder + IMU fusion (landerpi/bridge.py publishes
        # [left_rpm, right_rpm, imu_heading_deg, imu_calib, lateral_m_s] on
        # rover/rpm -- see module docstring). Reused unchanged from
        # nav_pipeline/odometry_logger.py (already validated: IMU-heading
        # gating on MAG calibration + wheel-moving, dead-reckoned x/y/theta).
        # Used two ways in the new stop-decide-move-stop cadence below: (1)
        # per-leg diagnostic logging of ACTUAL displacement vs. what was
        # commanded, (2) a hardware-grounded stall check -- far more
        # reliable than the old vision-only frame-diff heuristic, since it
        # directly measures whether the wheels actually turned the rover,
        # not just whether the scene looked different.
        self.odom = OdometryLogger(log_dir="odometry_log")
        self.odom.start_new_goal(self.instruction)
        self._last_rpm_rx_t: Optional[float] = None

        self.pub_cmd = session.declare_publisher("cmd_vel")
        self.pub_explain = session.declare_publisher("omnivla/explanation")
        self.pub_trajectory = session.declare_publisher("omnivla/trajectory")
        # Subscribe to every camera key any backend in this project actually
        # uses (matches nav_pipeline/zenoh_node.py's CAMERA_KEYS): the
        # Hiwonder Pi bridge (landerpi/bridge.py) publishes ONLY
        # image_raw/compressed (JPEG); the real rover publishes plain
        # image_raw; Isaac Sim's zenoh-bridge-ros2dds (confirmed live
        # 2026-08-11 via a wildcard "**" subscriber while testing this node
        # against Isaac for the first time) publishes plain rover_camera
        # instead -- neither image_raw nor image_raw/compressed. Listening
        # on all three means this node works against any of them unmodified.
        self.sub_image = session.declare_subscriber("image_raw", self._on_image)
        self.sub_image_compressed = session.declare_subscriber(
            "image_raw/compressed", self._on_image_compressed
        )
        self.sub_image_isaac = session.declare_subscriber("rover_camera", self._on_image)
        self.sub_goal = session.declare_subscriber("omnivla/goal_text", self._on_goal)
        # Both plain and "rt/"-prefixed (see nav_pipeline/zenoh_node.py's
        # RPM_KEYS) -- different Zenoh bridges on this project use either.
        self.sub_rpm = session.declare_subscriber("rover/rpm", self._on_rpm)
        self.sub_rpm_rt = session.declare_subscriber("rt/rover/rpm", self._on_rpm)

        print("[INFO] Zenoh subs: image_raw, image_raw/compressed, rover_camera, omnivla/goal_text, "
              "rover/rpm (encoder+IMU odometry)")
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

    def _on_image_compressed(self, sample: zenoh.Sample):
        try:
            rgb = parse_compressed_image(bytes(sample.payload))
            if rgb is not None:
                with self.lock:
                    self.latest_rgb = rgb
                    self._frame_count += 1
        except Exception as e:
            print(f"[WARN] compressed image parse failed: {e}")

    def _on_rpm(self, sample: zenoh.Sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) < 2:
                return
            left_rpm, right_rpm = data[0], data[1]
            imu_heading = data[2] if len(data) > 2 else None
            imu_calib = data[3] if len(data) > 3 else None
            lateral = data[4] if len(data) > 4 else None
            self.odom.update(left_rpm, right_rpm, imu_heading_deg=imu_heading,
                              imu_calib=imu_calib, lateral_m_s=lateral)
            self._last_rpm_rx_t = time.time()
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    def _on_goal(self, sample: zenoh.Sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if text and text != self.instruction:
                print(f"[INFO] New instruction: '{text}'")
                self.instruction = text
                self._goal_reached = False
                self._stop_streak = 0
                self._stop_grounding_rejects = 0
                self.odom.start_new_goal(text)  # new CSV file; pose stays continuous (see odometry_logger.py docstring)
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

    def publish_trajectory(self, traj_xy: Optional[np.ndarray]):
        if traj_xy is not None and len(traj_xy) > 0:
            self.pub_trajectory.put(serialize_trajectory(traj_xy))

    def _verify_stop_grounding(self, rgb: np.ndarray) -> bool:
        """External check on the model's own STOP decision -- see
        NodeConfig's grounding_* comment for why. Returns True (trust the
        model) if: no target phrase could be extracted, Grounding DINO isn't
        loaded, the target IS detected, or too many consecutive rejections
        have piled up (see grounding_max_reject_before_trust)."""
        target = _extract_target_phrase(self.instruction)
        dino = AGENT.get("dino")
        if target is None or dino is None:
            return True

        det = dino.detect_best(rgb, target)
        if det is not None:
            print(f"[GROUNDING] confirmed '{target}' (score={det.score:.2f}) -- latching goal-reached")
            self._stop_grounding_rejects = 0
            return True

        self._stop_grounding_rejects += 1
        print(f"[GROUNDING] model says STOP but '{target}' was NOT detected in the current frame "
              f"({self._stop_grounding_rejects}/{CFG.grounding_max_reject_before_trust}) -- NOT latching")
        self.publish_explanation(
            f"InternVLA-N1 [stop] rejected: '{target}' not visually confirmed "
            f"({self._stop_grounding_rejects}/{CFG.grounding_max_reject_before_trust})"
        )
        if self._stop_grounding_rejects >= CFG.grounding_max_reject_before_trust:
            print(f"[GROUNDING] {self._stop_grounding_rejects} consecutive rejections -- "
                  f"trusting the model's STOP anyway rather than loop forever")
            self._stop_grounding_rejects = 0
            return True
        return False

    def spin(self):
        """Stop -> settle -> ONE model decision -> execute that single
        decision for at most CFG.move_duration_s -> full stop -> repeat.
        See NodeConfig's "Stop-decide-move-stop control cadence" comment for
        why this replaced the old continuous per-tick loop."""
        last_status = time.time()
        print(f"[INFO] Stop-decide-move-stop cadence: settle={CFG.settle_time_s}s, "
              f"discrete-leg<={CFG.discrete_move_duration_s}s, trajectory-leg<={CFG.move_duration_s}s")
        print(f"[INFO] Instruction: '{self.instruction}'")
        print("[INFO] Waiting for camera frames via Zenoh...")
        try:
            while self._running:
                if self._goal_reached:
                    self.publish_cmd(0.0, 0.0)
                    time.sleep(0.2)
                    continue

                # 1. Full stop, then wait for the settle window so the next
                #    frame we trust was captured while genuinely stationary
                #    (no motion blur, no stale-frame-while-moving risk --
                #    see the 2026-08-10 Pi-dropout incident this closes).
                self.publish_cmd(0.0, 0.0)
                time.sleep(CFG.settle_time_s)

                with self.lock:
                    rgb = self.latest_rgb
                if rgb is None:
                    time.sleep(0.1)
                    continue

                # 2. ONE model decision for this leg.
                t0 = time.time()
                lin, ang, kind, detail = infer_cmd(rgb, self.instruction)
                infer_ms = (time.time() - t0) * 1000.0
                self.publish_trajectory(AGENT.get("last_trajectory_xy"))

                if kind == "stop":
                    self._stop_streak += 1
                    self.publish_explanation(
                        f"InternVLA-N1 [stop:{detail}] ({self._stop_streak}/{CFG.stop_confirm_count}) "
                        f"| instruction='{self.instruction}'"
                    )
                    if self._stop_streak >= CFG.stop_confirm_count:
                        if self._verify_stop_grounding(rgb):
                            print(f"\n{'=' * 50}\n  GOAL REACHED: '{self.instruction}'\n{'=' * 50}\n")
                            self._goal_reached = True
                            self.publish_cmd(0.0, 0.0)
                            self.publish_explanation(f"GOAL REACHED: '{self.instruction}'. Stopping.")
                        else:
                            self._stop_streak = 0  # not confirmed -- keep going, re-earn the streak on fresh frames
                    continue
                self._stop_streak = 0
                self._stop_grounding_rejects = 0

                self._infer_count += 1
                print(f"[PRED #{self._infer_count}] {kind}:{detail} "
                      f"lin={lin:.3f} ang={ang:.3f} ({infer_ms:.0f}ms)")
                self.publish_explanation(
                    f"InternVLA-N1 [{kind}:{detail}] -> lin={lin:.3f} ang={ang:.3f} "
                    f"| instruction='{self.instruction}'"
                )

                # 3. Execute this single decision for the leg -- lightweight
                #    independent safety checks keep running (depth hard-stop),
                #    but the heavy System-2/System-1 model is NOT re-invoked
                #    until the next leg.
                self._execute_leg(lin, ang, kind, detail)

                if time.time() - last_status > 10.0:
                    print(f"[STATUS] frames_rx={self._frame_count} "
                          f"inferences={self._infer_count} "
                          f"goal={'REACHED' if self._goal_reached else 'tracking'}")
                    last_status = time.time()
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
        finally:
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.publish_cmd(0.0, 0.0)
            self.odom.close()
            print("[INFO] Sent zero velocity. Goodbye.")

    def _execute_leg(self, lin: float, ang: float, kind: str, detail: str):
        """Hold ONE decided (lin, ang) for up to a bounded window, with a
        cheap per-poll depth hard-stop (no model call) and, at the end, a
        hardware-grounded stall check using real encoder+IMU displacement
        (self.odom) -- falls back to a simple vision frame-diff if fresh
        rover/rpm data isn't available. Ends with a full stop either way.

        Discrete actions get CFG.discrete_move_duration_s (short -- a
        bounded "glance", matching the paper's own "view adjustment action"
        framing, see that constant's comment), everything else (trajectory
        legs, System-1 actually ran with real model consensus) gets the
        full CFG.move_duration_s."""
        leg_cap = CFG.discrete_move_duration_s if kind == "discrete" else CFG.move_duration_s
        start_pose = (self.odom.x, self.odom.y, self.odom.theta)
        start_rpm_fresh = (self._last_rpm_rx_t is not None
                            and time.time() - self._last_rpm_rx_t < 2.0)
        with self.lock:
            start_frame = self.latest_rgb

        self.publish_cmd(lin, ang)
        leg_start = time.time()
        cut_short = None
        frame_now = None  # stays None if the loop body never runs (e.g. shutdown mid-leg)
        while self._running and (time.time() - leg_start) < leg_cap:
            time.sleep(CFG.leg_safety_poll_s)
            with self.lock:
                frame_now = self.latest_rgb
            if frame_now is None or AGENT.get("depth_model") is None:
                continue
            depth_now = AGENT["depth_model"].infer_image(frame_now[:, :, ::-1])
            obstacle_dist = _forward_cone_min_depth(depth_now)
            if obstacle_dist < CFG.navdp_estop_distance_m:
                cut_short = f"obstacle @ {obstacle_dist:.2f}m mid-leg"
                break

        elapsed = time.time() - leg_start
        self.publish_cmd(0.0, 0.0)  # full stop -- always, whether the leg ran to completion or was cut short

        dx = self.odom.x - start_pose[0]
        dy = self.odom.y - start_pose[1]
        dtheta = self.odom.theta - start_pose[2]
        achieved_dist = math.hypot(dx, dy)
        achieved_rot_deg = abs(math.degrees(dtheta))
        commanded_significant = abs(lin) > 0.02 or abs(ang) > 0.05
        rpm_still_fresh = (self._last_rpm_rx_t is not None
                            and time.time() - self._last_rpm_rx_t < 2.0)

        if cut_short:
            print(f"[SAFETY] {cut_short} -- leg cut short after {elapsed:.2f}s")
        print(f"[LEG] commanded lin={lin:+.3f} ang={ang:+.3f} for {elapsed:.2f}s -> "
              f"actual dist={achieved_dist:.3f}m dtheta={math.degrees(dtheta):+.1f}deg "
              f"(theta_src={self.odom.theta_source}, rpm_fresh={rpm_still_fresh})")

        if not commanded_significant or cut_short:
            return

        stalled = False
        if start_rpm_fresh and rpm_still_fresh and elapsed > leg_cap * 0.5:
            # Hardware-grounded: the wheels' own encoders (+ IMU for heading)
            # say how far the rover ACTUALLY moved, independent of whatever
            # the camera looked like -- see NodeConfig's odometry comment.
            stalled = achieved_dist < 0.02 and achieved_rot_deg < 3.0
        elif start_frame is not None:
            # Fallback: rover/rpm not (yet) publishing -- same grayscale
            # frame-diff heuristic the old per-tick loop used.
            end_frame = frame_now if frame_now is not None else start_frame
            small_a = np.asarray(Image.fromarray(start_frame).convert("L").resize((32, 24)), dtype=np.float32)
            small_b = np.asarray(Image.fromarray(end_frame).convert("L").resize((32, 24)), dtype=np.float32)
            stalled = float(np.abs(small_b - small_a).mean()) < CFG.stall_frame_diff_thresh

        if stalled:
            recover_ang = CFG.stall_backup_ang if ang >= 0 else -CFG.stall_backup_ang
            print(f"[STALL] commanded real motion but rover didn't move (dist={achieved_dist:.3f}m, "
                  f"dtheta={achieved_rot_deg:.1f}deg) -- backing up")
            self.publish_explanation(f"STALL detected -- backing up and re-assessing")
            self.publish_cmd(-CFG.stall_backup_lin, recover_ang)
            time.sleep(CFG.stall_recover_ticks * CFG.leg_safety_poll_s * 4)
            self.publish_cmd(0.0, 0.0)


# ================================================================
#  CLI & entry point
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="InternVLA-N1 DualVLN Zenoh inference node (GPU side, camera-only, "
                     "native System-2+System-1 dual-system, arXiv 2512.08186)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pi-ip", type=str, default=None,
                   help="Pi/Isaac IP for explicit Zenoh peer; omit for multicast scouting")
    p.add_argument("--instruction", type=str, default=CFG.instruction)
    p.add_argument("--predict-hz", type=float, default=CFG.predict_hz)
    p.add_argument("--max-linear", type=float, default=CFG.max_linear)
    p.add_argument("--max-angular", type=float, default=CFG.max_angular)
    p.add_argument("--model-path", type=str, default=CFG.model_path,
                   help="InternVLA-N1-DualVLN checkpoint (contains both System-2 and System-1 weights)")
    p.add_argument("--internnav-repo", type=str, default=CFG.internnav_repo)
    return p.parse_args()


def main():
    args = parse_args()
    CFG.instruction = args.instruction
    CFG.predict_hz = args.predict_hz
    CFG.max_linear = args.max_linear
    CFG.max_angular = args.max_angular
    CFG.model_path = args.model_path
    CFG.internnav_repo = args.internnav_repo

    load_model()

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting (auto-discover)")
    session = zenoh.open(config)
    print("[INFO] Zenoh session opened.")

    node = InternVLADualVLNZenohNode(session)
    node.spin()
    session.close()


if __name__ == "__main__":
    main()
