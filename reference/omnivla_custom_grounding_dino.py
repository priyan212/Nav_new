#!/usr/bin/env python3
"""
OmniVLA Custom — Grounded Observer Navigation Node
===================================================
Runs on the GPU machine.  Connects to the Pi's zenoh-bridge-ros2dds.

Features
--------
  • Dynamic-prompting GUI  (Tkinter dark-theme window)
    - Change instruction at any time; ↑/↓ for history
    - Live status indicators: Camera, Paused, Goal
  • Terminal live-command interface  (fallback / parallel)
  • Goal-reached detection (waypoint heuristic + optional OWL-ViT/CLIP visual observer)
  • Publishes omnivla/waypoints as nav_msgs/Path  →  shows in RViz
  • RViz auto-launch with rover config

Usage
-----
  conda activate omnivla
  cd /mnt/bigdisk/motion_planning/OmniVLA

  python inference/omnivla_custom.py --pi-ip 10.203.123.125
  python inference/omnivla_custom.py --pi-ip 10.203.123.125 --no-gui
  python inference/omnivla_custom.py --pi-ip 10.203.123.125 --no-rviz

On the Pi
---------
  ros2 launch omnivla_nav rover_bringup.launch.py

RViz topics
-----------
  /omnivla/waypoints   – predicted path (nav_msgs/Path, frame: base_link)
  /cmd_vel             – velocity commands
  /omnivla/grounding   – OWL-ViT/CLIP observer status
"""

import sys
import os
import struct
import time
import math
import argparse
import subprocess
import queue
from collections import deque
from threading import Lock, Thread
from typing import Type, Optional, List, Tuple
import tkinter as tk

# ── Enable ROS2 network discovery (GPU ↔ Pi communication) ──
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")  # Allow network communication
os.environ.setdefault("ROS_DOMAIN_ID", "0")       # Match Pi's domain

import numpy as np
try:
    import cv2  # fast JPEG decode that releases the GIL (vs PIL)
except Exception:
    cv2 = None
from PIL import Image, ImageDraw, ImageFont
try:
    from PIL import ImageTk as _ImageTk
except ImportError:
    _ImageTk = None
import torch
import torch.nn as nn

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found.  pip install eclipse-zenoh")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoImageProcessor,
)

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None

try:
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    CLIPProcessor = None
    CLIPModel = None

try:
    from transformers import AutoModelForDepthEstimation
except Exception:
    AutoModelForDepthEstimation = None


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ================================================================
#  CDR Helpers  (Common Data Representation — DDS wire format)
# ================================================================
class CDRReader:
    """Minimal CDR deserialiser for ROS 2 messages."""

    def __init__(self, data: bytes):
        self.data   = data
        self.le     = data[1] in (0x01, 0x11)
        self.end    = "<" if self.le else ">"
        self.offset = 4       # skip 4-byte encapsulation header
        self.base   = 4

    def _align(self, n: int):
        pos = self.offset - self.base
        rem = pos % n
        if rem:
            self.offset += n - rem

    def read_uint8(self) -> int:
        v = self.data[self.offset]; self.offset += 1; return v

    def read_int32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "i", self.data, self.offset)
        self.offset += 4; return v

    def read_uint32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "I", self.data, self.offset)
        self.offset += 4; return v

    def read_float32(self) -> float:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "f", self.data, self.offset)
        self.offset += 4; return v

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8; return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset: self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length; return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data  = self.data[self.offset: self.offset + count]
        self.offset += count; return data

    def read_sequence_float32(self) -> tuple:
        count = self.read_uint32()
        self._align(4)
        vals = struct.unpack_from(f"{self.end}{count}f", self.data, self.offset)
        self.offset += count * 4; return vals


class CDRWriter:
    """Minimal CDR serialiser (little-endian)."""

    def __init__(self):
        self.buf  = bytearray(b"\x00\x01\x00\x00")   # CDR LE header
        self.base = 4

    def _align(self, n: int):
        pos = len(self.buf) - self.base
        rem = pos % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_int32(self, v: int):
        self._align(4); self.buf += struct.pack("<i", v)

    def write_uint32(self, v: int):
        self._align(4); self.buf += struct.pack("<I", v)

    def write_float64(self, v: float):
        self._align(8); self.buf += struct.pack("<d", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded)); self.buf += encoded

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


# ================================================================
#  Message parsers / serialisers
# ================================================================
def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/Image CDR → numpy RGB (H, W, 3)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()   # header
    height = r.read_uint32(); width = r.read_uint32()
    encoding = r.read_string()
    r.read_uint8(); r._align(4); r.read_uint32()       # is_bigendian, step
    pixel_data = r.read_sequence_uint8()
    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None
    if encoding.lower() == "rgb8":
        return img[:, :, :3]
    elif encoding.lower() == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    elif img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/CompressedImage CDR → numpy RGB (H, W, 3).

    Used as a fallback when the raw image_raw stream is not available.
    JPEG-compressed images are ~10-15× smaller than raw YUYV→RGB8, so
    this also reduces Zenoh bandwidth significantly.
    """
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()   # header (seq, stamp, frame_id)
    _fmt = r.read_string()                              # "jpeg" / "png"
    jpeg_bytes = r.read_sequence_uint8()
    try:
        # cv2.imdecode is a C call that RELEASES the GIL, so the Zenoh callback
        # thread no longer competes with the (slow, GIL-holding) DINO/torch work
        # on the inference thread — this keeps the received frame rate high.
        # cv2 decodes to BGR; OmniVLA expects RGB, so flip channels.
        if cv2 is not None:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Fallback: PIL (slower, holds GIL longer) if cv2 is unavailable.
        import io as _io
        img = Image.open(_io.BytesIO(bytes(jpeg_bytes))).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def parse_string(cdr_data: bytes) -> str:
    """std_msgs/String CDR → str."""
    return CDRReader(cdr_data).read_string()


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    """geometry_msgs/Twist → CDR bytes."""
    w = CDRWriter()
    w.write_float64(linear_x); w.write_float64(0.0); w.write_float64(0.0)
    w.write_float64(0.0);      w.write_float64(0.0); w.write_float64(angular_z)
    return w.to_bytes()


def serialize_string(text: str) -> bytes:
    """std_msgs/String → CDR bytes."""
    w = CDRWriter(); w.write_string(text); return w.to_bytes()


def serialize_path(points_xy: List[Tuple[float, float]],
                   frame_id: str = "base_link") -> bytes:
    """nav_msgs/Path → CDR bytes.
    points_xy: list of (x, y) in robot-frame meters."""
    w = CDRWriter()
    # Path.header
    w.write_int32(0); w.write_uint32(0)
    w.write_string(frame_id)
    # Path.poses: geometry_msgs/PoseStamped[]
    w.write_uint32(len(points_xy))
    for px, py in points_xy:
        # PoseStamped.header
        w.write_int32(0); w.write_uint32(0)
        w.write_string(frame_id)
        # PoseStamped.pose.position
        w.write_float64(float(px))
        w.write_float64(float(py))
        w.write_float64(0.0)
        # PoseStamped.pose.orientation (identity quaternion)
        w.write_float64(0.0)   # qx
        w.write_float64(0.0)   # qy
        w.write_float64(0.0)   # qz
        w.write_float64(1.0)   # qw
    return w.to_bytes()


# ================================================================
#  Robot Geometry  (display metadata only; LiDAR logic removed)
# ================================================================
ROVER_LENGTH = 0.482   # m  overall (+X = forward)
ROVER_WIDTH  = 0.380   # m  overall (+Y = left)


# ================================================================
#  Configuration
# ================================================================
class NodeConfig:
    vla_path: str = "./omnivla-original"
    resume_step: int = 120_000
    num_images_in_input: int = 2
    # PD-controller
    max_linear: float = 0.15
    max_angular: float = 1.2   # raised from 0.8 — gives full differential steer; ESP32 normalises by max_angular too
    tick_dt: float = 1.0 / 3.0
    metric_waypoint_spacing: float = 0.1
    waypoint_select: int = 2   # lowered from 4 — nearer look-ahead → more reactive turning
    # Angular slew-rate limiter — maximum change in angular velocity per tick (rad/s).
    # Prevents abrupt full-speed turns that swing the camera past the target object.
    # At 3 Hz with ang_slew_rate=1.5 it takes ~3 ticks (~1 s) to ramp from 0 → max_angular=1.2.
    # Increase for faster response, decrease for smoother (less wobbly) turns.
    ang_slew_rate: float = 1.5   # rad/s per tick  (set very high to effectively disable)
    # Angular boost — a tiny constant added in the direction of any commanded turn.
    # Compensates for the rover's skid-steer friction: the model often predicts a
    # small ang that is physically too weak to rotate the chassis.  This adds just
    # enough extra differential to make every turn slightly more aggressive without
    # affecting straight-line runs (boost is only applied when |ang| > 0.01 rad/s).
    ang_boost: float = 0.05   # rad/s extra on top of model command  (tune 0.0 – 0.15)
    # Navigation
    predict_hz: float = 3.0
    instruction: str = " " # "go towards the nearest door"

    # Goal-reached detection
    goal_reached_threshold: float = 0.015   # waypoint magnitude (m) = "at goal"
    goal_reached_count: int = 1            # consecutive ticks to confirm

    # ----------------------------------------------------------------
    # Trajectory-derivative stopping (OmniVLA semantic convergence) (experimental)
    # ----------------------------------------------------------------
    trajectory_buffer_size       = 10
    trajectory_delta_threshold   = 0.008
    trajectory_compression_ratio = 0.80   #Experimental

    # ----------------------------------------------------------------
    # Convergence-aware steering damping (experimental)
    # ----------------------------------------------------------------

    steering_damping_start_ratio = 0.85
    steering_damping_end_ratio   = 0.45

    # Steering-damping caps are expressed as FRACTIONS of the active
    # max_angular (set via --max-angular) so the CLI flag actually controls
    # turning speed. far = full authority away from the goal, near = reduced
    # authority as the trajectory converges.
    max_angular_far_frac  = 1.0    # full max_angular far from convergence
    max_angular_near_frac = 1.0    # was 0.5 — that halved turning near the goal
                                   # and made angular feel weak/sluggish. The
                                   # earlier 'working fine' driver had no such
                                   # damping; keep full authority (set <1.0 only
                                   # if you see last-second steering oscillation).


    # ----------------------------------------------------------------
    # Trajectory freezing near convergence (experimental)
    # ----------------------------------------------------------------
    # OFF by default: this reuses OLD waypoints when OmniVLA's path changes a
    # lot, which distorts OmniVLA's own (already-good) navigation and hurts
    # path quality. Trust OmniVLA's fresh prediction each tick instead.
    enable_trajectory_freeze: bool = False
    freeze_ratio_threshold = 0.45
    freeze_max_delta_E = 0.03
    freeze_hold_ticks = 4
    # Only freeze on a genuinely violent steering flip, not on an intentional
    # turn. Raised from 0.25 so normal turns aren't suppressed near the goal.
    freeze_ang_delta_threshold = 0.6




    trajectory_variance_thresh   = 0.002
    trajectory_stable_count      = 5
    trajectory_min_runtime_sec   = 3.0


    
    # Frame freshness: warn / skip inference if camera frame is older than this
    max_frame_age_s: float = 5.0   # relaxed from 3s — WiFi jitter can cause gaps
                                    # without the camera actually dying; 5s = 10 frames @ 2fps

    # Hardware I/O fix from enhanced version: prefer the JPEG-compressed
    # camera stream and restart the Pi camera node if frames go stale.
    # This keeps Zenoh bandwidth low so camera + ESP32 cmd_vel stay responsive.
    prefer_compressed: bool = True
    # Camera auto-restart watchdog — DISABLED by default.
    # Restarting the Pi camera on every stale frame did far more harm than good:
    # each restart is a ~13 s SSH kill+relaunch during which the rover stops, and
    # repeated restarts piled up stale v4l2_camera processes that fought over the
    # device (mmap contention) → MORE stalls → more restarts (a vicious cycle that
    # caused the "moves a little then stops" stutter). Transient frame gaps (WiFi
    # jitter, brief hiccups) recover on their own, so we simply tolerate them.
    # Only enable this for a genuinely dead camera, via ENABLE_CAM_WATCHDOG=1.
    camera_watchdog_enabled: bool = False
    camera_stale_restart_s: float = 60.0  # only used if the watchdog is enabled

    # Grounded observer layer (Grounding DINO + CLIP).
    # OmniVLA still produces the motion; this layer verifies target visibility
    # and decides when the requested object is visually close enough to stop.
    enable_grounding: bool = True
    dino_model: str = "IDEA-Research/grounding-dino-base"
    clip_model: str = "openai/clip-vit-base-patch32"
    grounding_device: str = "cpu"       # use "cuda" only if GPU memory allows
    dino_conf_threshold: float = 0.10       # detector score threshold
    dino_stop_area_ratio: float = 0.15      # stop when target box covers this fraction
    dino_max_stop_area: float = 0.90        # boxes >= this are degenerate full-frame
                                            # false detections; never treat as a stop
    dino_center_tolerance: float = 0.85     # normalized center offset
    dino_stop_frames: int = 2               # consecutive detections required to stop
    grounding_period: int = 2               # run Grounding DINO every N ticks (was
                                            # 1). DINO's CUDA kernel is missing so
                                            # it runs slow pure-PyTorch attention
                                            # that hogs the GIL and starves the
                                            # camera callback; every-2-ticks keeps
                                            # target tracking while freeing the loop
    clip_bias_enabled: bool = True
    # clip_bias_gain: float = 0.18           # angular nudge from CLIP left/right crop similarity
    clip_bias_gain: float = 0.05   #Experimental
    # clip_bias_max: float = 0.12            # max rad/s added to angular velocity
    clip_bias_max: float = 0.04  #Experimental

    # ----------------------------------------------------------------
    # Target-object centering (Grounding DINO bounding box)
    # ----------------------------------------------------------------
    # Steer to keep the detected target horizontally centered in the frame.
    # center_offset_signed = (cx - w/2)/(w/2)  in [-1, 1]:
    #   >0 => target is to the RIGHT => turn right (negative ang, since
    #         positive angular turns left).
    # The correction is proportional and rides on top of OmniVLA's steering;
    # the dynamic steering-damping clamp bounds the total ang to max_angular.
    center_enabled: bool = True
    # Geometry-based centering: the target's pixel offset is converted into a
    # TRUE angular error using the camera field of view, so the turn command is
    # physically correct rather than a magic gain. See the control block below.
    #
    #   center_offset_signed ∈ [-1, 1]   (+ = target to the RIGHT of center)
    #   theta_err = center_offset_signed · (HFOV/2)      [rad]
    #   center_corr = -center_kp · theta_err             [rad/s]  (final clamp = max_angular)
    #
    camera_hfov_deg: float = 60.0   # camera horizontal FOV (Logitech C615 ≈ 60° at 4:3).
                                    # MUST match the real camera for accurate turns.
    center_kp: float = 2.2          # proportional gain (1/s). With HFOV=60°, an edge
                                    # target (30° error) → ~1.15 rad/s ≈ full max_angular,
                                    # tapering smoothly to 0 as the object nears center.
    center_deadband: float = 0.05   # |offset| below this = "centered enough" (no turn)
    center_min_cmd: float = 0.12    # rad/s stiction floor: minimum turn applied OUTSIDE the
                                    # deadband so small residual offsets still get corrected
                                    # instead of stalling. 0 disables.
    # (legacy, retained for CLI back-compat; no longer the centering mechanism)
    center_gain: float = 0.7
    center_max: float = 0.8

    # ----------------------------------------------------------------
    # Vision-only static-obstacle avoidance (monocular depth)
    # ----------------------------------------------------------------
    # No LiDAR on this rover. A small monocular depth model estimates a
    # RELATIVE disparity map (larger value = nearer). We look at a horizontal
    # band at rover height, split it left/center/right, and steer AWAY from
    # whichever side is closer while slowing down if the center is looming.
    # Depth is relative (not metric), so thresholds are normalized 0..1 and
    # need on-robot tuning via the --avoid-* flags.
    avoid_enabled: bool = False   # disabled by default — monocular depth reads floor/walls as
                                    # "near" in indoor environments and wrongly cuts angular velocity.
                                    # Re-enable with ENABLE_AVOID=1 ./launch_eye_vlm.sh
    depth_model: str = "LiheYoung/depth-anything-small-hf"
    avoid_period: int = 2           # run depth every N ticks (was 1 — halves per-tick load)
    # Vertical band of the frame to treat as the forward obstacle zone
    # (0 = top of image, 1 = bottom). Excludes ceiling/sky and the floor at
    # the rover's feet.
    avoid_band_top: float = 0.35
    avoid_band_bottom: float = 0.80
    # Horizontal region splits (fractions of frame width)
    avoid_left_edge: float = 0.40   # left region  = [0, 0.40]
    avoid_right_edge: float = 0.60  # right region = [0.60, 1.0]
    avoid_center_lo: float = 0.30   # center region for looming = [0.30, 0.70]
    avoid_center_hi: float = 0.70
    # Closeness (normalized disparity percentile) considered a near obstacle
    avoid_percentile: float = 90.0
    avoid_near_thresh: float = 0.80  # raised further — floor/walls read as 0.7-0.75
                                     # in typical indoor scenes; 0.80 prevents them
                                     # from spuriously cutting speed/angular
    avoid_gain: float = 1.2          # rad/s per unit relative-closeness difference
    avoid_rel_margin: float = 0.08   # min L/R closeness difference before steering
                                     # (suppresses monocular-depth noise; open or
                                     # symmetric scenes → zero avoidance)
    avoid_max: float = 0.7           # cap on the avoidance angular term
    avoid_slow_scale: float = 0.4    # how strongly a looming center cuts speed
                                     # (was 1.0 → could drop lin to 0 on a false
                                     # "near" reading; 0.4 slows but never stalls)
    # Don't treat the detected goal object as a forward obstacle when it is
    # roughly centered ahead and big enough (so the rover can approach it).
    avoid_target_center_tol: float = 0.35
    avoid_target_min_area: float = 0.05

CFG = NodeConfig()


# ================================================================
#  Grounded observer helpers (OWL-ViT + CLIP)
# ================================================================
def extract_target_phrase(instruction: str) -> str:
    """Best-effort extraction of the object/place phrase for open-vocab detection.

    The original instruction is still sent to OmniVLA unchanged. This function only
    creates a detector prompt such as "red bucket" from "go to the red bucket".
    """
    import re as _re
    s = instruction.strip().lower()

    # If the command is a simple navigation direction or behavior that does NOT specify a target object,
    # return an empty string to indicate that Grounding DINO should be skipped.
    direction_patterns = [
        r"^(?:please\s+)?(?:turn|steer|rotate|spin|veer|face|look)\s+(?:left|right|around|away)\b",
        r"^(?:please\s+)?(?:go|move|drive|walk|run|heading|head|travel|step)\s+(?:forward|backward|backwards|back|straight|ahead|left|right)\b",
        r"^(?:please\s+)?(?:stop|halt|freeze|unfreeze|pause|resume|quit|exit)\b",
    ]
    for pat in direction_patterns:
        if _re.match(pat, s):
            return ""

    s = _re.sub(r"\b(and then|then)\b.*$", "", s).strip()
    s = _re.sub(r"\band\s+stop\b", "", s).strip()
    s = _re.sub(r"\bstop\s+when\s+you\s+see\b", "", s).strip()
    patterns = [
        r"^(?:please\s+)?(?:go|move|drive|navigate|head)\s+(?:to|toward|towards|near|into|inside)\s+(?:the\s+)?(.+)$",
        r"^(?:please\s+)?(?:find|approach|reach|follow)\s+(?:the\s+)?(.+)$",
        r"^(?:please\s+)?(?:look\s+for)\s+(?:the\s+)?(.+)$",
    ]
    for pat in patterns:
        m = _re.match(pat, s, flags=_re.IGNORECASE)
        if m:
            target = m.group(1).strip()
            break
    else:
        target = s
    # Remove route/behavior tails that are not the object name.
    target = _re.split(r"\b(?:after|before|while|along|around|between|through|via|past)\b", target)[0].strip()
    target = _re.sub(r"^(?:the|a|an)\s+", "", target).strip()

    if target in ("left", "right", "forward", "backward", "backwards", "straight", "stop", "halt", "none"):
        return ""

    return target or instruction.strip()


class GroundedObserver:
    """Open-vocabulary target observer.

    - Grounding DINO grounds the target phrase in the current image.
    - CLIP left/center/right crop similarity provides a small steering nudge.
    - A large, centered, confident target box becomes a hard visual stop.
    """

    def __init__(self):
        self.enabled = bool(CFG.enable_grounding)
        self.device = torch.device(CFG.grounding_device if CFG.grounding_device == "cuda" and torch.cuda.is_available() else "cpu")
        self.dino_processor = self.dino_model = None
        self.clip_processor = self.clip_model = None
        self.last = {
            "target": "", "score": 0.0, "area": 0.0, "center_offset": 9.9,
            "clip_bias": 0.0, "stop": False, "ok": False, "reason": "not-run",
        }
        self._stop_count = 0
        if not self.enabled:
            return
        try:
            if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
                raise RuntimeError("transformers Grounding DINO classes are unavailable")
            log(f"Loading Grounding DINO observer on {self.device}: {CFG.dino_model}")
            self.dino_processor = AutoProcessor.from_pretrained(CFG.dino_model)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(CFG.dino_model).to(self.device).eval()
        except Exception as e:
            log(f"WARN: Grounding DINO observer disabled: {e}")
            self.enabled = False
            return
        try:
            if CLIPProcessor is None or CLIPModel is None:
                raise RuntimeError("transformers CLIP classes are unavailable")
            log(f"Loading CLIP crop observer on {self.device}: {CFG.clip_model}")
            self.clip_processor = CLIPProcessor.from_pretrained(CFG.clip_model)
            self.clip_model = CLIPModel.from_pretrained(CFG.clip_model).to(self.device).eval()
        except Exception as e:
            log(f"WARN: CLIP crop bias disabled: {e}")
            self.clip_processor = self.clip_model = None

    def reset(self):
        self._stop_count = 0
        self.last.update({"score": 0.0, "area": 0.0, "center_offset": 9.9,
                          "clip_bias": 0.0, "stop": False, "ok": False, "reason": "reset"})

    def _clip_bias(self, image: Image.Image, target: str) -> float:
        if not CFG.clip_bias_enabled or self.clip_model is None or not target:
            return 0.0
        w, h = image.size
        # Overlapping crops: left, center, right.
        crops = [
            image.crop((0, 0, int(w * 0.58), h)),
            image.crop((int(w * 0.21), 0, int(w * 0.79), h)),
            image.crop((int(w * 0.42), 0, w, h)),
        ]
        try:
            inputs = self.clip_processor(text=[target], images=crops, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.clip_model(**inputs).logits_per_image[:, 0]
                probs = logits.softmax(dim=0).detach().float().cpu().numpy()
            # Positive angular velocity turns left. If left crop is most similar, add positive bias.
            bias = float((probs[0] - probs[2]) * CFG.clip_bias_gain)
            return float(np.clip(bias, -CFG.clip_bias_max, CFG.clip_bias_max))
        except Exception as e:
            self.last["reason"] = f"clip-error:{e}"
            return 0.0

    def observe(self, rgb: np.ndarray, target: str) -> dict:
        if not self.enabled or not target:
            self.last.update({"target": target, "ok": False, "stop": False, "reason": "disabled-or-empty-target"})
            return self.last
        image = Image.fromarray(rgb).convert("RGB")
        w, h = image.size
        bias = self._clip_bias(image, target)
        try:
            inputs = self.dino_processor(text=target, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=CFG.dino_conf_threshold,
                text_threshold=CFG.dino_conf_threshold,
                target_sizes=[(h, w)]
            )[0]
            
            if len(results["scores"]) == 0:
                self._stop_count = 0
                self.last = {"target": target, "score": 0.0, "area": 0.0, "center_offset": 9.9,
                             "clip_bias": bias, "stop": False, "ok": False, "reason": "not-detected"}
                return self.last
            scores = results["scores"].detach().float().cpu().numpy()
            boxes = results["boxes"].detach().float().cpu().numpy()
            idx = int(np.argmax(scores))
            x1, y1, x2, y2 = boxes[idx]
            box_w = max(0.0, float(x2 - x1)); box_h = max(0.0, float(y2 - y1))
            area = (box_w * box_h) / float(w * h)
            cx = float((x1 + x2) / 2.0)
            center_offset_signed = (cx - w / 2.0) / (w / 2.0)
            center_offset = abs(center_offset_signed)
            centered = center_offset <= CFG.dino_center_tolerance
            # A near-full-frame box (area ~1.0) is a degenerate false detection
            # (Grounding DINO returns the whole image for vague prompts / bad
            # frames). It must NOT count as "reached", or the rover stops on the
            # first frame and never moves.
            degenerate = area >= CFG.dino_max_stop_area
            large = (CFG.dino_stop_area_ratio <= area) and not degenerate
            confident = float(scores[idx]) >= CFG.dino_conf_threshold
            if confident and large and centered:
                self._stop_count += 1
            else:
                self._stop_count = 0
            stop = self._stop_count >= CFG.dino_stop_frames
            reason = "stop" if stop else (
                f"tracking centered={centered} large={large} degenerate={degenerate} "
                f"score={float(scores[idx]):.2f}/{CFG.dino_conf_threshold:.2f} "
                f"area={area:.3f}/{CFG.dino_stop_area_ratio:.3f} "
                f"center={center_offset:.2f}/{CFG.dino_center_tolerance:.2f}"
            )
            self.last = {"target": target, "score": float(scores[idx]), "area": float(area),
                         "center_offset": float(center_offset), "center_offset_signed": float(center_offset_signed), 
                         "clip_bias": bias,
                         "stop": bool(stop), "ok": True, "reason": reason,
                         "box": [float(x1), float(y1), float(x2), float(y2)]}
            return self.last
        except Exception as e:
            self._stop_count = 0
            self.last = {"target": target, "score": 0.0, "area": 0.0, "center_offset": 9.9,
                         "clip_bias": bias, "stop": False, "ok": False, "reason": f"dino-error:{e}"}
            return self.last


class DepthObstacleAvoider:
    """Vision-only static-obstacle avoidance via monocular depth.

    Runs a small monocular depth model on the camera frame to get a RELATIVE
    disparity map (larger = nearer). A horizontal band at rover height is split
    into left / center / right regions; the rover steers away from whichever
    side is closer and slows down when the center is looming.

    Sign convention matches the rest of the driver: positive angular = turn
    left. So a near obstacle on the LEFT returns a negative (right) correction.
    """

    def __init__(self):
        self.enabled = bool(CFG.avoid_enabled)
        self.device = torch.device(
            CFG.grounding_device if CFG.grounding_device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.processor = None
        self.model = None
        self.last = {
            "avoid": 0.0, "block": 0.0, "left": 0.0, "center": 0.0, "right": 0.0,
            "ok": False, "reason": "not-run",
        }
        if not self.enabled:
            return
        try:
            if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
                raise RuntimeError("transformers depth-estimation classes are unavailable")
            log(f"Loading depth obstacle avoider on {self.device}: {CFG.depth_model}")
            self.processor = AutoImageProcessor.from_pretrained(CFG.depth_model)
            self.model = AutoModelForDepthEstimation.from_pretrained(CFG.depth_model).to(self.device).eval()
        except Exception as e:
            log(f"WARN: depth obstacle avoider disabled: {e}")
            self.enabled = False

    def observe(self, rgb: np.ndarray) -> dict:
        if not self.enabled or self.model is None:
            return self.last
        try:
            img = Image.fromarray(rgb)
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                depth = self.model(**inputs).predicted_depth  # (1, H, W), larger = nearer
            d = depth[0].float().cpu().numpy()
            dmin, dmax = float(d.min()), float(d.max())
            dn = (d - dmin) / (dmax - dmin + 1e-6)            # normalized disparity 0..1

            H, W = dn.shape
            y0 = int(H * CFG.avoid_band_top)
            y1 = int(H * CFG.avoid_band_bottom)
            band = dn[max(0, y0):max(y0 + 1, y1), :]
            bw = band.shape[1]

            left   = band[:, :max(1, int(bw * CFG.avoid_left_edge))]
            right  = band[:, int(bw * CFG.avoid_right_edge):]
            center = band[:, int(bw * CFG.avoid_center_lo):int(bw * CFG.avoid_center_hi)]

            pct = CFG.avoid_percentile
            pl = float(np.percentile(left,   pct)) if left.size   else 0.0
            pr = float(np.percentile(right,  pct)) if right.size  else 0.0
            pc = float(np.percentile(center, pct)) if center.size else 0.0

            # COMPARATIVE avoidance. Monocular depth is per-frame min/max
            # normalized, so the ABSOLUTE closeness clusters ~0.5 regardless of
            # scene (measured: L≈C≈R≈0.51 in an open room) and a fixed threshold
            # never fires. Instead steer away from whichever side is RELATIVELY
            # closer than the other, and slow when the CENTER is closer than the
            # more-open side (something looming ahead). A margin suppresses noise
            # so open/symmetric scenes correctly produce zero avoidance.
            m = CFG.avoid_rel_margin
            avoid = 0.0
            side_diff = pr - pl                       # >0: right nearer, <0: left nearer
            if side_diff > m:                         # right nearer  → steer left (positive)
                avoid += CFG.avoid_gain * (side_diff - m)
            elif -side_diff > m:                      # left nearer   → steer right (negative)
                avoid -= CFG.avoid_gain * (-side_diff - m)
            avoid = float(np.clip(avoid, -CFG.avoid_max, CFG.avoid_max))

            # Forward-blocked 0..1: how much the center is nearer than the most
            # open side (a wall/obstacle straight ahead), past the noise margin.
            opener = min(pl, pr)
            block = float(np.clip((pc - opener - m) / max(1e-6, 1.0 - m), 0.0, 1.0))

            self.last = {"avoid": avoid, "block": block, "left": pl, "center": pc,
                         "right": pr, "ok": True, "reason": "ok"}
            return self.last
        except Exception as e:
            self.last = {"avoid": 0.0, "block": 0.0, "left": 0.0, "center": 0.0,
                         "right": 0.0, "ok": False, "reason": f"depth-error:{e}"}
            return self.last


# ================================================================
#  Model helpers
# ================================================================
def _strip_ddp(sd: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _load_ckpt(name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{name}--{step}_checkpoint.pt")) \
            and name == "pose_projector":
        name = "proprio_projector"
    p = os.path.join(path, f"{name}--{step}_checkpoint.pt")
    log(f"  <- {p}")
    return _strip_ddp(torch.load(p, map_location=device))


def _init_module(cls: Type[nn.Module], name: str, path: str, step: int,
                 device: torch.device, kwargs: dict,
                 bf16: bool = False) -> nn.Module:
    m = cls(**kwargs)
    m.load_state_dict(_load_ckpt(name, path, step))
    if bf16:
        m = m.to(torch.bfloat16)
    return m.to(device).eval()


# ================================================================
#  Model loading
# ================================================================
MODEL: dict = {}


def load_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Loading OmniVLA on {device} ...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    processor = AutoProcessor.from_pretrained(CFG.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CFG.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(CFG.num_images_in_input)
    vla.eval()
    pose_proj = _init_module(
        ProprioProjector, "pose_projector", CFG.vla_path, CFG.resume_step, device,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},
    )
    action_head = _init_module(
        L1RegressionActionHead_idcat, "action_head", CFG.vla_path, CFG.resume_step, device,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
        bf16=True,
    )
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1
    )
    MODEL.update(
        vla=vla, action_head=action_head, pose_projector=pose_proj,
        device=device, num_patches=num_patches,
        action_tokenizer=ActionTokenizer(processor.tokenizer),
        processor=processor,
    )
    log("All models loaded.\n")


# ================================================================
#  Inference
# ================================================================
def predict_actions(image_pil: Image.Image, instruction: str,
                    modality_id: int = 7,
                    prev_image_pil: Optional[Image.Image] = None) -> np.ndarray:
    """Run OmniVLA -> return waypoints (8, 4).
    
    Args:
        image_pil: Current camera frame.
        instruction: Navigation instruction string.
        modality_id: Modality identifier.
        prev_image_pil: Previous camera frame for temporal reasoning.
                       If None, current frame is duplicated (degraded mode).
    """
    vla         = MODEL["vla"]
    action_head = MODEL["action_head"]
    pose_proj   = MODEL["pose_projector"]
    device      = MODEL["device"]
    num_patches = MODEL["num_patches"]
    action_tok  = MODEL["action_tokenizer"]
    processor   = MODEL["processor"]

    dummy     = np.random.rand(NUM_ACTIONS_CHUNK, ACTION_DIM)
    chunk_str = action_tok(dummy[0]) + "".join(action_tok(dummy[1:]))
    conversation = [
        {"from": "human", "value": f"What action should the robot take to {instruction}?"},
        {"from": "gpt",   "value": chunk_str},
    ]
    pb = PurePromptBuilder("openvla")
    for t in conversation:
        pb.add_turn(t["from"], t["value"])

    input_ids = torch.tensor(
        processor.tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
    )
    labels = input_ids.clone()
    labels[:-(len(chunk_str) + 1)] = -100

    pv = processor.image_processor.apply_transform(image_pil)
    if prev_image_pil is not None:
        pv_prev = processor.image_processor.apply_transform(prev_image_pil)
    else:
        pv_prev = pv  # fallback: duplicate current frame (degraded)
    pixel_values = torch.cat((pv_prev.unsqueeze(0), pv.unsqueeze(0)), dim=1)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=input_ids.unsqueeze(0).to(device),
            attention_mask=input_ids.unsqueeze(0).ne(
                processor.tokenizer.pad_token_id).to(device),
            pixel_values=pixel_values.to(torch.bfloat16).to(device),
            modality_id=torch.as_tensor(
                [modality_id], device=device, dtype=torch.bfloat16).unsqueeze(0),
            labels=labels.unsqueeze(0).to(device),
            output_hidden_states=True,
            proprio=torch.zeros(1, POSE_DIM, device=device, dtype=torch.bfloat16),
            proprio_projector=pose_proj,
        )

    gt_ids = labels.unsqueeze(0)[:, 1:].to(device)
    c_mask  = get_current_action_mask(gt_ids)
    n_mask  = get_next_actions_mask(gt_ids)
    act_hs  = output.hidden_states[-1][:, num_patches:-1]
    act_hs  = (
        act_hs[c_mask | n_mask]
        .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    predicted = action_head.predict_action(
        act_hs,
        torch.as_tensor([float(modality_id)], device=device, dtype=torch.bfloat16),
    )
    return predicted.detach().float().cpu().numpy()[0]   # (8, 4)


def waypoints_to_path(waypoints: np.ndarray) -> List[Tuple[float, float]]:
    """Convert OmniVLA waypoints (8,4) -> cumulative (x, y) in robot frame (m)."""
    pts: List[Tuple[float, float]] = []
    cx = cy = 0.0
    for wp in waypoints:
        cx += wp[0] * CFG.metric_waypoint_spacing
        cy += wp[1] * CFG.metric_waypoint_spacing
        pts.append((cx, cy))
    return pts


# ================================================================
#  PD controller
# ================================================================
def _clip_angle(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def pd_controller(waypoints: np.ndarray) -> Tuple[float, float]:
    wp = waypoints[CFG.waypoint_select].copy()
    wp[:2] *= CFG.metric_waypoint_spacing
    dx, dy, hx, hy = wp
    EPS, DT = 1e-8, CFG.tick_dt

    if abs(dx) < EPS and abs(dy) < EPS:
        lin, ang = 0.0, 1.0 * _clip_angle(math.atan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        lin, ang = 0.0, 1.0 * np.sign(dy) * math.pi / (2.0 * DT)
    else:
        # atan2 handles all quadrants correctly (atan(dy/dx) is wrong when dx<0)
        lin, ang = dx / DT, math.atan2(dy, dx) / DT

    # Clip linear and angular velocities directly to their maximum limits
    # to avoid excessive downscaling that prevents responsive turning.
    lin = float(np.clip(lin, 0.0, CFG.max_linear))
    ang = float(np.clip(ang, -CFG.max_angular, CFG.max_angular))

    # Angular boost: add a small constant in the turn direction so the chassis
    # actually rotates even when the model predicts a hair-fine angular command.
    if abs(ang) > 0.01:   # only boost genuine turn commands, not dead-band noise
        ang = float(np.clip(
            ang + math.copysign(CFG.ang_boost, ang),
            -CFG.max_angular, CFG.max_angular
        ))

    return lin, ang



def make_explanation(instruction: str, waypoints: np.ndarray,
                     lin: float, ang: float) -> str:
    wp = waypoints[CFG.waypoint_select]
    if wp[0] < -0.1:
        direction = "backward"
    elif abs(wp[1]) > abs(wp[0]):
        direction = "left" if wp[1] > 0 else "right"
    else:
        direction = "forward"
    return (
        f"Instruction: '{instruction}'. "
        f"OmniVLA -> {direction} (lin={lin:.3f} m/s, ang={ang:.3f} rad/s). "
        f"WP#{CFG.waypoint_select}: dx={wp[0]:.3f} dy={wp[1]:.3f}."
    )


# ================================================================
#  GUI theme
# ================================================================
BG     = "#1a1a2e"
BG2    = "#16213e"
FG     = "#e2e8f0"
FG_DIM = "#64748b"
GREEN  = "#4ade80"
RED    = "#f87171"
YELLOW = "#fbbf24"
ACCENT = "#7c3aed"


# ================================================================
#  Main node
# ================================================================
class OmniVLACustomNode:
    """
    Full-featured OmniVLA Zenoh node with dynamic-prompting GUI,
    goal-reached detection, and RViz waypoint path.
    """

    def __init__(self, session: zenoh.Session, pi_ip: Optional[str] = None):
        self.session   = session
        self.lock      = Lock()
        self._pi_ip    = pi_ip   # used by camera watchdog for SSH restart

        # State
        self.latest_rgb:  Optional[np.ndarray] = None
        self.instruction  = CFG.instruction
        self._target_phrase = extract_target_phrase(self.instruction)
        self._ground_tick = 0
        self._running         = True
        self._paused          = False
        # ── Manual teleop (GUI D-pad) ──────────────────────────────────────
        # When a manual-drive panel is open, _manual_drive is True and the
        # autonomous _tick() yields the cmd_vel publisher entirely so it never
        # fights the operator. _teleop_dir holds (forward, turn) signs in
        # {-1,0,1}; a 10 Hz pump turns them into cmd_vel (keeps the ESP32's
        # 500 ms watchdog fed while a direction is held).
        self._manual_drive    = False
        self._manual_win      = None
        self._teleop_dir      = (0, 0)
        self._teleop_stop_job = None
        self._frame_count     = 0
        self._gui_last_drawn_frame = -1   # last frame index rendered to the canvas
        # self._last_frame_t    = 0.0   # epoch of most-recent camera frame
        self._last_frame_t = time.time()  #Experimental
        self._infer_count     = 0
        self._infer_hz        = 0.0
        self._last_infer_t    = time.time()
        self._last_stale_warn_t    = 0.0   # throttle stale-frame log (once/5s)

        # Camera hardware fix: compressed-first tracking + watchdog
        self._last_compressed_t   = 0.0   # epoch of most recent JPEG frame
        self._compressed_count    = 0     # JPEG frames received
        self._cam_watchdog_last_t = 0.0   # last watchdog check
        self._cam_restart_last_t  = 0.0   # last SSH restart attempt
        self._cam_restart_count   = 0

        # Previous frame for temporal input (OmniVLA needs 2 consecutive frames)
        self._prev_rgb: Optional[np.ndarray] = None

        # Goal-reached
        self._goal_small_count = 0
        self._goal_reached     = False

        # ----------------------------------------------------------------
        # Trajectory convergence stopping state (experimental)
        # ----------------------------------------------------------------
        self._trajectory_energy_buffer = deque(
            maxlen=CFG.trajectory_buffer_size
        )

        self._trajectory_delta_buffer = deque(
            maxlen=CFG.trajectory_buffer_size
        )

        self._trajectory_stop_counter = 0

        self._prev_e78 = None
        self._initial_e78 = None   #Experimental

        self._prev_ang = 0.0    #Experimental
        self._ang_slew = 0.0    # current smoothed angular command (slew-rate limited)

        self._instruction_start_time = time.time()



        # ------------------------------------------------------------
        # Trajectory freezing state  (experimental)
        # ------------------------------------------------------------

        self._frozen_waypoints = None
        self._freeze_ticks_remaining = 0






        # Direct velocity override (bypasses model — for stop/backwards/etc.)
        self._direct_vel_until: float = 0.0
        self._direct_lin: float = 0.0
        self._direct_ang: float = 0.0

        # Command queue (GUI / terminal -> inference thread)
        self._cmd_queue: "queue.Queue[str]" = queue.Queue()

        # Command history for GUI up/down
        self._cmd_history: List[str] = []
        self._hist_idx = -1

        # Zenoh publishers
        self.pub_cmd       = session.declare_publisher("cmd_vel")
        self.pub_explain   = session.declare_publisher("omnivla/explanation")
        self.pub_waypoints = session.declare_publisher("omnivla/waypoints")
        self.pub_grounding = session.declare_publisher("omnivla/grounding")

        # Grounded observer: optional OWL-ViT/CLIP target verification.
        self.grounding = GroundedObserver()

        # Vision-only static-obstacle avoidance (monocular depth).
        self.depth_avoider = DepthObstacleAvoider()
        self._avoid_tick = 0

        # Zenoh subscribers.
        # Subscribe ONLY to the JPEG-compressed camera stream. We deliberately do
        # NOT subscribe to raw "image_raw": as soon as this node declares a raw
        # subscriber, the Pi's Zenoh bridge starts shipping the uncompressed
        # frames over the network. At 480x360 that is ~518 KB/frame * 30 fps ~=
        # 124 Mbps — it saturates the WiFi link and collapses the effective
        # frame rate for OmniVLA AND the Eye-VLM feed. The camera always
        # publishes image_raw/compressed, so the raw stream is never needed.
        # Set OMNIVLA_ALLOW_RAW_IMAGE=1 only on a wired/high-bandwidth link.
        # LATENCY FIX: subscribe to the compressed camera stream through a Zenoh
        # RingChannel (drop-to-newest) instead of a push callback. A push callback
        # has an unbounded FIFO queue: while inference (OmniVLA + DINO) is busy,
        # frames pile up and are processed oldest-first, so the displayed/used
        # frame falls seconds behind (observed ~18 s stale). RingChannel keeps only
        # the freshest frame(s) and DROPS the backlog at ingress, so a drainer
        # thread always decodes the newest frame — latency stays ~1 frame no matter
        # how slow the consumer is. Falls back to the push callback on any error.
        self.sub_image_jpg    = self._subscribe_image_newest(session, "image_raw/compressed")
        self.sub_goal         = session.declare_subscriber("omnivla/goal_text", self._on_goal)
        self.sub_image = self.sub_image_rt = None
        # rt/ prefix variant of the COMPRESSED stream (newer zenoh-bridge-ros2dds).
        try:
            self.sub_image_jpg_rt = self._subscribe_image_newest(session, "rt/image_raw/compressed")
        except Exception as e:
            log(f"rt/ prefix subscribers not supported (OK if using older bridge): {e}")
        # Opt-in raw fallback for wired links only (off by default — see above).
        if os.environ.get("OMNIVLA_ALLOW_RAW_IMAGE") == "1":
            log("OMNIVLA_ALLOW_RAW_IMAGE=1 → also subscribing to raw image_raw (high bandwidth!)")
            self.sub_image = session.declare_subscriber("image_raw", self._on_image)
            try:
                self.sub_image_rt = session.declare_subscriber("rt/image_raw", self._on_image)
            except Exception:
                pass

        # ── startup diagnostic: log all Zenoh traffic for 15 seconds ──
        self._diag_keys: set = set()

        def _diag_cb(sample: zenoh.Sample):
            key = str(sample.key_expr)
            if key not in self._diag_keys:
                self._diag_keys.add(key)
                log(f"[DIAG] Zenoh traffic detected: '{key}'")

        try:
            self._sub_diag = session.declare_subscriber("**", _diag_cb)
            log("[DIAG] Wildcard probe active for 15 s — will show all arriving Zenoh keys ...")
            Thread(target=self._stop_diag_after_delay, daemon=True).start()
        except Exception as e:
            log(f"[DIAG] Wildcard probe failed: {e}")
            self._sub_diag = None

        log("Subscribers : image_raw/compressed (PRIMARY), image_raw (fallback), omnivla/goal_text  [+ rt/ variants]")
        log("Publishers  : cmd_vel, omnivla/explanation, omnivla/waypoints, omnivla/grounding")

    # ----------------------------------------------------------------
    #  Zenoh callbacks
    # ----------------------------------------------------------------
    def _stop_diag_after_delay(self):
        """Undeclare the wildcard diagnostic subscriber after 15 seconds.

        After the probe, emit a structured report so the user immediately
        reports whether camera data arrived.
        """
        time.sleep(15.0)
        try:
            if self._sub_diag is not None:
                self._sub_diag.undeclare()
                self._sub_diag = None

            keys = sorted(self._diag_keys)

            if not keys:
                log(
                    "\n╔══════════════════════════════════════════════════════╗\n"
                    "║  [DIAG] NO Zenoh traffic received in 15 s            ║\n"
                    "╠══════════════════════════════════════════════════════╣\n"
                    "║  → Pi bringup is NOT running (or wrong Pi IP)        ║\n"
                    "║  → On Pi: ros2 launch omnivla_nav rover_bringup.launch.py ║\n"
                    "╚══════════════════════════════════════════════════════╝"
                )
                return

            # Classify keys (camera only; LiDAR logic intentionally removed)
            has_jpg   = any("compressed" in k for k in keys)
            has_raw   = any("image_raw" in k and "compressed" not in k for k in keys)
            has_image = has_jpg or has_raw

            log(f"[DIAG] Probe done. Unique Zenoh keys seen: {keys}")

            lines = ["\n┌──────────────────────────────────────────────────────┐"]
            lines.append("│  [DIAG] 15 s probe summary                          │")
            lines.append("├──────────────────────────────────────────────────────┤")
            lines.append(f"│  Camera           : {'JPEG✓' if has_jpg else 'raw-only' if has_raw else '✗ MISSING'}{' ' * 24}│")
            lines.append("├──────────────────────────────────────────────────────┤")
            if has_jpg:
                lines.append("│  ✓ JPEG compressed stream active — fast camera.      │")
            elif has_raw:
                lines.append("│  ⚠ Only raw camera stream seen — using fallback.     │")
            elif not has_image:
                lines.append("│  ✗ IMAGE MISSING — camera not streaming              │")
            lines.append("│  LiDAR disabled in this build — no /scan required.   │")
            lines.append("└──────────────────────────────────────────────────────┘")
            log("\n".join(lines))

        except Exception:
            pass

    def _subscribe_image_newest(self, session, key: str):
        """Subscribe to a compressed-image key with drop-to-newest semantics.

        Uses a Zenoh RingChannel so only the freshest frame(s) are retained; a
        daemon drainer thread pulls the newest and feeds the normal handler, so
        a busy inference loop can never build a stale frame backlog. Falls back
        to the plain push callback if RingChannel isn't available.
        """
        try:
            sub = session.declare_subscriber(key, zenoh.handlers.RingChannel(2))
        except Exception as e:
            log(f"RingChannel unavailable for '{key}' ({e}); using push callback")
            return session.declare_subscriber(key, self._on_image_compressed)

        def _drain():
            while self._running:
                try:
                    sample = sub.recv()          # blocks; ring keeps only newest
                    if sample is not None:
                        self._on_image_compressed(sample)
                except Exception as e:
                    if self._running:
                        log(f"image drainer '{key}' error: {e}")
                        time.sleep(0.1)
        Thread(target=_drain, daemon=True, name=f"img-drain-{key.replace('/', '_')}").start()
        log(f"Camera '{key}': drop-to-newest (RingChannel) active")
        return sub

    def _on_image_compressed(self, sample: zenoh.Sample):
        """JPEG compressed — PRIMARY camera source.

        Updates latest_rgb unconditionally; marks last_compressed_t so
        _on_image (raw) knows to stay silent. This mirrors the enhanced
        version hardware path, without changing navigation logic.
        """
        try:
            rgb = parse_compressed_image(bytes(sample.payload))
            if rgb is not None:
                with self.lock:
                    self.latest_rgb         = rgb
                    self._frame_count      += 1
                    now = time.time()
                    self._last_frame_t      = now
                    self._last_compressed_t = now
                    self._compressed_count += 1
                if self._compressed_count == 1:
                    log("INFO: JPEG compressed stream active — fast camera ✓")
        except Exception as e:
            log(f"CompressedImage parse error: {e}")

    def _on_image(self, sample: zenoh.Sample):
        """Raw image — FALLBACK only.

        Only used when JPEG compressed has not arrived in the last 1 s.
        This prevents raw frames from consuming bandwidth and starving
        other hardware traffic such as ESP32 /cmd_vel.
        """
        try:
            with self.lock:
                compressed_age = time.time() - self._last_compressed_t
            if compressed_age < 1.0:
                return   # JPEG stream is healthy — skip raw
            rgb = parse_image(bytes(sample.payload))
            if rgb is not None:
                with self.lock:
                    self.latest_rgb    = rgb
                    self._frame_count += 1
                    self._last_frame_t = time.time()
                if self._frame_count == 1:
                    log("INFO: Using raw image_raw fallback (JPEG not arriving).")
            else:
                if self._frame_count == 0:
                    log("WARN: image_raw received but parse returned None (bad encoding?)")
        except Exception as e:
            log(f"Image parse error: {e}")

    def _on_goal(self, sample: zenoh.Sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if text and text != self.instruction:
                log(f"Goal via Zenoh: '{text}'")
                self.instruction       = text
                self._target_phrase    = extract_target_phrase(text)
                self.grounding.reset()
                self._goal_reached     = False
                self._goal_small_count = 0
        except Exception:
            pass

    # ----------------------------------------------------------------
    #  Camera watchdog  (SSH restart when frames stale)
    # ----------------------------------------------------------------
    def _camera_watchdog_tick(self):
        """Restart the Pi camera node if no frame has arrived recently.

        This is copied from the enhanced version's hardware path and only
        affects camera recovery; it does not alter OmniVLA control logic.
        """
        if not CFG.camera_watchdog_enabled:
            return   # auto-restart disabled — transient frame gaps recover on their own
        if self._pi_ip is None:
            return
        now = time.time()
        if now - self._cam_watchdog_last_t < 5.0:
            return
        self._cam_watchdog_last_t = now

        frame_age = now - self._last_frame_t
        if frame_age < CFG.camera_stale_restart_s:
            return
        if now - self._cam_restart_last_t < 25.0:   # cooldown: a restart takes ~13 s; don't storm
            return

        log(f"CAMERA WATCHDOG: No frame for {frame_age:.0f} s → "
            f"restarting v4l2_camera on Pi (attempt #{self._cam_restart_count + 1})")
        self._cam_restart_last_t = now
        self._cam_restart_count += 1
        Thread(target=self._do_camera_restart_ssh, daemon=True).start()

    def _do_camera_restart_ssh(self):
        """Background thread: SSH to Pi, re-probe V4L2 device, restart node."""
        # Use sshpass for password auth (no SSH key needed)
        pi_password = os.environ.get("PI_PASS", "hri")  # Default password
        
        # Free the device BEFORE restarting. A plain SIGTERM + 2 s wait was not
        # enough: the stale node kept /dev/videoN open, so the new node failed
        # "mapping device memory" (mmap) and produced zero frames — which made
        # the watchdog restart again, forever. Kill -9 and wait for the device
        # to actually be released (fuser) before starting a single new node.
        # Params match the launch bringup (800x600, compressed plugin) so the
        # PRIMARY image_raw/compressed stream is republished after a restart.
        # NB: do NOT use "pkill -f v4l2_camera_node" here — this script string
        # contains "v4l2_camera_node", so pkill -f would match (and kill) this
        # very "bash -c <script>" shell before it restarts the camera, leaving
        # a stale node holding the device. Free the device by fd with fuser -k
        # (name-independent, so it can't match this shell) per device instead.
        restart_script = (
            "source /opt/ros/humble/setup.bash; source ~/rover_ws/install/setup.bash 2>/dev/null; "
            # Free EVERY video device first, not just the one we pick below.
            # The camera re-enumerates (video0->video2->...) across restarts, so
            # old v4l2_camera nodes accumulate holding stale nodes and contend for
            # the live device (observed 7 procs → mmap contention → stalls). fuser
            # matches by open fd (not process name), so it cannot kill this shell.
            "for v in /dev/video*; do fuser -k $v 2>/dev/null; done; sleep 1; "
            "for n in $(seq 0 20); do "
            "  d=/dev/video$n; [ -e $d ] || continue; "
            "  INFO=$(v4l2-ctl --device=$d --info 2>/dev/null); "
            "  echo \"$INFO\" | grep -q 'Video Capture' || continue; "
            "  echo \"$INFO\" | grep -q 'Streaming'     || continue; "
            "  fuser -k $d 2>/dev/null; "
            "  for w in $(seq 1 10); do fuser $d >/dev/null 2>&1 || break; sleep 1; done; "
            "  nohup ros2 run v4l2_camera v4l2_camera_node --ros-args"
            "    -p video_device:=$d -p image_size:=[640,480]"
            # 15 fps (not 30): we only forward ~11 fps and infer at 3 Hz, so
            # 30 fps capture is wasted CPU/USB load that feeds the stalls.
            "    -p pixel_format:=YUYV -p time_per_frame:=[1,15]"
            "    -p image_raw.disable_pub_plugins:=[theora,compressedDepth]"
            "    -p image_raw.compressed.jpeg_quality:=65"
            "    >> /tmp/camera_restart.log 2>&1 &"
            "  echo restarted:$d; break; "
            "done"
        )
        try:
            r = subprocess.run(
                ["sshpass", "-p", pi_password, "ssh",
                 "-o", "ConnectTimeout=10",
                 "-o", "StrictHostKeyChecking=no",
                 f"pi@{self._pi_ip}", "bash", "-c", restart_script],
                capture_output=True, text=True, timeout=25,
            )
            out = r.stdout.strip()
            if "restarted:" in out:
                dev = out.split("restarted:")[-1].split()[0]
                log(f"CAMERA WATCHDOG: Restarted v4l2_camera on {dev} ✓")
            else:
                log(f"CAMERA WATCHDOG: SSH ran but no 'restarted' ack. "
                    f"stdout='{out[:120]}' stderr='{r.stderr.strip()[:80]}'")
        except subprocess.TimeoutExpired:
            log("CAMERA WATCHDOG: SSH timed out (Pi unreachable?)")
        except Exception as e:
            log(f"CAMERA WATCHDOG: Error: {e}")

    # ----------------------------------------------------------------
    #  E-stop  (geometry-aware path-rectangle check)
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    #  Publish helpers
    # ----------------------------------------------------------------
    def publish_cmd(self, lin: float, ang: float):
        self.pub_cmd.put(serialize_twist(lin, ang))

    def publish_explanation(self, text: str):
        self.pub_explain.put(serialize_string(text))

    def publish_waypoints(self, waypoints: np.ndarray):
        pts = waypoints_to_path(waypoints)
        self.pub_waypoints.put(serialize_path(pts, frame_id="base_link"))

    def publish_grounding(self, info: dict):
        try:
            msg = (
                f"target='{info.get('target','')}' "
                f"ok={info.get('ok', False)} "
                f"score={info.get('score', 0.0):.3f} "
                f"area={info.get('area', 0.0):.3f} "
                f"center_offset={info.get('center_offset', 9.9):.3f} "
                f"clip_bias={info.get('clip_bias', 0.0):.3f} "
                f"stop={info.get('stop', False)} "
                f"reason={info.get('reason','')}"
            )
            self.pub_grounding.put(serialize_string(msg))
        except Exception:
            pass

    # ----------------------------------------------------------------
    #  Main inference loop
    # ----------------------------------------------------------------
    def spin(self):
        period      = 1.0 / CFG.predict_hz
        last_status = time.time()
        log(f"Inference loop @ {CFG.predict_hz} Hz -- '{self.instruction}'")
        log("Waiting for camera frames ...")
        # Reset the camera frame time on startup to prevent triggering the watchdog
        # during initial model loading and Grounding DINO/CLIP loading times.
        with self.lock:
            self._last_frame_t = time.time()
        try:
            while self._running:
                t0 = time.time()
                self._camera_watchdog_tick()
                self._tick()
                self._drain_cmd_queue()
                if time.time() - last_status > 15.0:
                    log(
                        f"frames={self._frame_count} (JPEG={self._compressed_count})  infer={self._infer_count}  "
                        f"hz={self._infer_hz:.1f}  "
                        f"goal={'REACHED' if self._goal_reached else 'tracking'}"
                    )
                    last_status = time.time()
                elapsed = time.time() - t0
                if period - elapsed > 0:
                    time.sleep(period - elapsed)
        except KeyboardInterrupt:
            log("Shutdown (Ctrl-C).")
        finally:
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.publish_cmd(0.0, 0.0)
            log("Zero velocity sent. Bye.")

    def _tick(self):
        # Manual teleop owns cmd_vel while its panel is open — the autonomous
        # loop yields completely so it can't fight the operator's commands.
        if self._manual_drive:
            return
        with self.lock:
            rgb          = self.latest_rgb
            last_frame_t = self._last_frame_t
        if rgb is None:
            return

        # Frame freshness check — skip inference if camera feed is stale
        age = time.time() - last_frame_t
        if age > CFG.max_frame_age_s:
            now = time.time()
            if now - self._last_stale_warn_t > 5.0:   # warn at most once per 5 s
                log(f"WARN: Camera frame is {age:.1f}s old (>{CFG.max_frame_age_s}s). "
                    f"Is the Pi bringup running? (v4l2_camera may have crashed)")
                self._last_stale_warn_t = now
            self.publish_cmd(0.0, 0.0)
            return

        # Holds
        # Prevent the robot from turning or drifting in the rest position when there is no instruction,
        # or when the instruction is explicitly "stop" or "halt".
        is_rest = (not self.instruction) or (self.instruction.strip().lower() in ("", "stop", "halt"))
        if self._goal_reached or self._paused or is_rest:
            self.publish_cmd(0.0, 0.0)
            return

        # Direct velocity override (stop / backwards / etc. — bypasses model)
        if time.time() < self._direct_vel_until:
            self.publish_cmd(self._direct_lin, self._direct_ang)
            return

        # Dynamic prompt steering based on recent bounding box
        # active_instruction = self.instruction
        # if CFG.enable_grounding and self.grounding.enabled and self.grounding.last.get("ok", False):
        #     cx_signed = self.grounding.last.get("center_offset_signed", 0.0)
        #     # Positive cx_signed means object is on the right -> turn right
        #     if cx_signed > 0.15:
        #         active_instruction = f"turn slightly right and {self.instruction}"
        #     elif cx_signed < -0.15:
        #         active_instruction = f"turn slightly left and {self.instruction}"

        # Inference
        t0        = time.time()
        prev_pil  = Image.fromarray(self._prev_rgb) if self._prev_rgb is not None else None
        waypoints = predict_actions(Image.fromarray(rgb), self.instruction, prev_image_pil=prev_pil)


        # Preview steering from NEW trajectory
        new_lin_preview, new_ang_preview = pd_controller(waypoints)
            


        self._prev_rgb = rgb.copy()  # store for next inference tick
        lin, ang  = pd_controller(waypoints)

        
        


        infer_ms  = (time.time() - t0) * 1000.0

        # Rolling inference rate (EMA)
        dt = max(time.time() - self._last_infer_t, 1e-3)
        self._last_infer_t = time.time()
        self._infer_hz = 0.8 * self._infer_hz + 0.2 * (1.0 / dt)

        # # Goal-reached detection
        # wp     = waypoints[-1]
        # wp_mag = math.sqrt(wp[0] ** 2 + wp[1] ** 2) * CFG.metric_waypoint_spacing
        # if wp_mag < CFG.goal_reached_threshold:
        #     self._goal_small_count += 1
        #     if self._goal_small_count >= CFG.goal_reached_count:
        #         log(f"=== GOAL REACHED: '{self.instruction}' (wp_mag={wp_mag:.4f}m) ===")
        #         self._goal_reached = True
        #         self.publish_cmd(0.0, 0.0)
        #         self.publish_explanation(f"GOAL REACHED: '{self.instruction}'.")
        #         self.publish_waypoints(waypoints)
        #         return
        # else:
        #     self._goal_small_count = 0


        # ----------------------------------------------------------------
        # OmniVLA trajectory-convergence stopping
        # ----------------------------------------------------------------

        # Use long-horizon future waypoints
        wp7 = waypoints[6]
        wp8 = waypoints[7]

        # Metric magnitudes
        m7 = math.sqrt(wp7[0] ** 2 + wp7[1] ** 2)
        m8 = math.sqrt(wp8[0] ** 2 + wp8[1] ** 2)

        m7 *= CFG.metric_waypoint_spacing
        m8 *= CFG.metric_waypoint_spacing

        # ----------------------------------------------------------------
        # Future trajectory energy
        # ----------------------------------------------------------------
        # Weight waypoint 8 slightly more heavily because it represents
        # farther semantic intent.
        # ----------------------------------------------------------------

        E78 = (0.4 * m7) + (0.6 * m8)


        # Store initial semantic energy   #Experimental
        if self._initial_e78 is None:     #Experimental
            self._initial_e78 = E78       #Experimental

        # Relative semantic compression   #Experimental
        energy_ratio = E78 / max(self._initial_e78, 1e-6)   #Experimental


        self._trajectory_energy_buffer.append(E78)

        # ----------------------------------------------------------------
        # Trajectory derivative
        # ----------------------------------------------------------------
        # We now measure HOW MUCH the future trajectory is still changing.
        # Near semantic completion:
        #   trajectory converges
        #   derivative shrinks
        # ----------------------------------------------------------------

        if self._prev_e78 is None:
            delta_E = 999.0
        else:
            delta_E = abs(E78 - self._prev_e78)

        self._prev_e78 = E78

        self._trajectory_delta_buffer.append(delta_E)


        # ------------------------------------------------------------
        # Trajectory freezing near convergence (experimental)
        # ------------------------------------------------------------
        # Near target:
        # if OmniVLA suddenly changes trajectory violently,
        # keep using last stable trajectory temporarily.
        # ------------------------------------------------------------

        freeze_active = False

        if (
            CFG.enable_trajectory_freeze
            and energy_ratio < CFG.freeze_ratio_threshold
            and delta_E > CFG.freeze_max_delta_E
            and self._freeze_ticks_remaining == 0
            and abs(new_ang_preview - self._prev_ang) > CFG.freeze_ang_delta_threshold
        ):

            # Freeze previous stable trajectory
            if self._frozen_waypoints is not None:

                waypoints = self._frozen_waypoints.copy()

                lin, ang = pd_controller(waypoints) #Experimental

                self._freeze_ticks_remaining = CFG.freeze_hold_ticks

                freeze_active = True

                log(
                    f"[TRAJ_FREEZE] "
                    f"UNSTABLE TRAJECTORY DETECTED "
                    f"(ratio={energy_ratio:.3f}, dE={delta_E:.3f}) "
                    f"→ reusing stable trajectory"
                )

        # Countdown freeze hold
        elif self._freeze_ticks_remaining > 0:

            if self._frozen_waypoints is not None:

                waypoints = self._frozen_waypoints.copy()

                lin, ang = pd_controller(waypoints) #Experimental

                self._freeze_ticks_remaining -= 1

                freeze_active = True

        # Store stable trajectory
        else:

            self._frozen_waypoints = waypoints.copy()

        # ----------------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------------

        avg_E78 = np.mean(self._trajectory_energy_buffer)
        avg_delta = np.mean(self._trajectory_delta_buffer)
        var_E78 = np.var(self._trajectory_energy_buffer)

        # ----------------------------------------------------------------
        # Trajectory shape stability
        # ----------------------------------------------------------------
        # Measure total future trajectory length.
        # Near convergence:
        #   trajectory shape changes less.
        # ----------------------------------------------------------------

        trajectory_length = 0.0

        for i in range(7):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]

            segment = math.sqrt(dx * dx + dy * dy)
            trajectory_length += segment

        trajectory_length *= CFG.metric_waypoint_spacing


        # ----------------------------------------------------------------
        # Runtime guard
        # ----------------------------------------------------------------
        # Prevent immediate stopping right after instruction begins.
        # ----------------------------------------------------------------

        runtime_sec = time.time() - self._instruction_start_time

            # ------------------------------------------------------------
            # Robot itself must also be moving very little
            # Prevents endless spinning near target (experimental)
            # ------------------------------------------------------------

        low_motion = (
                abs(lin) < 0.03
                and abs(ang) < 0.08
            )
        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------

        if self._infer_count <= 5 or self._infer_count % 5 == 0:
            log(
                f"[TRAJ_STOP] "
                f"m7={m7:.4f}  "
                f"m8={m8:.4f}  "
                f"E78={E78:.4f}  "
                f"ratio={energy_ratio:.3f}  "
                f"dE={delta_E:.5f}  "
                f"avg_dE={avg_delta:.5f}  "
                f"var={var_E78:.6f}  "
                f"traj_len={trajectory_length:.4f}  "
                f"runtime={runtime_sec:.1f}s  "
                f"count={self._trajectory_stop_counter}"
            )

            # ----------------------------------------------------------------
            # Convergence-based stopping
            # ----------------------------------------------------------------
            # IMPORTANT:
            # We NO LONGER expect magnitudes to go to zero.
            # Instead we detect SEMANTIC SATURATION:
            #   - future trajectory stops evolving
            #   - trajectory becomes stable over time
            # ----------------------------------------------------------------

            # stable_convergence = (
            #     avg_delta < CFG.trajectory_delta_threshold
            #     and var_E78 < CFG.trajectory_variance_thresh
            #     and runtime_sec > CFG.trajectory_min_runtime_sec
            # )






        # Experimental
        stable_convergence = (
            avg_delta < CFG.trajectory_delta_threshold
            and var_E78 < CFG.trajectory_variance_thresh
            and runtime_sec > CFG.trajectory_min_runtime_sec
            and energy_ratio < CFG.trajectory_compression_ratio
            and low_motion
        )


        if stable_convergence:
            self._trajectory_stop_counter += 1
        else:
            self._trajectory_stop_counter = 0

            
        # ----------------------------------------------------------------
        # FINAL STOP
        # ----------------------------------------------------------------

        if self._trajectory_stop_counter >= CFG.trajectory_stable_count:

            log(
            f"=== TRAJECTORY CONVERGENCE STOP === "
            f"'{self.instruction}' "
            f"avg_dE={avg_delta:.5f} "
            f"var={var_E78:.6f}"
            )

            self._goal_reached = True

            self.publish_cmd(0.0, 0.0)

            self.publish_explanation(
            f"TRAJECTORY CONVERGENCE STOP: '{self.instruction}'."
            )

            self.publish_waypoints(waypoints)

            return




        # Vision-only static-obstacle avoidance: refresh the depth reading.
        # depth_info holds the steer-away angular term and a forward "block"
        # measure; it is applied further below, inside the steering block.
        depth_info = self.depth_avoider.last
        if CFG.avoid_enabled and self.depth_avoider.enabled:
            self._avoid_tick += 1
            if self._avoid_tick % max(1, CFG.avoid_period) == 0:
                depth_info = self.depth_avoider.observe(rgb)
                if self._avoid_tick <= 5 or self._avoid_tick % 5 == 0:
                    log(
                        f"[AVOID] ok={depth_info.get('ok', False)} "
                        f"L={depth_info.get('left', 0):.2f} "
                        f"C={depth_info.get('center', 0):.2f} "
                        f"R={depth_info.get('right', 0):.2f} "
                        f"block={depth_info.get('block', 0):.2f} "
                        f"avoid={depth_info.get('avoid', 0):+.3f} "
                        f"reason={depth_info.get('reason', '')}"
                    )

        # Grounded observer stop/centering gate.
        # OmniVLA still provides lin/ang; OWL-ViT/CLIP only verifies the target
        # and optionally adds a small angular nudge to keep it centered.
        ground_info = self.grounding.last
        if CFG.enable_grounding and self.grounding.enabled:
            self._ground_tick += 1
            if self._ground_tick % max(1, CFG.grounding_period) == 0:
                ground_info = self.grounding.observe(rgb, self._target_phrase)
                self.publish_grounding(ground_info)
                if self._ground_tick <= 5 or self._ground_tick % 5 == 0:
                    log(
                        f"[GROUND] target='{self._target_phrase}' "
                        f"ok={ground_info.get('ok',False)} "
                        f"score={ground_info.get('score',0):.2f} "
                        f"area={ground_info.get('area',0):.3f} "
                        f"center={ground_info.get('center_offset',9.9):.2f} "
                        f"bias={ground_info.get('clip_bias',0):+.3f} "
                        f"stop={ground_info.get('stop',False)} "
                        f"reason={ground_info.get('reason','')}"
                    )
            if ground_info.get("stop", False):
                log(
                    f"=== VISUAL GOAL REACHED: target='{self._target_phrase}' "
                    f"score={ground_info.get('score',0):.2f} "
                    f"area={ground_info.get('area',0):.2f} ==="
                )
                self._goal_reached = True
                self.publish_cmd(0.0, 0.0)
                self.publish_explanation(f"VISUAL GOAL REACHED: target='{self._target_phrase}'.")
                self.publish_waypoints(waypoints)
                return


            # # CLIP crop bias: positive means target is stronger on the left; add left turn.
            # ang = float(np.clip(ang + float(ground_info.get("clip_bias", 0.0)), -0.08, 0.08))

            # clip_bias = float(ground_info.get("clip_bias", 0.0))

            # # Safety limit ONLY on visual bias
            # clip_bias = float(np.clip(clip_bias, -0.04, 0.04))

            # # Add small visual correction to OmniVLA steering
            # ang = ang + clip_bias

            # # Final safety limit
            # ang = float(np.clip(ang, -0.35, 0.35))



            clip_bias = float(ground_info.get("clip_bias", 0.0))


            # During trajectory freeze: (experimental)
            # disable visual steering correction
            # otherwise CLIP can destabilize steering again

            if freeze_active:
                clip_bias = 0.0




            # ------------------------------------------------------------
            # Convergence-aware steering damping
            # ------------------------------------------------------------
            # Near semantic convergence:
            #   reduce steering authority smoothly
            # to prevent last-second semantic instability.
            # ------------------------------------------------------------

            ratio = energy_ratio

            start_r = CFG.steering_damping_start_ratio
            end_r   = CFG.steering_damping_end_ratio

            if ratio >= start_r:
                damping_alpha = 0.0

            elif ratio <= end_r:
                damping_alpha = 1.0

            else:
                damping_alpha = (
                    (start_r - ratio)
                    / (start_r - end_r)
                )

            # Smoothly interpolate allowed angular velocity.
            # Caps derive from the active max_angular (--max-angular) so the
            # CLI flag governs turn speed instead of a hard-coded 0.35/0.16.
            max_angular_far  = CFG.max_angular_far_frac  * CFG.max_angular
            max_angular_near = CFG.max_angular_near_frac * CFG.max_angular
            dynamic_max_ang = (
                (1.0 - damping_alpha) * max_angular_far
                + damping_alpha * max_angular_near
            )

            # Limit visual correction separately
            clip_bias = float(np.clip(clip_bias, -0.04, 0.04))

            # Apply visual correction
            ang = ang + clip_bias

            # ------------------------------------------------------------
            # Target-object centering (Grounding DINO bounding box)
            # ------------------------------------------------------------
            # Steer to bring the detected target to the horizontal center of
            # the frame. center_offset_signed > 0 => target on the right =>
            # turn right (negative ang, since positive angular turns left).
            center_corr = 0.0
            if (
                CFG.center_enabled
                and not freeze_active
                and ground_info.get("ok", False)
            ):
                cx_signed = float(ground_info.get("center_offset_signed", 0.0))  # [-1,1], + = right
                if abs(cx_signed) > CFG.center_deadband:
                    # Convert the pixel offset into a TRUE angular error using the
                    # camera FOV, then command a turn proportional to it. This is
                    # computed from geometry, not a hard-coded rate.
                    #   theta_err > 0  ⇒ target is to the RIGHT of center.
                    theta_err = cx_signed * (math.radians(CFG.camera_hfov_deg) / 2.0)  # rad
                    # Positive angular turns LEFT, so a right-offset target needs a
                    # negative (right) command to null the error.
                    center_corr = -CFG.center_kp * theta_err
                    # Stiction floor: guarantee enough differential to actually
                    # rotate the skid-steer chassis for small residual offsets.
                    if 0.0 < abs(center_corr) < CFG.center_min_cmd:
                        center_corr = math.copysign(CFG.center_min_cmd, center_corr)
                    ang = ang + center_corr
                    # The final np.clip(ang, ±dynamic_max_ang) below bounds this to
                    # the full max_angular authority — no separate sub-cap throttles it.

            # ------------------------------------------------------------
            # Vision-only static-obstacle avoidance (monocular depth)
            # ------------------------------------------------------------
            # Steer away from the closer side and slow when the center looms.
            # Avoidance rides on top of OmniVLA + centering and is bounded by
            # the same dynamic clamp below, so it can't run away.
            avoid_ang = 0.0
            block = 0.0
            if CFG.avoid_enabled and not freeze_active and depth_info.get("ok", False):
                avoid_ang = float(depth_info.get("avoid", 0.0))
                block = float(depth_info.get("block", 0.0))
                # Don't treat the goal object as a forward obstacle: if the DINO
                # target is detected and roughly centered ahead, the "looming
                # center" is the target itself, so suppress the forward-block
                # slowdown (side avoidance still applies) and let it approach.
                target_ahead = (
                    ground_info.get("ok", False)
                    and ground_info.get("center_offset", 9.9) < CFG.avoid_target_center_tol
                    and ground_info.get("area", 0.0) >= CFG.avoid_target_min_area
                )
                if target_ahead:
                    block = 0.0
                ang = ang + avoid_ang
                # Looming obstacle dead ahead → cut forward speed (down to 0).
                if block > 0.0:
                    lin = lin * (1.0 - min(1.0, block) * CFG.avoid_slow_scale)

            # FINAL dynamic steering clamp
            ang = float(np.clip(
                ang,
                -dynamic_max_ang,
                dynamic_max_ang
            ))

            # Debug
            if self._infer_count <= 5 or self._infer_count % 5 == 0:
                log(
                    f"[STEERING_DAMP] "
                    f"ratio={ratio:.3f} "
                    f"alpha={damping_alpha:.3f} "
                    f"max_ang={dynamic_max_ang:.3f} "
                    f"center_off={ground_info.get('center_offset_signed', 0.0):+.3f} "
                    f"center_corr={center_corr:+.3f} "
                    f"avoid={avoid_ang:+.3f} "
                    f"block={block:.2f} "
                    f"lin={lin:.3f} "
                    f"ang={ang:.3f}"
                )



        # ── Angular slew-rate limiter ────────────────────────────────────────
        # Ramp _ang_slew toward target `ang` by at most ang_slew_rate * tick_dt
        # per inference tick.  This prevents the rover from snapping to full
        # turning speed instantly, which would swing the camera past the object.
        max_step = CFG.ang_slew_rate * CFG.tick_dt
        delta    = ang - self._ang_slew
        if abs(delta) <= max_step:
            self._ang_slew = ang          # already within one step — snap
        else:
            self._ang_slew += math.copysign(max_step, delta)   # ramp
        smoothed_ang = self._ang_slew

        self.publish_cmd(lin, smoothed_ang)

        self._prev_ang = smoothed_ang    #Experimental

        self.publish_explanation(make_explanation(self.instruction, waypoints, lin, ang))
        self.publish_waypoints(waypoints)

        self._infer_count += 1
        if self._infer_count <= 5 or self._infer_count % 30 == 0:
            # Show full predicted path so we can see if model is turning toward target
            path_str = "  ".join(
                f"wp{i}=({waypoints[i][0]:.2f},{waypoints[i][1]:.2f})"
                for i in range(8)
            )
            log(
                f"[#{self._infer_count}] lin={lin:.3f} ang={ang:.3f}  "
                f"instr='{self.instruction}'\n"
                f"  PATH: {path_str}  ({infer_ms:.0f}ms)"
            )

    # ----------------------------------------------------------------
    #  Command processing
    # ----------------------------------------------------------------
    def _drain_cmd_queue(self):
        try:
            while True:
                self._handle_command(self._cmd_queue.get_nowait())
        except queue.Empty:
            pass

    def _handle_command(self, raw: str):
        """
        System meta-commands are handled here.
        EVERYTHING ELSE is forwarded to OmniVLA as the navigation instruction.
        OmniVLA decides what 'stop', 'go backwards', 'turn left', etc. means.
        """
        cmd = raw.strip().lower()

        # ── System / tuning commands ──────────────────────────────────
        if cmd in ("quit", "exit"):
            log("Quit requested."); self._running = False; return

        if cmd == "status":
            log(
                f"instr='{self.instruction}'  paused={self._paused}  "
                f"frames={self._frame_count} (JPEG={self._compressed_count})  "
                f"infer={self._infer_count}  hz={self._infer_hz:.1f}  "
                f"v={CFG.max_linear:.2f}  w={CFG.max_angular:.2f}  wp={CFG.waypoint_select}  "
                f"target='{self._target_phrase}'  grounding={self.grounding.last}"
            ); return

        if cmd == "help":
            self._print_help(); return

        if cmd in ("grounding status", "observer status"):
            log(f"Grounding target='{self._target_phrase}' last={self.grounding.last}")
            return

        if cmd.startswith("waypoint "):
            try:
                idx = int(cmd.split()[1])
                assert 0 <= idx <= 7
                CFG.waypoint_select = idx
                log(f"Waypoint index -> {idx}")
            except Exception:
                log("Usage: waypoint <0-7>")
            return

        if cmd.startswith("speed "):
            try:
                CFG.max_linear = max(0.0, min(1.0, float(cmd.split()[1])))
                log(f"Max linear -> {CFG.max_linear:.3f} m/s")
            except Exception:
                log("Usage: speed <0.0-1.0>")
            return

        if cmd.startswith("angular "):
            try:
                CFG.max_angular = max(0.0, min(2.0, float(cmd.split()[1])))
                log(f"Max angular -> {CFG.max_angular:.3f} rad/s")
            except Exception:
                log("Usage: angular <0.0-2.0>")
            return

        # ── Direct motion primitives (bypass model — model never outputs backwards) ──
        _DIRECT: dict = {
            "stop":         (0.0,               0.0),
            "halt":         (0.0,               0.0),
            "back":         (-CFG.max_linear,   0.0),
            "backward":     (-CFG.max_linear,   0.0),
            "backwards":    (-CFG.max_linear,   0.0),
            "go back":      (-CFG.max_linear,   0.0),
            "go backward":  (-CFG.max_linear,   0.0),
            "go backwards": (-CFG.max_linear,   0.0),
            "reverse":      (-CFG.max_linear,   0.0),
        }
        if cmd in _DIRECT:
            lin_d, ang_d = _DIRECT[cmd]
            dur = 0.5 if lin_d == 0.0 else 3.0
            self._direct_lin       = lin_d
            self._direct_ang       = ang_d
            self._direct_vel_until = time.time() + dur
            self._goal_reached     = False

            self._trajectory_energy_buffer.clear() #Experimental
            self._trajectory_delta_buffer.clear()  #Experimental
            self._trajectory_stop_counter = 0  #Experimental
            self._prev_e78 = None  #Experimental
            self._initial_e78 = None   #Experimental
            self._instruction_start_time = time.time()  #Experimental

            self._goal_small_count = 0

            

            self._paused           = False
            action = "STOP" if lin_d == 0.0 else f"backwards  lin={lin_d:.2f} m/s  for {dur:.0f} s"
            log(f"[DIRECT] {action}  (model bypassed)")
            return

        # ── EVERYTHING ELSE → OmniVLA instruction ────────────────────
        # Preserve the user's original language exactly.
        # No regex remapping / normalization is applied here; predict_actions()
        # will insert this text directly into:
        #   "What action should the robot take to {instruction}?"
        instruction = raw.strip()
        self.instruction       = instruction
        self._target_phrase    = extract_target_phrase(instruction)
        self.grounding.reset()
        self._goal_reached     = False

        self._frozen_waypoints = None   #Experimental
        self._freeze_ticks_remaining = 0    #Experimental

        self._ang_slew         = 0.0    # reset slew so new goal starts from zero turn rate

        self._goal_small_count = 0
        self._direct_vel_until = 0.0   # cancel any active direct override
        self._paused           = False   # always un-pause when new instruction given
        log(f"OmniVLA instruction -> '{self.instruction}'  | observer target='{self._target_phrase}'")

    @staticmethod
    def _print_help():
        print("""
+------------------------------------------------------------------+
|  OmniVLA Controller — Command Reference                         |
+------------------------------------------------------------------+
|  TYPED TEXT (Enter/Send)  → sent directly to OmniVLA as the     |
|    navigation instruction. Examples:                             |
|    "go towards the door"    "turn left"    "stop"                |
|    "go backwards"           "follow the corridor"               |
|    OmniVLA decides what velocity to output.                      |
+------------------------------------------------------------------+
|  GUI BUTTONS (direct hardware control, bypass OmniVLA):          |
|    [⏸ Freeze]  – zero velocity, hold position (safety pause)   |
|    [▷ Unfreeze] – resume last instruction                        |
|    [↺ Reset Goal] – clear goal-reached latch                    |
|    [✕ Quit]   – shutdown                                        |
+------------------------------------------------------------------+
|  SYSTEM / TUNING commands (typed):                               |
|    status              – print current state                     |
|    waypoint <0-7>      – which waypoint to track (default 4)    |
|    speed <m/s>         – max linear speed  (0.0 – 1.0)          |
|    angular <rad/s>     – max angular speed (0.0 – 2.0)          |
|    help                – this message                            |
|    grounding status    – print OWL-ViT/CLIP observer state        |
|    LiDAR/e-stop logic is removed in this build.                   |
|    Visual stop: target box area/confidence from OWL-ViT.          |
|    quit / exit         – shutdown                                |
+------------------------------------------------------------------+""")

    # ----------------------------------------------------------------
    #  Tkinter GUI
    # ----------------------------------------------------------------
    _CAM_W = 320   # matches Pi camera resolution (320×240)
    _CAM_H = 240

    def _build_gui(self):
        root = tk.Tk()
        root.title("OmniVLA Controller  —  Grounded Vision")
        root.configure(bg=BG)
        root.resizable(True, True)
        screen_w = max(1, root.winfo_screenwidth())
        screen_h = max(1, root.winfo_screenheight())
        window_w = max(960, int(screen_w * 0.92))
        window_h = max(720, int(screen_h * 0.90))
        root.geometry(f"{window_w}x{window_h}")
        self.root = root
        self._cam_photo = None

        tk.Label(root, text="OmniVLA Controller - Grounded Observer", bg=BG, fg=ACCENT, font=("Helvetica", 16, "bold")).pack(pady=(8, 2))

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=10, pady=(0, 6))

        fi = tk.Frame(top, bg=BG2)
        # fi.pack(fill="x", pady=(0, 4))
        tk.Label(fi, text="Current Instruction", bg=BG2, fg=FG_DIM, font=("Helvetica", 9)).pack(anchor="w", padx=8, pady=(4, 0))
        self._lbl_instruction = tk.Label(fi, text=self.instruction, bg=BG2, fg=YELLOW, font=("Helvetica", 11, "bold"), wraplength=max(640, int(window_w * 0.80)), justify="left")
        self._lbl_instruction.pack(anchor="w", padx=8, pady=(0, 6))

        fs = tk.Frame(top, bg=BG2)
        fs.pack(fill="x", pady=(0, 2))
        dot_font = ("Helvetica", 10)
        self._dot_camera = tk.Label(fs, text="* Camera", bg=BG2, fg=RED, font=dot_font)
        self._dot_paused = tk.Label(fs, text="* Paused", bg=BG2, fg=FG_DIM, font=dot_font)
        self._dot_goal   = tk.Label(fs, text="* Goal",   bg=BG2, fg=FG_DIM, font=dot_font)
        for w in (self._dot_camera, self._dot_paused, self._dot_goal):
            w.pack(side="left", padx=6, pady=4)
        self._lbl_stats = tk.Label(fs, text="", bg=BG2, fg=FG_DIM, font=("Helvetica", 9))
        self._lbl_stats.pack(side="right", padx=8)

        camera_wrap = tk.Frame(root, bg=BG)
        camera_wrap.pack(fill="both", expand=True, padx=10, pady=4)
        camera_wrap.columnconfigure(0, weight=1)
        camera_wrap.rowconfigure(0, weight=1)

        fc = tk.Frame(camera_wrap, bg=BG2, bd=2, relief="flat")
        fc.grid(row=0, column=0, sticky="nsew")
        fc.columnconfigure(0, weight=1)
        fc.rowconfigure(1, weight=1)
        tk.Label(fc, text="Camera Feed", bg=BG2, fg=FG_DIM, font=("Helvetica", 9)).grid(row=0, column=0, pady=(6, 0), sticky="n")
        
        self._cam_container = tk.Frame(fc, bg="#000000")
        self._cam_container.grid(row=1, column=0, padx=12, pady=10, sticky="nsew")
        self._cam_container.columnconfigure(0, weight=1)
        self._cam_container.rowconfigure(0, weight=1)
        self._cam_canvas = tk.Canvas(self._cam_container, bg="#000000", highlightthickness=0)
        self._cam_canvas.grid(row=0, column=0, sticky="nsew")
        self._cam_image_id = self._cam_canvas.create_text(10, 10, anchor="nw", fill=FG_DIM, font=("Helvetica", 10), text="No camera")

        bottom = tk.Frame(root, bg=BG)
        bottom.pack(fill="x", padx=10, pady=(0, 8))

        fin = tk.Frame(bottom, bg=BG)
        fin.pack(fill="x", pady=6)
        tk.Label(fin, text="Instruction / command  (Up/Down = history, Enter = send):", bg=BG, fg=FG_DIM, font=("Helvetica", 9)).pack(anchor="w")
        self._entry_var = tk.StringVar()
        self._entry = tk.Entry(fin, textvariable=self._entry_var, bg=BG2, fg=FG, insertbackground=FG, font=("Helvetica", 24, "bold"), relief="flat", bd=4)
        self._entry.pack(fill="x", ipady=8, pady=(2, 4))
        self._entry.focus()
        self._entry.bind("<Return>", lambda _e: self._gui_submit())
        self._entry.bind("<Up>",     lambda _e: self._history_up())
        self._entry.bind("<Down>",   lambda _e: self._history_down())

        fb = tk.Frame(bottom, bg=BG)
        fb.pack(fill="x", pady=2)
        _b = dict(bg=ACCENT, fg=FG, font=("Helvetica", 10), relief="flat", activebackground="#5b21b6", activeforeground=FG, padx=6, pady=4)
        _b_stop = dict(bg="#92400e", fg=FG, font=("Helvetica", 10), relief="flat", activebackground="#78350f", activeforeground=FG, padx=6, pady=4)
        tk.Button(fb, text="Send",       command=self._gui_submit, **_b).pack(side="left", padx=2)
        tk.Button(fb, text="⏸ Freeze",  command=self._gui_freeze, **_b_stop).pack(side="left", padx=2)
        tk.Button(fb, text="▷ Unfreeze",command=self._gui_unfreeze, **_b).pack(side="left", padx=2)
        tk.Button(fb, text="↺ Reset Goal", command=lambda: self._do_reset_goal(), **_b).pack(side="left", padx=2)
        tk.Button(fb, text="🎮 Manual Drive", command=self._open_manual_drive,
                  bg="#0369a1", fg=FG, font=("Helvetica", 10, "bold"), relief="flat",
                  activebackground="#075985", activeforeground=FG, padx=6, pady=4).pack(side="left", padx=2)
        tk.Button(fb, text="Help",         command=lambda: self._cmd_queue.put("help"), **_b).pack(side="left", padx=2)
        
        self._btn_object = tk.Button(fb, text="Object: ON" if CFG.enable_grounding else "Object: OFF", command=self._toggle_object, bg="#15803d" if CFG.enable_grounding else "#475569", fg=FG, font=("Helvetica", 10), relief="flat", padx=6, pady=4)
        self._btn_object.pack(side="left", padx=10)
        
        tk.Button(fb, text="✕ Quit", command=self._gui_quit, bg="#7f1d1d", fg=FG, font=("Helvetica", 10), relief="flat", activebackground="#991b1b", activeforeground=FG, padx=6, pady=4).pack(side="right", padx=2)


        fh = tk.Frame(bottom, bg=BG2)
        fh.pack(fill="x", pady=4)
        tk.Label(fh, text="Recent", bg=BG2, fg=FG_DIM, font=("Helvetica", 9)).pack(anchor="w", padx=6, pady=(4, 0))
        self._lbl_history = tk.Label(fh, text="", bg=BG2, fg=FG_DIM, font=("Helvetica", 9), justify="left", anchor="w")
        self._lbl_history.pack(fill="x", padx=6, pady=(0, 4))

        root.protocol("WM_DELETE_WINDOW", self._gui_quit)
        root.after(33, self._update_gui)

    def _toggle_object(self):
        CFG.enable_grounding = not CFG.enable_grounding
        if CFG.enable_grounding:
            self._btn_object.config(text="Object: ON", bg="#15803d")
            log("Object detection enabled.")
        else:
            self._btn_object.config(text="Object: OFF", bg="#475569")
            log("Object detection disabled.")
            
    def _gui_submit(self):

        text = self._entry_var.get().strip()
        if text:
            self._cmd_queue.put(text)
            self._cmd_history.insert(0, text)
            self._cmd_history = self._cmd_history[:20]
            self._hist_idx = -1
            self._entry_var.set("")

    def _gui_freeze(self):
        """Safety pause — directly zeroes velocity, bypasses OmniVLA."""
        self._paused = True
        self.publish_cmd(0.0, 0.0)
        log("FREEZE: robot halted (hardware override). Click Unfreeze to resume.")

    def _gui_unfreeze(self):
        """Resume inference with the current OmniVLA instruction."""
        self._paused = False
        log(f"UNFREEZE: resuming OmniVLA instruction '{self.instruction}'")

    def _gui_quit(self):
        """Quit button / window close — stop robot and shut down."""
        self._running = False
        self.publish_cmd(0.0, 0.0)
        log("Quit requested.")
        try:
            self.root.destroy()
        except Exception:
            pass

    def _do_reset_goal(self):
        self._goal_reached     = False

        self._trajectory_energy_buffer.clear() #Experimental
        self._trajectory_delta_buffer.clear() #Experimental
        self._trajectory_stop_counter = 0 #Experimental
        self._prev_e78 = None #Experimental
        self._initial_e78 = None   #Experimental
        self._instruction_start_time = time.time() #Experimental

        self._frozen_waypoints = None   #Experimental
        self._freeze_ticks_remaining = 0    #Experimental

        self._frozen_waypoints = None   #Experimental
        self._freeze_ticks_remaining = 0    #Experimental


        self._goal_small_count = 0

       

        log("Goal state reset.")

    # ──────────────────────────────────────────────────────────────────────
    #  Manual teleoperation  (D-pad panel for repositioning the rover by hand)
    # ──────────────────────────────────────────────────────────────────────
    def _open_manual_drive(self):
        """Pop up a small four-direction manual-drive panel.

        While it is open the autonomous loop yields the cmd_vel publisher, so
        the operator can drive the rover back to its start position without
        physically carrying it. Closing the panel (or Stop) returns control.
        """
        if self._manual_win is not None:          # already open → just focus it
            try:
                self._manual_win.deiconify(); self._manual_win.lift()
                self._manual_win.focus_force()
            except Exception:
                self._manual_win = None
            if self._manual_win is not None:
                return

        BG, BG2, FG, ACCENT = "#0f172a", "#1e293b", "#e2e8f0", "#7c3aed"
        win = tk.Toplevel(self.root)
        win.title("Manual Drive")
        win.configure(bg=BG)
        win.resizable(False, False)
        try:
            win.transient(self.root)
        except Exception:
            pass
        self._manual_win = win
        self._manual_drive = True
        self._teleop_dir = (0, 0)

        tk.Label(win, text="Manual Drive  —  hold a button (or arrow keys)",
                 bg=BG, fg=FG, font=("Helvetica", 11, "bold")).grid(
                     row=0, column=0, columnspan=3, padx=10, pady=(10, 4))

        # Speed sliders (live-adjustable). Defaults are gentle for precise
        # repositioning; capped at the configured maxima.
        self._teleop_lin_speed = tk.DoubleVar(value=min(0.10, CFG.max_linear))
        self._teleop_ang_speed = tk.DoubleVar(value=min(0.7, CFG.max_angular))
        sl = tk.Frame(win, bg=BG); sl.grid(row=1, column=0, columnspan=3, padx=10, pady=2, sticky="ew")
        tk.Label(sl, text="Speed", bg=BG, fg="#94a3b8", font=("Helvetica", 9)).pack(anchor="w")
        tk.Scale(sl, from_=0.03, to=max(0.05, CFG.max_linear), resolution=0.01,
                 orient="horizontal", label="linear m/s", variable=self._teleop_lin_speed,
                 bg=BG2, fg=FG, troughcolor="#334155", highlightthickness=0, length=220).pack(fill="x")
        tk.Scale(sl, from_=0.1, to=max(0.2, CFG.max_angular), resolution=0.05,
                 orient="horizontal", label="turn rad/s", variable=self._teleop_ang_speed,
                 bg=BG2, fg=FG, troughcolor="#334155", highlightthickness=0, length=220).pack(fill="x")

        # D-pad:            [ ↑ ]
        #             [ ← ] [ ■ ] [ → ]
        #                   [ ↓ ]
        btn = dict(width=5, height=2, font=("Helvetica", 16, "bold"),
                   bg=ACCENT, fg=FG, activebackground="#5b21b6", activeforeground=FG, relief="flat")
        b_fwd   = tk.Button(win, text="↑", **btn)
        b_left  = tk.Button(win, text="←", **btn)
        b_right = tk.Button(win, text="→", **btn)
        b_back  = tk.Button(win, text="↓", **btn)
        b_stop  = tk.Button(win, text="■", width=5, height=2, font=("Helvetica", 16, "bold"),
                            bg="#b91c1c", fg=FG, activebackground="#7f1d1d", activeforeground=FG, relief="flat",
                            command=self._teleop_release)
        b_fwd.grid(row=2, column=1, padx=4, pady=4)
        b_left.grid(row=3, column=0, padx=4, pady=4)
        b_stop.grid(row=3, column=1, padx=4, pady=4)
        b_right.grid(row=3, column=2, padx=4, pady=4)
        b_back.grid(row=4, column=1, padx=4, pady=(4, 10))

        # Press-and-hold: press sets the direction, release stops.
        for b, d in ((b_fwd, (1, 0)), (b_back, (-1, 0)),
                     (b_left, (0, 1)), (b_right, (0, -1))):   # +turn = LEFT (ROS z)
            b.bind("<ButtonPress-1>",   lambda e, dd=d: self._teleop_press(dd))
            b.bind("<ButtonRelease-1>", lambda e: self._teleop_release())

        # Keyboard: arrow keys / WASD while the panel is focused.
        keymap = {"Up": (1, 0), "Down": (-1, 0), "Left": (0, 1), "Right": (0, -1),
                  "w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}
        for key, d in keymap.items():
            win.bind(f"<KeyPress-{key}>",   lambda e, dd=d: self._teleop_press(dd))
            win.bind(f"<KeyRelease-{key}>", lambda e: self._teleop_release_soft())
        win.bind("<space>", lambda e: self._teleop_release())

        win.protocol("WM_DELETE_WINDOW", self._close_manual_drive)
        win.focus_force()
        log("Manual drive ON — autonomous paused. Hold arrows/buttons to drive.")
        self._manual_pump()

    def _teleop_press(self, direction):
        if self._teleop_stop_job is not None:      # cancel any pending soft-stop
            try: self.root.after_cancel(self._teleop_stop_job)
            except Exception: pass
            self._teleop_stop_job = None
        self._teleop_dir = direction

    def _teleop_release(self):
        """Hard stop (button release / Stop / spacebar)."""
        self._teleop_dir = (0, 0)
        try: self.publish_cmd(0.0, 0.0)
        except Exception: pass

    def _teleop_release_soft(self):
        """Key release: defer the stop ~120 ms so X11 auto-repeat (which fires
        KeyRelease+KeyPress rapidly while a key is held) doesn't cause stutter."""
        if self._teleop_stop_job is not None:
            try: self.root.after_cancel(self._teleop_stop_job)
            except Exception: pass
        self._teleop_stop_job = self.root.after(120, self._teleop_release)

    def _manual_pump(self):
        """10 Hz publisher: turns the held direction into cmd_vel and keeps the
        ESP32 watchdog fed. Runs only while the panel is open."""
        if self._manual_win is None:
            return
        fwd, turn = self._teleop_dir
        lin = fwd  * float(self._teleop_lin_speed.get())
        ang = turn * float(self._teleop_ang_speed.get())
        try:
            self.publish_cmd(lin, ang)
        except Exception:
            pass
        self.root.after(100, self._manual_pump)

    def _close_manual_drive(self):
        """Close the panel, stop the rover, and hand control back to autonomy."""
        self._teleop_dir = (0, 0)
        for _ in range(2):
            try: self.publish_cmd(0.0, 0.0)
            except Exception: pass
        self._manual_drive = False
        win, self._manual_win = self._manual_win, None
        try:
            if win is not None: win.destroy()
        except Exception:
            pass
        log("Manual drive OFF — autonomous control resumed.")

    def _history_up(self):
        if not self._cmd_history:
            return
        self._hist_idx = min(self._hist_idx + 1, len(self._cmd_history) - 1)
        self._entry_var.set(self._cmd_history[self._hist_idx])
        self._entry.icursor(tk.END)

    def _history_down(self):
        if self._hist_idx <= 0:
            self._hist_idx = -1
            self._entry_var.set("")
            return
        self._hist_idx -= 1
        self._entry_var.set(self._cmd_history[self._hist_idx])
        self._entry.icursor(tk.END)

    def _update_gui(self):
        try:
            self._lbl_instruction.config(text=self.instruction[:80])
            frame_age = time.time() - self._last_frame_t
            has_cam   = self.latest_rgb is not None and frame_age < CFG.max_frame_age_s
            self._dot_camera.config(fg=GREEN if has_cam else RED)
            self._dot_paused.config(fg=YELLOW if self._paused       else FG_DIM)
            self._dot_goal.config(  fg=GREEN  if self._goal_reached else FG_DIM)
            self._lbl_stats.config(
                text=f"frames={self._frame_count}  "
                     f"infer={self._infer_count}  "
                     f"{self._infer_hz:.1f}Hz  "
                     f"camera={'OK' if has_cam else 'WAIT'}"
            )
            hist_text = "\\n".join(f"  {c}" for c in self._cmd_history[:4])
            self._lbl_history.config(text=hist_text)

            if _ImageTk is not None:
                with self.lock:
                    rgb = self.latest_rgb
                # Only rebuild the canvas image when a NEW frame arrived. The
                # 30 Hz redraw of a ~6 fps stream was converting the same frame
                # ~5× and its PIL work held the GIL, starving the camera drainer
                # (→ very low update rate). Now the heavy work runs at the frame
                # rate; the cheap status labels above still update every tick.
                if rgb is not None and self._frame_count != self._gui_last_drawn_frame:
                    self._gui_last_drawn_frame = self._frame_count
                    img = Image.fromarray(rgb)
                    
                    # Draw bounding box ON THE ORIGINAL IMAGE before resizing!
                    try:
                        g = getattr(self, "grounding", None).last if getattr(self, "grounding", None) is not None else {}
                        if CFG.enable_grounding and g.get("ok") and "box" in g:
                            x1, y1, x2, y2 = g["box"]
                            draw = ImageDraw.Draw(img)
                            label = f"{g.get('target','target')} {g.get('score',0):.2f} area={g.get('area',0):.2f}"
                            # Background for text at top-right corner of the whole image instead of bounding box
                            draw.rectangle([img.width - 250, 0, img.width, 30], fill=(0, 255, 0))
                            draw.text((img.width - 245, 8), label, fill=(0, 0, 0))
                    except Exception as e:
                        pass
                        
                    # Now resize the image to the canvas size
                    target_w = max(640, self._cam_container.winfo_width() - 24)
                    target_h = max(360, self._cam_container.winfo_height() - 24)
                    img_ratio = img.width / max(1, img.height)
                    target_ratio = target_w / max(1, target_h)
                    if img_ratio > target_ratio:
                        resized_w = target_w
                        resized_h = max(1, int(target_w / img_ratio))
                    else:
                        resized_h = target_h
                        resized_w = max(1, int(target_h * img_ratio))
                    img = img.resize((resized_w, resized_h), Image.BILINEAR)

                    if frame_age > 1.0:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([0, 0, img.width, 20], fill=(180, 0, 0))
                        draw.text((4, 3), f"STALE {frame_age:.1f}s", fill=(255, 255, 255))
                    photo = _ImageTk.PhotoImage(img)
                    canvas_w = max(1, self._cam_canvas.winfo_width())
                    canvas_h = max(1, self._cam_canvas.winfo_height())
                    self._cam_canvas.delete("camera_image")
                    self._cam_canvas.delete("camera_text")
                    self._cam_canvas.create_image(
                        canvas_w // 2, canvas_h // 2,
                        image=photo, anchor="center", tags=("camera_image",)
                    )
                    self._cam_canvas.image = photo
                    self._cam_photo = photo
                elif rgb is None:
                    age_txt = f"No camera  ({frame_age:.0f}s ago)" if self._last_frame_t > 0 else "No camera"
                    self._cam_canvas.delete("camera_image")
                    self._cam_canvas.delete("camera_text")
                    self._cam_canvas.create_text(
                        max(10, self._cam_canvas.winfo_width() // 2),
                        max(10, self._cam_canvas.winfo_height() // 2),
                        text=age_txt, fill=FG_DIM, font=("Helvetica", 10), tags=("camera_text",)
                    )
            else:
                pass

        except tk.TclError:
            return

        if self._running:
            try:
                self.root.after(33, self._update_gui)
            except tk.TclError:
                pass

    # ----------------------------------------------------------------
    #  Entry points
    # ----------------------------------------------------------------
    def start_with_gui(self):
        """Build GUI, run inference in a daemon thread, block on mainloop."""
        self._build_gui()
        Thread(target=self.spin,                 daemon=True).start()
        Thread(target=self._terminal_input_loop, daemon=True).start()
        self.root.mainloop()

    def start_headless(self):
        """No GUI -- terminal input in a daemon thread, inference on this thread."""
        Thread(target=self._terminal_input_loop, daemon=True).start()
        self.spin()

    def _terminal_input_loop(self):
        """Background thread: reads terminal stdin and pushes to cmd queue."""
        self._print_help()
        import select, re as _re
        _burst_times: list = []
        _BURST_WINDOW   = 1.5   # seconds
        _MAX_BURST      = 3     # max lines accepted per window
        _flood_warned   = False
        # Pattern: lines that look like log/stderr output, not user commands
        _LOG_PAT = _re.compile(
            r'^(\[\d{2}:\d{2}:\d{2}\]'      # [HH:MM:SS]
            r'|\d{4}-\d{2}-\d{2} \d{2}:'    # 2026-04-06 12:
            r'|[A-Z]+\s+external/'           # I external/...
            r'|Loading checkpoint'           # Loading checkpoint shards
            r'|\+--+\+$'                     # table borders
            r'|\|.*\|$'                      # table rows
            r'|WARNING|FutureWarning'        # warnings
            r'|tensorflow|TF-TRT'            # TF noise
            r'|NUMA node)'                   # CUDA NUMA
        )
        while self._running:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    raw = sys.stdin.readline().strip()
                    if not raw:
                        continue
                    # Skip lines that look like log/code output, not commands
                    if _LOG_PAT.search(raw):
                        continue
                    # Skip very long lines (pasted code/logs)
                    if len(raw) > 250:
                        continue
                    # Rate-limit: detect paste floods (>_MAX_BURST lines in _BURST_WINDOW s)
                    now = time.time()
                    _burst_times = [t for t in _burst_times if now - t < _BURST_WINDOW]
                    if len(_burst_times) >= _MAX_BURST:
                        if not _flood_warned:
                            log("WARN: stdin input flood detected — ignoring paste. "
                                "Type commands one at a time.")
                            _flood_warned = True
                        continue
                    _flood_warned = False
                    _burst_times.append(now)
                    self._cmd_queue.put(raw)
            except Exception:
                time.sleep(0.1)


# ================================================================
#  CLI & entry point
# ================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniVLA Custom -- OmniVLA + OWL-ViT/CLIP grounded visual observer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pi-ip",           type=str,   default=None,
                   help="Pi IP for Zenoh (omit = multicast scouting)")
    p.add_argument("--instruction",     type=str,   default=CFG.instruction)
    p.add_argument("--predict-hz",      type=float, default=CFG.predict_hz)
    p.add_argument("--max-linear",      type=float, default=CFG.max_linear)
    p.add_argument("--max-angular",     type=float, default=CFG.max_angular)
    p.add_argument("--waypoint",        type=int,   default=CFG.waypoint_select,
                   help="Waypoint index to track (0-7)")
    p.add_argument("--goal-threshold",  type=float, default=CFG.goal_reached_threshold,
                   help="Waypoint magnitude threshold for goal-reached (m)")
    p.add_argument("--goal-count",      type=int,   default=CFG.goal_reached_count,
                   help="Consecutive ticks below threshold to confirm goal")
    p.add_argument("--vla-path",        type=str,   default=CFG.vla_path)
    p.add_argument("--no-grounding",    action="store_true",
                   help="Disable Grounding DINO/CLIP visual target observer")
    p.add_argument("--grounding-device", type=str, default=CFG.grounding_device, choices=["cpu", "cuda"],
                   help="Device for Grounding DINO/CLIP observer")
    p.add_argument("--dino-model",       type=str, default=CFG.dino_model)
    p.add_argument("--clip-model",      type=str, default=CFG.clip_model)
    p.add_argument("--dino-conf",        type=float, default=CFG.dino_conf_threshold,
                   help="Grounding DINO confidence threshold")
    p.add_argument("--dino-stop-area",   type=float, default=CFG.dino_stop_area_ratio,
                   help="Stop when detected target box covers this fraction of frame")
    p.add_argument("--dino-center",      type=float, default=CFG.dino_center_tolerance,
                   help="Stop only if normalized box center offset is below this value")
    p.add_argument("--no-clip-bias",    action="store_true",
                   help="Disable CLIP left/center/right steering bias")
    p.add_argument("--no-centering",    action="store_true",
                   help="Disable steering to center the detected target in view")
    p.add_argument("--center-kp",       type=float, default=CFG.center_kp,
                   help="Proportional gain (1/s) mapping target angular error → turn rate")
    p.add_argument("--camera-hfov",     type=float, default=CFG.camera_hfov_deg,
                   help="Camera horizontal FOV (deg) — for geometrically accurate centering")
    p.add_argument("--center-min-cmd",  type=float, default=CFG.center_min_cmd,
                   help="Min turn rate (rad/s) applied outside the deadband (stiction floor)")
    p.add_argument("--center-gain",     type=float, default=CFG.center_gain,
                   help="(legacy, unused) kept for CLI back-compat")
    p.add_argument("--center-deadband", type=float, default=CFG.center_deadband,
                   help="Ignore target center offsets below this (centered-enough band)")
    p.add_argument("--no-avoid",        action="store_true",
                   help="Disable vision-only static-obstacle avoidance")
    p.add_argument("--depth-model",     type=str, default=CFG.depth_model,
                   help="HF monocular depth model for obstacle avoidance")
    p.add_argument("--avoid-gain",      type=float, default=CFG.avoid_gain,
                   help="rad/s steer-away per unit (closeness - threshold)")
    p.add_argument("--avoid-near-thresh", type=float, default=CFG.avoid_near_thresh,
                   help="Normalized disparity above which a side counts as close (0-1)")
    p.add_argument("--avoid-period",    type=int, default=CFG.avoid_period,
                   help="Run depth model every N inference ticks")
    p.add_argument("--no-gui",          action="store_true",
                   help="Headless mode -- no Tkinter window")
    p.add_argument("--no-rviz",         action="store_true",
                   help="Skip RViz auto-launch")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI args to config
    CFG.instruction            = args.instruction
    CFG.predict_hz             = args.predict_hz
    CFG.max_linear             = args.max_linear
    CFG.max_angular            = args.max_angular
    CFG.waypoint_select        = args.waypoint
    CFG.goal_reached_threshold = args.goal_threshold
    CFG.goal_reached_count     = args.goal_count
    CFG.vla_path               = args.vla_path
    CFG.enable_grounding       = not args.no_grounding
    CFG.grounding_device       = args.grounding_device
    CFG.dino_model             = args.dino_model
    CFG.clip_model             = args.clip_model
    CFG.dino_conf_threshold    = args.dino_conf
    CFG.dino_stop_area_ratio   = args.dino_stop_area
    CFG.dino_center_tolerance  = args.dino_center
    CFG.clip_bias_enabled      = not args.no_clip_bias
    CFG.center_enabled         = not args.no_centering
    CFG.center_kp              = args.center_kp
    CFG.camera_hfov_deg        = args.camera_hfov
    CFG.center_min_cmd         = args.center_min_cmd
    CFG.center_gain            = args.center_gain
    CFG.center_deadband        = args.center_deadband
    CFG.avoid_enabled          = not args.no_avoid
    CFG.depth_model            = args.depth_model
    CFG.avoid_gain             = args.avoid_gain
    CFG.avoid_near_thresh      = args.avoid_near_thresh
    CFG.avoid_period           = args.avoid_period
    # Camera auto-restart is off by default (restarts caused more harm than good).
    # Opt in only for a genuinely dead camera: ENABLE_CAM_WATCHDOG=1 ./launch...
    CFG.camera_watchdog_enabled = os.environ.get("ENABLE_CAM_WATCHDOG") == "1"
    CFG.tick_dt                = 1.0 / CFG.predict_hz

    load_models()

    # RViz auto-launch
    if not args.no_rviz:
        rviz_cfg = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../files/rover_nav.rviz")
        )
        cmd = ["rviz2", "-d", rviz_cfg] if os.path.exists(rviz_cfg) else ["rviz2"]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"RViz launched: {' '.join(cmd)}")
        except FileNotFoundError:
            log("rviz2 not found -- skipping (run on Pi or install ros-humble-rviz2).")
        except Exception as e:
            log(f"RViz launch failed: {e}")

    # Zenoh session
    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        # Disable UDP multicast scouting — we connect via explicit TCP to the Pi.
        # This avoids EADDRINUSE conflicts with any local DDS nodes on the GPU.
        config.insert_json5("scouting/multicast/enabled", "false")
        log(f"Zenoh -> tcp/{args.pi_ip}:7447")
    else:
        log("Zenoh -> multicast scouting")
    session = zenoh.open(config)
    log("Zenoh session opened.\n")

    node = OmniVLACustomNode(session, pi_ip=args.pi_ip)
    try:
        if args.no_gui:
            node.start_headless()
        else:
            node.start_with_gui()
    finally:
        session.close()
        log("Session closed.")


if __name__ == "__main__":
    main()
