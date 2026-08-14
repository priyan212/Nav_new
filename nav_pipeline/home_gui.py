#!/usr/bin/env python3
"""Nav_new — manual control + "Go Home" GUI (encoder+IMU fused odometry).

Still deliberately independent of pipeline.py / isaac_gui.py / zenoh_node.py
(all three import DinoNavDPPipeline, which additionally pulls in Grounding
DINO / NavDP / CLIP checkpoint loading -- multi-model GPU load none of which
this GUI needs, since Go Home has no target phrase to detect or trajectory
policy to sample). The small CDR (DDS wire format) codec bits it needs from
zenoh_node.py's Zenoh contract (camera image, RPM) are duplicated below
rather than imported, for exactly that reason -- keep them in sync with
zenoh_node.py if the wire contract (RPM_KEYS / CAMERA_KEYS / Float32MultiArray
/ Twist) ever changes.

By DEFAULT this is still exactly the original camera-free, no-GPU, ~1s
startup: no image subscription, no NavDP/DINO/depth models, no NavDP loop.

Obstacle avoidance while homing (NavDP, OPT-IN via --enable-obstacle-avoidance)
--------------------------------------------------------------------------------
Passing --enable-obstacle-avoidance starts the Pi camera subscription and
loads pipeline.py's full DinoNavDPPipeline (DINO detector -- unconditionally
loaded by that class even though never actually invoked for detection here,
see below -- the NavDP policy, and Depth Anything V2), so launch no longer
starts in ~1s/no-GPU -- expect a real model-load pause comparable to
launch_rover.sh's, plus continuous Wi-Fi/CPU load on the Pi from the camera
stream that a plain Go Home run never had (compressed JPEG only by default,
see --compressed-only, but still real added traffic competing with the
ESP32 serial/cmd_vel/rpm path -- keep this off unless you actually need it).

An earlier version of this feature ran its own simple depth-only reactive
guard (single forward-distance threshold + a one-shot escape turn): it
worked, but monocular depth is noisy enough to false-trigger it on nothing,
and its "recovery" was just resuming the bearing-servo from wherever the
turn left the rover -- not a real return to progress toward home. This was
replaced with pipeline.py's "GOTO" mechanism instead (the same one
remind_gui.py uses to drive back to a remembered OBJECT location, just
pointed at a fixed home point instead): every tick, navdp_home_loop feeds
DinoNavDPPipeline.step() a local-frame goal computed from odometry
(object_map.py's world_to_local) via its external_goal argument, which
skips DINO detection entirely (external_dets=None + external_goal set) and
drives via NavDP's diffusion-sampled candidate trajectories, scored by both
goal progress AND a footprint-aware obstacle-clearance veto
(obstacle_guard.py's swept_clearance over the SAME depth-based obstacle
points the old guard used, plus a hard forward_guard/AVOID stop) -- a strict
superset of the old mechanism, not an alternative to it. Because it
re-samples toward the SAME fixed goal every tick, once clear of an obstacle
it naturally keeps making progress toward home again; there's no separate
"resume" logic to get wrong.

pipeline.py's GOTO state deliberately never self-declares "arrived" (stale
dead-reckoning over a long blind leg makes trusting proximity alone unsafe
-- see its own docstring) -- so navdp_home_loop owns that check itself:
once within --home-arrival-radius of home, it hands off to
home_control_loop's existing (unchanged, vision-free) FACE/ARRIVED phase
for the precise final approach + heading match, via st.home_leg ("navdp" |
"servo") -- see navdp_home_loop's and home_control_loop's docstrings for
the exact hand-off/ownership contract. Manual drive is NEVER gated by any
of this, regardless of the flag: it writes st.last_cmd directly through a
completely separate code path (App._manual_update) that neither loop
touches.

Pose estimate
-------------
Position (x, y) is dead-reckoned from wheel-encoder speed (encoders measure
translation distance well). Heading (theta) comes from the BNO055 IMU's
fused Euler heading whenever it's calibrated (esp32/rover_6wd_complete.ino
rpm_data[2]/[3]) instead of being integrated from the wheel differential --
skid-steer wheel-diff heading drifts hard (wheel slip during in-place turns
is exactly where it's worst), while the IMU's onboard NDOF fusion has no
accumulating drift once its magnetometer is calibrated. See
OdometryLogger._imu_theta() in odometry_logger.py for the gating/conversion.
No separate Kalman filter is layered on top of that: the BNO055 already runs
its own internal sensor fusion (accel+gyro+mag) to produce that heading, and
the encoder speed reading is not itself noisy enough to need state
estimation -- a second filter here would add tuning risk without a clear
accuracy win. If wheel slip during straight-line driving (not just turns)
ever turns out to corrupt the encoder *distance* estimate too, a real EKF
fusing encoder+IMU with a slip covariance term would be the right upgrade;
nothing here precludes bolting that onto OdometryLogger later.

Go Home
-------
Continuously-corrected go-to-goal control (recomputes bearing/distance from
the fused pose every tick), not an open-loop "turn 180, drive N meters":
  - ROTATE phase: heading error > ~20 deg -> pure in-place rotation only.
  - DRIVE phase: heading error < ~8 deg -> drive forward, with the same
    turn-rate steering correction folded in to hold the line (hysteresis
    band between the two thresholds stops phase-chattering right at the
    boundary).
  - Once within --home-dist-tol of home (default 10 cm): FACE phase --
    stop translating and rotate in place to match the heading recorded at
    "Set Home Here" (or theta=0, the launch heading, if home was never
    re-set), within --home-heading-tol (default 5 deg). Only then does it
    report ARRIVED -- the rover ends up facing the same way it started,
    not just at the same spot.
Since it re-reads the fused pose every tick, drift or a bump mid-return gets
corrected on the fly rather than compounding, which is what makes this more
reliable than a single fixed rotate-then-drive plan.

Run (from Nav_new root, internnav conda env -- needed now for the NavDP
obstacle-avoidance models, see above):
    python -m nav_pipeline.home_gui [--pi-ip <IP>]
"""

import argparse
import math
import os
import signal
import struct
import sys
import time
import traceback
from collections import deque
from threading import Lock, Thread
from typing import List, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import ttk

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# DinoNavDPPipeline/PipelineConfig (nav_pipeline.pipeline) are NOT imported
# here: they pull in torch/DINO/NavDP checkpoint loading, which the original
# ~1s/no-GPU startup this file exists for (see above) doesn't have and
# shouldn't need by default. Imported lazily in main(), only inside the
# --enable-obstacle-avoidance branch.
from nav_pipeline.object_map import world_to_local  # noqa: E402 -- pure trig, no heavy deps
from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402 -- dataclass only, no heavy deps
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402

# ================================================================
#  CDR (DDS wire format) codec -- duplicated from zenoh_node.py, see the
#  module docstring for why. Only the pieces this file actually needs.
# ================================================================
RPM_KEYS = ["rover/rpm", "rt/rover/rpm"]
CAMERA_KEYS = ["image_raw", "rt/image_raw", "rover_camera", "rt/rover_camera"]
CAMERA_COMPRESSED_KEYS = ["image_raw/compressed", "rt/image_raw/compressed"]


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
        v = self.data[self.offset]
        self.offset += 1
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

    def read_float32(self) -> float:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "f", self.data, self.offset)
        self.offset += 4
        return v

    def read_string(self, skip: bool = True):
        length = self.read_uint32()
        if skip:
            self.offset += length
            return None
        s = self.data[self.offset : self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data = self.data[self.offset : self.offset + count]
        self.offset += count
        return data


def parse_float32_multiarray(cdr_data: bytes) -> List[float]:
    """std_msgs/Float32MultiArray CDR -> list[float] (the .data field)."""
    r = CDRReader(cdr_data)
    dim_count = r.read_uint32()
    for _ in range(dim_count):
        r.read_string()
        r.read_uint32()
        r.read_uint32()
    r.read_uint32()  # data_offset
    n = r.read_uint32()
    return [r.read_float32() for _ in range(n)]


def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/Image CDR -> RGB uint8 (H,W,3), or None (depth is not
    consumed here -- Go Home estimates its own monocular depth, see
    depth_estimator.py; unlike zenoh_node.py this never subscribes to an
    external depth topic)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    height = r.read_uint32()
    width = r.read_uint32()
    encoding = r.read_string(skip=False)
    r.read_uint8(); r._align(4); r.read_uint32()  # is_bigendian, step
    pixel_data = r.read_sequence_uint8()

    if encoding.lower() not in ("rgb8", "bgr8"):
        return None
    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None
    if img.shape[2] < 3:
        return None
    if encoding.lower() == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    return img[:, :, :3]


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/CompressedImage CDR -> RGB uint8 (H,W,3)."""
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

        from PIL import Image

        return np.array(Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")
        self.base = 4

    def write_float64(self, v: float):
        rem = (len(self.buf) - self.base) % 8
        if rem:
            self.buf += b"\x00" * (8 - rem)
        self.buf += struct.pack("<d", v)

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    w = CDRWriter()
    w.write_float64(linear_x); w.write_float64(0.0); w.write_float64(0.0)
    w.write_float64(0.0); w.write_float64(0.0); w.write_float64(angular_z)
    return w.to_bytes()


# ================================================================
#  Pure control-law helpers (duplicated from pipeline.py's bearing_to_angular
#  for the same import-weight reason -- see module docstring)
# ================================================================
def bearing_to_angular(bearing: float, max_angular: float, ang_min_cmd: float,
                       deadband: float, ramp_rad: float) -> float:
    """Smooth heading-error -> angular-velocity command (tanh ramp from the
    stiction floor up to max_angular). See pipeline.py's version for the
    full rationale -- identical shape, just torch/numpy-free here."""
    mag = abs(bearing)
    if mag < deadband:
        return 0.0
    span = max(max_angular - ang_min_cmd, 0.0)
    x = 2.0 * (mag - deadband) / max(ramp_rad, 1e-6)
    magnitude = min(ang_min_cmd + span * math.tanh(x), max_angular)
    return math.copysign(magnitude, bearing)


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def decode_calib(c: float) -> str:
    v = int(round(c))
    sys_c, gyr_c, acc_c, mag_c = (v // 1000) % 10, (v // 100) % 10, (v // 10) % 10, v % 10
    return f"SYS{sys_c} GYR{gyr_c} ACC{acc_c} MAG{mag_c}"


HEARTBEAT_PERIOD_S = 0.15
HOMING_CONTROL_HZ = 10.0
ROTATE_ENTER_RAD = math.radians(20.0)
DRIVE_ENTER_RAD = math.radians(8.0)
HOMING_TIMEOUT_S = 180.0
RPM_STALE_S = 1.5
# navdp_home_loop's spin-stall watchdog -- same thresholds as isaac_gui.py's
# (a real documented incident there: 145s/17 turns spinning against nothing,
# 493deg peak turn for 0.36m net travel). Duplicated rather than imported,
# same import-weight reason as everything else in this file's module
# docstring -- isaac_gui.py pulls in DinoNavDPPipeline unconditionally at
# module level.
# Loosened 2026-08-14 in lockstep with isaac_gui.py's copy (15s/1 turn ->
# 20s/1.5 turns) at user request -- see isaac_gui.py's comment for why this
# still catches the documented incident.
NAVDP_SPIN_WINDOW_S = 20.0
NAVDP_SPIN_ROT_THRESH_RAD = 3.0 * math.pi
NAVDP_SPIN_DIST_THRESH_M = 0.3


# ================================================================
#  Shared state
# ================================================================
class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.x = self.y = self.theta = 0.0
        self.imu_heading_deg = float("nan")
        self.imu_calib = 0.0
        self.theta_source = "enc"
        self.home: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.path: deque = deque(maxlen=8000)
        self._last_path_pt: Optional[Tuple[float, float]] = None
        self.mode = "idle"  # "manual" | "homing" | "idle"
        self.stopped = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.15
        self.max_angular = 0.5
        self.home_phase = "-"
        self.home_dist = 0.0
        self.home_heading_err = 0.0
        self.rpm_count = 0
        self.last_rpm_t = 0.0
        # NavDP obstacle avoidance while homing (see navdp_home_loop)
        self.latest_rgb: Optional[np.ndarray] = None
        self.frame_count = 0
        self.home_leg = "servo"  # "navdp" | "servo" -- who's allowed to write last_cmd while
        #                          homing; see navdp_home_loop's and home_control_loop's
        #                          docstrings for the hand-off contract. "servo" (a no-op gate)
        #                          when --enable-obstacle-avoidance is off.
        self.navdp_state = "-"  # "-" | "GOTO" | "AVOID" (pipe.step()'s StepResult.state)
        self.navdp_min_forward = float("inf")
        self.navdp_infer_count = 0
        self.navdp_lat_text = ""


def zenoh_setup(session: zenoh.Session, st: SharedState, odom: OdometryLogger, compressed_only: bool = False,
                enable_camera: bool = False):
    def on_rpm(sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) < 2:
                return
            imu_heading = data[2] if len(data) >= 3 else None
            imu_calib = data[3] if len(data) >= 4 else None
            lateral_m_s = data[4] if len(data) >= 5 else None  # holonomic chassis only, see landerpi/bridge.py
            odom.update(data[0], data[1], imu_heading_deg=imu_heading, imu_calib=imu_calib,
                        lateral_m_s=lateral_m_s)
            with st.lock:
                st.x, st.y, st.theta = odom.x, odom.y, odom.theta
                st.theta_source = odom.theta_source
                if imu_heading is not None:
                    st.imu_heading_deg = imu_heading
                if imu_calib is not None:
                    st.imu_calib = imu_calib
                st.rpm_count += 1
                st.last_rpm_t = time.time()
                if (st._last_path_pt is None
                        or math.hypot(st.x - st._last_path_pt[0], st.y - st._last_path_pt[1]) > 0.02):
                    st.path.append((st.x, st.y))
                    st._last_path_pt = (st.x, st.y)
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    def on_image(sample):
        try:
            img = parse_image(bytes(sample.payload))
            if img is not None:
                with st.lock:
                    st.latest_rgb = img
                    st.frame_count += 1
        except Exception as e:
            print(f"[WARN] image parse failed: {e}")

    def on_image_compressed(sample):
        try:
            img = parse_compressed_image(bytes(sample.payload))
            if img is not None:
                with st.lock:
                    st.latest_rgb = img
                    st.frame_count += 1
        except Exception as e:
            print(f"[WARN] compressed image parse failed: {e}")

    # Camera subscription is opt-in (--enable-obstacle-avoidance): even
    # compressed JPEG is continuous Wi-Fi traffic + Pi CPU load competing
    # with the ESP32 serial/cmd_vel/rpm path that plain manual-drive/Go-Home
    # never had before obstacle avoidance existed -- default stays exactly
    # the original camera-free behavior. On the real rover Wi-Fi, raw
    # 640x480 rgb8 (~8 MB/s) additionally saturates the link and starves
    # cmd_vel/rpm outright (same failure mode documented in isaac_gui.py's
    # zenoh_setup), hence compressed-only within that.
    if enable_camera:
        raw_keys = [] if compressed_only else CAMERA_KEYS
        compressed_keys = CAMERA_COMPRESSED_KEYS
    else:
        raw_keys = []
        compressed_keys = []
    subs = (
        [session.declare_subscriber(k, on_rpm) for k in RPM_KEYS]
        + [session.declare_subscriber(k, on_image) for k in raw_keys]
        + [session.declare_subscriber(k, on_image_compressed) for k in compressed_keys]
    )
    pubs = {"cmd": session.declare_publisher("cmd_vel")}
    return subs, pubs


def navdp_home_loop(pipe: "DinoNavDPPipeline", st: SharedState, running: dict,
                    odom: OdometryLogger, predict_hz: float,
                    arrival_radius: float, navdp_timeout_s: float) -> None:
    """NavDP-driven outer Go-Home leg (see module docstring): drives toward
    st.home via pipeline.py's GOTO state (external_goal, no DINO detection
    involved) at its own predict_hz cadence -- NavDP + depth estimation +
    the obstacle guard all run synchronously inside pipe.step() each tick,
    no separate depth thread needed here (unlike the old simpler guard).

    Ownership contract with home_control_loop: this loop owns st.last_cmd
    (and holds st.home_leg == "navdp") for the long cross-room leg. It never
    self-declares arrival (pipeline.py's GOTO state deliberately doesn't --
    see its step() docstring: dead-reckoning drift over a long blind walk
    makes trusting proximity alone unsafe) -- instead, the INSTANT the
    odometry distance to home drops to arrival_radius, it sets
    st.home_leg = "servo" and stops touching st.last_cmd for the rest of
    this homing session, handing off to home_control_loop's existing (and
    already correct) FACE/ARRIVED phase for the precise final approach +
    heading match. Only ever runs while st.mode == "homing" AND
    args.enable_obstacle_avoidance is set (the thread itself is never
    started otherwise -- see main()); never touches manual drive.
    """
    period = 1.0 / predict_hz
    was_homing = False
    start_t: Optional[float] = None

    while running["on"]:
        t0 = time.time()
        with st.lock:
            mode = st.mode
            x, y, theta = st.x, st.y, st.theta
            hx, hy, _ = st.home
            rgb = st.latest_rgb

        if mode != "homing":
            if was_homing:
                pipe.reset()  # don't let stale memory/avoid-state leak into the next Go-Home press
            was_homing = False
            start_t = None
            time.sleep(period)
            continue

        if not was_homing:
            pipe.reset()
            start_t = time.time()
        was_homing = True

        dist = math.hypot(hx - x, hy - y)
        if dist <= arrival_radius:
            with st.lock:
                st.home_leg = "servo"
                st.navdp_state = "-"
            time.sleep(period)
            continue
        with st.lock:
            st.home_leg = "navdp"

        if time.time() - start_t > navdp_timeout_s:
            print(f"[WARN] navdp-home timeout after {navdp_timeout_s:.0f}s ({dist:.2f}m left) "
                  f"-- stopping, check for a persistent obstacle")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.mode = "idle"
                st.home_leg = "servo"
                st.home_phase = f"NAVDP TIMEOUT ({dist:.2f}m left)"
            time.sleep(period)
            continue

        if rgb is None:
            # can't plan trajectories without a frame -- hold rather than drive blind
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.home_phase = "WAITING FOR CAMERA"
                st.home_dist = dist
            time.sleep(period)
            continue

        spin = odom.spin_delta(NAVDP_SPIN_WINDOW_S)
        if spin is not None and spin[0] > NAVDP_SPIN_ROT_THRESH_RAD and spin[1] < NAVDP_SPIN_DIST_THRESH_M:
            print(f"[WARN] navdp-home spin-stall: {math.degrees(spin[0]):.0f}deg turned in "
                  f"{NAVDP_SPIN_WINDOW_S:.0f}s, only {spin[1]:.2f}m net travel -- stopping")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.mode = "idle"
                st.home_leg = "servo"
                st.home_phase = "SPIN STALL"
            time.sleep(period)
            continue

        lx, ly = world_to_local((hx, hy), (x, y, theta))
        goal = np.array([lx, ly, 0.0], dtype=np.float32)
        try:
            res = pipe.step(rgb, "home", depth=None, pose=(x, y, theta),
                            external_dets=None, external_goal=goal)
        except Exception as e:
            print(f"[ERROR] navdp-home step failed: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.home_phase = f"NAVDP ERROR: {e}"
            time.sleep(period)
            continue

        with st.lock:
            st.last_cmd = (res.linear, res.angular)
            st.home_phase = f"NAVDP-{res.state}"
            st.home_dist = dist
            st.navdp_state = res.state
            st.navdp_min_forward = res.min_forward
            st.navdp_infer_count += 1
            st.navdp_lat_text = "  ".join(f"{k} {v * 1000:.0f}ms" for k, v in res.timing.items())

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def heartbeat_loop(st: SharedState, pubs, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = st.last_cmd
        pubs["cmd"].put(serialize_twist(lin, ang))


def home_control_loop(st: SharedState, running, args):
    """Go-to-goal controller driving toward st.home. Only actually commands
    anything while st.mode == "homing" AND it currently owns st.home_leg
    ("servo") -- otherwise idles and keeps its phase state reset so the next
    "Go Home" press (or the next navdp_home_loop hand-off) always starts
    from ROTATE.

    When --enable-obstacle-avoidance is on, navdp_home_loop owns the long
    cross-room leg (st.home_leg == "navdp") and this loop stands down until
    it hands off at the arrival radius -- see navdp_home_loop's docstring
    for the exact contract. When the flag is off, st.home_leg never leaves
    its default "servo", so this loop behaves exactly as it always did
    (this whole function is otherwise untouched from before obstacle
    avoidance existed): pure vision-free bearing-servo ROTATE/DRIVE/FACE/
    ARRIVED, no obstacle awareness of its own."""
    period = 1.0 / HOMING_CONTROL_HZ
    phase = "rotate"
    prev_ang = 0.0
    start_t: Optional[float] = None

    while running["on"]:
        t0 = time.time()
        with st.lock:
            mode = st.mode
            x, y, theta = st.x, st.y, st.theta
            hx, hy, home_theta = st.home
            home_leg = st.home_leg

        if mode != "homing" or (args.enable_obstacle_avoidance and home_leg == "navdp"):
            phase = "rotate"
            prev_ang = 0.0
            start_t = None
            time.sleep(period)
            continue

        if start_t is None:
            start_t = time.time()
            phase = "rotate"

        dx, dy = hx - x, hy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        err = normalize_angle(bearing - theta)
        home_heading_err = normalize_angle(home_theta - theta)

        if time.time() - start_t > HOMING_TIMEOUT_S:
            print(f"[WARN] go-home timeout after {HOMING_TIMEOUT_S:.0f}s "
                  f"({dist:.2f}m still remaining) -- stopping, check IMU calib / retry")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.mode = "idle"
                st.home_phase = f"TIMEOUT ({dist:.2f}m left)"
            time.sleep(period)
            continue

        if dist <= args.home_dist_tol:
            # Position reached. Final phase: rotate in place to match the
            # heading recorded at "Set Home Here" (or the launch heading,
            # theta=0, if home was never re-set) -- so the rover ends up
            # facing the same way it started rather than whatever direction
            # the last approach leg happened to leave it pointing.
            phase = "face"
            if abs(home_heading_err) < math.radians(args.home_heading_tol):
                with st.lock:
                    st.last_cmd = (0.0, 0.0)
                    st.mode = "idle"
                    st.home_phase = "ARRIVED"
                    st.home_dist = dist
                    st.home_heading_err = home_heading_err
                time.sleep(period)
                continue

            ang = bearing_to_angular(home_heading_err, args.home_max_angular, args.home_ang_min_cmd,
                                      args.home_deadband, math.radians(args.home_ramp_deg))
            if args.home_angular_slew_max > 0:
                max_delta = args.home_angular_slew_max * period
                ang = max(prev_ang - max_delta, min(prev_ang + max_delta, ang))
            prev_ang = ang

            with st.lock:
                st.last_cmd = (0.0, ang)
                st.home_phase = "FACING HOME DIR"
                st.home_dist = dist
                st.home_heading_err = home_heading_err
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            continue

        # hysteresis: cheap turn-in-place until roughly facing home, then
        # drive with a gentler steering correction -- avoids chattering
        # right at a single fixed threshold (also applies coming out of the
        # "face" phase if the rover drifted back outside home_dist_tol)
        if phase in ("rotate", "face"):
            phase = "rotate" if abs(err) >= DRIVE_ENTER_RAD else "drive"
        elif abs(err) > ROTATE_ENTER_RAD:
            phase = "rotate"

        ang = bearing_to_angular(err, args.home_max_angular, args.home_ang_min_cmd,
                                  args.home_deadband, math.radians(args.home_ramp_deg))
        if args.home_angular_slew_max > 0:
            max_delta = args.home_angular_slew_max * period
            ang = max(prev_ang - max_delta, min(prev_ang + max_delta, ang))
        prev_ang = ang

        if phase == "rotate":
            lin = 0.0
        else:
            lin = min(args.home_max_linear, max(args.home_lin_min_cmd, args.home_kp_lin * dist))
            lin *= max(0.15, 1.0 - 0.8 * abs(ang) / max(args.home_max_angular, 1e-6))

        with st.lock:
            st.last_cmd = (lin, ang)
            st.home_phase = "ROTATING" if phase == "rotate" else "DRIVING"
            st.home_dist = dist
            st.home_heading_err = err

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ================================================================
#  GUI
# ================================================================
class App:
    PLOT_SIZE = 520
    MIN_SPAN_M = 1.0  # floor on the autoscale range so a stationary rover isn't shown zoomed to a point

    def __init__(self, root: tk.Tk, st: SharedState, obstacle_avoidance_enabled: bool = False,
                 imu_min_mag_calib: int = 3):
        self.root = root
        self.st = st
        self.obstacle_avoidance_enabled = obstacle_avoidance_enabled
        self.imu_min_mag_calib = imu_min_mag_calib
        root.title("Nav_new — Manual Control + Go Home")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=0, padx=4, pady=4)

        # -------- manual drive (same hold-button / arrow-key pattern as isaac_gui.py) --------
        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=1, column=0, sticky="w", pady=(6, 2))
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        # -------- home controls --------
        home_bar = ttk.Frame(main)
        home_bar.grid(row=2, column=0, sticky="w", pady=2)
        ttk.Button(home_bar, text="GO HOME", command=self.go_home).pack(side="left", padx=2)
        ttk.Button(home_bar, text="Set Home Here", command=self.set_home_here).pack(side="left", padx=2)
        ttk.Button(home_bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=90, anchor="w")
        self.status.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=90, anchor="w")
        self.info.grid(row=4, column=0, sticky="w")
        self.warn = ttk.Label(main, text="", width=90, anchor="w", foreground="#b00000")
        self.warn.grid(row=5, column=0, sticky="w")

        self.root.after(66, self.refresh)

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str):
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str):
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self):
        with self.st.lock:
            lin = 0.0
            ang = 0.0
            if "fwd" in self._manual_held:
                lin += self.st.max_linear
            if "back" in self._manual_held:
                lin -= 0.5 * self.st.max_linear
            if "left" in self._manual_held:
                ang += self.st.max_angular
            if "right" in self._manual_held:
                ang -= self.st.max_angular
            self.st.mode = "manual"
            self.st.stopped = False
            self.st.last_cmd = (lin, ang)

    # ---------------- home / stop ---------------- #
    def go_home(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.mode = "homing"
            self.st.stopped = False
            # Set in the SAME lock as st.mode so navdp_home_loop can never
            # lose the race to home_control_loop on a fresh press -- see
            # navdp_home_loop's docstring for the ownership contract.
            self.st.home_leg = "navdp" if self.obstacle_avoidance_enabled else "servo"
            self.st.home_phase = "ROTATING"

    def set_home_here(self):
        with self.st.lock:
            self.st.home = (self.st.x, self.st.y, self.st.theta)
            self.st.home_phase = "-"

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.mode = "idle"
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)
            self.st.home_leg = "servo"
            if self.st.home_phase not in ("ARRIVED",):
                self.st.home_phase = "STOPPED (user)"

    def on_close(self):
        self.closed = True
        self.root.destroy()

    # ---------------- drawing ---------------- #
    def _bounds(self, path, home, x, y):
        xs = [p[0] for p in path] + [home[0], x]
        ys = [p[1] for p in path] + [home[1], y]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span = max(maxx - minx, maxy - miny, self.MIN_SPAN_M) * 1.25
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        return cx, cy, span

    def refresh(self):
        if self.closed:
            return
        try:
            self._refresh_body()
        except Exception:
            # See isaac_gui.py's refresh() for why this guard exists: without
            # it, a single uncaught exception here permanently kills this
            # self-rescheduling redraw loop -- the GUI freezes on whatever
            # was last drawn while the homing loop keeps running fine in its
            # own thread. Symptom this fixes: "GUI pauses on one frame mid
            # run."
            traceback.print_exc()
        finally:
            if not self.closed:
                self.root.after(66, self.refresh)

    def _refresh_body(self):
        with self.st.lock:
            x, y, theta = self.st.x, self.st.y, self.st.theta
            home = self.st.home
            path = list(self.st.path)
            mode, stopped = self.st.mode, self.st.stopped
            phase, dist, herr = self.st.home_phase, self.st.home_dist, self.st.home_heading_err
            lin, ang = self.st.last_cmd
            imu_heading, imu_calib, theta_src = self.st.imu_heading_deg, self.st.imu_calib, self.st.theta_source
            rpm_count, last_rpm_t = self.st.rpm_count, self.st.last_rpm_t
            frame_count = self.st.frame_count
            navdp_state, min_fwd = self.st.navdp_state, self.st.navdp_min_forward
            home_leg = self.st.home_leg

        S = self.PLOT_SIZE
        cx, cy, span = self._bounds(path, home, x, y)
        scale = (S * 0.9) / span

        def to_px(wx, wy):
            return S / 2 + (wx - cx) * scale, S / 2 - (wy - cy) * scale

        self.plot.delete("all")
        # faint crosshair through home
        hpx, hpy = to_px(home[0], home[1])
        self.plot.create_line(0, hpy, S, hpy, fill="#eee")
        self.plot.create_line(hpx, 0, hpx, S, fill="#eee")

        if len(path) >= 2:
            pts = [to_px(px, py) for px, py in path]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="#4a90d9", width=2)

        # home marker
        self.plot.create_oval(hpx - 7, hpy - 7, hpx + 7, hpy + 7, outline="#1a7a1a", width=2)
        self.plot.create_text(hpx, hpy - 14, text="HOME", fill="#1a7a1a", font=("TkDefaultFont", 9, "bold"))

        if mode == "homing":
            xpx, ypx = to_px(x, y)
            self._dashed_line(xpx, ypx, hpx, hpy, fill="#d4a017")

        # robot marker + heading arrow
        xpx, ypx = to_px(x, y)
        alen = 16
        hx2, hy2 = xpx + alen * math.cos(theta), ypx - alen * math.sin(theta)
        self.plot.create_oval(xpx - 6, ypx - 6, xpx + 6, ypx + 6, fill="#1a1a1a")
        self.plot.create_line(xpx, ypx, hx2, hy2, fill="#1a1a1a", width=3, arrow="last")

        mode_txt = "STOPPED" if stopped and mode == "idle" else mode.upper()
        if self.obstacle_avoidance_enabled:
            min_fwd_txt = f"{min_fwd:.2f}m" if math.isfinite(min_fwd) else "clear"
            obstacle_txt = f"navdp[{home_leg}]: {navdp_state} ({min_fwd_txt})"
        else:
            obstacle_txt = "obstacle avoidance: off"
        self.status.configure(
            text=f"[{mode_txt}]  home-phase: {phase}   dist {dist:.2f}m   heading-err {math.degrees(herr):+.1f}°"
                 f"   lin {lin:.3f}  ang {ang:+.3f}   {obstacle_txt}"
        )
        heading_txt = f"{imu_heading:.1f}°" if math.isfinite(imu_heading) else "n/a"
        self.info.configure(
            text=f"pose x={x:.2f} y={y:.2f} theta={math.degrees(theta):+.1f}°  "
                 f"(theta src: {theta_src})   imu heading {heading_txt}  calib [{decode_calib(imu_calib)}]"
        )
        stale = last_rpm_t == 0.0 or (time.time() - last_rpm_t) > RPM_STALE_S
        if stale:
            self.warn.configure(text="⚠ NO /rover/rpm DATA — check rover-agent on the Pi (see launch script output)")
        elif self.obstacle_avoidance_enabled and frame_count == 0:
            self.warn.configure(
                text="⚠ NO CAMERA FRAMES — NavDP obstacle avoidance is blind, Go Home has no collision "
                     "protection until frames arrive (check rover-camera on the Pi)")
        elif (int(round(imu_calib)) % 10 if math.isfinite(imu_calib) else 0) < self.imu_min_mag_calib:
            # Mag-calib check, not theta_src -- theta_src also reads "enc"
            # whenever the wheels aren't turning (OdometryLogger._imu_theta's
            # motion gate, see landerpi/README.md's 2026-08-07 drift fix),
            # which is unrelated to whether the IMU is actually calibrated.
            self.warn.configure(
                text="⚠ heading source: wheel encoders only (IMU not calibrated yet — "
                     f"tilt/rotate the rover until magnetometer calib >={self.imu_min_mag_calib})")
        else:
            self.warn.configure(text=f"rpm samples: {rpm_count}   camera frames: {frame_count}")

    def _dashed_line(self, x0, y0, x1, y1, fill, dash_len=8):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        n = max(1, int(length // (dash_len * 2)))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        for i in range(n + 1):
            sx = x0 + ux * i * dash_len * 2
            sy = y0 + uy * i * dash_len * 2
            ex = sx + ux * dash_len
            ey = sy + uy * dash_len
            self.plot.create_line(sx, sy, ex, ey, fill=fill, width=2)


def main():
    ap = argparse.ArgumentParser(description="Nav_new manual control + Go Home GUI")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--max-linear", type=float, default=0.15, help="manual-drive m/s cap")
    ap.add_argument("--max-angular", type=float, default=0.5, help="manual-drive rad/s cap")
    ap.add_argument("--home-max-linear", type=float, default=0.15)
    ap.add_argument("--home-max-angular", type=float, default=0.5,
                    help="gentler than the 1.2 rad/s reactive-search cap elsewhere -- this is a "
                         "precision return-to-point maneuver, also fed into PipelineConfig for the "
                         "NavDP leg when --enable-obstacle-avoidance is on")
    ap.add_argument("--home-ang-min-cmd", type=float, default=0.12,
                    help="stiction floor, rad/s (matches pipeline.py's tuned real-rover value)")
    ap.add_argument("--home-lin-min-cmd", type=float, default=0.08, help="stiction floor, m/s")
    ap.add_argument("--home-kp-lin", type=float, default=0.15, help="linear speed = clip(kp*dist, floor, max)")
    ap.add_argument("--home-deadband", type=float, default=0.05, help="rad, matches pipeline.py servo_deadband")
    ap.add_argument("--home-ramp-deg", type=float, default=45.0)
    ap.add_argument("--home-angular-slew-max", type=float, default=0.6,
                    help="rad/s^2 max angular acceleration while homing (0 disables)")
    ap.add_argument("--home-dist-tol", type=float, default=0.10, help="meters -- stop within this of home")
    ap.add_argument("--home-heading-tol", type=float, default=5.0,
                    help="deg -- final in-place rotation to match the home heading stops within this")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3,
                    help="0-3; magnetometer calib required before trusting IMU heading over wheel-diff "
                         "(default 3, Bosch's own bar for a trustworthy absolute heading -- lower values "
                         "have been observed to produce 100+ deg heading swings while stationary)")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    # ── NavDP obstacle avoidance while homing (see navdp_home_loop) --
    #    OPT-IN: default Go Home is exactly the original camera-free, no-GPU
    #    behavior. Pass --enable-obstacle-avoidance to turn the camera +
    #    DINO/NavDP/depth models + NavDP-driven GOTO leg on. ──
    ap.add_argument("--enable-obstacle-avoidance", action="store_true",
                    help="start the Pi camera + load DinoNavDPPipeline (DINO + NavDP + depth) + "
                         "drive the long cross-room leg via NavDP's obstacle-aware GOTO state "
                         "instead of the plain bearing-servo. Off by default: this is real added "
                         "Wi-Fi/CPU/GPU load Go Home never had before, and manual drive never uses "
                         "it anyway")
    ap.add_argument("--compressed-only", action="store_true",
                    help="subscribe to image_raw/compressed only, not raw image_raw -- avoids "
                         "saturating the rover's Wi-Fi link and starving cmd_vel/rpm (see "
                         "zenoh_setup's comment); the launch script always passes this")
    ap.add_argument("--fov", type=float, default=60.0,
                    help="camera horizontal FOV (deg) -- 60 matches the Logitech camera on the "
                         "real rover, see launch_rover.sh's comment")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--depth-encoder", type=str, default="vits", choices=["vits", "vitb"],
                    help="Depth Anything V2 encoder; vitb is more accurate but ~2x slower")
    ap.add_argument("--hard-stop-dist", type=float, default=0.60, help="meters, forward obstacle -> AVOID")
    ap.add_argument("--reverse-dist", type=float, default=0.35,
                    help="meters -- closer than this, back up before turning (too close to rotate safely)")
    ap.add_argument("--slow-dist", type=float, default=2.5,
                     help="meters -- start slowing + curving below this (also where NavDP's "
                          "clearance-aware trajectory selection takes over from pure goal-bearing "
                          "servo) -- the main knob for how far out avoidance visibly starts")
    ap.add_argument("--corridor-half-width", type=float, default=0.35, help="meters, forward-corridor half-width")
    ap.add_argument("--max-range", type=float, default=4.0, help="meters -- ignore obstacle points beyond this")
    ap.add_argument("--avoid-confirm-ticks", type=int, default=2,
                    help="consecutive obstacle-guard hits required before AVOID engages")
    ap.add_argument("--avoid-cooldown-ticks", type=int, default=8,
                    help="ticks to keep biasing steering toward the escape side after AVOID releases")
    ap.add_argument("--avoid-bias-gain", type=float, default=0.15, help="rad/s added during the cooldown window")
    ap.add_argument("--home-arrival-radius", type=float, default=1.0,
                    help="meters -- navdp_home_loop hands off to the precise vision-free servo "
                         "(home_control_loop's FACE/ARRIVED phase) once within this of home")
    ap.add_argument("--home-navdp-timeout-s", type=float, default=240.0,
                    help="seconds -- wall-clock cap on the long NavDP-driven leg only (separate from "
                         "the short servo leg's own --home-dist-tol-scoped timeout)")
    ap.add_argument("--home-navdp-predict-hz", type=float, default=2.5,
                    help="tick rate of navdp_home_loop -- matches the 2.5-3 Hz convention elsewhere "
                         "in this project (isaac_gui.py/remind_gui.py/zenoh_node.py)")
    ap.add_argument("--home-navdp-sample-num", type=int, default=32, help="NavDP candidate trajectories per tick")
    ap.add_argument("--home-navdp-policy-type", type=str, default="crossmodal",
                    choices=["crossmodal", "extracted"], help="see pipeline.py's PipelineConfig docstring")
    ap.add_argument("--home-navdp-angular-slew-max", type=float, default=0.10,
                    help="rad/s max angular delta per navdp_home_loop tick -- deliberately separate "
                         "from --home-angular-slew-max: that one is a rad/s^2 cap applied at 10 Hz, "
                         "this is a raw per-tick delta at ~2.5 Hz, not unit-compatible")
    ap.add_argument("--home-navdp-invert-angular", action="store_true",
                    help="flip NavDP leg turn direction (real-rover wiring escape hatch, matches "
                         "--invert-angular elsewhere)")
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length,
                    help="robot length (m) for obstacle_guard's swept-footprint clearance -- "
                         "defaults to the ESP32 rover's real size, override for a different robot "
                         "(e.g. the LanderPi, see landerpi/README.md) before trusting obstacle avoidance")
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width,
                    help="robot width (m), see --footprint-length")
    args = ap.parse_args()

    guard_cfg = GuardConfig(
        hard_stop_dist=args.hard_stop_dist, reverse_dist=args.reverse_dist, slow_dist=args.slow_dist,
        corridor_half_width=args.corridor_half_width, max_range=args.max_range,
        footprint_length=args.footprint_length, footprint_width=args.footprint_width,
    )

    navdp_pipe = None
    if args.enable_obstacle_avoidance:
        from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

        print("[INFO] Loading NavDP go-home models (DINO + NavDP + depth)...")
        t0 = time.time()
        navdp_cfg = PipelineConfig(
            device=args.device,
            horizontal_fov_deg=args.fov,
            sample_num=args.home_navdp_sample_num,
            policy_type=args.home_navdp_policy_type,
            avoid_enabled=True,
            guard=guard_cfg,
            invert_angular=args.home_navdp_invert_angular,
            max_linear=args.home_max_linear, max_angular=args.home_max_angular,
            ang_min_cmd=args.home_ang_min_cmd,
            servo_deadband=args.home_deadband, servo_ramp_deg=args.home_ramp_deg,
            angular_slew_max=args.home_navdp_angular_slew_max,
            avoid_confirm_ticks=args.avoid_confirm_ticks,
            avoid_cooldown_ticks=args.avoid_cooldown_ticks,
            avoid_bias_gain=args.avoid_bias_gain,
            depth_encoder=args.depth_encoder,
            # No live detection target ever exists for Go Home (external_goal
            # is always used instead) -- skip loading SAM/CLIP/DINOv2-ReID/
            # SceneTagger entirely, none of them are ever invoked in GOTO.
            use_sam=False, use_clip=False, use_appearance_reid=False, use_scene_tagger=False,
        )
        navdp_pipe = DinoNavDPPipeline(navdp_cfg, use_depth_estimator=True)
        print(f"[INFO] NavDP go-home models loaded in {time.time() - t0:.1f}s")

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)
    odom.start_new_goal("home_session")

    st = SharedState()
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    _subs, pubs = zenoh_setup(session, st, odom, compressed_only=args.compressed_only,
                              enable_camera=args.enable_obstacle_avoidance)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=home_control_loop, args=(st, running, args), daemon=True).start()
    if args.enable_obstacle_avoidance:
        Thread(target=navdp_home_loop, args=(navdp_pipe, st, running, odom),
               kwargs={"predict_hz": args.home_navdp_predict_hz,
                       "arrival_radius": args.home_arrival_radius,
                       "navdp_timeout_s": args.home_navdp_timeout_s},
               daemon=True).start()

    root = tk.Tk()
    App(root, st, obstacle_avoidance_enabled=args.enable_obstacle_avoidance,
        imu_min_mag_calib=args.imu_min_mag_calib)

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
        try:
            session.close()
        except zenoh.ZError as e:
            print(f"[WARN] zenoh session close timed out/failed: {e}")
        odom.close()
        print("[INFO] zero velocity sent, session closed")


if __name__ == "__main__":
    main()
