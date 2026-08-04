#!/usr/bin/env python3
"""Nav_new — manual control + "Go Home" GUI (encoder+IMU fused odometry).

Deliberately independent of pipeline.py / isaac_gui.py / zenoh_node.py (all
three import DinoNavDPPipeline, which pulls in torch/cv2/GroundingDINO at
import time -- multi-second GPU model load just to teleop and drive back to
a point). This GUI never touches DINO/NavDP/SAM/CLIP/depth at all, so it
starts in under a second and needs no GPU. The small CDR (DDS wire format)
codec bits it needs from zenoh_node.py's Zenoh contract are duplicated below
rather than imported, for exactly that reason -- keep them in sync with
zenoh_node.py if the wire contract (RPM_KEYS / Float32MultiArray / Twist)
ever changes.

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

Run (from Nav_new root, internnav conda env not required -- see launch script):
    python -m nav_pipeline.home_gui [--pi-ip <IP>]
"""

import argparse
import math
import os
import signal
import struct
import sys
import time
from collections import deque
from threading import Lock, Thread
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402

# ================================================================
#  CDR (DDS wire format) codec -- duplicated from zenoh_node.py, see the
#  module docstring for why. Only the pieces this file actually needs.
# ================================================================
RPM_KEYS = ["rover/rpm", "rt/rover/rpm"]


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

    def read_string(self):
        length = self.read_uint32()
        self.offset += length


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


def zenoh_setup(session: zenoh.Session, st: SharedState, odom: OdometryLogger):
    def on_rpm(sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) < 2:
                return
            imu_heading = data[2] if len(data) >= 3 else None
            imu_calib = data[3] if len(data) >= 4 else None
            odom.update(data[0], data[1], imu_heading_deg=imu_heading, imu_calib=imu_calib)
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

    subs = [session.declare_subscriber(k, on_rpm) for k in RPM_KEYS]
    pubs = {"cmd": session.declare_publisher("cmd_vel")}
    return subs, pubs


def heartbeat_loop(st: SharedState, pubs, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = st.last_cmd
        pubs["cmd"].put(serialize_twist(lin, ang))


def home_control_loop(st: SharedState, running, args):
    """Go-to-goal controller driving toward st.home. Only actually commands
    anything while st.mode == "homing"; otherwise idles and keeps its phase
    state reset so the next "Go Home" press always starts from ROTATE."""
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

        if mode != "homing":
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

    def __init__(self, root: tk.Tk, st: SharedState):
        self.root = root
        self.st = st
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
        with self.st.lock:
            x, y, theta = self.st.x, self.st.y, self.st.theta
            home = self.st.home
            path = list(self.st.path)
            mode, stopped = self.st.mode, self.st.stopped
            phase, dist, herr = self.st.home_phase, self.st.home_dist, self.st.home_heading_err
            lin, ang = self.st.last_cmd
            imu_heading, imu_calib, theta_src = self.st.imu_heading_deg, self.st.imu_calib, self.st.theta_source
            rpm_count, last_rpm_t = self.st.rpm_count, self.st.last_rpm_t

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
        self.status.configure(
            text=f"[{mode_txt}]  home-phase: {phase}   dist {dist:.2f}m   heading-err {math.degrees(herr):+.1f}°"
                 f"   lin {lin:.3f}  ang {ang:+.3f}"
        )
        heading_txt = f"{imu_heading:.1f}°" if math.isfinite(imu_heading) else "n/a"
        self.info.configure(
            text=f"pose x={x:.2f} y={y:.2f} theta={math.degrees(theta):+.1f}°  "
                 f"(theta src: {theta_src})   imu heading {heading_txt}  calib [{decode_calib(imu_calib)}]"
        )
        stale = last_rpm_t == 0.0 or (time.time() - last_rpm_t) > RPM_STALE_S
        if stale:
            self.warn.configure(text="⚠ NO /rover/rpm DATA — check rover-agent on the Pi (see launch script output)")
        elif theta_src != "imu":
            self.warn.configure(
                text="⚠ heading source: wheel encoders only (IMU not calibrated yet — "
                     "tilt/rotate the rover until magnetometer calib >=1)")
        else:
            self.warn.configure(text=f"rpm samples: {rpm_count}")

        self.root.after(66, self.refresh)

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
                         "precision return-to-point maneuver, not obstacle avoidance")
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
    ap.add_argument("--imu-min-mag-calib", type=int, default=1,
                    help="0-3; magnetometer calib required before trusting IMU heading over wheel-diff")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    args = ap.parse_args()

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
    _subs, pubs = zenoh_setup(session, st, odom)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=home_control_loop, args=(st, running, args), daemon=True).start()

    root = tk.Tk()
    App(root, st)

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
