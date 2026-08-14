#!/usr/bin/env python3
"""Point-goal obstacle-avoidance benchmark -- Nav_new's DINO+NavDP test harness.

Same Zenoh contract, same DINO+NavDP pipeline, same S2Diff-HTTP-capable
policy backend as nav_pipeline.isaac_gui (LAUNCH/launch_rover_s2diff_http.sh)
-- the ONLY difference is what drives the goal. Instead of a DINO-detected
target object, the operator enters a straight-line distance: the goal point
is fixed in the WORLD frame the instant "Set Goal" is pressed (that many
meters directly ahead of the rover's heading at that moment) and
re-projected into the rover's current local frame every tick via odometry
(pipeline.py's `external_goal`/"GOTO" path -- the exact mechanism
nav_pipeline/remind_gui.py already uses to drive back to a remembered
object location, see object_map.py's world_to_local/local_to_world).

Because the goal is re-derived from a fixed world anchor every tick rather
than tracked visually, it is remembered exactly: it never moves, isn't lost
to a DINO false negative, and once an obstacle-avoid excursion clears,
NavDP's open-space visual servo (deterministic bearing-to-goal, see
pipeline.py's _step_inner) naturally re-lines the rover back up on the
original straight-line bearing to it -- no extra "remember the path" logic
needed beyond keeping the anchor fixed. Depth-based obstacle_guard (AVOID
state, swept-trajectory clearance veto) is fully active the whole time,
identically to every other real-rover launcher; pipeline.py's GOTO state
never self-declares arrival from proximity alone (dead-reckoning drift over
distance makes that unsafe to trust -- see step()'s docstring), so this
file's own inference_loop checks distance-to-goal itself every tick and
declares GOAL REACHED within --arrival-radius.

A second field, "Obstacle to avoid", feeds avoid_text -- DINO-detected
every tick same as a normal target, but only marks a named-obstacle box
(res.avoid_detection); only the S2Diff HTTP policy backend
(nav_pipeline/s2diff_http_client.py) actually acts on it as pixel-obstacle
guidance (see pipeline.py's step() docstring) -- pair this file with
nav_pipeline.s2diff_http_runner's policy patch (see LAUNCH/
launch_point_goal.sh) to get that. General depth-based physical obstacle
avoidance (AVOID/swept_clearance) works regardless of this field or what
the obstacle actually is; naming it lets S2Diff bias trajectory sampling
away from that recognized object earlier/more assertively.

Run (from Nav_new root, same server-first requirement as
launch_rover_s2diff_http.sh):
    conda activate internnav
    python -m nav_pipeline.point_goal_gui --pi-ip <IP> --server-url http://127.0.0.1:8888
"""

import argparse
import os
import signal
import sys
import time
import traceback
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.isaac_gui import (  # noqa: E402
    DEPTH_STALE_S,
    INTRINSICS_STALE_S,
    SPIN_DIST_THRESH_M,
    SPIN_ROT_THRESH_RAD,
    SPIN_WINDOW_S,
    heartbeat_loop,
    zenoh_setup,
)
from nav_pipeline.object_map import local_to_world, world_to_local  # noqa: E402
from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.zenoh_node import serialize_path, serialize_string, serialize_twist  # noqa: E402


class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.latest_intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy)
        self.latest_intrinsics_t = 0.0
        self.frame_count = 0
        self.mode = "manual"                     # "text" | "manual" -- starts inert, same as isaac_gui.py
        self.world_goal: Optional[tuple] = None   # (x, y) world-frame, set once by "Set Goal"; None = no goal yet
        self.goal_distance = 3.0                  # last-entered distance (m), for the GUI field's default
        self.avoid = ""                           # named obstacle text, live-editable, "" = disabled
        self.stopped = False
        self.goal_reached = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.5                     # manual-drive caps; set from CLI args in main()
        self.max_angular = 0.4
        # for display
        self.display_rgb: Optional[np.ndarray] = None
        self.avoid_detection = None
        self.state_text = "waiting for camera"
        self.vel_text = "lin 0.000  ang +0.000"
        self.lat_text = ""
        self.trajs = None
        self.chosen = None
        self.goal_pt = None
        self.obstacles = None
        self.min_forward = float("inf")
        self.infer_count = 0
        # IMU status -- stashed by isaac_gui.py's zenoh_setup()/on_rpm (this
        # GUI reuses it, see the import above), surfaced in the info bar
        # same as isaac_gui.py/home_gui.py/remind_gui.py.
        self.imu_heading_deg = float("nan")
        self.imu_calib = 0.0
        self.theta_source = "enc"


def inference_loop(pipe: DinoNavDPPipeline, st: SharedState, pubs, running,
                   predict_hz: float, arrival_radius: float, odom: OdometryLogger):
    period = 1.0 / predict_hz
    last_world_goal = None
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            intrinsics = st.latest_intrinsics
            intrinsics_age = time.time() - st.latest_intrinsics_t
            mode = st.mode
            world_goal = st.world_goal
            avoid = st.avoid
            paused = st.stopped or st.goal_reached

        if world_goal != last_world_goal:
            # A fresh "Set Goal" press: reset any belief/tracked-box state
            # left over from a previous run (same reason isaac_gui.py resets
            # on a new target -- stale pose/goal state from the PREVIOUS run
            # must not leak its first ego-motion delta into this one).
            if last_world_goal is not None:
                pipe.reset()
            if world_goal is not None:
                odom.start_new_goal(f"point_goal_x{world_goal[0]:.2f}_y{world_goal[1]:.2f}")
            last_world_goal = world_goal

        if rgb is None:
            time.sleep(0.1)
            continue

        if mode == "manual":
            # bypass GOTO/NavDP entirely -- last_cmd is set directly by the
            # GUI's manual-drive buttons/keys; use this to reposition the
            # rover before pressing "Set Goal". Still honors STOP.
            with st.lock:
                if st.stopped:
                    st.last_cmd = (0.0, 0.0)
                lin, ang = st.last_cmd
                st.display_rgb = rgb
                st.state_text = "MANUAL (stopped)" if st.stopped else "MANUAL DRIVE"
                st.vel_text = f"lin {lin:.3f}  ang {ang:+.3f}"
            time.sleep(0.05)
            continue

        if paused or world_goal is None:
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.vel_text = "lin 0.000  ang +0.000"
                if world_goal is None:
                    st.state_text = "waiting for goal -- enter distance and press Set Goal"
            time.sleep(0.1)
            continue

        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None
        if intrinsics is not None and intrinsics_age > INTRINSICS_STALE_S:
            intrinsics = None

        pose = (odom.x, odom.y, odom.theta)
        # odom always has SOME pose (starts at the origin) -- this never
        # actually goes missing, but stay defensive rather than assume.
        lx, ly = world_to_local(world_goal, pose)
        dist = float(np.hypot(lx, ly))

        if dist <= arrival_radius:
            with st.lock:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = f"GOAL REACHED ({dist:.2f}m, radius {arrival_radius:.2f}m)"
                st.vel_text = "lin 0.000  ang +0.000"
            pubs["explain"].put(serialize_string(f"POINT GOAL REACHED at {dist:.2f}m. Stopping."))
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            continue

        external_goal = np.array([lx, ly, 0.0], dtype=np.float32)

        try:
            # external_dets=[] (not None) skips DINO target-detection/SAM/
            # CLIP/belief entirely -- there IS no target object, only an
            # imaginary point -- and drops straight to pipeline.py's
            # external_goal/"GOTO" fallback, which reuses the exact same
            # obstacle-guard/NavDP trajectory machinery as a live TRACK
            # (see pipeline.py step()'s docstring).
            res = pipe.step(rgb, "", depth=depth, pose=pose, external_dets=[],
                            external_goal=external_goal, avoid_text=avoid, intrinsics=intrinsics)
        except Exception as e:
            print(f"[ERROR] pipeline step: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = f"ERROR: {e}"
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.5)
            continue

        spin = odom.spin_delta(SPIN_WINDOW_S)
        if spin[0] > SPIN_ROT_THRESH_RAD and spin[1] < SPIN_DIST_THRESH_M:
            print(f"[WARN] spin-stall watchdog: turned {spin[0]:.1f}rad in {SPIN_WINDOW_S:.0f}s, "
                  f"only {spin[1]:.2f}m net travel -- forcing stop until a new goal is sent")
            with st.lock:
                st.goal_reached = True  # reuses the existing pause-until-new-goal gate
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = (f"SPIN STALL: {np.degrees(spin[0]):.0f}° turned, "
                                 f"{spin[1]:.2f}m travel -- send a new goal")
                st.vel_text = "lin 0.000  ang +0.000"
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            continue

        with st.lock:
            st.display_rgb = rgb
            st.avoid_detection = res.avoid_detection
            st.trajs = res.all_trajectories
            st.chosen = res.trajectory
            st.goal_pt = res.goal_point
            st.obstacles = res.obstacle_points
            st.min_forward = res.min_forward
            st.infer_count += 1
            st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
            st.state_text = f"{res.state} -- {dist:.2f}m straight-line to goal"
            st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v * 1000:.0f}ms" for k, v in res.timing.items())

        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        avoid_score = f"{res.avoid_detection.score:.2f}" if res.avoid_detection else "-"
        pubs["explain"].put(serialize_string(
            f"POINT-GOAL [{res.state}] dist={dist:.2f}m avoid={avoid_score} "
            f"-> lin={res.linear:.3f} ang={res.angular:.3f}"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448          # camera panel (px)
    PLOT_SIZE = 448         # top-down plot (px)
    PLOT_RANGE = 3.5        # meters shown ahead

    def __init__(self, root: tk.Tk, st: SharedState, odom: OdometryLogger):
        self.root = root
        self.st = st
        self.odom = odom
        root.title("Nav_new — Point-Goal Obstacle-Avoidance Benchmark")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)
        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        goal_bar = ttk.Frame(main)
        goal_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(goal_bar, text="Goal distance (m):").pack(side="left")
        self.dist_entry = ttk.Entry(goal_bar, width=8)
        self.dist_entry.insert(0, f"{st.goal_distance:.1f}")
        self.dist_entry.pack(side="left", padx=4)
        self.dist_entry.bind("<Return>", lambda e: self.send_goal())
        ttk.Button(goal_bar, text="Set Goal (straight ahead)", command=self.send_goal).pack(side="left", padx=2)
        ttk.Button(goal_bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        avoid_bar = ttk.Frame(main)
        avoid_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(avoid_bar, text="Obstacle to avoid:").pack(side="left")
        self.avoid_entry = ttk.Entry(avoid_bar, width=28)
        self.avoid_entry.bind("<Return>", lambda e: self.send_avoid())
        self.avoid_entry.pack(side="left", padx=4)
        ttk.Button(avoid_bar, text="Set", command=self.send_avoid).pack(side="left", padx=2)
        ttk.Button(avoid_bar, text="Clear", command=self.clear_avoid).pack(side="left", padx=2)

        # Manual drive: reposition the rover (e.g. face it toward open space,
        # or toward the obstacle you want in its path) before pressing
        # "Set Goal" -- the goal is anchored straight ahead of whatever
        # heading the rover has AT THAT MOMENT.
        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys) -- aim heading before Set Goal:").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=100, anchor="w")
        self.status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=100, anchor="w")
        self.info.grid(row=5, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(66, self.refresh)

    def send_goal(self):
        text = self.dist_entry.get().strip()
        try:
            d = float(text)
        except ValueError:
            self.status.configure(text=f"[ERROR] '{text}' isn't a number -- enter a distance in meters")
            return
        if d <= 0:
            self.status.configure(text="[ERROR] distance must be > 0")
            return
        self._manual_held.clear()
        pose = (self.odom.x, self.odom.y, self.odom.theta)
        wx, wy = local_to_world((d, 0.0), pose)
        with self.st.lock:
            self.st.mode = "text"
            self.st.world_goal = (wx, wy)
            self.st.goal_distance = d
            self.st.stopped = False
            self.st.goal_reached = False

    def send_avoid(self):
        t = self.avoid_entry.get().strip()
        with self.st.lock:
            self.st.avoid = t

    def clear_avoid(self):
        self.avoid_entry.delete(0, "end")
        with self.st.lock:
            self.st.avoid = ""

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

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
            self.st.goal_reached = False
            self.st.last_cmd = (lin, ang)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    # ------------------------------------------------------------------ #
    def refresh(self):
        if self.closed:
            return
        try:
            self._refresh_body()
        except Exception:
            # See isaac_gui.py's refresh() for why this guard exists: without
            # it, a single uncaught exception here permanently kills this
            # self-rescheduling redraw loop -- the GUI freezes on whatever
            # was last drawn while the inference/movement loop keeps running
            # fine in its own thread.
            traceback.print_exc()
        finally:
            if not self.closed:
                self.root.after(66, self.refresh)

    def _refresh_body(self):
        with self.st.lock:
            rgb = self.st.latest_rgb if self.st.latest_rgb is not None else self.st.display_rgb
            avoid_det = self.st.avoid_detection
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers = self.st.frame_count, self.st.infer_count
            drive_mode = self.st.mode
            stopped = self.st.stopped
            world_goal = self.st.world_goal
            goal_distance = self.st.goal_distance
            imu_heading, imu_calib, theta_source = self.st.imu_heading_deg, self.st.imu_calib, self.st.theta_source

        if rgb is not None:
            img = Image.fromarray(rgb).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            if avoid_det is not None:
                d = ImageDraw.Draw(img)
                x0, y0, x1, y1 = avoid_det.box
                d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 40, 40), width=3)
                d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), f"avoid: {avoid_det.label} {avoid_det.score:.2f}",
                        fill=(255, 40, 40))
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
        goal_txt = "manual drive" if drive_mode == "manual" else (
            f"point goal ({goal_distance:.1f}m straight-line)" if world_goal is not None else "no goal set")
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  {goal_txt}   {vel_text}{fwd}")
        heading_txt = f"{imu_heading:.1f}°" if np.isfinite(imu_heading) else "n/a"
        imu_txt = (f"theta src: {theta_source}   imu heading {heading_txt}"
                   f"  calib [{OdometryLogger.decode_calib(imu_calib)}]")
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}   {imu_txt}")


def main():
    ap = argparse.ArgumentParser(description="Nav_new point-goal obstacle-avoidance benchmark")
    ap.add_argument("--avoid", default="",
                    help="named obstacle to steer away from, e.g. 'trash bin' -- DINO-detected "
                         "every tick, only fed into S2Diff pixel-obstacle avoidance "
                         "(nav_pipeline/s2diff_http_client.py); no effect on the plain or "
                         "in-process-S2Diff policy backends. '' (default) disables it -- the "
                         "GUI's 'Obstacle to avoid' field overrides this live")
    ap.add_argument("--goal-distance", type=float, default=3.0,
                    help="meters straight ahead of the rover's heading at the moment 'Set "
                         "Goal' is pressed -- pre-fills the GUI field, doesn't auto-start")
    ap.add_argument("--arrival-radius", type=float, default=0.4,
                    help="meters from the point goal at which the rover declares GOAL REACHED "
                         "and stops -- pipeline.py's GOTO state never self-declares arrival "
                         "from proximity alone (drift-safety, see step()'s docstring), so this "
                         "file checks distance itself every tick")
    ap.add_argument("--policy-type", choices=["crossmodal", "extracted"], default="crossmodal",
                    help="NavDP policy backend (see PipelineConfig.policy_type). "
                         "nav_pipeline.s2diff_http_runner REQUIRES \"extracted\" here -- see "
                         "isaac_gui.py's --policy-type help for the full explanation")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 0.25 on the real rover)")
    ap.add_argument("--servo-ramp-deg", type=float, default=35.0,
                    help="see isaac_gui.py's --servo-ramp-deg help")
    ap.add_argument("--angular-slew-max", type=float, default=0.10,
                    help="rad/s hard cap on angular command change per tick, every state. 0 disables.")
    ap.add_argument("--invert-angular", action="store_true",
                    help="flip turn direction (use if the rover steers the wrong way)")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vits",
                    help="RGB-only metric depth model size, used whenever real sensor depth "
                         "isn't fresh -- see isaac_gui.py's --depth-encoder help")
    ap.add_argument("--compressed-only", action="store_true",
                    help="subscribe only the JPEG camera stream (REQUIRED over rover Wi-Fi)")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log",
                    help="dead-reckoned pose CSV log dir (from /rover/rpm) -- also anchors "
                         "the point goal's world frame")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3,
                    help="IMU calibration digit (0-3) required before theta rides the IMU "
                         "heading instead of wheel-diff dead reckoning -- see OdometryLogger. "
                         "This tool's whole point is isolating goal-reaching accuracy, which "
                         "the world-frame heading feeds directly, so it's worth tuning here too.")
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length,
                    help="robot length (m) for obstacle_guard's swept-footprint clearance")
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width,
                    help="robot width (m), see --footprint-length")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        policy_type=args.policy_type,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        servo_ramp_deg=args.servo_ramp_deg,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        # No target object ever exists on this path (external_dets=[] every
        # tick) -- SAM/CLIP/scene-tagger/belief all exist only to resolve
        # DINO target ambiguity or coast a lost visual target, none of which
        # applies here. Off by default keeps this benchmark's own timing
        # clean (just DINO-for-avoid + depth + NavDP + obstacle guard).
        use_sam=False,
        use_clip=False,
        use_scene_tagger=False,
        use_belief_goal=False,
        depth_encoder=args.depth_encoder,
        guard=GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width),
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState()
    st.avoid = args.avoid
    st.goal_distance = args.goal_distance
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz, args.arrival_radius, odom),
           daemon=True).start()

    root = tk.Tk()
    App(root, st, odom)

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
