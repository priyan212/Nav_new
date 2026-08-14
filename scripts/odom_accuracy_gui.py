#!/usr/bin/env python3
"""Standalone GUI for the real-rover odometry-accuracy check (see
belief_eval_20260730/RESULTS.md, "The caveat that matters most for the real
rover"): does dead-reckoned pose from /rover/rpm (nav_pipeline/odometry_logger.py)
stay close to the truth through a turn, or drift enough to undermine porting
SubgoalBeliefBank's ego-motion correction to the real rover?

No DINO/NavDP/depth models are loaded -- this only needs cmd_vel + /rover/rpm,
so it starts instantly. Workflow (matches the procedure already agreed):

  1. Mark the rover's start position/heading on the floor.
  2. Click "New Test" (give it a label) -- resets the dead-reckoned pose to
     (0,0,0) and opens a fresh odometry_log/odom_<label>_<timestamp>.csv,
     exactly like sending a target does in the real pipeline.
  3. Drive with the manual controls (buttons or arrow keys) through the
     maneuver (e.g. a 90 deg in-place spin, or a timed drive+turn).
  4. Stop, mark the rover's true ending position/heading on the floor,
     physically measure true (x, y, theta) relative to the start heading.
  5. Type those measured numbers into the "measured ground truth" fields and
     click "Record Measurement" -- this reads the logger's own dead-reckoned
     (x, y, theta) at that instant, computes position/heading error against
     what you measured, shows it, and appends a row to
     odometry_log/odom_accuracy_results.csv.
  6. Repeat for as many trials as useful (spin-only, drive+turn, different
     durations); every trial is one row, comparable directly against the
     synthetic belief_eval_20260730 numbers.

Run (from Nav_new root, internnav conda env):
    python scripts/odom_accuracy_gui.py [--pi-ip <IP>]
"""

import argparse
import csv
import os
import signal
import sys
import time
from threading import Lock, Thread

import numpy as np

import tkinter as tk
from tkinter import ttk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.zenoh_node import (  # noqa: E402
    RPM_KEYS,
    parse_float32_multiarray,
    serialize_twist,
)

HEARTBEAT_PERIOD_S = 0.15  # rover firmware zeroes cmd_vel if nothing arrives within ~500ms
RESULTS_CSV = "odom_accuracy_results.csv"
RESULTS_HEADER = ["timestamp", "label", "odom_csv", "logged_x", "logged_y", "logged_theta_deg",
                  "measured_x", "measured_y", "measured_theta_deg",
                  "pos_error_m", "heading_error_deg"]


class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.last_cmd = (0.0, 0.0)
        self.stopped = True


def zenoh_setup(session: zenoh.Session, st: SharedState, odom: OdometryLogger):
    def on_rpm(sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) >= 2:
                imu_heading = data[2] if len(data) >= 3 else None
                imu_calib = data[3] if len(data) >= 4 else None
                odom.update(data[0], data[1], imu_heading_deg=imu_heading, imu_calib=imu_calib)
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    subs = [session.declare_subscriber(k, on_rpm) for k in RPM_KEYS]
    pub = session.declare_publisher("cmd_vel")
    return subs, pub


def heartbeat_loop(st: SharedState, pub, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = (0.0, 0.0) if st.stopped else st.last_cmd
        pub.put(serialize_twist(lin, ang))


class App:
    def __init__(self, root, st: SharedState, odom: OdometryLogger, pub,
                 max_linear: float, max_angular: float, log_dir: str):
        self.root = root
        self.st = st
        self.odom = odom
        self.pub = pub
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.results_path = os.path.join(log_dir, RESULTS_CSV)
        self._ensure_results_csv()
        self.closed = False
        self._manual_held: set = set()
        self._trail: list = []  # [(x, y), ...] for the live path canvas
        self._test_label = "untitled"

        root.title("Odometry accuracy test")
        main = ttk.Frame(root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        # --- new test --- #
        row0 = ttk.Frame(main)
        row0.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(row0, text="Test label:").pack(side="left")
        self.label_entry = ttk.Entry(row0, width=24)
        self.label_entry.insert(0, "spin_90deg")
        self.label_entry.pack(side="left", padx=4)
        ttk.Button(row0, text="New Test (reset origin)", command=self.new_test).pack(side="left", padx=4)
        self.csv_label = ttk.Label(row0, text="(no test started)")
        self.csv_label.pack(side="left", padx=8)

        # --- live pose readout + path canvas --- #
        row1 = ttk.Frame(main)
        row1.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        self.pose_label = ttk.Label(row1, text="x=0.000  y=0.000  theta=0.0deg",
                                    font=("TkDefaultFont", 14, "bold"))
        self.pose_label.pack(side="left")

        self.canvas = tk.Canvas(main, width=320, height=320, bg="white", highlightthickness=1,
                                highlightbackground="gray")
        self.canvas.grid(row=2, column=0, rowspan=6, sticky="nw", padx=(0, 10))
        self.px_per_m = 60.0  # canvas scale

        # --- manual drive --- #
        drive = ttk.LabelFrame(main, text="Manual drive (hold, or arrow keys)")
        drive.grid(row=2, column=1, sticky="ew", pady=4)
        for label, direction in (("<", "left"), ("^", "fwd"), ("v", "back"), (">", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2, pady=4)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))
        ttk.Button(drive, text="STOP", command=self.stop).pack(side="left", padx=8)

        # --- measured ground truth entry --- #
        gt = ttk.LabelFrame(main, text="Measured ground truth (relative to start heading)")
        gt.grid(row=3, column=1, sticky="ew", pady=4)
        self.gt_x = self._labeled_entry(gt, "true x (m, forward+):", "0.0")
        self.gt_y = self._labeled_entry(gt, "true y (m, left+):", "0.0")
        self.gt_theta = self._labeled_entry(gt, "true theta (deg):", "0.0")
        ttk.Button(gt, text="Record Measurement", command=self.record_measurement).pack(
            anchor="w", padx=4, pady=6)

        self.error_label = ttk.Label(main, text="", font=("TkDefaultFont", 10, "bold"))
        self.error_label.grid(row=4, column=1, sticky="w")

        # --- results table --- #
        cols = ("label", "logged", "measured", "pos_err_m", "heading_err_deg")
        self.tree = ttk.Treeview(main, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (110, 170, 170, 90, 110)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        self.tree.grid(row=5, column=1, sticky="nsew", pady=4)

        self.status = ttk.Label(main, text=f"results -> {self.results_path}")
        self.status.grid(row=6, column=1, sticky="w")

        self._load_existing_rows()
        self.root.after(100, self.refresh)

    def _labeled_entry(self, parent, text, default):
        row = ttk.Frame(parent)
        row.pack(anchor="w", padx=4, pady=2)
        ttk.Label(row, text=text, width=22).pack(side="left")
        e = ttk.Entry(row, width=10)
        e.insert(0, default)
        e.pack(side="left")
        return e

    def _ensure_results_csv(self):
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w", newline="") as f:
                csv.writer(f).writerow(RESULTS_HEADER)

    def _load_existing_rows(self):
        try:
            with open(self.results_path, newline="") as f:
                for row in csv.DictReader(f):
                    self.tree.insert("", "end", values=(
                        row["label"],
                        f"{float(row['logged_x']):.3f},{float(row['logged_y']):.3f},{float(row['logged_theta_deg']):.1f}",
                        f"{float(row['measured_x']):.3f},{float(row['measured_y']):.3f},{float(row['measured_theta_deg']):.1f}",
                        f"{float(row['pos_error_m']):.3f}",
                        f"{float(row['heading_error_deg']):.1f}",
                    ))
        except FileNotFoundError:
            pass

    # ---------------- actions ---------------- #
    def new_test(self):
        label = self.label_entry.get().strip() or "untitled"
        self._test_label = label
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)
        self.odom.start_new_goal(label)
        self._trail = [(0.0, 0.0)]
        self.canvas.delete("all")
        self.csv_label.config(text=f"-> {os.path.basename(self.odom.path)}")

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

    def manual_press(self, direction: str):
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str):
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self):
        lin = ang = 0.0
        if "fwd" in self._manual_held:
            lin += self.max_linear
        if "back" in self._manual_held:
            lin -= 0.5 * self.max_linear
        if "left" in self._manual_held:
            ang += self.max_angular
        if "right" in self._manual_held:
            ang -= self.max_angular
        with self.st.lock:
            self.st.stopped = False
            self.st.last_cmd = (lin, ang)

    def record_measurement(self):
        if self.odom.path is None:
            self.error_label.config(text="click 'New Test' first")
            return
        try:
            mx = float(self.gt_x.get())
            my = float(self.gt_y.get())
            mtheta_deg = float(self.gt_theta.get())
        except ValueError:
            self.error_label.config(text="measured x/y/theta must be numbers")
            return

        lx, ly, ltheta = self.odom.x, self.odom.y, self.odom.theta
        ltheta_deg = float(np.degrees(ltheta))
        pos_err = float(np.hypot(lx - mx, ly - my))
        heading_err = float(((ltheta_deg - mtheta_deg) + 180) % 360 - 180)  # wrapped to [-180, 180]

        row = dict(timestamp=f"{time.time():.3f}", label=self._test_label,
                  odom_csv=os.path.basename(self.odom.path),
                  logged_x=f"{lx:.4f}", logged_y=f"{ly:.4f}", logged_theta_deg=f"{ltheta_deg:.2f}",
                  measured_x=f"{mx:.4f}", measured_y=f"{my:.4f}", measured_theta_deg=f"{mtheta_deg:.2f}",
                  pos_error_m=f"{pos_err:.4f}", heading_error_deg=f"{heading_err:.2f}")
        with open(self.results_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_HEADER).writerow(row)

        self.tree.insert("", "end", values=(
            self._test_label, f"{lx:.3f},{ly:.3f},{ltheta_deg:.1f}",
            f"{mx:.3f},{my:.3f},{mtheta_deg:.1f}", f"{pos_err:.3f}", f"{heading_err:.1f}"))
        self.error_label.config(
            text=f"pos error = {pos_err:.3f} m   heading error = {heading_err:+.1f} deg")

    def _to_px(self, x, y):
        cx, cy = 160, 160
        return cx + y * self.px_per_m, cy - x * self.px_per_m  # x fwd = up, y left = left

    def refresh(self):
        if self.closed:
            return
        x, y, theta = self.odom.x, self.odom.y, self.odom.theta
        self.pose_label.config(text=f"x={x:.3f}  y={y:.3f}  theta={np.degrees(theta):.1f}deg")

        self._trail.append((x, y))
        self.canvas.delete("all")
        ox, oy = self._to_px(0, 0)
        self.canvas.create_oval(ox - 4, oy - 4, ox + 4, oy + 4, fill="green", outline="")
        pts = [self._to_px(px, py) for px, py in self._trail]
        if len(pts) > 1:
            self.canvas.create_line(*[c for p in pts for c in p], fill="blue", width=2)
        cxp, cyp = pts[-1]
        hx, hy = cxp + 12 * np.sin(theta), cyp - 12 * np.cos(theta)
        self.canvas.create_line(cxp, cyp, hx, hy, fill="red", width=2, arrow="last")
        self.canvas.create_oval(cxp - 4, cyp - 4, cxp + 4, cyp + 4, fill="red", outline="")

        self.root.after(100, self.refresh)

    def on_close(self):
        self.closed = True
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Real-rover odometry-accuracy test GUI")
    ap.add_argument("--pi-ip", default=None)
    # matches launch_rover.sh's real-rover manual-drive caps: max-angular 1.2 (not
    # PipelineConfig's 0.25) because the ESP32 firmware normalizes angular commands
    # by this value -- 0.25 there gave only ~1/5 of real steering authority.
    ap.add_argument("--max-linear", type=float, default=0.15)
    ap.add_argument("--max-angular", type=float, default=1.2)
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3,
                    help="IMU calibration digit (0-3) required before theta rides the IMU "
                         "heading instead of wheel-diff dead reckoning -- see OdometryLogger. "
                         "This tool exists specifically to measure odometry accuracy, so "
                         "lowering it here is one way to see how much the IMU gate itself "
                         "is costing/buying accuracy vs. wheel-diff-only.")
    args = ap.parse_args()

    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting (auto-discover)")
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState()
    _subs, pub = zenoh_setup(session, st, odom)
    running = {"on": True}
    Thread(target=heartbeat_loop, args=(st, pub, running), daemon=True).start()

    root = tk.Tk()
    app = App(root, st, odom, pub, args.max_linear, args.max_angular, args.odometry_log_dir)

    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, root.destroy))

    try:
        root.mainloop()
    finally:
        running["on"] = False
        time.sleep(0.2)
        pub.put(serialize_twist(0.0, 0.0))
        time.sleep(0.1)
        pub.put(serialize_twist(0.0, 0.0))
        odom.close()
        session.close()
        print("[INFO] zero velocity sent, session closed")


if __name__ == "__main__":
    main()
