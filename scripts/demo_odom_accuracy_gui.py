#!/usr/bin/env python3
"""Scripted, fully-SYNTHETIC walkthrough of odom_accuracy_gui.py, for
learning the UI before running it on the real rover. NOT connected to zenoh
or any real rover -- fabricates left/right wheel RPM and feeds it into the
exact same OdometryLogger.update() path a real /rover/rpm sample would hit,
then drives the GUI through two example trials (a 90deg spin, then a
drive+turn) by calling the same App methods the buttons call. What you see
is the real UI; only the "sensor data" is made up.

Everything it writes goes to odometry_log_DEMO_SIMULATED_DATA/ (never the
real odometry_log/), and the window title/status line say DEMO throughout.

Run: python scripts/demo_odom_accuracy_gui.py
"""
import math
import os
import sys
import time
from threading import Thread

import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nav_pipeline.odometry_logger import OdometryLogger, TRACK_WIDTH_M, WHEEL_RADIUS_M  # noqa: E402
import odom_accuracy_gui as gui  # noqa: E402

DEMO_LOG_DIR = "odometry_log_DEMO_SIMULATED_DATA"


class NullPub:
    """Stands in for the real zenoh cmd_vel publisher -- demo mode drives
    the rover's simulated pose directly, so outgoing cmd_vel goes nowhere."""

    def put(self, *_a, **_k):
        pass


def _rpm_for(v: float, w: float):
    v_left = v - w * TRACK_WIDTH_M / 2.0
    v_right = v + w * TRACK_WIDTH_M / 2.0
    to_rpm = lambda vw: vw * 60.0 / (2.0 * math.pi * WHEEL_RADIUS_M)
    return to_rpm(v_left), to_rpm(v_right)


def demo_feed(odom: OdometryLogger, running: dict):
    """Fabricated wheel RPM at 10Hz (matches the real /rover/rpm publish
    rate). Phase 1 mirrors belief_eval_20260730's 'spin' scenario (~90deg in
    place); phase 2 mirrors its 'turn' scenario (drive+turn, v=0.15 m/s,
    w=0.25 rad/s, matching real-rover PipelineConfig defaults)."""
    phase1 = [(0.0, 0.0)] * 10 + [(0.0, 0.30)] * 52 + [(0.0, 0.0)] * 15
    for v, w in phase1:
        if not running["on"]:
            return
        odom.update(*_rpm_for(v, w))
        time.sleep(0.1)

    phase2 = [(0.0, 0.0)] * 10 + [(0.15, 0.25)] * 30 + [(0.0, 0.0)] * 15
    for v, w in phase2:
        if not running["on"]:
            return
        odom.update(*_rpm_for(v, w))
        time.sleep(0.1)


def main():
    odom = OdometryLogger(DEMO_LOG_DIR)
    st = gui.SharedState()
    running = {"on": True}

    root = tk.Tk()
    app = gui.App(root, st, odom, NullPub(), max_linear=0.15, max_angular=1.2, log_dir=DEMO_LOG_DIR)
    root.title("Odometry accuracy test  —  DEMO MODE: SIMULATED DATA, NOT A REAL ROVER")

    Thread(target=demo_feed, args=(odom, running), daemon=True).start()

    def set_label_and_start(label):
        app.label_entry.delete(0, "end")
        app.label_entry.insert(0, label)
        app.new_test()

    def fill_and_record(x, y, theta_deg):
        for entry, val in ((app.gt_x, x), (app.gt_y, y), (app.gt_theta, theta_deg)):
            entry.delete(0, "end")
            entry.insert(0, str(val))
        app.record_measurement()

    # timeline (ms), matched to demo_feed's own timing above
    root.after(500, lambda: set_label_and_start("DEMO_spin_90deg"))
    # phase 1 ends at 1.0+5.2+1.5=7.7s; "measured" values are close to the
    # phase's true analytic motion (v=0 throughout -> x=y=0 exactly; w*t =
    # 0.3*5.2=1.56rad=89.4deg) with a small illustrative discrepancy
    root.after(7700, lambda: fill_and_record(0.0, 0.0, 90.0))
    root.after(8200, lambda: set_label_and_start("DEMO_drive_turn_3s"))
    # phase 2 ends 5.5s after it starts (~13.2s absolute); true analytic
    # motion at v=0.15,w=0.25,t=3s: theta=43.0deg, x=0.41m, y=0.16m -- again
    # a small illustrative discrepancy from what gets typed in as "measured"
    root.after(13600, lambda: fill_and_record(0.40, 0.15, 44.0))

    def auto_close():
        print("[DEMO] finished -- closing shortly")
        root.after(2500, root.destroy)
    root.after(16500, auto_close)

    def on_close():
        running["on"] = False
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.mainloop()
    finally:
        running["on"] = False
        odom.close()


if __name__ == "__main__":
    main()
