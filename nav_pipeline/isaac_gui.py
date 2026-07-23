#!/usr/bin/env python3
"""DINO + NavDP Isaac Sim GUI — Nav_new's own control panel.

Replaces the old OmniVLA isaac_gui.py: same Zenoh contract, new inference
core (Grounding DINO -> point goal -> NavDP cross-modal -> trajectory
selection -> cmd_vel).

Features
--------
- Live camera feed from rover_camera / image_raw via Zenoh
- DINO detection bbox + score overlaid on the feed
- Target text entry (Enter or Send) + preset buttons
- State (TRACK / SEARCH / STOP), velocity readout, per-stage latency
- Top-down plot of the 32 sampled NavDP trajectories + the selected one
- STOP button (zero velocity, inference paused until new target)

Run (from Nav_new root):
    conda activate internnav
    python -m nav_pipeline.isaac_gui [--target "trash bin"] [--pi-ip <IP>]
"""

import argparse
import os
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.zenoh_node import (  # noqa: E402
    CAMERA_COMPRESSED_KEYS,
    CAMERA_KEYS,
    DEPTH_KEYS,
    parse_compressed_image,
    parse_image,
    serialize_path,
    serialize_string,
    serialize_twist,
)

PRESETS = ["trash bin", "cardboard box", "wooden pallet", "door", "chair"]
HEARTBEAT_PERIOD_S = 0.15
DEPTH_STALE_S = 1.0


class SharedState:
    def __init__(self, target: str):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.frame_count = 0
        self.mode = "text"                      # "text" | "manual"
        self.target = target
        self.stopped = False
        self.goal_reached = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.5                   # manual-drive caps; set from CLI args in main()
        self.max_angular = 0.6
        # for display
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


def zenoh_setup(session: zenoh.Session, st: SharedState, compressed_only: bool = False):
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

    # On the real rover Wi-Fi, raw 640x480 rgb8 (~8 MB/s) saturates the link
    # (starving cmd_vel/rpm and even SSH) — subscribe compressed JPEG only.
    raw_keys = [] if compressed_only else CAMERA_KEYS
    subs = (
        [session.declare_subscriber(k, on_image) for k in raw_keys]
        + [session.declare_subscriber(k, on_compressed) for k in CAMERA_COMPRESSED_KEYS]
        + [session.declare_subscriber(k, on_depth) for k in DEPTH_KEYS]
    )
    pubs = {
        "cmd": session.declare_publisher("cmd_vel"),
        "explain": session.declare_publisher("omnivla/explanation"),
        "path": session.declare_publisher("omnivla/waypoints"),
    }
    return subs, pubs


def heartbeat_loop(st: SharedState, pubs, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = st.last_cmd
        pubs["cmd"].put(serialize_twist(lin, ang))


def inference_loop(pipe: DinoNavDPPipeline, st: SharedState, pubs, running,
                   predict_hz: float, stop_confirm: int = 3):
    period = 1.0 / predict_hz
    stop_streak = 0
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            mode = st.mode
            target = st.target
            paused = st.stopped or st.goal_reached
        if rgb is None:
            time.sleep(0.1)
            continue
        if mode == "manual":
            # bypass detection/NavDP entirely -- last_cmd is set directly by
            # the GUI's manual-drive button/key handlers; this loop only
            # keeps the camera preview and status text current, and still
            # honors STOP as an emergency zero
            with st.lock:
                if st.stopped:
                    st.last_cmd = (0.0, 0.0)
                lin, ang = st.last_cmd
                st.display_rgb = rgb
                st.state_text = "MANUAL (stopped)" if st.stopped else "MANUAL DRIVE"
                st.vel_text = f"lin {lin:.3f}  ang {ang:+.3f}"
            time.sleep(0.05)
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

        try:
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
        reached = stop_streak >= stop_confirm

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
                st.state_text = f"GOAL REACHED: '{target}'"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                st.state_text = res.state
                st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v*1000:.0f}ms" for k, v in res.timing.items())

        if reached:
            pubs["explain"].put(serialize_string(f"GOAL REACHED: '{target}'. Stopping."))
        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        score = f"{res.detection.score:.2f}" if res.detection else "-"
        pubs["explain"].put(serialize_string(
            f"DINO+NavDP [{res.state}] det={score} -> lin={res.linear:.3f} ang={res.angular:.3f} "
            f"| target='{target}'"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448          # camera panel (px)
    PLOT_SIZE = 448         # top-down plot (px)
    PLOT_RANGE = 3.5        # meters shown ahead

    def __init__(self, root: tk.Tk, st: SharedState):
        self.root = root
        self.st = st
        root.title("Nav_new — DINO + NavDP")
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
        self.entry = ttk.Entry(bar, width=36)
        self.entry.insert(0, st.target)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", lambda e: self.send_target())
        ttk.Button(bar, text="Send", command=self.send_target).pack(side="left", padx=2)
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        presets = ttk.Frame(main)
        presets.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        for p in PRESETS:
            ttk.Button(presets, text=p, command=lambda t=p: self.send_target(t)).pack(side="left", padx=2)

        # Manual drive: hold a button (or an arrow key, once the window has
        # focus) to drive directly, bypassing detection/NavDP entirely.
        # Releasing zeros that axis; it does NOT hand control back to
        # whatever autonomous target was running before -- send a target
        # for that. STOP still works as usual.
        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        # Fixed character width on the two dynamic-text rows: their content
        # (state name, counters, latency numbers) changes length on every
        # refresh tick, and an unconstrained Label makes the whole window
        # resize to match on every tick. Width is a floor, not a wrap limit,
        # so longer text (e.g. a long custom target) just doesn't shrink
        # back below it.
        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=100, anchor="w")
        self.status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=100, anchor="w")
        self.info.grid(row=5, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(66, self.refresh)

    def send_target(self, text: Optional[str] = None):
        t = text if text is not None else self.entry.get().strip()
        if not t:
            return
        if text is not None:
            self.entry.delete(0, "end")
            self.entry.insert(0, t)
        self._manual_held.clear()
        with self.st.lock:
            self.st.mode = "text"
            self.st.target = t
            self.st.stopped = False
            self.st.goal_reached = False

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
        with self.st.lock:
            rgb = self.st.display_rgb if self.st.display_rgb is not None else self.st.latest_rgb
            det = self.st.detection
            mask = self.st.mask
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers, target = self.st.frame_count, self.st.infer_count, self.st.target
            drive_mode = self.st.mode
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
        target_txt = "manual drive" if drive_mode == "manual" else f"'{target}'"
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  target: {target_txt}   {vel_text}{fwd}")
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}")
        self.root.after(66, self.refresh)


def main():
    ap = argparse.ArgumentParser(description="Nav_new DINO+NavDP Isaac GUI")
    ap.add_argument("--target", default="trash bin")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 0.25 on the real rover)")
    ap.add_argument("--invert-angular", action="store_true",
                    help="flip turn direction (use if the rover steers away from the target)")
    ap.add_argument("--compressed-only", action="store_true",
                    help="subscribe only the JPEG camera stream (REQUIRED over rover Wi-Fi)")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(0.15, args.max_angular),
        invert_angular=args.invert_angular,
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz), daemon=True).start()

    root = tk.Tk()
    App(root, st)

    # Ctrl-C in the launching terminal closes the GUI cleanly. Tk's mainloop
    # only delivers Python signals while Python code runs, so keep a periodic
    # no-op tick alive for the handler to fire promptly.
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
