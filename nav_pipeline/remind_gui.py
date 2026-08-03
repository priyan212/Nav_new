#!/usr/bin/env python3
"""REMIND + NavDP rover GUI -- Nav_new's isaac_gui.py, retargeted to drive by
persistent object ID instead of a bare text phrase.

Same control panel as nav_pipeline.isaac_gui (camera feed, top-down NavDP
trajectory plot, state/velocity readout, manual drive, STOP), but:

- Target selection: instead of a free-text DINO phrase, this sends every
  camera frame to a REMIND live-tracking server (REMIND/remind-reid-tracker,
  a separate process/conda env -- see remind_client.py and
  launch_rover_remind.sh) and overlays EVERY currently-tracked object with
  REMIND's own persistent label, e.g. "CHAIR ID 1". The operator reads an ID
  off the video and types it back (or double-clicks it in the "known
  objects" list) as "CHAIR ID 1" -- see remind_target.parse_object_target.
- Navigation: unchanged. Once the requested (class, id) is resolved to a
  detection this tick, it's handed to DinoNavDPPipeline.step() via the
  external_dets hook (see pipeline.py) -- same belief/SEARCH/AVOID/STOP
  state machine, same 1.5 m default stop_distance, as every other launcher.
- Depth: RGB-only metric depth via Depth Anything V2 ViT-B by default (more
  accurate than the vits default used elsewhere; depth error feeds directly
  into the STOP distance decision -- see depth_estimator.py).

Run (from Nav_new root, after the REMIND live server is up -- see
launch_rover_remind.sh, which brings up both):
    conda activate internnav
    python -m nav_pipeline.remind_gui --pi-ip <IP> --remind-server http://127.0.0.1:8765
"""

import argparse
import colorsys
import os
import signal
import sys
import time
from threading import Lock, Thread
from typing import List, Optional

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

from nav_pipeline.dino_detector import Detection  # noqa: E402
from nav_pipeline.isaac_gui import (  # noqa: E402
    DEPTH_STALE_S,
    HEARTBEAT_PERIOD_S,
    SPIN_DIST_THRESH_M,
    SPIN_ROT_THRESH_RAD,
    SPIN_WINDOW_S,
    heartbeat_loop,
    zenoh_setup,
)
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.remind_client import RemindClient, RemindObject  # noqa: E402
from nav_pipeline.remind_target import parse_object_target  # noqa: E402
from nav_pipeline.zenoh_node import serialize_path, serialize_string, serialize_twist  # noqa: E402


def _color_for_id(object_id: Optional[int]) -> tuple:
    """Deterministic per-ID color (RGB) so the same object keeps the same
    box color tick to tick -- matches REMIND's own render_frame convention
    (REMIND/remind-reid-tracker/scripts/run_video_tracking.py's
    _color_for_id) enough to feel consistent, without importing across the
    two separate environments."""
    if object_id is None:
        return (150, 150, 150)
    hue = (int(object_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


class SharedState:
    def __init__(self, target: str):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.frame_count = 0
        self.mode = "manual"                    # "text" | "manual" -- starts inert
        self.target = target                     # display string, e.g. "CHAIR ID 1"
        self.target_class: Optional[str] = None
        self.target_id: Optional[int] = None
        self.stopped = False
        self.goal_reached = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.5
        self.max_angular = 0.6
        # for display
        self.display_rgb: Optional[np.ndarray] = None
        self.detection = None
        self.mask: Optional[np.ndarray] = None
        self.remind_objects: List[RemindObject] = []
        self.remind_ok = True
        self.state_text = "waiting for camera"
        self.vel_text = "lin 0.000  ang +0.000"
        self.lat_text = ""
        self.trajs = None
        self.chosen = None
        self.goal_pt = None
        self.obstacles = None
        self.min_forward = float("inf")
        self.infer_count = 0

        if target:
            parsed = parse_object_target(target)
            if parsed:
                self.target_class, self.target_id = parsed
                self.mode = "text"


def remind_poll_loop(remind: RemindClient, st: SharedState, running, remind_period_s: float = 0.4):
    """Runs independently of the nav control tick, in its own thread.

    REMIND's own latency (~0.2-0.5s measured on an RTX 3090 Ti; see
    REMIND_METHOD.md) is heavier than the nav loop's tick budget -- calling
    it synchronously from remind_inference_loop put that latency directly
    on the control loop's critical path (measured: dropped the effective
    loop rate from the requested ~2.5 Hz to ~1.5-1.9 Hz, eroding the
    obstacle-guard confirm-tick timing margin). This thread just keeps
    st.remind_objects fresh at REMIND's own achievable rate; the nav loop
    always reads whatever's latest instead of waiting on it.
    """
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
        if rgb is not None:
            try:
                objects = remind.infer(rgb)
                with st.lock:
                    st.remind_objects = objects
                    st.remind_ok = True
            except Exception as e:
                print(f"[WARN] REMIND inference failed: {e}")
                with st.lock:
                    st.remind_ok = False
        dt = remind_period_s - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def remind_inference_loop(pipe: DinoNavDPPipeline, st: SharedState, pubs,
                          running, predict_hz: float,
                          stop_confirm: int = 3, odom: Optional[OdometryLogger] = None):
    period = 1.0 / predict_hz
    stop_streak = 0
    last_target_key = (None, None)

    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            mode = st.mode
            target_class, target_id, target_text = st.target_class, st.target_id, st.target
            paused = st.stopped or st.goal_reached
            last_objects = st.remind_objects  # kept fresh by remind_poll_loop

        target_key = (target_class, target_id)
        if target_key != (None, None) and target_key != last_target_key:
            if last_target_key != (None, None):
                # new goal: don't let tracked-box/goal-belief state from the
                # PREVIOUS target leak into this one
                pipe.reset()
            if odom is not None:
                odom.start_new_goal(target_text)
            last_target_key = target_key

        if rgb is None:
            time.sleep(0.1)
            continue

        if mode == "manual":
            with st.lock:
                if st.stopped:
                    st.last_cmd = (0.0, 0.0)
                lin, ang = st.last_cmd
                st.display_rgb = rgb
                st.state_text = "MANUAL (stopped)" if st.stopped else "MANUAL DRIVE"
                st.vel_text = f"lin {lin:.3f}  ang {ang:+.3f}"
            time.sleep(0.05)
            continue

        if paused or target_key == (None, None):
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.vel_text = "lin 0.000  ang +0.000"
                if target_key == (None, None):
                    st.state_text = "waiting for target, e.g. 'CHAIR ID 1'"
            time.sleep(0.1)
            continue

        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None

        matched = [o for o in last_objects
                  if o.object_id == target_id and (o.class_name or "").lower() == target_class]
        external_dets = []
        if matched:
            det = Detection(box=matched[0].bbox, score=max(matched[0].confidence, 0.01), label=matched[0].label)
            # REMIND already segmented this object (its own YOLO-seg
            # backend) -- attach the mask so pipeline.py's external_dets
            # branch reuses it instead of running a second SAM2 pass (see
            # pipeline.py's goal computation, `getattr(det, "mask", None)`).
            det.mask = matched[0].mask
            external_dets = [det]

        try:
            pose = (odom.x, odom.y, odom.theta) if odom is not None else None
            res = pipe.step(rgb, target_class, depth=depth, pose=pose, external_dets=external_dets)
        except Exception as e:
            print(f"[ERROR] pipeline step: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = f"ERROR: {e}"
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.5)
            continue

        spin = odom.spin_delta(SPIN_WINDOW_S) if odom is not None else None
        if spin is not None and spin[0] > SPIN_ROT_THRESH_RAD and spin[1] < SPIN_DIST_THRESH_M:
            print(f"[WARN] spin-stall watchdog: turned {spin[0]:.1f}rad in {SPIN_WINDOW_S:.0f}s, "
                  f"only {spin[1]:.2f}m net travel -- forcing stop until a new target is sent")
            with st.lock:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = (f"SPIN STALL: {np.degrees(spin[0]):.0f}deg turned, "
                                 f"{spin[1]:.2f}m travel -- send a new target")
                st.vel_text = "lin 0.000  ang +0.000"
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
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
                st.state_text = f"GOAL REACHED: '{target_text}' (stopped at {pipe.cfg.stop_distance:.1f}m)"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                st.state_text = res.state if matched else f"{res.state} ('{target_text}' not currently visible)"
                st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v * 1000:.0f}ms" for k, v in res.timing.items())

        if reached:
            pubs["explain"].put(serialize_string(f"GOAL REACHED: '{target_text}'. Stopping."))
        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        pubs["explain"].put(serialize_string(
            f"REMIND+NavDP [{res.state}] target='{target_text}' -> lin={res.linear:.3f} ang={res.angular:.3f}"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448
    PLOT_SIZE = 448
    PLOT_RANGE = 3.5

    def __init__(self, root: tk.Tk, st: SharedState, remind: RemindClient):
        self.root = root
        self.st = st
        self.remind = remind
        root.title("Nav_new — REMIND + NavDP")
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

        known = ttk.Frame(main)
        known.grid(row=0, column=2, padx=4, pady=4, sticky="ns")
        ttk.Label(known, text="Known objects (double-click to target):").pack(anchor="w")
        self.known_list = tk.Listbox(known, width=28, height=22)
        self.known_list.pack(fill="y", expand=True)
        self.known_list.bind("<Double-Button-1>", self._on_known_double_click)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="Object:").pack(side="left")
        self.name_entry = ttk.Entry(bar, width=18)
        if st.target_class:
            self.name_entry.insert(0, st.target_class)
        self.name_entry.pack(side="left", padx=(4, 12))
        self.name_entry.bind("<Return>", lambda e: self.send_target())

        ttk.Label(bar, text="ID:").pack(side="left")
        self.id_entry = ttk.Entry(bar, width=6)
        if st.target_id is not None:
            self.id_entry.insert(0, str(st.target_id))
        self.id_entry.pack(side="left", padx=4)
        self.id_entry.bind("<Return>", lambda e: self.send_target())

        ttk.Button(bar, text="Send", command=self.send_target).pack(side="left", padx=(8, 2))
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)
        ttk.Button(bar, text="Reset REMIND memory", command=self.reset_memory).pack(side="left", padx=10)

        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                width=110, anchor="w")
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=110, anchor="w")
        self.info.grid(row=4, column=0, columnspan=3, sticky="w")

        self._photo = None
        self._known_ids: List[str] = []
        self.root.after(66, self.refresh)

    def send_target(self):
        """Reads the two separate Object/ID fields directly and maps them
        to the model's target (st.target_class, st.target_id) -- no string
        parsing involved on this path, unlike the old single combined-text
        entry (parse_object_target is still used for the known-objects
        list, which has to split REMIND's own "CLASS ID n" labels)."""
        name = self.name_entry.get().strip().lower()
        id_text = self.id_entry.get().strip()
        if not name or not id_text:
            with self.st.lock:
                self.st.state_text = "enter both an object name and an ID"
            return
        try:
            target_id = int(id_text)
        except ValueError:
            with self.st.lock:
                self.st.state_text = f"ID must be a whole number, got '{id_text}'"
            return
        canonical = f"{name.upper()} ID {target_id}"
        self._manual_held.clear()
        with self.st.lock:
            self.st.mode = "text"
            self.st.target_class = name
            self.st.target_id = target_id
            self.st.target = canonical
            self.st.stopped = False
            self.st.goal_reached = False

    def _set_fields(self, class_name: str, object_id: int):
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, class_name)
        self.id_entry.delete(0, "end")
        self.id_entry.insert(0, str(object_id))

    def _on_known_double_click(self, _event):
        sel = self.known_list.curselection()
        if not sel:
            return
        label = self._known_ids[sel[0]]  # e.g. "CHAIR ID 1"
        parsed = parse_object_target(label)
        if parsed is None:
            return
        class_name, object_id = parsed
        self._set_fields(class_name, object_id)
        self.send_target()

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

    def reset_memory(self):
        def _do():
            try:
                self.remind.reset()
                with self.st.lock:
                    self.st.state_text = "REMIND memory reset -- re-explore to rebuild the catalogue"
            except Exception as e:
                with self.st.lock:
                    self.st.state_text = f"REMIND reset failed: {e}"
        Thread(target=_do, daemon=True).start()

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
            rgb = self.st.latest_rgb if self.st.latest_rgb is not None else self.st.display_rgb
            det = self.st.detection
            mask = self.st.mask
            objects = list(self.st.remind_objects)
            remind_ok = self.st.remind_ok
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers, target = self.st.frame_count, self.st.infer_count, self.st.target
            drive_mode = self.st.mode
            stopped = self.st.stopped
            target_class, target_id = self.st.target_class, self.st.target_id

        if rgb is not None:
            frame = rgb
            if mask is not None and mask.shape[:2] == rgb.shape[:2]:
                frame = rgb.copy()
                frame[mask] = (0.55 * frame[mask] + 0.45 * np.array([0, 255, 60])).astype(np.uint8)
            img = Image.fromarray(frame).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            d = ImageDraw.Draw(img)
            for o in objects:
                is_target = (o.object_id == target_id and (o.class_name or "").lower() == target_class)
                color = (0, 255, 60) if is_target else _color_for_id(o.object_id)
                x0, y0, x1, y1 = o.bbox
                d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=color, width=3 if is_target else 2)
                d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), o.label, fill=color)
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        # known-objects list: every object_id ever confirmed. Only rebuild
        # when the content actually changes, and restore the selection
        # afterward -- refresh() runs every 66ms, well inside a double-click
        # gesture's ~300-500ms window, so an unconditional delete()+insert()
        # every tick wiped selection state and could shift indices mid-click
        # if REMIND's response order shifted between polls (double-click
        # then acted on a different row than the one clicked). The
        # "REMIND server unreachable" state is shown in the info line below
        # instead of as a fake extra row here -- inserting it into this
        # listbox at index 0 previously shifted every real row's index by
        # one relative to self._known_ids, an off-by-one on top of the above.
        new_known_ids = [f"{(o.class_name or '?').upper()} ID {o.object_id}"
                         for o in objects if o.object_id is not None]
        if new_known_ids != self._known_ids:
            sel = self.known_list.curselection()
            selected_label = self._known_ids[sel[0]] if sel and sel[0] < len(self._known_ids) else None
            self.known_list.delete(0, "end")
            for label in new_known_ids:
                self.known_list.insert("end", label)
            self._known_ids = new_known_ids
            if selected_label is not None and selected_label in new_known_ids:
                self.known_list.selection_set(new_known_ids.index(selected_label))

        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

        def to_px(x, y):
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
        remind_txt = "" if remind_ok else "  [REMIND UNREACHABLE]"
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}{remind_txt}")
        self.root.after(66, self.refresh)


def main():
    ap = argparse.ArgumentParser(description="Nav_new REMIND+NavDP rover GUI")
    ap.add_argument("--target", default="",
                    help="initial target as 'CLASS ID N', e.g. 'chair id 1' -- starts empty "
                         "(manual drive) until a target is sent from the GUI")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--remind-server", default="http://127.0.0.1:8765",
                    help="base URL of the REMIND live server (see launch_rover_remind.sh)")
    ap.add_argument("--remind-period", type=float, default=0.4,
                    help="minimum seconds between REMIND inference calls; the nav loop reuses "
                         "the last response in between instead of blocking on every tick")
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 1.2 on the real rover)")
    ap.add_argument("--search-angular", type=float, default=0.15)
    ap.add_argument("--servo-ramp-deg", type=float, default=35.0)
    ap.add_argument("--angular-slew-max", type=float, default=0.10)
    ap.add_argument("--invert-angular", action="store_true")
    ap.add_argument("--no-belief-goal", action="store_true")
    ap.add_argument("--stop-distance", type=float, default=1.5,
                    help="meters from the object at which to stop (depth-based)")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vitb",
                    help="RGB-only metric depth model (no depth sensor on the real rover); "
                         "defaults to vitb here since depth error feeds directly into the "
                         "STOP distance decision -- needs checkpoints/depth_anything_v2_"
                         "metric_hypersim_vitb.pth (scripts/download_models.py --depth-encoder vitb)")
    ap.add_argument("--compressed-only", action="store_true")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    args = ap.parse_args()

    print(f"[INFO] checking REMIND server at {args.remind_server} ...")
    remind = RemindClient(args.remind_server)
    if not remind.health():
        print(f"[ERROR] REMIND server not reachable at {args.remind_server} -- "
              f"start it first (see launch_rover_remind.sh)")
        sys.exit(1)
    print("[INFO] REMIND server OK")

    print("[INFO] loading navigation models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(args.search_angular, args.max_angular),
        servo_ramp_deg=args.servo_ramp_deg,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        use_belief_goal=not args.no_belief_goal,
        depth_encoder=args.depth_encoder,
        stop_distance=args.stop_distance,
        # REMIND already provides persistent per-object identity; the
        # pipeline's own single-target DINOv2 appearance re-lock (tuned for
        # raw multi-candidate DINO streams) is redundant here and is fully
        # bypassed anyway by the external_dets hook (see pipeline.py).
        use_appearance_reid=False,
        # Same reasoning: REMIND's YOLO-seg backend already segments every
        # detection as part of its own pipeline, and the mask is forwarded
        # through external_dets (see remind_client.RemindObject.mask) --
        # SAM2 would just recompute the same thing a second time. CLIP
        # verification is dropped too: target_text here is just the class
        # name REMIND/YOLO already assigned, so it would only be
        # re-checking that same classification with a weaker model.
        use_sam=False,
        use_clip=False,
        # The periodic scene inventory (scene_log/) runs a separate
        # Grounding DINO pass over a large vocabulary purely for offline
        # logging -- it's not consulted by any navigation decision, and
        # REMIND's own catalogue already captures a strictly richer version
        # of the same information. Measured cost: this was the single
        # remaining latency spike on the nav loop (~170-285ms once a
        # second, pushing p95 to ~420ms); disabling it flattens the loop to
        # a steady ~222ms/tick with no navigation-relevant loss.
        use_scene_tagger=False,
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    odom = OdometryLogger(args.odometry_log_dir)
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=remind_poll_loop, args=(remind, st, running),
           kwargs={"remind_period_s": args.remind_period}, daemon=True).start()
    Thread(target=remind_inference_loop,
           args=(pipe, st, pubs, running, args.predict_hz),
           kwargs={"odom": odom}, daemon=True).start()

    root = tk.Tk()
    App(root, st, remind)

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
