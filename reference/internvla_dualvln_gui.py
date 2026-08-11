#!/usr/bin/env python3
"""
InternVLA-N1 DualVLN — standalone path viewer
==============================================
A lightweight Tkinter viewer for reference/internvla_dualvln_zenoh_node.py.
Shows the camera feed and a top-down bird's-eye plot of the trajectory the
bot is about to follow (System-1's predicted local-frame path), matching the
visual style already used by nav_pipeline/isaac_gui.py (camera panel + a
separate top-down tk.Canvas plot, not projected onto the camera image).

Deliberately a SEPARATE process from the inference node, not merged into it:
- Needs no GPU/model/torch -- just Zenoh + Tkinter + PIL, so it can run on a
  different, lighter machine (or the same one) than whatever is running the
  ~17GB model.
- Zero risk to the validated inference node: this only subscribes to topics
  the node already publishes (cmd_vel, omnivla/explanation, omnivla/trajectory)
  plus the same camera topics, and optionally publishes omnivla/goal_text
  (a topic the node already listens on for live instruction changes -- no
  node-side change needed for that).

Run (any machine with Zenoh connectivity to the GPU box / Pi):
    conda activate internnav   # only needs zenoh, numpy, pillow -- no torch
    python reference/internvla_dualvln_gui.py --pi-ip <gpu-or-pi-ip>

Subscribes (Zenoh, CDR):
  image_raw, image_raw/compressed  -- camera feed
  omnivla/trajectory                -- predicted local-frame path (custom format)
  cmd_vel                           -- current velocity command
  omnivla/explanation               -- status text from the node
Publishes (Zenoh, CDR):
  omnivla/goal_text                 -- optional: change instruction live
"""

import sys
import struct
import time
import argparse
from threading import Lock
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh Python library not found (pip install eclipse-zenoh).")
    sys.exit(1)


# ================================================================
#  CDR Helpers -- copied from internvla_dualvln_zenoh_node.py so this
#  viewer stays a fully independent, standalone process (same wire format).
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

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8
        return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset:self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data = self.data[self.offset:self.offset + count]
        self.offset += count
        return data


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
    Hiwonder's Pi bridge (landerpi/bridge.py) publishes ONLY this topic."""
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


def parse_twist_linear_angular(cdr_data: bytes) -> tuple:
    """geometry_msgs/msg/Twist CDR -> (linear.x, angular.z). Matches
    internvla_dualvln_zenoh_node.py's serialize_twist field order exactly."""
    r = CDRReader(cdr_data)
    lin_x = r.read_float64()
    r.read_float64(); r.read_float64()               # linear.y, linear.z
    r.read_float64(); r.read_float64()                # angular.x, angular.y
    ang_z = r.read_float64()
    return lin_x, ang_z


def parse_trajectory(cdr_data: bytes) -> np.ndarray:
    """Inverse of internvla_dualvln_zenoh_node.py's serialize_trajectory:
    count + flat float64 (x, y) pairs -> (N, 2) array, local frame meters."""
    r = CDRReader(cdr_data)
    n = r.read_uint32()
    pts = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        pts[i, 0] = r.read_float64()
        pts[i, 1] = r.read_float64()
    return pts


def serialize_string(text: str) -> bytes:
    w = CDRWriter()
    w.write_string(text)
    return w.to_bytes()


# ================================================================
#  Shared state (Zenoh callbacks write; Tkinter refresh loop reads)
# ================================================================
class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_trajectory: Optional[np.ndarray] = None
        self.linear = 0.0
        self.angular = 0.0
        self.explanation = ""
        self.frame_count = 0
        self.last_frame_ts = 0.0
        self.last_traj_ts = 0.0


# ================================================================
#  Tkinter App -- camera panel + top-down trajectory plot, matching
#  nav_pipeline/isaac_gui.py's layout/transform/color conventions.
# ================================================================
class App:
    CAM_SIZE = 448
    PLOT_SIZE = 448

    def __init__(self, root: tk.Tk, st: SharedState, session: "zenoh.Session", plot_range: float):
        self.root = root
        self.st = st
        self.session = session
        self.plot_range = plot_range
        self.closed = False
        self.pub_goal = session.declare_publisher("omnivla/goal_text")

        root.title("InternVLA-N1 DualVLN — path viewer")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)

        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Label(bar, text="Instruction:").pack(side="left")
        self.goal_entry = ttk.Entry(bar, width=60)
        self.goal_entry.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(bar, text="Set", command=self.on_set_goal).pack(side="left")
        self.goal_entry.bind("<Return>", lambda e: self.on_set_goal())

        self.status = ttk.Label(main, text="waiting for frames...", anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self.explain = ttk.Label(main, text="", anchor="w", wraplength=self.CAM_SIZE + self.PLOT_SIZE)
        self.explain.grid(row=3, column=0, columnspan=2, sticky="we")

        self.root.after(66, self.refresh)  # ~15Hz self-rescheduling, decoupled from Zenoh threads

    def on_set_goal(self):
        text = self.goal_entry.get().strip()
        if text:
            self.pub_goal.put(serialize_string(text))

    def on_close(self):
        self.closed = True
        self.root.destroy()

    def refresh(self):
        if self.closed:
            return
        with self.st.lock:
            rgb = self.st.latest_rgb
            traj = self.st.latest_trajectory
            lin, ang = self.st.linear, self.st.angular
            explanation = self.st.explanation
            frame_count = self.st.frame_count
            frame_age = time.time() - self.st.last_frame_ts if self.st.last_frame_ts else None
            traj_age = time.time() - self.st.last_traj_ts if self.st.last_traj_ts else None

        if rgb is not None:
            img = Image.fromarray(rgb).convert("RGB").resize((self.CAM_SIZE, self.CAM_SIZE))
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        self._draw_plot(traj)

        stale = " (STALE)" if frame_age is not None and frame_age > 2.0 else ""
        self.status.configure(
            text=f"frames={frame_count}{stale}  lin={lin:+.3f} m/s  ang={ang:+.3f} rad/s"
            + (f"  traj_age={traj_age:.1f}s" if traj_age is not None else "  traj=none")
        )
        self.explain.configure(text=explanation)

        self.root.after(66, self.refresh)

    def _draw_plot(self, traj: Optional[np.ndarray]):
        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.plot_range

        def to_px(x, y):  # robot-local frame (x fwd, y left) -> canvas pixels
            return S / 2 - (y / R) * (S / 2), S - (x / R) * S * 0.92 - 20

        # range rings every 1m, purely visual reference (isaac_gui.py has no
        # equivalent -- added here since this viewer has no obstacle points
        # to otherwise convey scale)
        for ring_m in range(1, int(R) + 1):
            _, top_y = to_px(ring_m, 0.0)
            self.plot.create_line(0, top_y, S, top_y, fill="#eee")

        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")                    # baseline
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")   # robot marker

        if traj is not None and len(traj) >= 2:
            pts = [to_px(p[0], p[1]) for p in traj]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)
            gx, gy = to_px(traj[-1][0], traj[-1][1])
            self.plot.create_oval(gx - 4, gy - 4, gx + 4, gy + 4, fill="red", outline="")


# ================================================================
#  Zenoh wiring
# ================================================================
def make_callbacks(st: SharedState):
    def on_image(sample: "zenoh.Sample"):
        rgb = parse_image(bytes(sample.payload))
        if rgb is not None:
            with st.lock:
                st.latest_rgb = rgb
                st.frame_count += 1
                st.last_frame_ts = time.time()

    def on_image_compressed(sample: "zenoh.Sample"):
        rgb = parse_compressed_image(bytes(sample.payload))
        if rgb is not None:
            with st.lock:
                st.latest_rgb = rgb
                st.frame_count += 1
                st.last_frame_ts = time.time()

    def on_trajectory(sample: "zenoh.Sample"):
        try:
            traj = parse_trajectory(bytes(sample.payload))
        except Exception:
            return
        with st.lock:
            st.latest_trajectory = traj
            st.last_traj_ts = time.time()

    def on_cmd(sample: "zenoh.Sample"):
        try:
            lin, ang = parse_twist_linear_angular(bytes(sample.payload))
        except Exception:
            return
        with st.lock:
            st.linear, st.angular = lin, ang

    def on_explanation(sample: "zenoh.Sample"):
        try:
            text = parse_string(bytes(sample.payload))
        except Exception:
            return
        with st.lock:
            st.explanation = text

    return on_image, on_image_compressed, on_trajectory, on_cmd, on_explanation


def main():
    p = argparse.ArgumentParser(description="Standalone path viewer for internvla_dualvln_zenoh_node.py")
    p.add_argument("--pi-ip", type=str, default=None,
                   help="IP of the GPU box (or Pi) running the DualVLN node / Zenoh; "
                        "omit for multicast scouting")
    p.add_argument("--plot-range", type=float, default=3.5,
                   help="meters of forward range shown in the top-down plot")
    args = p.parse_args()

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting (auto-discover)")
    session = zenoh.open(config)

    st = SharedState()
    on_image, on_image_compressed, on_trajectory, on_cmd, on_explanation = make_callbacks(st)
    subs = [
        session.declare_subscriber("image_raw", on_image),
        session.declare_subscriber("image_raw/compressed", on_image_compressed),
        session.declare_subscriber("omnivla/trajectory", on_trajectory),
        session.declare_subscriber("cmd_vel", on_cmd),
        session.declare_subscriber("omnivla/explanation", on_explanation),
    ]
    print("[INFO] Zenoh subs: image_raw, image_raw/compressed, omnivla/trajectory, cmd_vel, omnivla/explanation")

    root = tk.Tk()
    App(root, st, session, args.plot_range)
    try:
        root.mainloop()
    finally:
        for s in subs:
            s.undeclare()
        session.close()


if __name__ == "__main__":
    main()
