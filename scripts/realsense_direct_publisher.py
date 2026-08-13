#!/usr/bin/env python3
"""RealSense D435i -> Zenoh direct publisher (bypasses the ROS2 wrapper).

WHY THIS EXISTS (2026-08-13): realsense2_camera_node (the ROS2 wrapper)
cannot reliably run color+depth together on this Pi (Raspberry Pi 4 Model
B) -- enabling depth alongside color makes the wrapper's device bring-up
sequence (rs_node_setup.cpp) hang the whole node: 0% CPU, zero frames on
ANY stream, kernel log showing repeated UVC extension-unit control
timeouts ("usb 2-1: Failed to query (SET_CUR/GET_LEN) UVC control 1/7 on
unit 3: -110/-32"). Ruled out first: USB autosuspend, usbfs_memory_mb at
runtime AND at boot (reboot + kernel cmdline param), a different physical
USB3 port, a minimal depth profile (424x240@6fps -- same failure, so it
was never a bandwidth issue). This exact failure signature is a
well-documented, still-open issue specific to Raspberry Pi 4 + RealSense
D400-series combined streams through the ROS2 wrapper (see e.g.
https://github.com/IntelRealSense/realsense-ros/issues/3089,
https://github.com/IntelRealSense/realsense-ros/issues/2991) -- several
affected users independently converged on the same workaround adopted
here: talk to the camera directly via the raw librealsense SDK (pyrealsense2)
instead of through the ROS2 wrapper. Verified working on this exact Pi:
30/30 color + 30/30 depth frames, sustained, clean start/stop -- the ROS2
wrapper has never once succeeded at that combination here.

This script owns the D435i exclusively (do not also run
realsense_only_bringup.launch.py / realsense2_camera_node at the same
time -- two separate librealsense pipeline instances against the same
physical device WILL conflict). It talks Zenoh directly (eclipse-zenoh
Python client), connecting as a local client to the zenoh-bridge-ros2dds
router already running on this Pi (rover-zenoh.service, tcp/127.0.0.1:7447)
-- that router still relays these publications out to the GPU exactly like
any other Zenoh key, no ROS2 involved for the camera at all. rover-agent's
ESP32<->rover/rpm<->cmd_vel path is untouched and still goes through ROS2
via that same bridge.

CDR wire format matches nav_pipeline/zenoh_node.py's parsers exactly
(CDRReader/parse_image/parse_camera_info/parse_compressed_image) -- this
file can't import that module directly (it pulls in the full GPU-side
torch/DINO/NavDP pipeline, not installed here), so the small CDR-writing
subset needed is duplicated below, same as reference/
internvla_dualvln_zenoh_node.py already duplicates zenoh_node.py's parsers
for the same reason.

Depth is published as a custom "png16" encoding (PNG-compressed uint16
millimeters) on the SAME depth_raw key/Image-message shape zenoh_node.py's
DEPTH_KEYS/_on_depth already consume -- parse_image() has a matching
"png16" branch added alongside its existing 16uc1/32fc1 branches. This
keeps depth's Wi-Fi payload down near color's, instead of raw 16UC1's
~9MB/s (see isaac_gui.py's own comment: raw rgb8 alone was already enough
to saturate this link and starve cmd_vel/rpm/SSH).

Usage on Pi:
  python3 realsense_direct_publisher.py [--pi-ip 127.0.0.1]
"""

import argparse
import signal
import struct
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import zenoh


# ================================================================
#  CDR writer (DDS wire format) -- minimal subset, must byte-match
#  nav_pipeline/zenoh_node.py's CDRReader/parsers on the GPU side.
# ================================================================
class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n: int):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_uint8(self, v: int):
        self.buf += struct.pack("<B", v)

    def write_int32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<i", v)

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

    def write_sequence_uint8(self, data: bytes):
        self.write_uint32(len(data))
        self.buf += data

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def serialize_image(height: int, width: int, encoding: str, data: bytes, step: int = 0) -> bytes:
    """sensor_msgs/Image CDR -- matches zenoh_node.py's parse_image() read order."""
    w = CDRWriter()
    w.write_int32(0); w.write_uint32(0); w.write_string("")  # header (stamp, frame_id -- unused by parser)
    w.write_uint32(height)
    w.write_uint32(width)
    w.write_string(encoding)
    w.write_uint8(0)  # is_bigendian
    w._align(4)
    w.write_uint32(step)  # unused by parser, just needs to be present for correct CDR offsets
    w.write_sequence_uint8(data)
    return w.to_bytes()


def serialize_compressed_image(fmt: str, data: bytes) -> bytes:
    """sensor_msgs/CompressedImage CDR -- matches parse_compressed_image()."""
    w = CDRWriter()
    w.write_int32(0); w.write_uint32(0); w.write_string("")  # header
    w.write_string(fmt)
    w.write_sequence_uint8(data)
    return w.to_bytes()


def serialize_camera_info(height: int, width: int, fx: float, fy: float, cx: float, cy: float) -> bytes:
    """sensor_msgs/CameraInfo CDR -- matches parse_camera_info() (only reads through K[9])."""
    w = CDRWriter()
    w.write_int32(0); w.write_uint32(0); w.write_string("")  # header
    w.write_uint32(height)
    w.write_uint32(width)
    w.write_string("plumb_bob")
    w.write_uint32(0)  # d[] -- no distortion coeffs (RealSense factory-rectified)
    for v in (fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0):  # K[9]
        w.write_float64(v)
    return w.to_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-ip", default="127.0.0.1", help="Zenoh router to connect to (local bridge by default)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--jpeg-quality", type=int, default=65)  # lowered from a
    # typical 80 -- depth is now sharing this same Wi-Fi link (see the PNG
    # compression comment below), so color's own budget needed to shrink too
    ap.add_argument("--depth-hz", type=float, default=5.0, help=(
        "Depth publish rate cap, independent of the camera's own capture "
        "fps. 2026-08-13: publishing depth at the full 15fps alongside "
        "color made reception on the GPU side WORSE than a slower, "
        "CPU-bottlenecked test run -- confirmed genuine Wi-Fi bandwidth "
        "ceiling, not a compute problem (that was ruled out separately, "
        "see the PNG compression comment below). NavDP only infers at "
        "~2-3Hz (README.md), so publishing depth faster than that buys "
        "nothing downstream -- 5Hz leaves headroom while staying well "
        "inside isaac_gui.py's/zenoh_node.py's DEPTH_STALE_S=1.0s "
        "freshness window."))
    args = ap.parse_args()

    config = zenoh.Config()
    config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    print(f"[INFO] Connecting to Zenoh router at {args.pi_ip}:7447 ...", flush=True)
    session = zenoh.open(config)
    pub_color = session.declare_publisher("image_raw/compressed")
    pub_depth = session.declare_publisher("depth_raw")
    pub_info = session.declare_publisher("image_raw/camera_info")
    print("[INFO] Zenoh publishers ready: image_raw/compressed, depth_raw, image_raw/camera_info", flush=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    print("[INFO] Starting RealSense pipeline (color+depth, direct SDK)...", flush=True)
    pipe.start(cfg)
    # profile.get_stream(...).get_intrinsics() ALSO goes through the same
    # XU control channel as the ROS2 wrapper's failing calls (confirmed
    # 2026-08-13: it threw the identical "get_xu(...) UVCIOC_CTRL_QUERY ...
    # Connection timed out" even after a successful pipe.start()) -- so
    # this uses the already-measured constants for this exact D435i unit
    # at 640x480 (see LAUNCH/_backend.sh's BACKEND_FOV comment for the
    # derivation) instead of querying at runtime. Only valid at 640x480;
    # re-measure via realsense-viewer on a machine where the query works
    # (e.g. the GPU workstation) if width/height ever change.
    fx, fy, cx, cy = 607.79, 608.10, 320.38, 238.80
    print(f"[INFO] Started OK. Using known intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}", flush=True)

    running = {"on": True}
    def _stop(signum, frame):
        running["on"] = False
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    frame_count = 0
    depth_count = 0
    consecutive_failures = 0
    # 2026-08-13: a single transient glitch (kernel: "Non-zero status (-71)
    # in video completion handler") is normally recoverable one frame later
    # -- but sometimes it instead pushes the device into the SAME sustained
    # XU-control-timeout failure state that made the ROS2 wrapper hang
    # permanently (confirmed: after one such glitch, EVERY subsequent frame
    # failed for minutes straight, 0 real frames published, CPU dropped to
    # ~10%). Catching and looping forever in that state is worse than
    # crashing -- it keeps the process alive but silently produces nothing,
    # and blocks systemd's Restart=always from giving it the fresh
    # pipe.start() that has reliably recovered every single time tonight
    # (manual kill+restart, port replug, physical replug). So: tolerate a
    # few in a row, but exit and let systemd restart clean past that.
    MAX_CONSECUTIVE_FAILURES = 5
    last_info_pub = 0.0
    last_depth_pub = 0.0
    depth_period = 1.0 / args.depth_hz
    last_status = time.time()
    try:
        while running["on"]:
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
                frames = align.process(frames)
            except RuntimeError as e:
                consecutive_failures += 1
                print(f"[WARN] frame capture/align failed ({consecutive_failures}/"
                      f"{MAX_CONSECUTIVE_FAILURES}), skipping: {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[ERROR] too many consecutive failures -- exiting for a clean "
                          "systemd restart (fresh pipe.start() has always recovered so far)",
                          flush=True)
                    sys.exit(1)
                continue
            consecutive_failures = 0
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok:
                pub_color.put(serialize_compressed_image("jpeg", jpeg.tobytes()))

            # Depth wire size, not just rate, turned out to be the real
            # bottleneck (2026-08-13, measured with a raw byte-counting
            # subscriber, no decode overhead): PNG-compressed 640x480
            # 16-bit depth averaged ~210KB/msg vs color JPEG's ~24KB/msg --
            # PNG's lossless coding doesn't compress real, noisy sensor
            # depth nearly as well as JPEG compresses natural color images.
            # A depth-only rate cap alone (tried 5Hz) still lost ~75% of
            # messages at that size. Fix: halve the pixel count before
            # encoding (~4x smaller PNG) and upsample back to color's
            # resolution on the GPU side (parse_image's "png16" branch,
            # nav_pipeline/zenoh_node.py) so isaac_gui.py's/zenoh_node.py's
            # depth.shape==rgb.shape freshness gate still passes -- nearest-
            # neighbor both directions so no depth values are invented
            # across real object-edge discontinuities.
            now_d = time.time()
            if now_d - last_depth_pub >= depth_period:
                depth_mm = np.asanyarray(depth_frame.get_data())  # uint16, native mm
                h, w = depth_mm.shape
                small = cv2.resize(depth_mm, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                ok, png = cv2.imencode(".png", small)
                if ok:
                    pub_depth.put(serialize_image(small.shape[0], small.shape[1], "png16", png.tobytes()))
                    depth_count += 1
                last_depth_pub = now_d

            now = time.time()
            if now - last_info_pub > 2.0:
                pub_info.put(serialize_camera_info(args.height, args.width, fx, fy, cx, cy))
                last_info_pub = now

            frame_count += 1
            if now - last_status > 10.0:
                print(f"[STATUS] color_published={frame_count} depth_published={depth_count}", flush=True)
                last_status = now
    finally:
        print("[INFO] Shutting down...", flush=True)
        pipe.stop()
        session.close()
        print("[INFO] Stopped cleanly.", flush=True)


if __name__ == "__main__":
    main()
