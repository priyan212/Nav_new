"""Loopback test for the Zenoh node: fake camera in -> cmd_vel out, no robot.

Publishes the saved rover frame on `image_raw` at 10 Hz and prints every
`cmd_vel` / `omnivla/explanation` received. Run the node in another terminal
(or let this script offer to check topics only).

  Terminal A:  python -m nav_pipeline.zenoh_node --target "trash bin"
  Terminal B:  python scripts/test_zenoh_loopback.py
"""

import struct
import sys
import time

import numpy as np
import zenoh
from PIL import Image


def serialize_ros_image(rgb: np.ndarray) -> bytes:
    """numpy RGB -> sensor_msgs/Image CDR (rgb8)."""
    h, w_, _ = rgb.shape
    buf = bytearray(b"\x00\x01\x00\x00")
    base = 4

    def align(n):
        rem = (len(buf) - base) % n
        if rem:
            buf.extend(b"\x00" * (n - rem))

    def u32(v):
        align(4)
        buf.extend(struct.pack("<I", v))

    def i32(v):
        align(4)
        buf.extend(struct.pack("<i", v))

    def string(s):
        e = s.encode() + b"\x00"
        u32(len(e))
        buf.extend(e)

    i32(0); u32(0); string("camera")      # header
    u32(h); u32(w_)
    string("rgb8")
    buf.append(0)                          # is_bigendian
    u32(w_ * 3)                            # step
    data = rgb.astype(np.uint8).tobytes()
    u32(len(data))
    buf.extend(data)
    return bytes(buf)


def parse_twist(cdr: bytes):
    vals = struct.unpack_from("<6d", cdr, 4)
    return vals[0], vals[5]


def parse_string_msg(cdr: bytes) -> str:
    (length,) = struct.unpack_from("<I", cdr, 4)
    return cdr[8 : 8 + length - 1].decode(errors="replace")


def main():
    rgb = np.array(Image.open("data/current_img.jpg").convert("RGB"))
    payload = serialize_ros_image(rgb)

    session = zenoh.open(zenoh.Config())
    pub = session.declare_publisher("image_raw")

    got = {"cmd": 0, "expl": 0}

    def on_cmd(sample):
        lin, ang = parse_twist(bytes(sample.payload))
        got["cmd"] += 1
        if got["cmd"] % 10 == 1:
            print(f"  cmd_vel #{got['cmd']}: lin={lin:.3f} ang={ang:+.3f}")

    def on_expl(sample):
        got["expl"] += 1
        if got["expl"] % 5 == 1:
            print(f"  explanation: {parse_string_msg(bytes(sample.payload))}")

    session.declare_subscriber("cmd_vel", on_cmd)
    session.declare_subscriber("omnivla/explanation", on_expl)

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    print(f"publishing image_raw at 10 Hz for {duration:.0f}s ...")
    t_end = time.time() + duration
    while time.time() < t_end:
        pub.put(payload)
        time.sleep(0.1)

    print(f"done. received cmd_vel={got['cmd']} explanations={got['expl']}")
    session.close()
    sys.exit(0 if got["cmd"] > 0 and got["expl"] > 0 else 1)


if __name__ == "__main__":
    main()
