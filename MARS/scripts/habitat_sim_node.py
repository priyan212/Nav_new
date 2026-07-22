#!/usr/bin/env python3
"""Habitat Mars sim node — replaces Isaac Sim in the Nav_new Zenoh loop.

Runs the Marsyard terrain in headless habitat-sim, drives a kinematic rover
agent from cmd_vel, and speaks the same Zenoh contract as the Isaac bridge
scripts, so nav_pipeline (zenoh_node / GUIs) works unchanged:

Publishes (CDR):
  image_raw/compressed   sensor_msgs/CompressedImage (JPEG, rover camera)
  depth_raw              sensor_msgs/Image (32FC1, meters — perfect sim depth)
  mars/pose              std_msgs/String (JSON {"x","z","yaw","t"} habitat world)
Subscribes (CDR):
  cmd_vel, rt/cmd_vel    geometry_msgs/Twist (linear.x m/s, angular.z rad/s,
                         ROS convention: +angular = turn left). Zeroed if no
                         message within 0.5 s (mirrors rover firmware watchdog).
  mars/reset             std_msgs/String "x,z,yaw" (or empty = default start)

Frames: habitat world is Y-up; yaw=0 faces -Z, positive yaw turns left (CCW
from above), matching ROS. Ground height comes from a bullet raycast against
the stage mesh each step — no heightmap axis-mapping involved.

Run (mars_habitat env, from MARS/mars-habitatsim):
    python ../scripts/habitat_sim_node.py [--rocks rock_envs/run1/rock_field.json]
"""
import argparse
import json
import math
import os
import struct
import sys
import time
from threading import Lock

import numpy as np
import quaternion  # noqa: F401  (numpy-quaternion, needed for habitat AgentState)

import cv2
import habitat_sim
import zenoh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "mars-habitatsim"))
DEFAULT_SCENE = os.path.join(REPO, "assets", "marsyard2022.glb")

YARD_LIMIT = 23.0          # keep the rover inside the 50x50 yard (2 m margin)
CMD_TIMEOUT_S = 0.5        # rover-firmware-style watchdog


# ---------------------------------------------------------------- CDR helpers
class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_int32(self, v):
        self._align(4)
        self.buf += struct.pack("<i", v)

    def write_uint32(self, v):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_uint8(self, v):
        self.buf += struct.pack("<B", v)

    def write_string(self, s):
        b = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(b))
        self.buf += b

    def write_bytes_seq(self, data):
        self.write_uint32(len(data))
        self.buf += data

    def to_bytes(self):
        return bytes(self.buf)


def _write_header(w: CDRWriter, frame_id: str = "rover_camera"):
    t = time.time()
    w.write_int32(int(t))
    w.write_uint32(int((t % 1.0) * 1e9))
    w.write_string(frame_id)


def serialize_compressed_image(jpeg_bytes: bytes) -> bytes:
    w = CDRWriter()
    _write_header(w)
    w.write_string("jpeg")
    w.write_bytes_seq(jpeg_bytes)
    return w.to_bytes()


def serialize_depth_image(depth: np.ndarray) -> bytes:
    h, wd = depth.shape
    w = CDRWriter()
    _write_header(w)
    w.write_uint32(h)
    w.write_uint32(wd)
    w.write_string("32FC1")
    w.write_uint8(0)                 # is_bigendian
    w.write_uint32(wd * 4)           # step
    w.write_bytes_seq(depth.astype("<f4").tobytes())
    return w.to_bytes()


def serialize_string(s: str) -> bytes:
    w = CDRWriter()
    w.write_string(s)
    return w.to_bytes()


def parse_twist(data: bytes):
    """geometry_msgs/Twist: 6 little-endian float64 after the 4-byte header."""
    if len(data) < 52:
        return None
    vals = struct.unpack_from("<6d", data, 4)
    return vals[0], vals[5]          # linear.x, angular.z


def parse_string(data: bytes) -> str:
    (length,) = struct.unpack_from("<I", data, 4)
    return data[8 : 8 + length - 1].decode("utf-8", errors="replace")


# ---------------------------------------------------------------- simulation
def make_sim(scene: str, width: int, height: int, hfov: float, cam_height: float):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.enable_physics = True    # needed for cast_ray ground queries

    def cam(uuid, sensor_type):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = sensor_type
        s.resolution = [height, width]
        s.position = [0.0, cam_height, 0.0]
        s.hfov = hfov
        return s

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        cam("rgb", habitat_sim.SensorType.COLOR),
        cam("depth", habitat_sim.SensorType.DEPTH),
    ]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def load_rocks(sim, manifest_path: str):
    """Register the pre-placed rock meshes (render-only, non-collidable, so
    ground raycasts still hit the terrain). Mesh paths in the manifest are
    absolute from the original author's machine — remap next to the manifest."""
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    payload = json.loads(open(manifest_path).read())
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    n = 0
    for rock in payload["rocks"]:
        mesh = os.path.join(manifest_dir, "rocks", os.path.basename(rock["mesh_path"]))
        if not os.path.exists(mesh):
            print(f"[WARN] rock mesh missing: {mesh}")
            continue
        template = otm.create_new_template(mesh)
        template.render_asset_handle = mesh
        template.collision_asset_handle = mesh
        template.is_collidable = False
        tid = otm.register_template(template, f"rock_{rock['id']}")
        obj = rom.add_object_by_template_handle(otm.get_template_handle_by_id(tid))
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.collidable = False
        n += 1
    print(f"[INFO] registered {n} rocks from {manifest_path}")


def ground_height(sim, x: float, z: float):
    """Raycast straight down against the (collidable) terrain stage."""
    ray = habitat_sim.geo.Ray()
    ray.origin = np.array([x, 50.0, z], dtype=np.float32)
    ray.direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    hits = sim.cast_ray(ray, max_distance=200.0)
    if hits.has_hits():
        return float(hits.hits[0].point[1])
    return None


class RoverAgent:
    def __init__(self, sim, x: float, z: float, yaw: float):
        self.sim = sim
        self.agent = sim.get_agent(0)
        self.lock = Lock()
        self.cmd = (0.0, 0.0)
        self.cmd_t = 0.0
        self.reset(x, z, yaw)

    def reset(self, x: float, z: float, yaw: float):
        with self.lock:
            self.x, self.z, self.yaw = x, z, yaw
            self.cmd = (0.0, 0.0)
        self._apply()

    def set_cmd(self, lin: float, ang: float):
        with self.lock:
            self.cmd = (lin, ang)
            self.cmd_t = time.time()

    def step(self, dt: float):
        with self.lock:
            lin, ang = self.cmd
            stale = time.time() - self.cmd_t > CMD_TIMEOUT_S
        if stale:
            lin, ang = 0.0, 0.0
        self.yaw += ang * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)   # yaw=0 faces -Z
        self.x = float(np.clip(self.x + lin * fx * dt, -YARD_LIMIT, YARD_LIMIT))
        self.z = float(np.clip(self.z + lin * fz * dt, -YARD_LIMIT, YARD_LIMIT))
        self._apply()

    def _apply(self):
        y = ground_height(self.sim, self.x, self.z)
        if y is None:
            y = getattr(self, "_last_y", 0.0)
        self._last_y = y
        state = habitat_sim.AgentState()
        state.position = np.array([self.x, y, self.z], dtype=np.float32)
        state.rotation = np.quaternion(math.cos(self.yaw / 2), 0, math.sin(self.yaw / 2), 0)
        self.agent.set_state(state, reset_sensors=False)

    def pose_json(self) -> str:
        return json.dumps({"x": round(self.x, 4), "z": round(self.z, 4),
                           "yaw": round(self.yaw, 4), "t": time.time()})


def main():
    ap = argparse.ArgumentParser(description="Habitat Mars sim node (Zenoh)")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--rocks", default=None, help="rock_field.json manifest (optional)")
    ap.add_argument("--hz", type=float, default=10.0, help="sim/publish rate")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--cam-height", type=float, default=0.4, help="camera height above ground (m)")
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=8.0)
    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--snapshot", default=None,
                    help="save an overhead RGB view to this path at startup (sanity check)")
    args = ap.parse_args()

    print(f"[INFO] loading scene: {args.scene}")
    sim = make_sim(args.scene, args.width, args.height, args.fov, args.cam_height)
    if args.rocks:
        load_rocks(sim, args.rocks)

    if args.snapshot:
        st = habitat_sim.AgentState()
        st.position = np.array([0.0, 30.0, 0.0], dtype=np.float32)
        st.rotation = np.quaternion(math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0)
        sim.get_agent(0).set_state(st, reset_sensors=False)
        obs = sim.get_sensor_observations()
        cv2.imwrite(args.snapshot, cv2.cvtColor(obs["rgb"][..., :3], cv2.COLOR_RGB2BGR))
        print(f"[INFO] snapshot saved: {args.snapshot}")

    rover = RoverAgent(sim, args.start_x, args.start_z, args.start_yaw)
    print(f"[INFO] rover at ({args.start_x}, {args.start_z}) yaw={args.start_yaw}")

    session = zenoh.open(zenoh.Config())
    pub_img = session.declare_publisher("image_raw/compressed")
    pub_depth = session.declare_publisher("depth_raw")
    pub_pose = session.declare_publisher("mars/pose")

    def on_cmd(sample):
        v = parse_twist(bytes(sample.payload))
        if v is not None:
            rover.set_cmd(*v)

    def on_reset(sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if text:
                x, z, yaw = (float(t) for t in text.split(","))
            else:
                x, z, yaw = args.start_x, args.start_z, args.start_yaw
            rover.reset(x, z, yaw)
            print(f"[INFO] reset to ({x:.1f}, {z:.1f}) yaw={yaw:.2f}")
        except Exception as e:
            print(f"[WARN] bad reset msg: {e}")

    subs = [session.declare_subscriber(k, on_cmd) for k in ("cmd_vel", "rt/cmd_vel")]
    subs.append(session.declare_subscriber("mars/reset", on_reset))
    print("[INFO] zenoh up: pub image_raw/compressed, depth_raw, mars/pose | sub cmd_vel, mars/reset")

    period = 1.0 / args.hz
    frames = 0
    last_status = time.time()
    try:
        while True:
            t0 = time.time()
            rover.step(period)
            obs = sim.get_sensor_observations()

            rgb = obs["rgb"][..., :3]
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok:
                pub_img.put(serialize_compressed_image(jpg.tobytes()))
            pub_depth.put(serialize_depth_image(obs["depth"]))
            pub_pose.put(serialize_string(rover.pose_json()))

            frames += 1
            if time.time() - last_status > 10.0:
                lin, ang = rover.cmd
                print(f"[STATUS] frames={frames} pose=({rover.x:.2f},{rover.z:.2f},"
                      f"{rover.yaw:.2f}) cmd=({lin:.2f},{ang:.2f})")
                last_status = time.time()

            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[INFO] shutting down")
    finally:
        session.close()
        sim.close()


if __name__ == "__main__":
    main()
