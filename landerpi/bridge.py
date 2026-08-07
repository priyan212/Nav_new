#!/usr/bin/env python3
"""LanderPi <-> GPU Zenoh bridge. Runs INSIDE the existing, UNMODIFIED
`armpi_pro` ROS1 (Noetic) Docker container that Hiwonder ships this robot
with (see landerpi/README.md for how that stack was discovered). This file
is the only thing added to make the robot speak to Nav_new -- it does not
touch any Hiwonder package, topic, or service.

It speaks the SAME Zenoh contract nav_pipeline/zenoh_node.py already expects
from the old ESP32 rover (see that file's docstring), so pipeline.py,
obstacle_guard.py, home_gui.py, odometry_logger.py etc. need ZERO changes to
drive this robot -- this bridge is the entire adaptation layer.

  ROS1 (rospy) side -- talks to Hiwonder's existing nodes, unmodified:
    subscribes  /usb_cam/image_raw/compressed   (sensor_msgs/CompressedImage)
    publishes   /chassis_control/set_velocity   (chassis_control/SetVelocity:
                float64 velocity mm/s, float64 direction deg, float64 angular
                rad/s -- see armpi_pro/src/chassis_control/scripts/
                chassis_control_node.py's MecanumChassis.set_velocity; this
                bridge only ever drives it "forward/reverse + turn" style
                (direction 90=fwd / 270=rev), never commands lateral strafe,
                so the existing diff-drive-shaped pipeline.py/obstacle_guard.py
                logic applies unchanged)

  Zenoh side -- talks to the GPU, hand-rolled ROS2 CDR wire format (matches
  zenoh_node.py's CDRReader/CDRWriter byte-for-byte; no ROS2 install needed
  anywhere in this bridge):
    publishes   image_raw/compressed   (sensor_msgs/CompressedImage CDR)
                rover/rpm              (std_msgs/Float32MultiArray CDR:
                [left_rpm, right_rpm, imu_heading_deg, imu_calib,
                lateral_m_s]). imu_heading_deg/imu_calib come from an
                OPTIONAL BNO055 IMU wired directly to the Pi's I2C bus (NOT
                part of Hiwonder's stock hardware -- see read_bno055()/
                init_bno055() below and landerpi/README.md's "IMU" section);
                if none is detected at startup these are sent as 0.0/0.0,
                which is always below odometry_logger.py's MAG>=3 trust
                gate, so it cleanly falls back to encoder-only heading with
                no GPU-side change needed either way. lateral_m_s is always
                real (from the wheel encoders, not the IMU): measured
                Mecanum sideways velocity, +left, consumed by
                zenoh_node.py's/home_gui.py's on_rpm handlers and
                odometry_logger.py's holonomic pose update.
    subscribes  cmd_vel                (geometry_msgs/Twist CDR)

Known hardware differences from the old ESP32 rover (see landerpi/README.md
for the full investigation):
  - No wheel-encoder feedback is exposed over ROS on this stock image --
    chassis_control_node.py's I2C motor driver is write-only despite being
    "encoder motors". HOWEVER: the motor driver chip (I2C addr 0x34) DOES
    expose real per-wheel encoder totals at register 0x3C (4x int32 LE,
    confirmed via Hiwonder's own public PX4 driver source -- Hiwonder's own
    ROS code just never reads it). This bridge reads it directly, bypassing
    chassis_control_node.py entirely for odometry (read-only I2C traffic
    alongside its writes -- Linux i2c-dev serializes bus transactions, so
    this is safe to run concurrently). See _read_encoder_totals()/_rpm_loop
    for the empirically-calibrated channel mixing (the raw channel order/
    polarity does NOT match chassis_control_node.py's write-side motor
    indices -- verified live via isolated forward-only and turn-only test
    pulses, cross-checked for <1.5% cross-talk between the two).
  - This chassis is Mecanum, not 6WD skid-steer, but since we only ever
    command forward/reverse + turn (no strafe), real lateral slip is
    invisible to odometry_logger.py's diff-drive-shaped pose model (same
    limitation the old rover has on skid turns) -- we recover real (v, w)
    from the encoders and re-encode as "virtual" left/right wheel RPM using
    the exact same WHEEL_RADIUS_M / TRACK_WIDTH_M constants
    odometry_logger.py already has. Those numbers describe the OLD rover's
    real geometry; here they are purely a shared encoding constant with no
    physical meaning -- they just need to match on both ends of the
    round-trip, and reusing the existing ones means zero GPU-side changes.
  - No depth camera in this ROS1 package set (usb_cam only) -- depth_raw is
    never published, so pipeline.py's monocular Depth-Anything-V2 fallback
    kicks in, identical to the real rover (also RGB-only).
  - chassis_control_node.py has NO cmd_vel timeout of its own (unlike the
    ESP32 firmware's 500ms auto-zero) -- it will drive forever on the last
    command if the Zenoh link/GPU/Wi-Fi dies. This bridge's watchdog_loop
    replaces that missing safety net so the new bot doesn't regress on
    safety versus the old one.

Runs under a Python 3.10 venv (~/nav_new_bridge/venv310), NOT the
container's system Python 3.8 -- current eclipse-zenoh's PyO3 bindings need
CPython >=3.9 (see landerpi/README.md "Bridge dependencies" for why, and
the one-time setup). rospy itself is pure Python and works fine under 3.10
even though ROS Noetic's own system Python is 3.8.

Run inside the container (see landerpi/README.md / LAUNCH/launch_bot.sh
--hiwonder for the full deploy flow):
  docker exec -d -u ubuntu -w /home/ubuntu armpi_pro /bin/bash -c \
    "source /home/ubuntu/armpi_pro/src/armpi_pro_bringup/scripts/source_env.bash \
     /home/ubuntu/nav_new_bridge/venv310/bin/python3.10 /home/ubuntu/nav_new_bridge/bridge.py"
"""

import math
import os
import struct
import sys
import threading
import time

import rospy
import smbus2
from sensor_msgs.msg import CompressedImage
from chassis_control.msg import SetVelocity

try:
    import zenoh
except ImportError:
    print("ERROR: eclipse-zenoh not installed in this container's Python "
          "(pip3 install --user eclipse-zenoh==1.9.0 -- must match the GPU "
          "side's version, see landerpi/README.md for the Rust-toolchain "
          "build steps this required).")
    sys.exit(1)

# Must match nav_pipeline/odometry_logger.py -- shared encoding constant
# only, see module docstring above (this chassis is not diff-drive).
WHEEL_RADIUS_M = 0.056
TRACK_WIDTH_M = 0.345

CMD_VEL_TIMEOUT_S = 0.5   # ESP32-firmware parity -- see module docstring
RPM_PUBLISH_HZ = 10.0
LISTEN_PORT = 7447

# Real per-wheel encoder feedback (see module docstring) -- undocumented by
# Hiwonder's own ROS code but confirmed working via Hiwonder's public PX4
# driver source (register 0x3C, "total pulse value of 4 encoder motors").
ENCODER_I2C_BUS = 1
ENCODER_I2C_ADDR = 0x34
ENCODER_REG = 0x3C
# Physical wheel geometry (chassis_control_node.py's MecanumChassis
# defaults) -- used ONLY to convert raw pulses to real mm/s. Unrelated to
# WHEEL_RADIUS_M/TRACK_WIDTH_M above, which are a pure encoding constant.
WHEEL_DIAMETER_MM = 96.5
PULSE_PER_CYCLE = 44 * 178          # encoder CPR * gearbox ratio
MM_PER_PULSE = (math.pi * WHEEL_DIAMETER_MM) / PULSE_PER_CYCLE
HALF_WHEELBASE_MM = 110 + 97.5      # MecanumChassis a+b


def read_encoder_totals(bus: smbus2.SMBus):
    """Raw cumulative pulse totals for the 4 wheel encoders. Channel order/
    polarity is whatever this specific board returns over I2C -- NOT the
    same as chassis_control_node.py's write-side motor indices (verified
    live: isolating a pure-forward test pulse from a pure-turn test pulse
    needed a different channel grouping than the write-side mixing matrix
    predicts). See _rpm_loop for the empirically-calibrated combination."""
    data = bus.read_i2c_block_data(ENCODER_I2C_ADDR, ENCODER_REG, 16)
    return struct.unpack("<4i", bytes(data))


# Optional BNO055 IMU (see landerpi/README.md "IMU" section) -- NOT part of
# Hiwonder's stock hardware/firmware (that path is confirmed dead, see
# README), a separate breakout wired directly to the Pi's own I2C-1 bus
# (shares the bus with the encoder chip at 0x34 above -- fine, Linux i2c-dev
# serializes transactions). Same chip family as the old ESP32 rover's BNO055
# (esp32/rover_6wd_complete.ino), so odometry_logger.py's existing IMU
# heading-fusion path (calib-gated on the MAG sub-score) applies unchanged
# -- this is the only part of the bridge that needed zero new GPU-side code.
IMU_I2C_ADDR = 0x28
IMU_REG_CHIP_ID = 0x00
IMU_REG_CALIB_STAT = 0x35
IMU_REG_SYS_TRIGGER = 0x3F
IMU_REG_PWR_MODE = 0x3E
IMU_REG_OPR_MODE = 0x3D
IMU_REG_EUL_HEADING_LSB = 0x1A
IMU_CHIP_ID_EXPECTED = 0xA0
IMU_MODE_CONFIG = 0x00
IMU_MODE_NDOF = 0x0C   # full 9-DOF sensor fusion
IMU_RETRY_PERIOD_S = 5.0   # re-probe if not detected at startup (e.g. still being wired up)

# Accel/mag/gyro offset + radius calibration profile, 22 bytes -- readable in
# any operating mode, but only WRITABLE in CONFIG_MODE (Bosch BNO055
# datasheet 3.6.4). init_bno055() never previously saved/restored this, so
# every bridge restart forced the magnetometer to relearn calibration from
# zero -- live-tested (2026-08-07) driving the robot through 180+ degrees of
# turns in both directions across 25s and MAG calibration never moved off 2,
# staying below odometry_logger.py's MAG>=3 trust gate the whole time (pure
# yaw motion on a flat floor may not be enough for BNO055's calibration
# algorithm regardless -- persistence at least removes "just restarted" as a
# recurring cause).
IMU_REG_CALIB_PROFILE_START = 0x55
IMU_CALIB_PROFILE_LEN = 22
CALIB_SAVE_PERIOD_S = 15.0
BNO055_CALIB_FILE = os.path.expanduser("~/nav_new_bridge/bno055_calib.bin")


def load_saved_calib() -> "bytes | None":
    try:
        with open(BNO055_CALIB_FILE, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    return data if len(data) == IMU_CALIB_PROFILE_LEN else None


def save_calib_profile(bus: smbus2.SMBus):
    """Persist the BNO055's current calibration profile to disk (atomic
    write) so a future bridge restart can restore it in init_bno055()
    instead of relearning from scratch. Safe to call at any calibration
    completeness -- even a partially-learned profile (e.g. MAG stuck at 2)
    is a strictly better starting point than none. Called periodically from
    _rpm_loop and once more on rospy shutdown."""
    try:
        data = bytes(bus.read_i2c_block_data(IMU_I2C_ADDR, IMU_REG_CALIB_PROFILE_START,
                                              IMU_CALIB_PROFILE_LEN))
    except Exception:
        return
    try:
        os.makedirs(os.path.dirname(BNO055_CALIB_FILE), exist_ok=True)
        tmp = BNO055_CALIB_FILE + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, BNO055_CALIB_FILE)
    except OSError as e:
        rospy.logwarn_throttle(60, "BNO055 calib save failed: %s", e)


def init_bno055(bus: smbus2.SMBus) -> bool:
    """Bring the BNO055 up into NDOF fusion mode. Returns False (heading
    fusion just stays unavailable, same as before this chip existed) if it
    isn't present/responding -- e.g. not yet connected, or briefly
    disconnected during hand-wiring.

    Restores a previously-saved calibration profile (see save_calib_profile)
    if one exists, so calibration progress survives a bridge restart instead
    of starting from zero every time."""
    try:
        if bus.read_byte_data(IMU_I2C_ADDR, IMU_REG_CHIP_ID) != IMU_CHIP_ID_EXPECTED:
            return False
        bus.write_byte_data(IMU_I2C_ADDR, IMU_REG_OPR_MODE, IMU_MODE_CONFIG); time.sleep(0.025)
        bus.write_byte_data(IMU_I2C_ADDR, IMU_REG_PWR_MODE, 0x00); time.sleep(0.01)
        bus.write_byte_data(IMU_I2C_ADDR, IMU_REG_SYS_TRIGGER, 0x00); time.sleep(0.01)
        saved = load_saved_calib()
        if saved is not None:
            try:
                bus.write_i2c_block_data(IMU_I2C_ADDR, IMU_REG_CALIB_PROFILE_START, list(saved))
                rospy.loginfo("BNO055: restored saved calibration profile from %s", BNO055_CALIB_FILE)
            except Exception as e:
                rospy.logwarn("BNO055: failed to restore saved calibration (%s) -- "
                               "continuing, will relearn from scratch", e)
        bus.write_byte_data(IMU_I2C_ADDR, IMU_REG_OPR_MODE, IMU_MODE_NDOF); time.sleep(0.025)
        return True
    except Exception:
        return False


def read_bno055(bus: smbus2.SMBus):
    """(heading_deg, packed_calib) -- packed_calib matches the ESP32
    firmware's own sys*1000+gyr*100+acc*10+mag packing (see
    odometry_logger.py's decode_calib()/_mag_calib_ok()), so no GPU-side
    change was needed for this part. heading_deg is only numerically stable
    once the chip is mounted reasonably flat/level -- Euler yaw hits gimbal
    lock near +-90deg pitch/roll (verified live while hand-held before
    mounting: quaternion output stayed smooth through the same window that
    produced ~90deg heading jumps -- a mounting-angle artifact, not a wiring
    or calibration problem)."""
    data = bus.read_i2c_block_data(IMU_I2C_ADDR, IMU_REG_EUL_HEADING_LSB, 2)
    heading = struct.unpack("<h", bytes(data))[0] / 16.0
    calib = bus.read_byte_data(IMU_I2C_ADDR, IMU_REG_CALIB_STAT)
    sys_c, gyr_c, acc_c, mag_c = (calib >> 6) & 3, (calib >> 4) & 3, (calib >> 2) & 3, calib & 3
    return heading, sys_c * 1000 + gyr_c * 100 + acc_c * 10 + mag_c


# ================================================================
#  CDR helpers -- mirror nav_pipeline/zenoh_node.py's CDRReader/CDRWriter
#  byte-for-byte. Duplicated (not imported) so this file has no dependency
#  on the GPU-side repo layout and can be deployed standalone to the Pi.
# ================================================================
class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n: int):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_int32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<i", v)

    def write_uint32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_float32(self, v: float):
        self._align(4)
        self.buf += struct.pack("<f", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self.buf += encoded

    def write_sequence_uint8(self, data: bytes):
        self.write_uint32(len(data))
        self.buf += data

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


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

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8
        return v


def serialize_compressed_image(jpeg_bytes: bytes, frame_id: str = "usb_cam") -> bytes:
    """sensor_msgs/CompressedImage -> CDR. Mirrors parse_compressed_image
    in zenoh_node.py: header (stamp+frame_id), format string, uint8[] data."""
    w = CDRWriter()
    now = time.time()
    w.write_int32(int(now))
    w.write_uint32(int((now % 1) * 1e9))
    w.write_string(frame_id)
    w.write_string("jpeg")
    w.write_sequence_uint8(jpeg_bytes)
    return w.to_bytes()


def serialize_float32_multiarray(values) -> bytes:
    """std_msgs/Float32MultiArray -> CDR. Mirrors parse_float32_multiarray
    in zenoh_node.py: dim_count=0 (no labels), data_offset=0, then the data."""
    w = CDRWriter()
    w.write_uint32(0)             # dim_count
    w.write_uint32(0)             # data_offset
    w.write_uint32(len(values))   # sequence length
    for v in values:
        w.write_float32(float(v))
    return w.to_bytes()


def parse_twist(cdr_data: bytes):
    """geometry_msgs/Twist CDR -> (linear_x, angular_z). Mirrors
    serialize_twist in zenoh_node.py: 6x float64, linear xyz + angular xyz."""
    r = CDRReader(cdr_data)
    linear_x = r.read_float64()
    r.read_float64(); r.read_float64()          # linear.y, linear.z (unused)
    r.read_float64(); r.read_float64()          # angular.x, angular.y (unused)
    angular_z = r.read_float64()
    return linear_x, angular_z


# ================================================================
#  Bridge
# ================================================================
class LanderPiBridge:
    def __init__(self):
        rospy.init_node("nav_new_zenoh_bridge", anonymous=True)

        cfg = zenoh.Config()
        cfg.insert_json5("listen/endpoints", f'["tcp/0.0.0.0:{LISTEN_PORT}"]')
        self.session = zenoh.open(cfg)

        self.pub_img = self.session.declare_publisher("image_raw/compressed")
        self.pub_rpm = self.session.declare_publisher("rover/rpm")
        self.sub_cmd = self.session.declare_subscriber("cmd_vel", self._on_cmd_vel)

        self.set_vel_pub = rospy.Publisher(
            "/chassis_control/set_velocity", SetVelocity, queue_size=1)
        rospy.Subscriber(
            "/usb_cam/image_raw/compressed", CompressedImage, self._on_camera, queue_size=1)

        self._lock = threading.Lock()
        self._last_cmd = (0.0, 0.0)   # (linear_x m/s, angular_z rad/s) -- last CMD RECEIVED
        self._last_cmd_t = 0.0
        self._frame_count = 0

        threading.Thread(target=self._rpm_loop, daemon=True).start()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        rospy.loginfo(
            "nav_new bridge up on tcp/0.0.0.0:%d | /usb_cam -> image_raw/compressed | "
            "cmd_vel -> /chassis_control/set_velocity | synthetic rover/rpm @ %.0fHz | "
            "%.1fs cmd_vel watchdog", LISTEN_PORT, RPM_PUBLISH_HZ, CMD_VEL_TIMEOUT_S)

    # ---------------- ROS1 -> Zenoh ---------------- #
    def _on_camera(self, msg: CompressedImage):
        try:
            self.pub_img.put(serialize_compressed_image(bytes(msg.data)))
            self._frame_count += 1
        except Exception as e:
            rospy.logwarn_throttle(5, "camera publish failed: %s", e)

    def _rpm_loop(self):
        """Real closed-loop odometry from the I2C encoder register (see
        module docstring) -- NOT gated on cmd_vel freshness like the old
        open-loop version was; if the watchdog zeroes the motors, the real
        encoders naturally show the real (possibly-coasting) motion instead
        of an assumption. Also brings up the optional BNO055 IMU (see
        read_bno055 docstring) if one is connected -- shares the same I2C
        bus/loop rather than a separate thread, since bus transactions are
        serialized either way.
        """
        period = 1.0 / RPM_PUBLISH_HZ
        bus = smbus2.SMBus(ENCODER_I2C_BUS)
        imu_ok = init_bno055(bus)
        imu_last_probe_t = time.time()
        last_calib_save_t = time.time()
        rospy.on_shutdown(lambda: imu_ok and save_calib_profile(bus))
        rospy.loginfo("BNO055 IMU: %s", "detected, NDOF fusion enabled" if imu_ok
                       else "not detected -- retrying every %.0fs (odometry falls back to "
                            "encoder-only meanwhile)" % IMU_RETRY_PERIOD_S)
        prev = None
        prev_t = None
        while not rospy.is_shutdown():
            try:
                r = read_encoder_totals(bus)
                t = time.time()
                if imu_ok and t - last_calib_save_t >= CALIB_SAVE_PERIOD_S:
                    last_calib_save_t = t
                    save_calib_profile(bus)
                if prev is not None:
                    dt = t - prev_t
                    if dt > 0:
                        d1, d2, d3, d4 = (r[i] - prev[i] for i in range(4))
                        # Empirically-calibrated channel mixing (see
                        # read_encoder_totals docstring): isolates forward,
                        # turn, AND lateral (this chassis is Mecanum -- a 3rd
                        # real DOF that a naive 2-axis diff-drive-style
                        # reconstruction silently drops, which was the actual
                        # cause of "bogus" routes even after switching from
                        # open-loop to real encoder feedback: real sideways
                        # slip/motion was being measured then discarded).
                        # Each of the 3 combinations below isolates cleanly
                        # against BOTH other test motions (<1.5% cross-talk),
                        # verified via 3 isolated live test pulses (forward,
                        # turn, strafe). Signs already match the Twist/
                        # REP103 convention (+linear_x=fwd, +angular_z=left,
                        # +lateral=left) -- no extra inversion needed.
                        vx_pulses = (d1 - d2 - d3 + d4) / 4.0
                        vp_pulses = (d1 - d2 + d3 - d4) / 4.0
                        vy_pulses = (d1 + d2 + d3 + d4) / 4.0
                        v = (vx_pulses / dt) * MM_PER_PULSE / 1000.0  # m/s
                        vp_mm_s = (vp_pulses / dt) * MM_PER_PULSE
                        w = vp_mm_s / HALF_WHEELBASE_MM                # rad/s
                        lateral = (vy_pulses / dt) * MM_PER_PULSE / 1000.0  # m/s, +left

                        v_left = v - w * TRACK_WIDTH_M / 2.0
                        v_right = v + w * TRACK_WIDTH_M / 2.0
                        rpm_left = v_left / (2 * math.pi * WHEEL_RADIUS_M) * 60.0
                        rpm_right = v_right / (2 * math.pi * WHEEL_RADIUS_M) * 60.0

                        if not imu_ok and t - imu_last_probe_t >= IMU_RETRY_PERIOD_S:
                            imu_last_probe_t = t
                            imu_ok = init_bno055(bus)
                            if imu_ok:
                                rospy.loginfo("BNO055 IMU: connected, NDOF fusion enabled")

                        imu_heading, imu_calib = 0.0, 0.0  # 0.0 calib -> always below the
                        if imu_ok:                          # MAG>=3 gate, i.e. "no IMU" fallback
                            try:
                                imu_heading, imu_calib = read_bno055(bus)
                            except Exception as e:
                                rospy.logwarn_throttle(5, "BNO055 read failed: %s", e)

                        # [left_rpm, right_rpm, imu_heading_deg, imu_calib,
                        # lateral_m_s] -- matches zenoh_node.py/home_gui.py's
                        # on_rpm 5-element parse.
                        self.pub_rpm.put(serialize_float32_multiarray(
                            [rpm_left, rpm_right, imu_heading, imu_calib, lateral]))
                prev, prev_t = r, t
            except Exception as e:
                rospy.logwarn_throttle(5, "encoder read failed: %s", e)
            time.sleep(period)

    # ---------------- Zenoh -> ROS1 ---------------- #
    def _on_cmd_vel(self, sample):
        try:
            lin, ang = parse_twist(bytes(sample.payload))
        except Exception as e:
            rospy.logwarn_throttle(5, "cmd_vel parse failed: %s", e)
            return
        with self._lock:
            self._last_cmd = (lin, ang)
            self._last_cmd_t = time.time()
        self._drive(lin, ang)

    def _drive(self, lin: float, ang: float):
        msg = SetVelocity()
        msg.velocity = abs(lin) * 1000.0             # m/s -> mm/s
        msg.direction = 90.0 if lin >= 0 else 270.0   # this chassis' fwd axis; verify empirically
        msg.angular = ang                             # rad/s, matches MecanumChassis.set_velocity directly
        self.set_vel_pub.publish(msg)

    def _watchdog_loop(self):
        # chassis_control_node.py has no cmd_vel timeout of its own (see
        # module docstring) -- this replaces that missing safety net.
        while not rospy.is_shutdown():
            time.sleep(0.1)
            with self._lock:
                age = time.time() - self._last_cmd_t
            if age > CMD_VEL_TIMEOUT_S:
                self._drive(0.0, 0.0)

    def spin(self):
        try:
            rospy.spin()
        finally:
            self._drive(0.0, 0.0)


if __name__ == "__main__":
    LanderPiBridge().spin()
