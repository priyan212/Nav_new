#!/usr/bin/env python3
"""
Camera bringup for the rover using an Intel RealSense D435i (replaces the
old Logitech webcam / v4l2_camera_node, see camera_only_bringup.launch.py
which this supersedes for --rover).

Currently color-only. Depth + IMU are wired below (params + remappings) but
DISABLED -- see the block comment at enable_depth for why: enabling them
causes a reproducible hang on THIS Pi (realsense2_camera_node drops to ~0-3%
CPU, stops publishing on every stream including color) due to a USB
xHCI-controller-level issue between this Pi and the D435i under combined
color+depth streaming, not a launch-param mistake. Kernel log always shows
the same signature: "usb 2-1: Failed to query (SET_CUR/GET_LEN) UVC control
1/7 on unit 3: -110/-32 (exp. ...)" (via `sudo dmesg`).

Ruled out already (2026-08-13, both forms of the standard fix):
  - disabling USB autosuspend for the device (power/control: auto->on)
  - usbfs_memory_mb 16->1000, BOTH at runtime (sysfs) AND at boot
    (/boot/firmware/cmdline.txt + reboot -- confirmed active via
    /sys/module/usbcore/parameters/usbfs_memory_mb reading 1000 post-reboot)
None of these changed the failure. This does not look like a usbfs
buffer-size issue. Also ruled out (2026-08-13): reducing the depth profile all the way down to
424x240@6fps (from 640x480@15) -- identical failure. This is NOT a
bandwidth/buffer-size issue; it's specifically that this Pi 4's USB3
controller (VL805 -- Pi 4 has a well-documented, community-acknowledged
history of exactly this class of issue with RealSense D400-series combined
streams) can't reliably sustain the coordinated control-channel polling
needed once a second UVC stream (depth) joins color, regardless of size.
Also ruled out (2026-08-13): moving the camera to the Pi's other USB3 port
(2-1 -> 2-2) -- identical failure signature on the new port too, so it's
not a single-port fault. Remaining untried options, in order: (1) a powered
USB3 hub between the Pi and camera -- most-cited community fix for this
exact signature, (2) a different/better-shielded USB-C cable (not yet
tried -- only the port was swapped), (3) longer-shot: a Pi 5 or other SBC
with a more robust USB3 controller. Try those before flipping
enable_depth/enable_gyro/enable_accel back to True again.

The GPU-side pipeline (nav_pipeline/pipeline.py) already prefers a fresh
real depth frame over its monocular Depth Anything V2 estimate the instant
one arrives (DEPTH_STALE_S gate) -- turning depth on here (once the USB
issue above is resolved) is what activates that path; no further GPU-side
depth-preference code changes are needed for it.

Topic remapping: this realsense2_camera_node build namespaces every topic
under camera_name/node_name/stream (e.g. /camera/realsense_camera/color/
image_raw), NOT just camera_name/stream -- confirmed empirically via
`ros2 topic list` after restart on 2026-08-13 (an earlier version of this
file assumed the shorter form, which was a silent no-op -- see git
history). If realsense2_camera_node's exact topic names ever differ from
what's listed below (this has varied across releases, e.g.
aligned_depth_to_color/imu naming), re-verify with `ros2 topic list` and
fix the remap source on the LEFT side only -- the Zenoh-facing key on the
right must stay in sync with nav_pipeline/zenoh_node.py's KEYS lists
(CAMERA_KEYS/CAMERA_COMPRESSED_KEYS/DEPTH_KEYS/CAMERA_INFO_KEYS/IMU_KEYS).

Usage on Pi:
  source /opt/ros/humble/setup.bash
  ros2 launch realsense_only_bringup.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense_camera",
        output="screen",
        parameters=[{
            "camera_name": "camera",
            "camera_namespace": "",
            "enable_color": True,
            "rgb_camera.color_profile": "640,480,15",
            # See module docstring -- depth causes a reproducible hang on
            # this Pi (kernel: "usb 2-1: Failed to query (SET_CUR/GET_LEN)
            # UVC control 1/7 on unit 3: -110 (exp. ...)", confirmed via
            # `sudo dmesg`). Left False until the USB issue is fixed.
            # RE-TESTED 2026-08-13 after setting usbcore.usbfs_memory_mb=1000
            # at BOOT (via /boot/firmware/cmdline.txt + reboot -- a stronger
            # version of the runtime sysfs write already tried and already
            # ruled out) -- same failure, unchanged. Both forms of this fix
            # are now ruled out; the hang is not a usbfs buffer-size issue.
            "enable_depth": False,
            "depth_module.depth_profile": "640,480,15",
            # Aligned to color's pixel grid so the pipeline's single set of
            # intrinsics (from /image_raw/camera_info) applies to both --
            # native depth/color are physically offset sensors otherwise.
            "align_depth.enable": True,
            "enable_infra1": False,
            "enable_infra2": False,
            # Also implicated in the same hang (see module docstring) --
            # left False until the underlying USB issue is fixed. Not used
            # for heading/odometry even once re-enabled (the ESP32's own
            # IMU -- BNO085/BNO08x as of 2026-08-14, was BNO055 -- stays
            # the heading source; the D435i's IMU is a raw BMI055, no
            # magnetometer/fusion, so it's only intended as a per-frame
            # camera-tilt input to obstacle_guard's ground-plane estimate,
            # not anything integrated over time).
            "enable_gyro": False,
            "enable_accel": False,
            # Combines accel+gyro into one sensor_msgs/Imu topic instead of
            # two half-populated ones -- simpler for the GPU-side CDR parser.
            # NOTE: this build's unite_imu_method param is an INTEGER enum
            # (0=off/separate topics, 1=copy, 2=linear_interpolation), not a
            # string -- passing the string name silently failed to set the
            # param and left the Motion Module stopped entirely (verified
            # 2026-08-13 via journalctl: "expected [integer] got [string]").
            "unite_imu_method": 2,
            "enable_sync": False,
            "publish_tf": False,
            "initial_reset": True,
        }],
        remappings=[
            ("/camera/realsense_camera/color/image_raw", "/image_raw"),
            ("/camera/realsense_camera/color/image_raw/compressed", "/image_raw/compressed"),
            ("/camera/realsense_camera/color/camera_info", "/image_raw/camera_info"),
            ("/camera/realsense_camera/aligned_depth_to_color/image_raw", "/depth_raw"),
            ("/camera/realsense_camera/aligned_depth_to_color/image_raw/compressedDepth", "/depth_raw/compressed"),
            ("/camera/realsense_camera/imu", "/imu_raw"),
        ],
    )
    return LaunchDescription([realsense_node])
