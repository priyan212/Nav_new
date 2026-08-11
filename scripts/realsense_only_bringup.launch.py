#!/usr/bin/env python3
"""
Minimal camera-only bringup for the rover using an Intel RealSense D435i
(replaces the old Logitech webcam / v4l2_camera_node, see
camera_only_bringup.launch.py which this supersedes for --rover).

Starts:
  realsense2_camera_node - color stream only (depth/IR/IMU disabled, this
                            pipeline still uses monocular Depth Anything V2
                            on the GPU side, not the RealSense's own depth)
                            remapped to /image_raw + /image_raw/compressed
                            so it matches the existing Zenoh contract exactly
                            (see Nav_new/nav_pipeline/zenoh_node.py
                            CAMERA_KEYS/CAMERA_COMPRESSED_KEYS) -- no GPU-side
                            changes needed.

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
            "enable_depth": False,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_gyro": False,
            "enable_accel": False,
            "enable_sync": False,
            "publish_tf": False,
            "initial_reset": True,
        }],
        remappings=[
            ("/camera/color/image_raw", "/image_raw"),
            ("/camera/color/image_raw/compressed", "/image_raw/compressed"),
            ("/camera/color/camera_info", "/image_raw/camera_info"),
        ],
    )
    return LaunchDescription([realsense_node])
