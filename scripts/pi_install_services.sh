#!/bin/bash
# Runs ON THE PI (once). Installs systemd services for the three rover-side
# components so they survive SSH drops, Wi-Fi hiccups, crashes and reboots:
#
#   rover-camera.service : v4l2 camera -> /image_raw (ROS 2)
#   rover-agent.service  : micro-ROS agent (ESP32 motors), with an automatic
#                          ESP32 hardware reset (RTS pulse) before each start
#   rover-zenoh.service  : zenoh-bridge-ros2dds router :7447 (ROS_DISTRO=humble)
#
# systemd guarantees exactly ONE instance of each (no more duplicate agents)
# and Restart=always self-heals every component within seconds.
#
# Usage: bash pi_install_services.sh   (will sudo with the pi password)
set -e

SUDO="sudo -S"
PASS=hri

# ── helper scripts the services call ─────────────────────────────
cat > /home/pi/rover_camera_start.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/pi/rover_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=0 ROS_DOMAIN_ID=0
exec ros2 launch omnivla_nav camera_only_bringup.launch.py \
    video_device:=/dev/video0 pixel_format:=YUYV "image_size:=[640,480]"
EOF

cat > /home/pi/rover_agent_start.sh << 'EOF'
#!/bin/bash
# hardware-reset the ESP32 (RTS pulse) so the micro-ROS session comes up clean
python3 - << 'PYEOF'
import serial, time
try:
    s = serial.Serial("/dev/ttyUSB0", 115200)
    s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False; time.sleep(0.5)
    s.close()
    print("ESP32 reset pulse sent")
except Exception as e:
    print("reset skipped:", e)
PYEOF
source /opt/ros/humble/setup.bash
source /home/pi/uros_ws/install/setup.bash 2>/dev/null || source /home/pi/microros_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=0 ROS_DOMAIN_ID=0
exec ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 -v4
EOF

cat > /home/pi/rover_zenoh_start.sh << 'EOF'
#!/bin/bash
export ROS_DISTRO=humble
exec /home/pi/zenoh-bridge-ros2dds -c /home/pi/zenoh_pi_bridge.json5
EOF

chmod +x /home/pi/rover_camera_start.sh /home/pi/rover_agent_start.sh /home/pi/rover_zenoh_start.sh

# ── systemd units (built as pi, installed with ONE sudo call) ────
mkdir -p /home/pi/units
make_unit () {
cat > "/home/pi/units/$3" << EOF
[Unit]
Description=$2
After=network.target

[Service]
User=pi
ExecStart=/bin/bash /home/pi/$1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
}
make_unit rover_camera_start.sh "Rover camera (v4l2 -> /image_raw)" rover-camera.service
make_unit rover_agent_start.sh "Rover micro-ROS agent (ESP32 motors, auto-reset)" rover-agent.service
make_unit rover_zenoh_start.sh "Rover zenoh-bridge-ros2dds router :7447" rover-zenoh.service

# ── stop every ad-hoc leftover, then hand over to systemd ────────
pkill -9 -f micro_ros_agent 2>/dev/null || true
pkill -9 -f v4l2_camera_node 2>/dev/null || true
pkill -9 -f zenoh-bridge 2>/dev/null || true
pkill -9 -f rover_bringup 2>/dev/null || true
sleep 2

# single sudo invocation; password via stdin, commands via -c (no heredoc
# stealing sudo's stdin — that bug ate the first install attempt)
echo "$PASS" | $SUDO bash -c '
  cp /home/pi/units/rover-camera.service /home/pi/units/rover-agent.service /home/pi/units/rover-zenoh.service /etc/systemd/system/ &&
  systemctl daemon-reload &&
  systemctl enable --now rover-camera.service rover-agent.service rover-zenoh.service
'
sleep 6
systemctl is-active rover-camera rover-agent rover-zenoh || true
echo SERVICES_INSTALLED
