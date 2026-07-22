#!/usr/bin/env bash
# Launch the Nav_new DINO+SAM+NavDP GUI (Isaac Sim / rover control panel).
#
#   ./launch_gui.sh                                   # sim defaults (0.5 m/s, 0.6 rad/s)
#   ./launch_gui.sh --target "cardboard box"
#   ./launch_gui.sh --max-linear 0.15 --max-angular 0.25   # real-rover caps
#   ./launch_gui.sh --pi-ip 192.168.1.42              # explicit Zenoh peer
#
# Prerequisites (Isaac Sim testing):
#   1. Isaac Sim: scene loaded, Play pressed, and in the Script Editor:
#        exec(open("/mnt/bigdisk/motion_planning/scripts/isaac_cmdvel_bridge.py").read(), globals())
#        exec(open("/mnt/bigdisk/motion_planning/scripts/isaac_camera_ros2_pub.py").read(), globals())
#   2. zenoh-bridge-ros2dds running:
#        source /opt/ros/jazzy/setup.bash
#        /mnt/bigdisk/motion_planning/zenoh-bridge-ros2dds \
#            -c /mnt/bigdisk/Priyan/OmniVLA_safe/configs/zenoh_isaac_bridge.json5
#
# NOTE: run only ONE controller at a time (this GUI, zenoh_node.py, or the old
# OmniVLA isaac_gui.py) — they all publish cmd_vel and will fight each other.

set -eo pipefail
cd "$(dirname "$0")"

source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav

export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

# stop any controller instance already running (mine or a previous yours)
pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1 || true
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1 || true

exec python -u -m nav_pipeline.isaac_gui --max-linear 0.5 --max-angular 0.6 "$@"
