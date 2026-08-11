#!/usr/bin/env bash
# ============================================================
#  Nav_new — manual control + Go Home launcher (real rover or LanderPi)
#  Run on the GPU machine:
#    ./launch_rover_home.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Thin wrapper around LAUNCH/launch_bot.sh (same manual-drive + "GO HOME"
#  panel, fused wheel-encoder + IMU odometry, optional
#  --enable-obstacle-avoidance NavDP homing -- see nav_pipeline/home_gui.py's
#  module docstring for the full mechanics) -- kept as a separate name for
#  backward compatibility with existing muscle memory/scripts. Defaults to
#  --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_rover_home.sh                          # default rover, default Pi IP
#    ./launch_rover_home.sh 192.168.21.125 --home-dist-tol 0.05
#    ./launch_rover_home.sh --hiwonder --enable-obstacle-avoidance
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"
exec ./launch_bot.sh "$@"
