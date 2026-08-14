#!/usr/bin/env bash
# ============================================================
#  Nav_new -- REAL ROVER / LanderPi launcher, point-goal obstacle-avoidance
#  benchmark (HTTP-served S2Diff NavDP variant).
#
#  Same underlying code as LAUNCH/launch_rover_s2diff_http.sh (same GUI
#  skeleton, same server, same NavDP policy patch) -- the only difference:
#  instead of a DINO-detected target object, you enter a straight-line
#  distance in the GUI and the rover drives toward a FIXED, imaginary point
#  goal that many meters directly ahead of wherever it's currently facing.
#  The goal never moves and is never lost/re-acquired visually (it's
#  re-derived from odometry every tick, see nav_pipeline/point_goal_gui.py's
#  module docstring) -- so after avoiding an obstacle, the rover always
#  re-lines up on the original straight line to it. A second GUI field lets
#  you name an obstacle for NavDP's S2Diff pixel-obstacle guidance to
#  specifically avoid. Built for benchmarking obstacle-avoidance quality in
#  isolation from target-detection quality.
#
#  ⚠ REQUIRES the server already running in a separate terminal first:
#      source /home/i3d/exit/etc/profile.d/conda.sh && conda activate internnav
#      cd tryout && python navdp_s2diff_server.py \
#          --checkpoint ../checkpoints/navdp_extracted.pth --port 8888
#
#  Examples:
#    ./launch_point_goal.sh                          # default rover, default Pi IP, server on localhost:8888
#    ./launch_point_goal.sh 192.168.21.125
#    ./launch_point_goal.sh 192.168.21.125 --arrival-radius 0.3
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.point_goal_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.point_goal_http_runner" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.s2diff_runner" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.s2diff_http_runner" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# Any HTTP response (even 404 on "/") means something's listening -- avoid
# hitting a real endpoint here, /navigator_reset actually loads the model.
if ! curl -s -o /dev/null -m 2 "http://127.0.0.1:8888/" 2>/dev/null; then
    warn "navdp_s2diff_server.py doesn't look reachable on :8888 -- start it first (see this script's header comment)"
fi

EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new point-goal benchmark [$BACKEND, HTTP-served S2Diff NavDP] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV)..."
exec python -u -m nav_pipeline.point_goal_http_runner \
    --pi-ip "$PI_IP" \
    --policy-type extracted \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
