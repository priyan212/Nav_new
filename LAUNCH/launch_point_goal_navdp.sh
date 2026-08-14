#!/usr/bin/env bash
# ============================================================
#  Nav_new -- REAL ROVER / LanderPi launcher, point-goal obstacle-avoidance
#  benchmark, PLAIN in-process NavDP variant (no S2Diff HTTP server).
#
#  Same GUI as LAUNCH/launch_point_goal.sh (fixed straight-line world-frame
#  point goal, named obstacle-to-avoid field, top-down trajectory/obstacle
#  plot) -- it's the literal same nav_pipeline/point_goal_gui.py underneath.
#  The only difference: NavDP trajectory sampling runs in-process with the
#  official standalone crossmodal checkpoint (policy_type="crossmodal", same
#  as plain LAUNCH/launch_rover.sh), NOT routed through
#  tryout/navdp_s2diff_server.py -- no separate server to start first.
#
#  Trade-off: the named "Obstacle to avoid" field's DINO detection still
#  runs and is shown in the camera view, but nothing consumes
#  res.avoid_detection/avoid_pixels as pixel-obstacle guidance on this path
#  (only nav_pipeline/s2diff_http_client.py's HTTP-patched sampler reads it
#  -- see pipeline.py's step() docstring). Obstacle avoidance here comes
#  entirely from real/estimated depth (obstacle_guard.py's AVOID state +
#  swept-trajectory clearance veto), same as every other plain launcher.
#
#  Examples:
#    ./launch_point_goal_navdp.sh                          # default rover, default Pi IP
#    ./launch_point_goal_navdp.sh 192.168.21.125
#    ./launch_point_goal_navdp.sh 192.168.21.125 --arrival-radius 0.3
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
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new point-goal benchmark [$BACKEND, in-process NavDP] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV)..."
exec python -u -m nav_pipeline.point_goal_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
