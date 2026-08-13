#!/usr/bin/env bash
# ============================================================
#  Nav_new -- REAL ROVER / LanderPi launcher, HTTP-served S2Diff NavDP variant.
#
#  Same GUI as LAUNCH/launch_rover.sh (camera feed, SAM mask, DINO bbox,
#  top-down trajectory/obstacle plot) -- it's the literal same isaac_gui.py
#  underneath. The only difference: NavDP trajectory sampling is routed over
#  HTTP to tryout/navdp_s2diff_server.py instead of running in-process.
#
#  ⚠ REQUIRES the server already running in a separate terminal first:
#      source /home/i3d/exit/etc/profile.d/conda.sh && conda activate internnav
#      cd tryout && python navdp_s2diff_server.py \
#          --checkpoint ../checkpoints/navdp_extracted.pth --port 8888
#
#  This is a THIRD, independent way to run S2Diff-guided NavDP (besides
#  plain launch_rover.sh and LAUNCH/launch_rover_s2diff.sh's in-process
#  path) -- useful for exercising tryout/navdp_s2diff_server.py itself, or
#  if something else needs to talk to that server too. None of the plain
#  launch_rover*.sh scripts are touched by this.
#
#  Examples:
#    ./launch_rover_s2diff_http.sh                          # default rover, default Pi IP, server on localhost:8888
#    ./launch_rover_s2diff_http.sh 192.168.21.125 --target "trash bin"
#    ./launch_rover_s2diff_http.sh 192.168.21.125 --server-url http://127.0.0.1:9000
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
info "Starting Nav_new GUI [$BACKEND, HTTP-served S2Diff NavDP] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, search $BACKEND_SEARCH_ANGULAR rad/s)..."
exec python -u -m nav_pipeline.s2diff_http_runner \
    --pi-ip "$PI_IP" \
    --policy-type extracted \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --search-angular "$BACKEND_SEARCH_ANGULAR" --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
