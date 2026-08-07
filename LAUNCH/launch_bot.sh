#!/usr/bin/env bash
# ============================================================
#  Nav_new — A/B robot launcher (old ESP32 rover  <->  Hiwonder LanderPi)
#  Run on the GPU machine:
#    ./launch_bot.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Same Nav_new manual-control + Go-Home GUI (nav_pipeline.home_gui) either
#  way -- pass --enable-obstacle-avoidance for full DINO+NavDP driving, same
#  as launch_rover_home.sh. The two backends speak the identical Zenoh
#  contract (see nav_pipeline/zenoh_node.py's docstring), so pipeline.py /
#  obstacle_guard.py / odometry_logger.py are unchanged either way -- only
#  what's running on the Pi differs (see LAUNCH/_backend.sh):
#    --rover     : old 6WD rover, ESP32 micro-ROS + zenoh-bridge-ros2dds,
#                  systemd services (rover-camera/agent/zenoh), REAL
#                  measured wheel-encoder RPM.
#    --hiwonder  : new Hiwonder LanderPi (Mecanum), stock ROS1 Noetic stack
#                  in Hiwonder's own `armpi_pro` Docker container (UNTOUCHED)
#                  + landerpi/bridge.py as the only added file -- REAL
#                  encoder+IMU odometry (see landerpi/README.md).
#  Defaults to --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_bot.sh --rover
#    ./launch_bot.sh --hiwonder
#    ./launch_bot.sh --hiwonder 10.47.234.228 --enable-obstacle-avoidance
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

ENABLE_AVOID=false
for arg in "$@"; do
    [[ "$arg" == "--enable-obstacle-avoidance" ]] && ENABLE_AVOID=true
done

$ENABLE_AVOID && backend_bringup camera || backend_bringup nocamera

# ── Launch the GUI ───────────────────────────────────────────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
if $ENABLE_AVOID; then
    export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
    export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}
fi

pkill -f "nav_pipeline.home_gui" 2>/dev/null && sleep 1

EXTRA_ARGS=()
if $ENABLE_AVOID; then
    EXTRA_ARGS+=(--fov "$BACKEND_FOV" --compressed-only)
    [[ "$BACKEND" == "hiwonder" ]] && EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new manual-control + Go-Home GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / 0.5 rad/s$($ENABLE_AVOID && echo ', obstacle avoidance on'))..."
exec python -u -m nav_pipeline.home_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 0.5 \
    --home-max-linear 0.15 --home-max-angular 0.5 \
    "${EXTRA_ARGS[@]}" \
    "$@"
