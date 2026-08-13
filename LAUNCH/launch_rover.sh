#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER / LanderPi launcher (DINO + SAM + NavDP)
#  Run on the GPU machine:
#    ./launch_rover.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Brings up the Pi (camera + ESP32 micro-ROS + Zenoh bridge via the
#  proven omnivla_nav rover_bringup.launch.py for --rover, or
#  landerpi/bridge.py inside Hiwonder's own armpi_pro container for
#  --hiwonder -- see LAUNCH/_backend.sh / landerpi/README.md), then starts
#  the SAME Nav_new GUI used in Isaac Sim — camera view, SAM mask, DINO
#  bbox, top-down trajectory/obstacle plot — with real-rover speed caps.
#  Defaults to --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_rover.sh                          # default rover, default Pi IP
#    ./launch_rover.sh 192.168.21.125 --target "trash bin"
#    ./launch_rover.sh --hiwonder --target "trash bin"
#    ./launch_rover.sh --hiwonder 10.47.234.228 --target "trash bin"
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# ── Launch the Nav_new GUI (real-rover caps) ───────────────
# conda activation hooks (isaacsim setup_conda_env.sh) reference unbound
# vars like ZSH_VERSION — relax nounset while sourcing them.
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# max-angular ($BACKEND_MAX_ANGULAR, see LAUNCH/_backend.sh): 1.2 for the
# rover matches the working OmniVLA node — the ESP32 firmware normalizes
# angular by this value, so 0.25 gave only ~1/5 of real steering. Reduced to
# 0.5 for --hiwonder (2026-08-07, see _backend.sh's comment).
# fov 60 matches the Logitech camera on the old rover (64.6 for the
# LanderPi's real usb_cam intrinsics, see LAUNCH/_backend.sh).
# search-angular ($BACKEND_SEARCH_ANGULAR, see LAUNCH/_backend.sh): 0.18 for
# the rover as of 2026-08-12 (was 0.15 default, tried 0.13 then 0.14, rover
# still didn't move at either; briefly 0.20). Root cause traced to the ESP32
# firmware actually flashed on this rover (esp32/rover_6wd_complete.ino), NOT
# PipelineConfig.ang_min_cmd (that floor is for TRACK's ramp and never even
# applies to SEARCH, which sets the angular command directly): drive_rover()
# there converts a pure-rotation angular_z into a per-wheel target of
# angular_z * TRACK_WIDTH_M/2 (TRACK_WIDTH_M=0.345m), and closed_loop_pwm()
# hard-zeros any per-wheel target below VEL_DEADBAND_MS=0.03 m/s ("stop
# request -> no windup"). 0.13-0.15 all produce 0.022-0.026 m/s per wheel --
# under the deadband, so the firmware silently commanded zero PWM regardless
# of what pipeline.py sent. The hard minimum to clear it is ~0.174 rad/s
# (0.03 * 2 / 0.345); 0.18 is the lowest value tested that still clears it
# (0.031 m/s/wheel, only ~3.5% margin -- deliberately chosen close to the
# floor per request rather than 0.20's larger ~15% margin). If it stops
# moving again, suspect this thin margin first; 0.20 is the safer fallback,
# and the floor to respect when going lower still is the hardware 0.174
# rad/s deadband threshold, not the old 0.12 software ang_min_cmd one.
# --hiwonder keeps the old carried-over 0.13 unchanged (different
# chassis/firmware, this deadband math doesn't apply -- see _backend.sh).
# servo-ramp-deg 70 (up from the 35 default) — at fov 60 / predict-hz 2.5,
# a 35deg ramp put a frame-edge detection (bearing ~30deg) at ~94% of
# max_angular, sweeping ~26deg in a single 0.4s tick -- almost the whole
# visible half-frame in one shot, so corner detections got spun straight
# out of view. 70deg brings that down to ~70% of max_angular at the edge.
# servo-ramp-deg was tuned on the OLD rover's camera+motor response --
# carried over verbatim for --hiwonder as an unvalidated starting point,
# see backend_bringup's warning.
EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, search $BACKEND_SEARCH_ANGULAR rad/s, ramp 70deg, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
exec python -u -m nav_pipeline.isaac_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --search-angular "$BACKEND_SEARCH_ANGULAR" --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
