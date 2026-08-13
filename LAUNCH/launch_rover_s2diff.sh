#!/usr/bin/env bash
# ============================================================
#  Nav_new -- REAL ROVER / LanderPi launcher, S2Diff-guided NavDP variant.
#
#  Identical to LAUNCH/launch_rover.sh (DINO + SAM + NavDP, same GUI, same
#  Pi bringup) except NavDP's action sampling runs through the S2Diff
#  in-loop obstacle guidance (nav_pipeline/s2diff_navdp.py) instead of the
#  original unguided DDPM sampler. launch_rover.sh itself is untouched --
#  this is a separate, new, opt-in entry point (nav_pipeline.s2diff_runner)
#  for comparing the two side by side. See tryout/S2DIFF_GUIDANCE.md and
#  nav_pipeline/s2diff_navdp.py's module docstring for what the guidance
#  does and does not change (pipeline.py's own obstacle veto/hard-stop/
#  anti-oscillation logic is unchanged and still has final say).
#
#  Examples:
#    ./launch_rover_s2diff.sh                          # default rover, default Pi IP
#    ./launch_rover_s2diff.sh 192.168.21.125 --target "trash bin"
#    ./launch_rover_s2diff.sh --hiwonder --target "trash bin"
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# ── Launch the Nav_new GUI (real-rover caps), S2Diff-guided NavDP ─────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.s2diff_runner" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# See LAUNCH/launch_rover.sh for why these particular tuning values.
EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new GUI [$BACKEND, S2Diff-guided NavDP] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, search $BACKEND_SEARCH_ANGULAR rad/s, ramp 70deg, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
exec python -u -m nav_pipeline.s2diff_runner \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --search-angular "$BACKEND_SEARCH_ANGULAR" --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
