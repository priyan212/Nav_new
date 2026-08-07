#!/usr/bin/env bash
# Launch the DINO + NavDP navigation node (GPU side).
#
# Usage:
#   ./launch_dino_navdp.sh                          # multicast discovery, default target
#   ./launch_dino_navdp.sh --target "red chair"     # custom target
#   ./launch_dino_navdp.sh --pi-ip 192.168.1.42 --target "trash bin"
#   ./launch_dino_navdp.sh --hiwonder --target "trash bin"   # LanderPi defaults (pi-ip/footprint)
#
# The rover/Isaac/LanderPi side is unchanged from the OmniVLA pipeline: it
# must already be brought up separately (zenoh-bridge-ros2dds / the Isaac
# zenoh bridge / landerpi/deploy_bridge.sh -- this script does NOT do that
# itself, unlike LAUNCH/launch_rover.sh -- use that instead if you also want
# Pi bring-up in one command).
#
# --rover/--hiwonder here ONLY fill in sensible --pi-ip/--footprint-length/
# --footprint-width defaults (still overridable) -- no bring-up, no camera/
# rpm verification. Defaults to --rover-shaped defaults if omitted (i.e.
# --pi-ip must be passed explicitly, same as before this flag existed).

set -eo pipefail
cd "$(dirname "$0")/.."

# Only inject --pi-ip/--fov/--footprint-* defaults if --rover/--hiwonder was
# actually passed -- preserves the original default (no flags at all -> no
# --pi-ip -> zenoh multicast discovery) exactly when neither is given.
BACKEND_EXPLICIT=false
for arg in "$@"; do
    [[ "$arg" == "--rover" || "$arg" == "--hiwonder" ]] && BACKEND_EXPLICIT=true
done

source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav

export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

if $BACKEND_EXPLICIT; then
    source LAUNCH/_backend.sh
    backend_parse_args "$@"
    set -- "${BACKEND_ARGS[@]}"
    EXTRA_ARGS=(--pi-ip "$PI_IP" --fov "$BACKEND_FOV")
    [[ "$BACKEND" == "hiwonder" ]] && EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
    exec python -m nav_pipeline.zenoh_node "${EXTRA_ARGS[@]}" "$@"
else
    exec python -m nav_pipeline.zenoh_node "$@"
fi
