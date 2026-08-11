#!/usr/bin/env bash
# ============================================================
#  Nav_new — REMIND + NavDP launcher (real rover or LanderPi)
#  Run on the GPU machine:
#    ./launch_rover_remind.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Same Pi bring-up as launch_rover.sh (camera + backend, see
#  LAUNCH/_backend.sh / landerpi/README.md), but target selection goes
#  through REMIND's persistent per-object re-identification instead of a
#  bare DINO text phrase:
#
#    1. Brings up the Pi.
#    2. Starts the REMIND live re-identification server as a background
#       process in ITS OWN conda env (REMIND/remind-reid-tracker/.venv --
#       its torch/transformers pins are incompatible with this project's
#       internnav env, see nav_pipeline/remind_client.py's docstring).
#    3. Launches nav_pipeline.remind_gui: the same camera/plot/manual-drive
#       control panel as isaac_gui.py, but every currently-tracked object is
#       overlaid with REMIND's own persistent label ("CHAIR ID 1", etc.) --
#       type that back (or double-click it in the "known objects" list) to
#       send the rover to that SPECIFIC object.
#    4. Navigates via the existing NavDP policy/obstacle-guard/goal-belief
#       stack, unchanged, and stops 1.5 m from the object using Depth
#       Anything V2 ViT-B monocular metric depth (more accurate than the
#       vits default elsewhere -- depth error feeds directly into the STOP
#       distance decision).
#  Defaults to --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_rover_remind.sh                          # default rover, default Pi IP
#    ./launch_rover_remind.sh 192.168.21.125 --target "chair id 1"
#    ./launch_rover_remind.sh --hiwonder --target "chair id 1"
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
REMIND_DIR="$REPO_DIR/REMIND/remind-reid-tracker"

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"
REMIND_PORT=${REMIND_PORT:-8765}

if [[ ! -d "$REMIND_DIR/.venv" ]]; then
    err "REMIND venv not found at $REMIND_DIR/.venv -- see REMIND/remind-reid-tracker/SETUP.md"
    exit 1
fi

backend_bringup camera

# ── Start the REMIND live re-identification server ─────────
# Own conda env (REMIND/remind-reid-tracker/.venv): its torch/transformers/
# ultralytics pins are incompatible with this project's internnav env (see
# nav_pipeline/remind_client.py's docstring) -- runs as a separate local
# process, talked to over loopback HTTP.
pkill -f "remind-reid-tracker/scripts/live_server.py" 2>/dev/null && sleep 1
mkdir -p "$REMIND_DIR/outputs"
REMIND_LOG="$REMIND_DIR/outputs/live_server_$(date +%Y%m%d_%H%M%S).log"
info "Starting REMIND live server (port $REMIND_PORT, log: $REMIND_LOG)..."
(
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$REMIND_DIR/.venv"
    set -u
    export HF_HOME="$REMIND_DIR/.cache/huggingface"
    export YOLO_CONFIG_DIR="$REMIND_DIR/.cache/ultralytics"
    cd "$REMIND_DIR"
    exec python scripts/live_server.py --port "$REMIND_PORT" --device cuda:0
) > "$REMIND_LOG" 2>&1 &
REMIND_PID=$!

cleanup() {
    info "Stopping REMIND live server (pid $REMIND_PID)..."
    kill "$REMIND_PID" 2>/dev/null
    wait "$REMIND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

info "Waiting for REMIND server to load models (up to 90 s)..."
REMIND_OK=false
for i in $(seq 1 90); do
    sleep 1
    curl -sf "http://127.0.0.1:${REMIND_PORT}/health" >/dev/null 2>&1 && { REMIND_OK=true; ok "REMIND server ready [${i}s]"; break; }
    kill -0 "$REMIND_PID" 2>/dev/null || { err "REMIND server process died — see $REMIND_LOG"; exit 1; }
done
$REMIND_OK || { err "REMIND server did not come up in time — see $REMIND_LOG"; exit 1; }

# ── Launch the Nav_new GUI (REMIND target selection, real-rover caps) ──
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.remind_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# Same tuned real-rover constants as launch_rover.sh (max-angular/fov/
# search-angular/servo-ramp-deg -- see that script's comments for the
# measurements behind each one; carried over unvalidated for --hiwonder,
# see backend_bringup's warning). --depth-encoder defaults to vitb inside
# remind_gui.py itself (requirement: accurate STOP distance), so it's not
# repeated here.
EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new REMIND+NavDP GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / $BACKEND_MAX_ANGULAR rad/s, fov $BACKEND_FOV, search 0.13 rad/s, ramp 70deg, slew $BACKEND_ANGULAR_SLEW_MAX rad/s/tick)..."
python -u -m nav_pipeline.remind_gui \
    --pi-ip "$PI_IP" \
    --remind-server "http://127.0.0.1:${REMIND_PORT}" \
    --max-linear 0.15 --max-angular "$BACKEND_MAX_ANGULAR" --fov "$BACKEND_FOV" \
    --search-angular 0.13 --servo-ramp-deg 70 --angular-slew-max "$BACKEND_ANGULAR_SLEW_MAX" \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
