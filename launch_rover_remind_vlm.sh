#!/usr/bin/env bash
# ============================================================
#  Nav_new — REMIND + NavDP launcher, VLM-confirmed arrival (real rover or LanderPi)
#  Run on the GPU machine:
#    ./launch_rover_remind_vlm.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Identical bring-up to LAUNCH/launch_rover_remind.sh (Pi camera/backend,
#  REMIND live re-identification server), but launches
#  nav_pipeline.remind_gui_vlm instead of nav_pipeline.remind_gui:
#
#    1-3. Same as LAUNCH/launch_rover_remind.sh -- Pi bring-up, REMIND live server.
#    4. Navigates via the same NavDP policy/obstacle-guard/goal-belief
#       stack and the same 1.5 m depth-based stop_distance trigger, but
#       that metric trigger is no longer the FINAL word on "arrived": once
#       it fires, the GUI asks REMIND's already-loaded InternVL model
#       (over the SAME /confirm_arrival HTTP endpoint the live server
#       exposes) whether the current full camera frame actually shows the
#       robot having reached the target, and only declares GOAL REACHED
#       once the VLM agrees (nav_pipeline/remind_gui_vlm.py's
#       VLMArrivalGate). Falls back automatically to the exact
#       launch_rover_remind.sh metric-only behavior if that endpoint is
#       ever unavailable -- the metric threshold still runs every tick and
#       is what actually zeroes the velocity command; the VLM only gates
#       the final "reached" confirmation on top of it, never replaces it.
#  Defaults to --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_rover_remind_vlm.sh                          # default rover, default Pi IP
#    ./launch_rover_remind_vlm.sh 10.47.234.125 --target "chair id 1"
#    ./launch_rover_remind_vlm.sh --hiwonder --target "chair id 1"
#    ./launch_rover_remind_vlm.sh --no-vlm-confirm          # A/B: pure metric, same as launch_rover_remind.sh
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"
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
# process, talked to over loopback HTTP. InternVL stays ENABLED (no
# --no-internvl/--use-blip) since this launcher's whole point is the
# /confirm_arrival endpoint it powers -- see remind_gui_vlm.py.
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

# ── Launch the Nav_new GUI (REMIND target selection, VLM-confirmed arrival, real-rover caps) ──
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "nav_pipeline.remind_gui_vlm" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

# Same tuned real-rover constants as launch_rover_remind.sh (max-angular/
# fov/search-angular/servo-ramp-deg -- see that script's comments for the
# measurements behind each one; carried over unvalidated for --hiwonder,
# see backend_bringup's warning). --depth-encoder defaults to vitb inside
# remind_gui_vlm.py itself, same reasoning as remind_gui.py.
EXTRA_ARGS=()
if [[ "$BACKEND" == "hiwonder" ]]; then
    EXTRA_ARGS+=(--footprint-length "$BACKEND_FOOTPRINT_LENGTH" --footprint-width "$BACKEND_FOOTPRINT_WIDTH")
fi
info "Starting Nav_new REMIND+NavDP GUI, VLM-confirmed arrival [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s, fov $BACKEND_FOV, search 0.13 rad/s, ramp 70deg)..."
python -u -m nav_pipeline.remind_gui_vlm \
    --pi-ip "$PI_IP" \
    --remind-server "http://127.0.0.1:${REMIND_PORT}" \
    --max-linear 0.15 --max-angular 1.2 --fov "$BACKEND_FOV" \
    --search-angular 0.13 --servo-ramp-deg 70 \
    --compressed-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
