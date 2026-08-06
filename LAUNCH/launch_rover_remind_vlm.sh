#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER launcher (REMIND + NavDP, VLM-confirmed arrival)
#  Run on the GPU machine:  ./launch_rover_remind_vlm.sh [PI_IP] [gui args...]
#
#  Identical bring-up to launch_rover_remind.sh (Pi camera + ESP32
#  micro-ROS + Zenoh bridge, REMIND live re-identification server), but
#  launches nav_pipeline.remind_gui_vlm instead of nav_pipeline.remind_gui:
#
#    1-3. Same as launch_rover_remind.sh -- Pi bring-up, REMIND live server.
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
#
#  Examples:
#    ./launch_rover_remind_vlm.sh                          # default Pi IP
#    ./launch_rover_remind_vlm.sh 10.47.234.125 --target "chair id 1"
#    ./launch_rover_remind_vlm.sh --no-vlm-confirm          # A/B: pure metric, same as launch_rover_remind.sh
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
REMIND_DIR="$REPO_DIR/REMIND/remind-reid-tracker"

if [[ "${1:-}" == -* ]]; then
    PI_IP=10.47.234.125
else
    PI_IP=${1:-10.47.234.125}; shift 2>/dev/null || true
fi
PI_USER=pi
PI_PASS=${PI_PASS:-hri}
SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $PI_USER@$PI_IP"
REMIND_PORT=${REMIND_PORT:-8765}

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

if [[ ! -d "$REMIND_DIR/.venv" ]]; then
    err "REMIND venv not found at $REMIND_DIR/.venv -- see REMIND/remind-reid-tracker/SETUP.md"
    exit 1
fi

# ── 1. Reach the Pi ───────────────────────────────────────────
info "Pinging Pi at $PI_IP ..."
ping -c1 -W2 "$PI_IP" >/dev/null || { warn "Pi unreachable — is the rover ON and on this network?"; exit 1; }
$SSH 'echo ssh_ok' | grep -q ssh_ok || { warn "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
ok "Pi reachable"

# ── 2. Restart the Pi systemd services ────────────────────────
info "Restarting rover services on Pi..."
$SSH "echo $PI_PASS | sudo -S systemctl restart rover-camera rover-agent rover-zenoh 2>/dev/null; sleep 4; systemctl is-active rover-camera rover-agent rover-zenoh" \
    | grep -c active | grep -q 3 && ok "services active: camera, agent (ESP32 auto-reset), zenoh" \
    || { warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"; }

# ── 3. Verify data is flowing ─────────────────────────────────
info "Waiting for camera topic (up to 25 s)..."
CAM_OK=false
for i in $(seq 1 25); do
    sleep 1
    $SSH 'bash -lc "source /opt/ros/humble/setup.bash; ros2 topic list 2>/dev/null"' 2>/dev/null \
        | grep -q "/image_raw" && { CAM_OK=true; ok "camera live [${i}s]"; break; }
done
$CAM_OK || warn "camera topic missing — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -n 20'"

info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
    | grep -q "layout\|data"; then
    ok "ESP32 alive (/rover/rpm publishing)"
else
    warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
fi

# ── 4. Start the REMIND live re-identification server ─────────
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

# ── 5. Launch the Nav_new GUI (REMIND target selection, VLM-confirmed arrival, real-rover caps) ──
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
# measurements behind each one). --depth-encoder defaults to vitb inside
# remind_gui_vlm.py itself, same reasoning as remind_gui.py.
info "Starting Nav_new REMIND+NavDP GUI, VLM-confirmed arrival (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s, fov 60, search 0.13 rad/s, ramp 70deg)..."
python -u -m nav_pipeline.remind_gui_vlm \
    --pi-ip "$PI_IP" \
    --remind-server "http://127.0.0.1:${REMIND_PORT}" \
    --max-linear 0.15 --max-angular 1.2 --fov 60 \
    --search-angular 0.13 --servo-ramp-deg 70 \
    --compressed-only \
    "$@"
