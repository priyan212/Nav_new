#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER launcher (REMIND + NavDP)
#  Run on the GPU machine:  ./launch_rover_remind.sh [PI_IP] [gui args...]
#
#  Same rover bring-up as launch_rover.sh (Pi camera + ESP32 micro-ROS +
#  Zenoh bridge), but target selection goes through REMIND's persistent
#  per-object re-identification instead of a bare DINO text phrase:
#
#    1. Brings up the Pi (camera + ESP32 + Zenoh), same as launch_rover.sh.
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
#
#  Examples:
#    ./launch_rover_remind.sh                          # default Pi IP
#    ./launch_rover_remind.sh 10.47.234.125 --target "chair id 1"
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
    exec python scripts/live_server.py --yolo-model yolo11l-seg.pt --port "$REMIND_PORT" --device cuda:0
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

# ── 5. Launch the Nav_new GUI (REMIND target selection, real-rover caps) ──
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
# measurements behind each one). --depth-encoder defaults to vitb inside
# remind_gui.py itself (requirement: accurate STOP distance), so it's not
# repeated here.
info "Starting Nav_new REMIND+NavDP GUI (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s, fov 60, search 0.13 rad/s, ramp 70deg)..."
python -u -m nav_pipeline.remind_gui \
    --pi-ip "$PI_IP" \
    --remind-server "http://127.0.0.1:${REMIND_PORT}" \
    --max-linear 0.15 --max-angular 1.2 --fov 60 \
    --search-angular 0.13 --servo-ramp-deg 70 \
    --compressed-only \
    "$@"
