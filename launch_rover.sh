#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER launcher (DINO + SAM + NavDP)
#  Run on the GPU machine:  ./launch_rover.sh [PI_IP] [gui args...]
#
#  Brings up the Pi (camera + ESP32 micro-ROS + Zenoh bridge via the
#  proven omnivla_nav rover_bringup.launch.py), then starts the SAME
#  Nav_new GUI used in Isaac Sim — camera view, SAM mask, DINO bbox,
#  top-down trajectory/obstacle plot — with real-rover speed caps.
#
#  Examples:
#    ./launch_rover.sh                          # default Pi IP (10.43.225.125)
#    ./launch_rover.sh 10.43.225.125 --target "trash bin"
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

PI_IP=${1:-10.183.202.125}; shift 2>/dev/null || true
PI_USER=pi
PI_PASS=${PI_PASS:-hri}
SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $PI_USER@$PI_IP"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }

# ── 1. Reach the Pi ───────────────────────────────────────────
info "Pinging Pi at $PI_IP ..."
ping -c1 -W2 "$PI_IP" >/dev/null || { warn "Pi unreachable — is the rover ON and on this network?"; exit 1; }
$SSH 'echo ssh_ok' | grep -q ssh_ok || { warn "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
ok "Pi reachable"


# ── 2. Restart the Pi systemd services ────────────────────────
# The Pi runs rover-camera / rover-agent / rover-zenoh as systemd services
# (installed by scripts/pi_install_services.sh): exactly one instance each,
# Restart=always, enabled at boot. rover-agent hardware-resets the ESP32
# (RTS pulse = auto-handshake) on every start. NO ad-hoc processes anymore.
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

info "Checking ESP32 heartbeat (/rover/rpm, up to 20 s)..."
RPM_OK=false
for i in $(seq 1 20); do
    sleep 1
    $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 2 ros2 topic echo /rover/rpm --once 2>/dev/null | head -1"' 2>/dev/null \
        | grep -q "layout\|data" && { RPM_OK=true; ok "ESP32 alive (/rover/rpm publishing) [${i}s]"; break; }
done
$RPM_OK || warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
# ── 6. Launch the Nav_new GUI (real-rover caps) ───────────────
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

# max-angular 1.2 matches the working OmniVLA node — the ESP32 firmware
# normalizes angular by this value, so 0.25 gave only ~1/5 of real steering.
# fov 60 matches the Logitech camera (90 was mis-scaling every bearing).
info "Starting Nav_new GUI (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s, fov 60)..."
exec python -u -m nav_pipeline.isaac_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 1.2 --fov 60 \
    --compressed-only \
    "$@"
