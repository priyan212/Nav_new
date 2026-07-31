#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER odometry-accuracy test launcher
#  Run on the GPU machine:  ./launch_odom_test.sh [PI_IP] [gui args...]
#
#  Brings up the Pi (ESP32 micro-ROS + Zenoh bridge only -- no camera
#  needed for this test), then starts scripts/odom_accuracy_gui.py: a
#  standalone GUI (no DINO/SAM/NavDP/depth models) for measuring how far
#  dead-reckoned /rover/rpm odometry drifts from hand-measured ground
#  truth. See belief_eval_20260730/RESULTS.md, "The caveat that matters
#  most for the real rover" -- this is the real-hardware check behind
#  whether porting SubgoalBeliefBank's ego-motion correction is worth it.
#
#  Examples:
#    ./launch_odom_test.sh                          # default Pi IP (10.47.234.125)
#    ./launch_odom_test.sh 10.47.234.125 --max-angular 1.0
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

PI_IP=${1:-10.47.234.125}; shift 2>/dev/null || true
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
# rover-agent (ESP32 bridge) and rover-zenoh are all this test needs --
# rover-camera is left alone (no camera reads happen here) but restarting
# it too is harmless and keeps all three services in the same known state.
info "Restarting rover services on Pi..."
$SSH "echo $PI_PASS | sudo -S systemctl restart rover-camera rover-agent rover-zenoh 2>/dev/null; sleep 4; systemctl is-active rover-camera rover-agent rover-zenoh" \
    | grep -c active | grep -q 3 && ok "services active: camera, agent (ESP32 auto-reset), zenoh" \
    || { warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"; }

# ── 3. Verify /rover/rpm is flowing ────────────────────────────
info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
# ONE continuous `ros2 topic echo`, not a loop of fresh 2s attempts -- the
# ESP32 gets hardware-reset (RTS pulse) on every rover-agent restart, so it
# needs real time to reboot + re-handshake micro-ROS before /rover/rpm even
# exists, and each fresh `ros2 topic echo` process has to redo DDS discovery
# of the remote publisher from cold.
if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
    | grep -q "layout\|data"; then
    ok "ESP32 alive (/rover/rpm publishing)"
else
    warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
fi

# ── 4. Launch the odometry-accuracy GUI ───────────────────────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u

pkill -f "scripts.odom_accuracy_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

info "Starting odometry-accuracy GUI (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s)..."
exec python -u scripts/odom_accuracy_gui.py \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 1.2 \
    "$@"
