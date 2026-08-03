#!/usr/bin/env bash
# ============================================================
#  Nav_new — REAL ROVER launcher (manual control + Go Home)
#  Run on the GPU machine:  ./launch_rover_home.sh [PI_IP] [gui args...]
#
#  Brings up the Pi (ESP32 micro-ROS + Zenoh bridge only -- no camera, this
#  GUI doesn't use vision at all), then starts nav_pipeline.home_gui: a
#  manual-drive control panel plus a "GO HOME" button that drives the rover
#  back to wherever it was when this launched (or wherever "Set Home Here"
#  was last pressed), using fused wheel-encoder + BNO055 IMU odometry.
#
#  No DINO/NavDP/SAM/CLIP/depth model loading -- starts in ~1s, no GPU
#  needed. Can be run from any Python env with `eclipse-zenoh` installed,
#  not just the internnav conda env (still activated below for consistency
#  with the other launch scripts / to guarantee zenoh is present).
#
#  Examples:
#    ./launch_rover_home.sh                          # default Pi IP
#    ./launch_rover_home.sh 10.47.234.125 --home-dist-tol 0.05
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == -* ]]; then
    PI_IP=10.47.234.125
else
    PI_IP=${1:-10.47.234.125}; shift 2>/dev/null || true
fi
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

# ── 2. Restart the ESP32 bridge + Zenoh bridge (no camera needed here) ──
info "Restarting rover-agent + rover-zenoh on Pi..."
$SSH "echo $PI_PASS | sudo -S systemctl restart rover-agent rover-zenoh 2>/dev/null; sleep 4; systemctl is-active rover-agent rover-zenoh" \
    | grep -c active | grep -q 2 && ok "services active: agent (ESP32 auto-reset), zenoh" \
    || { warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"; }

# ── 3. Verify the encoder+IMU feed is flowing ──────────────────
info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
    | grep -q "layout\|data"; then
    ok "ESP32 alive (/rover/rpm publishing: left_rpm, right_rpm, imu_heading_deg, imu_calib)"
else
    warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
fi

# ── 4. Launch the GUI ───────────────────────────────────────────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u

pkill -f "nav_pipeline.home_gui" 2>/dev/null && sleep 1

info "Starting Nav_new manual-control + Go-Home GUI (pi-ip=$PI_IP, caps 0.15 m/s / 0.5 rad/s)..."
exec python -u -m nav_pipeline.home_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 0.5 \
    --home-max-linear 0.15 --home-max-angular 0.5 \
    "$@"
