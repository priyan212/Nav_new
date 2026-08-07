#!/usr/bin/env bash
# ============================================================
#  Deploy/restart the Nav_new <-> LanderPi Zenoh bridge.
#  Run on the GPU machine:  ./deploy_bridge.sh [PI_IP]
#
#  Copies landerpi/bridge.py onto the Pi and (re)starts it inside the
#  existing, UNMODIFIED `armpi_pro` ROS1 Docker container via `docker exec
#  -d`. Never touches any Hiwonder file/package/service -- purely additive.
#  Idempotent: safe to re-run any time (kills any previous bridge instance
#  first).
#
#  Prereq (one-time, see landerpi/README.md): a Python 3.10 venv at
#  ~/nav_new_bridge/venv310 (via `uv`) with eclipse-zenoh==1.9.0 + rospy's
#  small pure-Python deps installed -- the container's system Python is 3.8,
#  which cannot run current eclipse-zenoh at all (PyO3 needs >=3.9). This
#  script does NOT set that up; it will fail loudly and tell you if it's
#  missing.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

PI_IP=${1:-10.47.234.228}
PI_USER=pi
PI_PASS=${PI_PASS:-raspberrypi}
SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new $PI_USER@$PI_IP"
SCP="sshpass -p $PI_PASS scp -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

info "Pinging LanderPi at $PI_IP ..."
ping -c1 -W2 "$PI_IP" >/dev/null || { err "Pi unreachable — is it ON and on this network? (IP may have changed, see landerpi/README.md)"; exit 1; }
$SSH 'echo ssh_ok' | grep -q ssh_ok || { err "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
ok "Pi reachable"

BRIDGE_PY="/home/ubuntu/nav_new_bridge/venv310/bin/python3.10"
ENV_SCRIPT="/home/ubuntu/armpi_pro/src/armpi_pro_bringup/scripts/source_env.bash"
info "Checking the bridge's Python 3.10 venv is set up in the armpi_pro container..."
$SSH "sudo -n docker exec -u ubuntu -w /home/ubuntu armpi_pro /bin/bash -c \
    'source $ENV_SCRIPT $BRIDGE_PY -c \"import zenoh, rospy\"'" >/dev/null 2>&1 \
    || { err "venv310 missing or broken. One-time setup needed — see landerpi/README.md ('Bridge dependencies')."; exit 1; }
ok "venv310 present (zenoh + rospy importable)"

info "Copying bridge.py to Pi, then into the armpi_pro container (separate filesystems -- only networking is shared via --network host)..."
$SSH "mkdir -p ~/nav_new_bridge" && $SCP bridge.py "$PI_USER@$PI_IP:~/nav_new_bridge/bridge.py" \
    && $SSH "sudo -n docker cp ~/nav_new_bridge/bridge.py armpi_pro:/home/ubuntu/nav_new_bridge/bridge.py && sudo -n docker exec -u root armpi_pro chown ubuntu:ubuntu /home/ubuntu/nav_new_bridge/bridge.py" \
    && ok "copied" || { err "scp/docker cp failed"; exit 1; }

info "Stopping any previous bridge instance..."
# -9/SIGKILL: rospy installs its own SIGTERM handler for graceful shutdown
# that doesn't reliably exit the process here (observed hanging, still
# holding tcp/7447, after a plain pkill claimed success) -- SIGKILL is
# unconditional and safe for a stateless bridge restart.
$SSH "sudo -n docker exec -u ubuntu armpi_pro pkill -9 -f nav_new_bridge/bridge.py 2>/dev/null; sleep 1; true"

info "Starting bridge inside armpi_pro container (detached, Python 3.10 venv)..."
$SSH "sudo -n docker exec -d -u ubuntu -w /home/ubuntu armpi_pro /bin/bash -c \
    'source /home/ubuntu/armpi_pro/src/armpi_pro_bringup/scripts/source_env.bash \
     $BRIDGE_PY /home/ubuntu/nav_new_bridge/bridge.py > /home/ubuntu/nav_new_bridge/bridge.log 2>&1'"
sleep 2

info "Verifying bridge is listening on tcp/$PI_IP:7447 ..."
for i in $(seq 1 10); do
    (echo > /dev/tcp/$PI_IP/7447) >/dev/null 2>&1 && { ok "bridge listening [${i}s]"; exit 0; }
    sleep 1
done
err "bridge did not come up — check: ssh $PI_USER@$PI_IP \"sudo docker exec -u ubuntu armpi_pro cat /home/ubuntu/nav_new_bridge/bridge.log\""
exit 1
