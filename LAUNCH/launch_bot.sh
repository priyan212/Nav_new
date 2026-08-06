#!/usr/bin/env bash
# ============================================================
#  Nav_new — A/B robot launcher (old ESP32 rover  <->  Hiwonder LanderPi)
#  Run on the GPU machine:
#    ./launch_bot.sh --rover    [PI_IP] [gui args...]
#    ./launch_bot.sh --hiwonder [PI_IP] [gui args...]
#
#  Same Nav_new manual-control + Go-Home GUI (nav_pipeline.home_gui) either
#  way -- pass --enable-obstacle-avoidance for full DINO+NavDP driving, same
#  as launch_rover_home.sh. The two backends speak the identical Zenoh
#  contract (see nav_pipeline/zenoh_node.py's docstring), so pipeline.py /
#  obstacle_guard.py / odometry_logger.py are unchanged either way -- only
#  what's running on the Pi differs:
#    --rover     : old 6WD rover, ESP32 micro-ROS + zenoh-bridge-ros2dds,
#                  systemd services (rover-camera/agent/zenoh), REAL
#                  measured wheel-encoder RPM.
#    --hiwonder  : new Hiwonder LanderPi (Mecanum), stock ROS1 Noetic stack
#                  in Hiwonder's own `armpi_pro` Docker container (UNTOUCHED)
#                  + landerpi/bridge.py as the only added file, OPEN-LOOP
#                  synthetic odometry (no wheel feedback on this hardware --
#                  see landerpi/README.md).
#
#  Examples:
#    ./launch_bot.sh --rover
#    ./launch_bot.sh --hiwonder
#    ./launch_bot.sh --hiwonder 192.168.0.8 --enable-obstacle-avoidance
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND=""
for arg in "$@"; do
    case "$arg" in
        --rover) BACKEND=rover ;;
        --hiwonder) BACKEND=hiwonder ;;
    esac
done
if [[ -z "$BACKEND" ]]; then
    echo "Usage: $0 --rover|--hiwonder [PI_IP] [gui args...]" >&2
    exit 1
fi
# strip the backend flag out of "$@" before passing the rest through
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--rover" || "$arg" == "--hiwonder" ]] || ARGS+=("$arg")
done
set -- "${ARGS[@]}"

if [[ "$BACKEND" == "rover" ]]; then
    DEFAULT_IP=10.47.234.125
    PI_PASS_DEFAULT=hri
    FOV=60
else
    DEFAULT_IP=192.168.0.8
    PI_PASS_DEFAULT=raspberrypi
    FOV=64.6   # from /usb_cam/camera_info: fx=507.2, width=640 -> 2*atan(320/507.2)
    # ArmPi Pro spec sheet (thinkrobotics.com): 298x256x521mm, 3.6kg -- 521mm
    # includes the raised arm, irrelevant to ground footprint. 298x256 is
    # from ONE retailer listing, not independently measured on this unit --
    # double-check with a tape measure before trusting obstacle avoidance.
    FOOTPRINT_LENGTH=0.298
    FOOTPRINT_WIDTH=0.256
fi

if [[ "${1:-}" == -* ]]; then
    PI_IP=$DEFAULT_IP
else
    PI_IP=${1:-$DEFAULT_IP}; shift 2>/dev/null || true
fi
PI_USER=pi
PI_PASS=${PI_PASS:-$PI_PASS_DEFAULT}
SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new $PI_USER@$PI_IP"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }

ENABLE_AVOID=false
for arg in "$@"; do
    [[ "$arg" == "--enable-obstacle-avoidance" ]] && ENABLE_AVOID=true
done

# ── 1. Reach the Pi ───────────────────────────────────────────
info "[$BACKEND] Pinging Pi at $PI_IP ..."
ping -c1 -W2 "$PI_IP" >/dev/null || { warn "Pi unreachable — is it ON and on this network?"; exit 1; }
$SSH 'echo ssh_ok' | grep -q ssh_ok || { warn "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
ok "Pi reachable"

if [[ "$BACKEND" == "rover" ]]; then
    # ── 2. Restart the Pi systemd services ──────────────────────
    info "Restarting rover services on Pi..."
    if $ENABLE_AVOID; then
        SERVICES="rover-camera rover-agent rover-zenoh"; N=3
    else
        SERVICES="rover-agent rover-zenoh"; N=2
    fi
    $SSH "echo $PI_PASS | sudo -S systemctl restart $SERVICES 2>/dev/null; sleep 4; systemctl is-active $SERVICES" \
        | grep -c active | grep -q $N && ok "services active: $SERVICES" \
        || warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"

    info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
    if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
        | grep -q "layout\|data"; then
        ok "ESP32 alive (/rover/rpm publishing)"
    else
        warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
    fi
else
    # ── 2. (Re)start landerpi/bridge.py inside the armpi_pro container ──
    info "Deploying/restarting the Nav_new bridge on the LanderPi..."
    # deploy_bridge.sh already confirms the bridge is listening on tcp/7447
    # (the authoritative check -- rospy routes loginfo through its own
    # logger, not reliably to this redirected stdout log, so grepping the
    # log file for a startup banner here was a flaky, redundant check).
    PI_PASS=$PI_PASS landerpi/deploy_bridge.sh "$PI_IP" || { warn "bridge deploy failed — see landerpi/README.md"; exit 1; }
fi

# ── 3. Launch the GUI ───────────────────────────────────────────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u
if $ENABLE_AVOID; then
    export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
    export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}
fi

pkill -f "nav_pipeline.home_gui" 2>/dev/null && sleep 1

EXTRA_ARGS=()
if $ENABLE_AVOID; then
    EXTRA_ARGS+=(--fov "$FOV" --compressed-only)
    [[ "$BACKEND" == "hiwonder" ]] && EXTRA_ARGS+=(--footprint-length "$FOOTPRINT_LENGTH" --footprint-width "$FOOTPRINT_WIDTH")
fi
info "Starting Nav_new manual-control + Go-Home GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / 0.5 rad/s$($ENABLE_AVOID && echo ', obstacle avoidance on'))..."
exec python -u -m nav_pipeline.home_gui \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 0.5 \
    --home-max-linear 0.15 --home-max-angular 0.5 \
    "${EXTRA_ARGS[@]}" \
    "$@"
