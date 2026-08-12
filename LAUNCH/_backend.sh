#!/usr/bin/env bash
# ============================================================
#  Shared --rover / --hiwonder backend selection + Pi bring-up.
#  Sourced by the other LAUNCH/*.sh scripts -- not meant to be run directly.
#
#  Usage in a launch script:
#    source "$(dirname "$0")/_backend.sh"
#    backend_parse_args "$@"
#    set -- "${BACKEND_ARGS[@]}"          # positional PI_IP + --rover/--hiwonder stripped
#    backend_bringup camera               # or: backend_bringup nocamera
#    ...
#    exec python -u -m nav_pipeline.xxx --pi-ip "$PI_IP" --fov "$BACKEND_FOV" ...
#
#  Defaults to --rover when neither flag is given, so every existing call
#  site (nobody passes --rover/--hiwonder today) is unaffected.
#
#  --hiwonder notes:
#    - Pi bring-up is landerpi/deploy_bridge.sh (bridge.py inside the
#      existing armpi_pro container), not systemd services -- see
#      landerpi/README.md.
#    - $BACKEND_FOV/$BACKEND_FOOTPRINT_LENGTH/$BACKEND_FOOTPRINT_WIDTH are
#      this robot's real measured/spec values (see landerpi/README.md).
#    - Steering-shape constants tuned specifically for the OLD rover's
#      camera/motor response (search-angular, servo-ramp-deg, the 1.2
#      max-angular normalization) are NOT re-validated for this chassis --
#      each launch script still passes its own tuned values verbatim, and
#      backend_bringup prints an explicit warning for --hiwonder callers
#      that pass them through unchanged, since they're a carried-over
#      starting point, not a measurement (see landerpi/README.md's "Known
#      caveats").
# ============================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

backend_parse_args() {
    BACKEND=""
    for arg in "$@"; do
        case "$arg" in
            --rover) BACKEND=rover ;;
            --hiwonder) BACKEND=hiwonder ;;
        esac
    done
    [[ -z "$BACKEND" ]] && BACKEND=rover

    BACKEND_ARGS=()
    for arg in "$@"; do
        [[ "$arg" == "--rover" || "$arg" == "--hiwonder" ]] || BACKEND_ARGS+=("$arg")
    done

    if [[ "$BACKEND" == "rover" ]]; then
        BACKEND_DEFAULT_IP=192.168.21.125   # updated 2026-08-11, Pi IP churns -- verify if unreachable
        BACKEND_PI_PASS_DEFAULT=hri
        # 60 is the Logitech webcam's FOV. Camera was briefly swapped to an
        # Intel RealSense D435i on 2026-08-11 (scripts/realsense_only_bringup.
        # launch.py, FOV=69, nominal HFOV at 640x480 -- see git history) but
        # that D435i never actually came up cleanly (USB2-only cable
        # link-negotiated at 480M in a USB3 port, failing its XU control
        # queries) and the Pi has been switched back to the RGB webcam
        # (v4l2_camera_node) as of 2026-08-11. Re-measure with
        # 2*atan(320/fx) from /image_raw/camera_info if this drifts.
        BACKEND_FOV=60
        # Matches the ESP32 firmware's own angular normalization (see
        # launch_rover.sh's comment) -- real, measured tuning for this bot.
        BACKEND_MAX_ANGULAR=1.2
        BACKEND_ANGULAR_SLEW_MAX=0.10   # pipeline.py's own default, unchanged
    else
        BACKEND_DEFAULT_IP=10.47.234.228
        BACKEND_PI_PASS_DEFAULT=raspberrypi
        BACKEND_FOV=64.6   # from /usb_cam/camera_info: fx=507.2, width=640 -> 2*atan(320/507.2)
        BACKEND_FOOTPRINT_LENGTH=0.298
        BACKEND_FOOTPRINT_WIDTH=0.256
        # 1.2 was only ever the ROVER's ESP32-normalization value, carried
        # over unvalidated (see warning below) -- reduced 2026-08-07 to the
        # same conservative cap already live-validated on this exact chassis
        # via launch_bot.sh/home_gui.py (README: "Velocity caps start at the
        # same conservative real-rover defaults"). Bump this specifically
        # (not the rover's) if 0.5 turns out too slow once re-tuned properly.
        BACKEND_MAX_ANGULAR=0.5
        # Reduced 2026-08-10 from pipeline.py's 0.10 default: a fresh
        # SEARCH->TRACK correction (bearing error is large right after a
        # glimpse at frame edge) commands a real turn, and that turn's own
        # viewpoint change was breaking the very re-identification continuity
        # (IoU + DINOv2 appearance similarity, see PipelineConfig's
        # appearance_min_similarity/appearance_iou_override comments) needed
        # to confirm the lock -- symptom: bot "gets excited" toward the
        # object, swings, loses it mid-swing, reverts to SEARCH, repeats.
        # Halving the slew rate spreads the same correction over ~2x the
        # ticks, so each tick's viewpoint change is smaller and more likely
        # to stay within the IoU-override/appearance-similarity bounds.
        BACKEND_ANGULAR_SLEW_MAX=0.05
    fi

    if [[ "${BACKEND_ARGS[0]:-}" == -* || -z "${BACKEND_ARGS[0]:-}" ]]; then
        PI_IP=$BACKEND_DEFAULT_IP
    else
        PI_IP=${BACKEND_ARGS[0]}
        BACKEND_ARGS=("${BACKEND_ARGS[@]:1}")
    fi
    PI_USER=pi
    PI_PASS=${PI_PASS:-$BACKEND_PI_PASS_DEFAULT}
    SSH="sshpass -p $PI_PASS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new $PI_USER@$PI_IP"
}

# backend_bringup camera|nocamera
backend_bringup() {
    local need_camera=$1

    info "[$BACKEND] Pinging Pi at $PI_IP ..."
    ping -c1 -W2 "$PI_IP" >/dev/null || { warn "Pi unreachable — is it ON and on this network?"; exit 1; }
    $SSH 'echo ssh_ok' | grep -q ssh_ok || { warn "SSH failed (user=$PI_USER pass=\$PI_PASS)"; exit 1; }
    ok "Pi reachable"

    if [[ "$BACKEND" == "rover" ]]; then
        local services n
        if [[ "$need_camera" == "camera" ]]; then
            services="rover-camera rover-agent rover-zenoh"; n=3
        else
            services="rover-agent rover-zenoh"; n=2
        fi
        info "Restarting rover services on Pi..."
        $SSH "echo $PI_PASS | sudo -S systemctl restart $services 2>/dev/null; sleep 4; systemctl is-active $services" \
            | grep -c active | grep -q "$n" && ok "services active: $services" \
            || warn "services not all active — run: bash scripts/pi_install_services.sh on the Pi"

        if [[ "$need_camera" == "camera" ]]; then
            info "Waiting for camera topic (up to 25 s)..."
            local cam_ok=false
            for i in $(seq 1 25); do
                sleep 1
                $SSH 'bash -lc "source /opt/ros/humble/setup.bash; ros2 topic list 2>/dev/null"' 2>/dev/null \
                    | grep -q "/image_raw" && { cam_ok=true; ok "camera live [${i}s]"; break; }
            done
            $cam_ok || warn "camera topic missing — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -n 20'"
        fi

        info "Checking ESP32 heartbeat (/rover/rpm, up to 25 s)..."
        if $SSH 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
            | grep -q "layout\|data"; then
            ok "ESP32 alive (/rover/rpm publishing)"
        else
            warn "No /rover/rpm — check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
        fi
    else
        info "Deploying/restarting the Nav_new bridge on the LanderPi..."
        PI_PASS=$PI_PASS "$(dirname "${BASH_SOURCE[0]}")/../landerpi/deploy_bridge.sh" "$PI_IP" \
            || { warn "bridge deploy failed — see landerpi/README.md"; exit 1; }
        warn "steering-shape constants below (search-angular/servo-ramp-deg/max-angular normalization) were tuned for the OLD rover's camera/motor response, NOT re-validated on this chassis -- see landerpi/README.md"
    fi
}
