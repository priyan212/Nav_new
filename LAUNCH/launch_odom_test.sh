#!/usr/bin/env bash
# ============================================================
#  Nav_new — odometry-accuracy test launcher (real rover or LanderPi)
#  Run on the GPU machine:
#    ./launch_odom_test.sh [--rover|--hiwonder] [PI_IP] [gui args...]
#
#  Brings up the Pi (no camera needed for this test), then starts
#  scripts/odom_accuracy_gui.py: a standalone GUI (no DINO/SAM/NavDP/depth
#  models) for measuring how far dead-reckoned /rover/rpm odometry drifts
#  from hand-measured ground truth. See belief_eval_20260730/RESULTS.md,
#  "The caveat that matters most for the real rover" -- this is the
#  real-hardware check behind whether porting SubgoalBeliefBank's
#  ego-motion correction is worth it. Also the right tool to re-validate
#  landerpi/bridge.py's encoder-based odometry after any change to it (see
#  landerpi/README.md's "Odometry" section for the story behind that).
#  Defaults to --rover (no flag = old behavior, unchanged).
#
#  Examples:
#    ./launch_odom_test.sh                          # default rover, default Pi IP
#    ./launch_odom_test.sh 10.47.234.125 --max-angular 1.0
#    ./launch_odom_test.sh --hiwonder
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source LAUNCH/_backend.sh
backend_parse_args "$@"
set -- "${BACKEND_ARGS[@]}"

backend_bringup nocamera

# ── Launch the odometry-accuracy GUI ───────────────────────
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u

pkill -f "scripts.odom_accuracy_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.isaac_gui" 2>/dev/null && sleep 1
pkill -f "nav_pipeline.zenoh_node" 2>/dev/null && sleep 1

info "Starting odometry-accuracy GUI [$BACKEND] (pi-ip=$PI_IP, caps 0.15 m/s / 1.2 rad/s)..."
exec python -u scripts/odom_accuracy_gui.py \
    --pi-ip "$PI_IP" \
    --max-linear 0.15 --max-angular 1.2 \
    "$@"
