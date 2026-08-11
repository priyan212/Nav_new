#!/usr/bin/env bash
# ============================================================
#  Nav_new — Hiwonder bot on InternVLA-N1's native DualVLN dual-system
#  Run on the GPU machine:
#    ./launch_hiwonder_dualvln.sh [--rover|--hiwonder] [--no-gui] [PI_IP] [node args...]
#
#  Brings up the Pi (landerpi/deploy_bridge.sh for --hiwonder, the default
#  here -- see LAUNCH/_backend.sh / landerpi/README.md), starts the
#  standalone path viewer (reference/internvla_dualvln_gui.py) as a
#  background helper, then starts reference/internvla_dualvln_zenoh_node.py
#  in the foreground: the checkpoint's OWN System-2 (QwenVL-2.5-7B
#  pixel+latent goal) + System-1 (Diffusion Transformer trajectory)
#  dual-system, exactly as released/trained ("Ground Slow, Move Fast",
#  arXiv 2512.08186) -- unlike reference/internvla_zenoh_node.py, which
#  forces System-1 off or swaps in a different (NavDP) checkpoint. Suited
#  to long, compound, multi-landmark instructions.
#
#  The GUI needs a display (Tkinter) -- pass --no-gui to skip it (e.g. over
#  SSH without X11 forwarding); its failure is non-fatal to the node either
#  way (it's a background helper, not required for navigation).
#
#  Examples:
#    ./launch_hiwonder_dualvln.sh --instruction "walk through the opening \
#        between the kitchen and the dining room, turn right, go through \
#        the doorway and stop next to the closet"
#    ./launch_hiwonder_dualvln.sh 10.47.234.228 --instruction "..."
#    ./launch_hiwonder_dualvln.sh --rover --instruction "..."   # override backend
#    ./launch_hiwonder_dualvln.sh --no-gui --instruction "..."  # headless
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# --no-gui is this script's own flag (not _backend.sh's) -- strip it before
# backend_parse_args sees "$@", same way --rover/--hiwonder get stripped.
SHOW_GUI=true
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-gui" ]]; then
        SHOW_GUI=false
    else
        ARGS+=("$arg")
    fi
done

source LAUNCH/_backend.sh
backend_parse_args --hiwonder "${ARGS[@]}"   # defaults this launcher to Hiwonder; a
                                              # later --rover in "${ARGS[@]}" still
                                              # overrides (backend_parse_args keeps
                                              # the LAST flag)
set -- "${BACKEND_ARGS[@]}"

backend_bringup camera

# conda activation hooks reference unbound vars — relax nounset while
# sourcing them (same pattern as launch_rover.sh).
set +u
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
set -u

# This checkpoint requires transformers==4.51.0, shadowed over the
# internnav env's own (incompatible) transformers install -- see the
# node's module docstring / _apply_n1_compat_shims(). Deliberately a CLEAN
# assignment, NOT an append onto any inherited PYTHONPATH: this shell may
# already have Isaac Sim's exts/omni.isaac.ml_archive/pip_prebundle on
# PYTHONPATH from an earlier sourced setup script, which bundles its own
# (older/different) torch -- since PYTHONPATH entries take priority over
# the conda env's own site-packages, an inherited Isaac Sim path ahead of
# the internnav env's torch silently shadows it (symptom: DepthAnythingV2
# state-dict loading warns "copying from a non-meta parameter ... to a
# meta parameter ... no-op" because the wrong torch/module machinery is
# in play). Matches the module docstring's documented invocation exactly.
export PYTHONPATH="/home/i3d/internnav_n1_tf451"
export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

pkill -f "reference/internvla_dualvln_zenoh_node.py" 2>/dev/null && sleep 1
pkill -f "reference/internvla_dualvln_gui.py" 2>/dev/null && sleep 1

# ── Start the path viewer (background helper, same pattern as
#    launch_rover_remind.sh's REMIND server: background + trap cleanup,
#    NOT exec'd so this script stays alive to kill it on exit) ──
GUI_PID=""
if [[ "$SHOW_GUI" == "true" ]]; then
    GUI_LOG="/tmp/internvla_dualvln_gui_$(date +%Y%m%d_%H%M%S).log"
    info "Starting path viewer GUI (pi-ip=$PI_IP, log: $GUI_LOG)..."
    python -u reference/internvla_dualvln_gui.py --pi-ip "$PI_IP" \
        > "$GUI_LOG" 2>&1 &
    GUI_PID=$!
else
    info "GUI disabled (--no-gui)."
fi

cleanup() {
    if [[ -n "$GUI_PID" ]]; then
        kill "$GUI_PID" 2>/dev/null
        wait "$GUI_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# max-linear/max-angular deliberately NOT overridden from $BACKEND_MAX_ANGULAR:
# that constant is tuned for pipeline.py's own bearing-servo/EMA control
# scheme (a different pipeline). This node has its own independent motion
# smoothing (motion_ramp_time_s) and its own validated defaults
# (max_linear=0.15, max_angular=0.25), already more conservative than
# BACKEND_MAX_ANGULAR's 0.5 ceiling for --hiwonder.
#
# NOT exec'd (unlike launch_rover.sh): this script needs to stay alive as
# the node's parent so the trap above can clean up the background GUI when
# the node exits/is interrupted -- exec would replace this shell entirely
# and the trap would never fire (same reasoning as launch_rover_remind.sh).
NODE_LOG="/tmp/internvla_dualvln_node_$(date +%Y%m%d_%H%M%S).log"
info "Starting InternVLA-N1 DualVLN node [$BACKEND] (pi-ip=$PI_IP, native System-2+System-1 dual-system, log: $NODE_LOG)..."
# Every [PRED #n]/[STATUS] line (the only diagnostic trail that's ever
# actually explained a real bug on this bot) previously only went to
# whichever terminal happened to run this script -- unrecoverable the moment
# that shell closes. Tee it to a timestamped file too, always, so any run by
# anyone is diagnosable after the fact. `set -uo pipefail` above already
# makes pipefail active, so $? through this pipe still reflects the Python
# process's own exit status, not tee's.
python -u reference/internvla_dualvln_zenoh_node.py \
    --pi-ip "$PI_IP" \
    "$@" 2>&1 | tee "$NODE_LOG"
