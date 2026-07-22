#!/usr/bin/env bash
# One-command bringup for the Mars habitat sim + DINO+NavDP GUI.
#   ./MARS/launch_mars.sh              # rock_envs/run1 rocks load by default
#   ./MARS/launch_mars.sh --no-rocks   # empty yard, no rocks
#   ./MARS/launch_mars.sh --belief-only                     # goal locked to belief (scripts/mars_belief_only_gui.py)
#   ./MARS/launch_mars.sh --belief-only --distractor-gate 2.0 --target "boulder"
#   ./MARS/launch_mars.sh --max-climb-deg 25               # rover balking at climbing slopes/hills? raise this
# --belief-only and --no-rocks/--rocks can be combined in any order; anything
# left over (--target, --distractor-gate, --max-climb-deg, --max-linear, ...)
# is forwarded to the GUI script unchanged.
# Habitat sim node runs in mars_habitat, the GUI in internnav.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/i3d/exit/etc/profile.d/conda.sh
# NOTE: no `set -u` around here on purpose -- the internnav env's
# activate.d/env_vars.sh chains into a stale Isaac Sim setup script
# (/mnt/bigdisk/motion_planning/Navigation/isaacsim/setup_conda_env.sh,
# a leftover from a deleted install) that references several unset shell
# vars (ZSH_VERSION, PYTHONPATH, ...). That script isn't ours to patch,
# so this launcher just tolerates unset vars instead.
set -e

ROCKS_ARG="--rocks $HERE/mars-habitatsim/rock_envs/run1/rock_field.json"
GUI_SCRIPT="$HERE/scripts/mars_gui.py"

# consume launcher-level flags (any order); everything left over is passed
# straight through to the GUI script as "$@" below.
while [[ $# -gt 0 ]]; do
    case "${1:-}" in
        --no-rocks)
            ROCKS_ARG=""
            shift
            ;;
        --rocks)
            shift   # accepted for backwards compatibility; rocks are now the default
            ;;
        --belief-only)
            # goal locked to the belief; DINO/SAM still run every frame but a
            # detection only updates the goal if it's near the belief's
            # predicted position (see scripts/mars_belief_only_gui.py)
            GUI_SCRIPT="$HERE/scripts/mars_belief_only_gui.py"
            shift
            ;;
        *)
            break
            ;;
    esac
done

echo "[1/2] starting habitat sim node (mars_habitat env)..."
conda activate mars_habitat
python -u "$HERE/scripts/habitat_sim_node.py" $ROCKS_ARG &> /tmp/mars_habitat_node.log &
NODE_PID=$!
trap 'echo "[cleanup] stopping sim node"; kill $NODE_PID 2>/dev/null' EXIT

# wait for the node to come up (scene + rocks can take ~15 s)
for i in $(seq 1 60); do
    grep -q "zenoh up" /tmp/mars_habitat_node.log 2>/dev/null && break
    if ! kill -0 $NODE_PID 2>/dev/null; then
        echo "ERROR: sim node died — tail of /tmp/mars_habitat_node.log:"
        tail -20 /tmp/mars_habitat_node.log
        exit 1
    fi
    sleep 1
done
echo "    sim node up (log: /tmp/mars_habitat_node.log)"

echo "[2/2] starting Mars GUI (internnav env)... ($(basename "$GUI_SCRIPT"))"
conda deactivate
conda activate internnav
cd "$HERE/.."
python "$GUI_SCRIPT" "$@"
