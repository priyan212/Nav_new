#!/usr/bin/env bash
# One-command bringup for the Earth (Indian Bend & Pima) habitat sim + DINO+NavDP GUI.
#   ./EARTH/launch_earth.sh
#   ./EARTH/launch_earth.sh --target "target sign"
#   ./EARTH/launch_earth.sh --max-climb-deg 25   # rover balking at climbing curbs/mounds? raise this
# Anything passed on the command line (--target, --max-linear, ...) is
# forwarded to the GUI script unchanged.
# Habitat sim node runs in mars_habitat (shared conda env with MARS -- same
# habitat-sim/habitat-lab install), the GUI in internnav.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/i3d/exit/etc/profile.d/conda.sh
# NOTE: no `set -u` around here on purpose -- see MARS/launch_mars.sh; the
# internnav env's activate.d/env_vars.sh chains into a stale Isaac Sim setup
# script that references unset shell vars, and that script isn't ours to patch.
set -e

echo "[1/2] starting habitat sim node (mars_habitat env)..."
conda activate mars_habitat
python -u "$HERE/scripts/habitat_sim_node.py" &> /tmp/earth_habitat_node.log &
NODE_PID=$!
trap 'echo "[cleanup] stopping sim node"; kill $NODE_PID 2>/dev/null' EXIT

# wait for the node to come up (scene load can take a few seconds)
for i in $(seq 1 60); do
    grep -q "zenoh up" /tmp/earth_habitat_node.log 2>/dev/null && break
    if ! kill -0 $NODE_PID 2>/dev/null; then
        echo "ERROR: sim node died — tail of /tmp/earth_habitat_node.log:"
        tail -20 /tmp/earth_habitat_node.log
        exit 1
    fi
    sleep 1
done
echo "    sim node up (log: /tmp/earth_habitat_node.log)"

echo "[2/2] starting Earth GUI (internnav env)..."
conda deactivate
conda activate internnav
cd "$HERE/.."
python "$HERE/scripts/earth_gui.py" "$@"
