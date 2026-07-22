#!/usr/bin/env bash
# Launch the DINO + NavDP navigation node (GPU side).
#
# Usage:
#   ./launch_dino_navdp.sh                          # multicast discovery, default target
#   ./launch_dino_navdp.sh --target "red chair"     # custom target
#   ./launch_dino_navdp.sh --pi-ip 192.168.1.42 --target "trash bin"
#
# The rover/Isaac side is unchanged from the OmniVLA pipeline: it must run the
# zenoh-bridge-ros2dds (or the Isaac zenoh bridge) publishing the camera and
# subscribing to cmd_vel.

set -eo pipefail
cd "$(dirname "$0")"

source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav

export HF_HOME=${HF_HOME:-/mnt/bigdisk/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/bigdisk/hf_cache/transformers}

exec python -m nav_pipeline.zenoh_node "$@"
