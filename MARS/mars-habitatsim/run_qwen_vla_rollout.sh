#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Qwen VLA rollout"

if ! conda env list | grep -q "^habitat "; then
    echo "[ERROR] 'habitat' conda environment not found."
    exit 1
fi

if ! conda env list | grep -q "^qwen_vlm "; then
    echo "[ERROR] 'qwen_vlm' conda environment not found."
    echo "The qwen_server subprocess (transformers + Qwen2.5-VL) needs it."
    exit 1
fi

echo "[inf] Activating habitat environment."
CONDA_BASE="$(conda info --base 2>/dev/null | grep '^/')"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate habitat

# qwen_server_manager spawns the qwen_server subprocess itself, using the qwen_vlm env's python directly (see sam_vla/vlm/qwen_server_manager.py). Nothing extra to activate here for that half.

echo "[inf] Starting run_qwen_vla_rollout"
echo ""
python -m sam_vla.run_qwen_vla_rollout "$@"

echo ""
echo "[inf] rollout done."
