#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "VLM nav interatctive frame capture?"
echo ""

if ! conda env list | grep -q "^habitat "; then
    echo "[ERROR] 'habitat' conda environment not found."
    echo "Please create it first with: conda install -c aihabitat -c conda-forge habitat-sim"
    exit 1
fi

echo "[inf] Running verification checks..."
python3 verify_vlm_nav_setup.py

echo ""
echo "[inf] Activating habitat environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate habitat

echo "[inf] Starting vlm_nav_interactive.py..."
echo ""
python vlm_nav_interactive.py

echo ""
echo "[inf] capture sesh done."
