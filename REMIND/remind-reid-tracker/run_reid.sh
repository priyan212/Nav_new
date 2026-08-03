#!/usr/bin/env bash
# Self-contained launcher for the two-video re-identification demo:
# video1 catalogues a room, video2 (a revisit / different angle) is matched
# against that same in-memory catalogue. See scripts/run_reid_across_videos.py.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${REPO_DIR}/.venv"

export HF_HOME="${REPO_DIR}/.cache/huggingface"
export YOLO_CONFIG_DIR="${REPO_DIR}/.cache/ultralytics"
mkdir -p "${HF_HOME}" "${YOLO_CONFIG_DIR}"

cd "${REPO_DIR}"
exec python scripts/run_reid_across_videos.py "$@"
