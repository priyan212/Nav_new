#!/usr/bin/env bash
# Self-contained REMIND launcher: activates the repo-local env and points
# every runtime cache (HuggingFace weights, Ultralytics config) inside this
# repo directory instead of the user's home directory.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${REPO_DIR}/.venv"

export HF_HOME="${REPO_DIR}/.cache/huggingface"
export YOLO_CONFIG_DIR="${REPO_DIR}/.cache/ultralytics"
mkdir -p "${HF_HOME}" "${YOLO_CONFIG_DIR}"

cd "${REPO_DIR}"
exec python main.py "$@"
