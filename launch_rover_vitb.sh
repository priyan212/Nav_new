#!/usr/bin/env bash
# Same as launch_rover.sh, but using the more accurate Depth Anything V2
# "Base" (vitb) encoder for monocular metric depth instead of the default
# "Small" (vits) one -- roughly 2x slower inference, but worth it since
# depth error feeds directly into the STOP distance decision (see
# nav_pipeline/depth_estimator.py's _ENCODER_CONFIGS).
#
# Needs checkpoints/depth_anything_v2_metric_hypersim_vitb.pth (already
# present in this repo; if missing elsewhere:
#   python scripts/download_models.py --depth-encoder vitb
#
# Examples:
#   ./launch_rover_vitb.sh                          # default Pi IP (10.47.234.125)
#   ./launch_rover_vitb.sh 10.47.234.125 --target "trash bin"
set -uo pipefail
cd "$(dirname "$0")"
exec ./launch_rover.sh "$@" --depth-encoder vitb
