#!/usr/bin/env bash
# Same as launch_rover_s2diff.sh, but using the more accurate Depth Anything
# V2 "Base" (vitb) encoder -- see LAUNCH/launch_rover_vitb.sh for why. Combine
# both experimental changes (better depth + S2Diff-guided NavDP) in one launch.
#
# Examples:
#   ./launch_rover_vitb_s2diff.sh                          # default Pi IP (192.168.21.125)
#   ./launch_rover_vitb_s2diff.sh 192.168.21.125 --target "trash bin"
set -uo pipefail
cd "$(dirname "$0")"
exec ./launch_rover_s2diff.sh "$@" --depth-encoder vitb
