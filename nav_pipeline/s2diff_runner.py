"""New, opt-in entry point: nav_pipeline.isaac_gui with S2Diff-guided NavDP sampling.

    python -u -m nav_pipeline.s2diff_runner [same args as nav_pipeline.isaac_gui]

Launched by LAUNCH/launch_rover_s2diff.sh. The plain LAUNCH/launch_rover*.sh
scripts still call `nav_pipeline.isaac_gui` directly and are untouched --
they keep NavDP's original unguided sampling.

This module does not edit nav_pipeline/isaac_gui.py, navdp_net.py, or
pipeline.py. It pre-parses the subset of isaac_gui's own CLI flags that
affect obstacle geometry (--fov, --footprint-length, --footprint-width) so
the guidance module's intrinsics/footprint match what pipeline.py itself
will use, installs the guided sample_pointgoal via
s2diff_navdp.patch_navdp_standalone(), then hands the untouched argv to
nav_pipeline.isaac_gui.main() exactly as the plain launcher would.
"""

from __future__ import annotations

import argparse
import sys

from .obstacle_guard import GuardConfig
from .s2diff_navdp import patch_navdp_standalone


def _parse_known_geometry_args(argv: list[str]):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length)
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width)
    args, _unknown = ap.parse_known_args(argv)
    return args


def main() -> None:
    args = _parse_known_geometry_args(sys.argv[1:])
    guard = GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width)
    patch_navdp_standalone(fov_deg=args.fov, guard=guard)
    print(f"[s2diff] guided NavDP sampling enabled (fov={args.fov}, footprint={args.footprint_length}x{args.footprint_width})")

    from .isaac_gui import main as isaac_gui_main

    isaac_gui_main()


if __name__ == "__main__":
    main()
