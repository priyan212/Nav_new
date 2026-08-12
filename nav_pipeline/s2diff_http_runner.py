"""New, opt-in entry point: nav_pipeline.isaac_gui with NavDP sampling routed
through tryout/navdp_s2diff_server.py over HTTP, instead of in-process.

    python -u -m nav_pipeline.s2diff_http_runner [same args as nav_pipeline.isaac_gui] \\
        --server-url http://127.0.0.1:8888

The server must already be running (see tryout/S2DIFF_GUIDANCE.md's Run
section) -- this only connects to it. Launched by
LAUNCH/launch_rover_s2diff_http.sh. Gives you the exact same GUI as
nav_pipeline.isaac_gui / LAUNCH/launch_rover.sh (camera feed, SAM mask, DINO
bbox, top-down trajectory/obstacle plot) since it's the same, unmodified
isaac_gui.main() underneath -- only nav_pipeline/s2diff_http_client.py's
sample_pointgoal patch differs from the plain launcher.
"""

from __future__ import annotations

import argparse
import sys

from .obstacle_guard import GuardConfig
from .s2diff_http_client import patch_navdp_standalone_http


def _parse_known_geometry_args(argv: list[str]):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--server-url", default="http://127.0.0.1:8888")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length)
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width)
    ap.add_argument("--stop-threshold", type=float, default=0.3)
    args, _unknown = ap.parse_known_args(argv)
    return args


def main() -> None:
    args = _parse_known_geometry_args(sys.argv[1:])
    guard = GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width)
    patch_navdp_standalone_http(
        server_url=args.server_url, fov_deg=args.fov, stop_threshold=args.stop_threshold, guard=guard
    )
    print(f"[s2diff-http] NavDP sampling routed to {args.server_url} (fov={args.fov})")

    # --server-url/--stop-threshold aren't among isaac_gui's own flags (its
    # argparse is strict, not parse_known_args) -- strip both the flag AND
    # its value before handing sys.argv off, or it'll error out.
    _strip_flags = {"--server-url", "--stop-threshold"}
    rest = sys.argv[1:]
    filtered = []
    skip_next = False
    for tok in rest:
        if skip_next:
            skip_next = False
            continue
        if tok in _strip_flags:
            skip_next = True
            continue
        filtered.append(tok)
    sys.argv = [sys.argv[0]] + filtered

    from .isaac_gui import main as isaac_gui_main

    isaac_gui_main()


if __name__ == "__main__":
    main()
