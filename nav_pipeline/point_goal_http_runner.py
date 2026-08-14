"""Point-goal benchmark (nav_pipeline.point_goal_gui) with NavDP trajectory
sampling routed through tryout/navdp_s2diff_server.py over HTTP, instead of
in-process -- the exact same monkeypatch nav_pipeline.s2diff_http_runner
uses for the normal DINO-target GUI (see that file's docstring for why).

    python -u -m nav_pipeline.point_goal_http_runner [same args as
        nav_pipeline.point_goal_gui] --server-url http://127.0.0.1:8888

The server must already be running (see tryout/S2DIFF_GUIDANCE.md's Run
section) -- this only connects to it. Launched by
LAUNCH/launch_point_goal.sh. Gives you the exact same point-goal GUI as
nav_pipeline.point_goal_gui -- only nav_pipeline/s2diff_http_client.py's
sample_pointgoal patch differs from a plain policy_type="extracted" run.
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

    # --server-url/--stop-threshold aren't among point_goal_gui's own flags
    # (its argparse is strict, not parse_known_args) -- strip both the flag
    # AND its value before handing sys.argv off, or it'll error out.
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

    from .point_goal_gui import main as point_goal_gui_main

    point_goal_gui_main()


if __name__ == "__main__":
    main()
