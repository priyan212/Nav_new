"""Offline test of DinoNavDPPipeline on a saved rover frame.

Checks state machine, command direction (should steer toward the detected
target), and per-stage latency. Saves data/pipeline_test.png.

  python scripts/test_pipeline_offline.py --target "trash bin"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/current_img.jpg")
    ap.add_argument("--target", default="trash bin")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    pipe = DinoNavDPPipeline(PipelineConfig())

    res = None
    for i in range(args.steps):  # repeated steps fill frame memory + warm cuda
        res = pipe.step(rgb, args.target)
        print(
            f"step {i}: state={res.state} v={res.linear:.3f} w={res.angular:+.3f} "
            f"goal={None if res.goal_point is None else np.round(res.goal_point, 2)} "
            f"timing={ {k: round(v, 2) for k, v in res.timing.items()} }"
        )

    if res.state == "TRACK":
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(rgb)
        if res.detection is not None:
            x0, y0, x1, y1 = res.detection.box
            axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, ec="lime", fc="none", lw=2))
        axes[0].set_title(f"target: {args.target} | v={res.linear:.2f} w={res.angular:+.2f}")
        axes[0].axis("off")

        ax = axes[1]
        for t in res.all_trajectories:
            ax.plot(-t[:, 1], t[:, 0], color="lightgray", lw=1)
        t = res.trajectory
        ax.plot(-t[:, 1], t[:, 0], color="tab:red", lw=2.5, label="selected")
        g = res.goal_point
        ax.scatter([-g[1]], [g[0]], marker="*", s=250, c="gold", ec="k", label="goal")
        ax.scatter([0], [0], marker="s", c="k", s=40)
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("selected trajectory (top-down, x fwd / y left)")
        ax.set_xlabel("y left->right flipped (m)")
        plt.tight_layout()
        plt.savefig("data/pipeline_test.png", dpi=110)
        print("saved -> data/pipeline_test.png")

        # direction sanity: goal right of center -> angular should be negative
        # (ROS convention: +z angular = CCW/left)
        print(f"SANITY: goal y={g[1]:+.2f} (left+), commanded w={res.angular:+.3f}")


if __name__ == "__main__":
    main()
