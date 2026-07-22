"""Offline smoke test: RGB image -> Grounding DINO -> goal -> NavDP -> trajectories.

Runs the full perception->policy chain on a saved rover frame with no robot,
no sim, no Zenoh. Validates:
  1. Grounding DINO finds the text target and returns a bbox.
  2. Depth Anything V2 produces metric depth (RGB-only fallback).
  3. NavDP (extracted System-1 weights) denoises trajectories for
     point-goal / pixel-goal / no-goal conditioning.
Outputs data/smoke_test_result.png with bbox, depth map and top-down
trajectory plots for visual inspection.

Run from Nav_new root:
  python scripts/smoke_test.py --image data/current_img.jpg --target "trash bin"
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nav_pipeline.dino_detector import GroundingDinoDetector
from nav_pipeline.depth_estimator import MetricDepthEstimator
from nav_pipeline.goal_utils import (
    goal_pixel_from_detection,
    goal_point_from_detection,
    intrinsics_from_fov,
    preprocess_depth,
    preprocess_rgb,
)
from nav_pipeline.navdp_net import NavDPStandalone


def plot_trajs(ax, best, worst, title):
    for t in worst:
        ax.plot(-t[:, 1], t[:, 0], color="lightgray", lw=1)
    for t in best:
        ax.plot(-t[:, 1], t[:, 0], color="tab:blue", lw=1.5)
    ax.plot(-best[0][:, 1], best[0][:, 0], color="tab:red", lw=2.5, label="best")
    ax.scatter([0], [0], marker="s", c="k", s=40)
    ax.set_title(title)
    ax.set_xlabel("y left->right (m)")
    ax.set_ylabel("x forward (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/current_img.jpg")
    ap.add_argument("--target", default="trash bin")
    ap.add_argument("--fov", type=float, default=90.0, help="horizontal FOV deg")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="data/smoke_test_result.png")
    args = ap.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    H, W = rgb.shape[:2]
    print(f"image {W}x{H}")

    # --- 1. detection -------------------------------------------------- #
    t0 = time.time()
    detector = GroundingDinoDetector(device=args.device)
    det = detector.detect_best(rgb, args.target)
    print(f"DINO load+detect {time.time()-t0:.1f}s")
    if det is None:
        print(f"NO DETECTION for '{args.target}' — aborting")
        sys.exit(1)
    print(f"detected '{det.label}' score={det.score:.3f} box={det.box.astype(int)}")

    # --- 2. depth ------------------------------------------------------ #
    t0 = time.time()
    depther = MetricDepthEstimator(device=args.device)
    depth = depther.estimate(rgb)
    print(f"depth {time.time()-t0:.1f}s  range=[{depth.min():.2f},{depth.max():.2f}]m")

    # --- 3. goals ------------------------------------------------------ #
    fx, fy, cx, cy = intrinsics_from_fov(W, H, args.fov)
    goal_pt = goal_point_from_detection(det.box, depth, fx, fy, cx, cy, stop_distance=0.5)
    goal_px = goal_pixel_from_detection(det.box, rgb.shape)
    print(f"goal point (robot frame) = {goal_pt}, goal pixel (norm) = {goal_px}")

    # --- 4. NavDP ------------------------------------------------------ #
    t0 = time.time()
    policy = NavDPStandalone.load(device=args.device)
    print(f"NavDP load {time.time()-t0:.1f}s")

    rgb_p = preprocess_rgb(rgb)
    dep_p = preprocess_depth(depth)
    # memory of 2 frames: duplicate the current frame at t=0
    images = torch.from_numpy(np.stack([rgb_p, rgb_p])).unsqueeze(0)
    depths = torch.from_numpy(np.stack([dep_p, dep_p])).unsqueeze(0)

    results = {}
    t0 = time.time()
    results["point-goal"] = policy.predict_pointgoal(goal_pt.reshape(1, 3), images, depths)
    results["pixel-goal"] = policy.predict_pixelgoal(goal_px.reshape(1, 2), images, depths)
    results["no-goal"] = policy.predict_nogoal(images, depths)
    print(f"NavDP 3x inference {time.time()-t0:.1f}s")

    # --- 5. visualize --------------------------------------------------- #
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(rgb)
    x0, y0, x1, y1 = det.box
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, ec="lime", fc="none", lw=2))
    axes[0].set_title(f"'{det.label}' {det.score:.2f}")
    axes[0].axis("off")
    im = axes[1].imshow(depth, cmap="turbo")
    axes[1].set_title("metric depth (m)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    for ax, (name, (best, worst, critic)) in zip(axes[2:], results.items()):
        b = best.float().cpu().numpy()
        w = worst.float().cpu().numpy()
        plot_trajs(ax, b, w, f"{name}\ncritic[{critic.min():.2f},{critic.max():.2f}]")
    if goal_pt is not None:
        for ax in axes[2:]:
            ax.scatter([-goal_pt[1]], [goal_pt[0]], marker="*", c="gold", ec="k", s=250, zorder=5, label="goal")
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"saved -> {args.out}")

    # numeric sanity: endpoint of best point-goal trajectory vs goal
    best_pt = results["point-goal"][0].float().cpu().numpy()[0]
    end = best_pt[-1]
    print(f"best point-goal trajectory endpoint = {end}, goal = {goal_pt[:2]}")


if __name__ == "__main__":
    main()
