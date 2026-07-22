"""Diagnostic: does goal conditioning actually steer the extracted NavDP?

Feeds identical observations with extreme LEFT / STRAIGHT / RIGHT goals
(fixed RNG seed) through the point-goal and pixel-goal heads and reports the
mean lateral displacement of the sampled trajectories. If the three goals
produce ~identical statistics, the goal heads are untrained/vestigial in the
InternVLA-N1-w-NavDP checkpoint and we must steer by trajectory selection
instead.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from nav_pipeline.depth_estimator import MetricDepthEstimator
from nav_pipeline.goal_utils import preprocess_depth, preprocess_rgb
from nav_pipeline.navdp_net import NavDPStandalone

DEVICE = "cuda:0"


def stats(trajs):
    t = trajs.float().cpu().numpy()
    end = t[:, -1, :]
    return f"end_x={end[:,0].mean():+.2f} end_y={end[:,1].mean():+.2f} (std {end[:,1].std():.2f})"


def main():
    rgb = np.array(Image.open("data/current_img.jpg").convert("RGB"))
    depth = MetricDepthEstimator(device=DEVICE).estimate(rgb)
    rgb_p, dep_p = preprocess_rgb(rgb), preprocess_depth(depth)
    images = torch.from_numpy(np.stack([rgb_p, rgb_p])).unsqueeze(0)
    depths = torch.from_numpy(np.stack([dep_p, dep_p])).unsqueeze(0)

    policy = NavDPStandalone.load(device=DEVICE)

    point_goals = {"LEFT": [1.0, 3.0, 0.0], "STRAIGHT": [3.0, 0.0, 0.0], "RIGHT": [1.0, -3.0, 0.0]}
    pixel_goals = {"LEFT": [0.1, 0.5], "CENTER": [0.5, 0.5], "RIGHT": [0.9, 0.5]}

    print("== point-goal head ==")
    for name, g in point_goals.items():
        torch.manual_seed(0)
        best, worst, critic = policy.predict_pointgoal(np.array(g).reshape(1, 3), images, depths)
        print(f"  {name:9s} goal={g}  best {stats(best)}  critic[{critic.min():.3f},{critic.max():.3f}]")

    print("== pixel-goal head ==")
    for name, g in pixel_goals.items():
        torch.manual_seed(0)
        best, worst, critic = policy.predict_pixelgoal(np.array(g).reshape(1, 2), images, depths)
        print(f"  {name:9s} goal={g}  best {stats(best)}")

    print("== goal embedding magnitudes ==")
    with torch.no_grad():
        rgbd = policy.rgbd_encoder(images.to(DEVICE), depths.to(DEVICE))
        pe = policy.point_encoder(torch.tensor([[3.0, 0.0, 0.0]], dtype=policy.input_dtype, device=DEVICE))
        pg = policy.pg_embed_mlp(torch.tensor([[0.5, 0.5]], dtype=policy.input_dtype, device=DEVICE))
        print(f"  |rgbd token| mean={rgbd.norm(dim=-1).float().mean():.2f}")
        print(f"  |point_embed|={pe.norm().float():.2f}  |pixel_embed|={pg.norm().float():.2f}")


if __name__ == "__main__":
    main()
