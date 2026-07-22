"""Monocular metric depth from RGB via Depth Anything V2 (ViT-S, hypersim).

Used when no depth sensor is available (the real rover is RGB-only).
In Isaac Sim, prefer the simulated depth camera and skip this module.
"""

import os

import numpy as np
import torch

from .depth_anything.depth_anything_v2.dpt import DepthAnythingV2

_DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth"
)


class MetricDepthEstimator:
    def __init__(self, checkpoint: str = _DEFAULT_CKPT, max_depth: float = 20.0, device: str = "cuda:0"):
        self.device = device
        self.model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=max_depth
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW float32 depth in meters."""
        bgr = rgb[:, :, ::-1].copy()  # infer_image expects BGR (cv2 convention)
        depth = self.model.infer_image(bgr)
        return depth.astype(np.float32)
