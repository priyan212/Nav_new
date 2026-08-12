from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .depth_obstacles import DepthObstacleBatch


@dataclass(frozen=True)
class PixelObstacleConfig:
    """Validation and sampling limits for externally supplied obstacle pixels."""

    minimum_depth: float = 0.10
    maximum_depth: float = 5.00
    maximum_points: int = 1536

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_depth < self.maximum_depth:
            raise ValueError("depth bounds must satisfy 0 < minimum < maximum")
        if self.maximum_points < 1:
            raise ValueError("maximum_points must be positive")


def pixels_to_local_obstacles(
    depths: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    obstacle_pixels: Sequence[Sequence[Sequence[int | float]]],
    config: PixelObstacleConfig = PixelObstacleConfig(),
    device: torch.device | str | None = None,
) -> DepthObstacleBatch:
    """Back-project only supplied ``[u,v]`` pixels into NavDP's local xy frame.

    ``obstacle_pixels`` must contain one pixel list per batch item. Pixels not in
    these lists are never considered obstacles. Invalid depth at a supplied
    pixel is ignored because it cannot be back-projected metrically.
    """

    depth = torch.as_tensor(depths, dtype=torch.float32, device=device)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError("depths must have shape [B,H,W] or [B,H,W,1]")
    if len(obstacle_pixels) != depth.shape[0]:
        raise ValueError("obstacle_pixels must contain one [u,v] list per batch item")

    intrinsic = torch.as_tensor(intrinsics, dtype=torch.float32, device=depth.device)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).expand(depth.shape[0], -1, -1)
    if intrinsic.shape != (depth.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [B,3,3]")
    if torch.any(intrinsic[:, 0, 0] <= 0) or torch.any(intrinsic[:, 1, 1] <= 0):
        raise ValueError("camera focal lengths must be positive")

    per_batch: list[torch.Tensor] = []
    height, width = depth.shape[1:]
    for batch_index, batch_pixels in enumerate(obstacle_pixels):
        pixels = torch.as_tensor(batch_pixels, dtype=torch.float32, device=depth.device)
        if pixels.numel() == 0:
            per_batch.append(torch.empty((0, 2), dtype=depth.dtype, device=depth.device))
            continue
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("each obstacle-pixel list must have shape [N,2]")
        if not torch.isfinite(pixels).all():
            raise ValueError("obstacle pixels must be finite")
        rounded = pixels.round()
        if not torch.allclose(pixels, rounded):
            raise ValueError("obstacle pixels must be integer [u,v] coordinates")
        uv = rounded.long()
        if (
            (uv[:, 0] < 0).any()
            or (uv[:, 0] >= width).any()
            or (uv[:, 1] < 0).any()
            or (uv[:, 1] >= height).any()
        ):
            raise ValueError("an obstacle pixel lies outside its depth image")

        u = uv[:, 0]
        v = uv[:, 1]
        metric_depth = depth[batch_index, v, u]
        valid = (
            torch.isfinite(metric_depth)
            & (metric_depth >= config.minimum_depth)
            & (metric_depth <= config.maximum_depth)
        )
        metric_depth = metric_depth[valid]
        u = u[valid].to(depth.dtype)
        fx = intrinsic[batch_index, 0, 0]
        cx = intrinsic[batch_index, 0, 2]
        forward = metric_depth
        lateral = -(u - cx) * metric_depth / fx
        points = torch.stack((forward, lateral), dim=-1)

        if points.shape[0] > config.maximum_points:
            indices = torch.linspace(
                0,
                points.shape[0] - 1,
                config.maximum_points,
                device=points.device,
            ).long()
            points = points[indices]
        per_batch.append(points)

    padded_size = max(1, max((points.shape[0] for points in per_batch), default=0))
    padded = torch.zeros(
        (depth.shape[0], padded_size, 2), dtype=depth.dtype, device=depth.device
    )
    mask = torch.zeros(
        (depth.shape[0], padded_size), dtype=torch.bool, device=depth.device
    )
    for batch_index, points in enumerate(per_batch):
        padded[batch_index, : points.shape[0]] = points
        mask[batch_index, : points.shape[0]] = True
    return DepthObstacleBatch(points_xy=padded, valid_mask=mask)
