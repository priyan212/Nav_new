from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DepthObstacleConfig:
    """Convert one metric depth image into local planar obstacle points."""

    minimum_depth: float = 0.10
    maximum_depth: float = 5.00
    pixel_stride: int = 4
    minimum_height: float = -0.45
    maximum_height: float = 0.55
    maximum_points: int = 1536

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_depth < self.maximum_depth:
            raise ValueError("depth bounds must satisfy 0 < minimum < maximum")
        if self.pixel_stride < 1:
            raise ValueError("pixel_stride must be positive")
        if self.minimum_height >= self.maximum_height:
            raise ValueError("minimum_height must be below maximum_height")
        if self.maximum_points < 1:
            raise ValueError("maximum_points must be positive")


@dataclass(frozen=True)
class DepthObstacleBatch:
    """Padded obstacle points in NavDP's x-forward/y-left local frame."""

    points_xy: torch.Tensor
    valid_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DepthObstacleBatch":
        return DepthObstacleBatch(
            points_xy=self.points_xy.to(device),
            valid_mask=self.valid_mask.to(device),
        )


def depth_to_local_obstacles(
    depths: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    config: DepthObstacleConfig = DepthObstacleConfig(),
    device: torch.device | str | None = None,
) -> DepthObstacleBatch:
    """Back-project depth pixels and retain points near the robot body height.

    ``depths`` is metric depth with shape ``[B,H,W]`` or ``[B,H,W,1]``.
    NavDP trajectories use x forward and y left.  The vertical coordinate is
    positive up relative to the camera optical centre; the height band removes
    most floor and ceiling returns without requiring a semantic obstacle model.
    """

    depth = torch.as_tensor(depths, dtype=torch.float32, device=device)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError("depths must have shape [B,H,W] or [B,H,W,1]")

    intrinsic = torch.as_tensor(intrinsics, dtype=torch.float32, device=depth.device)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).expand(depth.shape[0], -1, -1)
    if intrinsic.shape != (depth.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [B,3,3]")

    rows = torch.arange(0, depth.shape[1], config.pixel_stride, device=depth.device)
    cols = torch.arange(0, depth.shape[2], config.pixel_stride, device=depth.device)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    sampled = depth[:, :: config.pixel_stride, :: config.pixel_stride]

    fx = intrinsic[:, 0, 0, None, None]
    fy = intrinsic[:, 1, 1, None, None]
    cx = intrinsic[:, 0, 2, None, None]
    cy = intrinsic[:, 1, 2, None, None]
    if torch.any(fx <= 0) or torch.any(fy <= 0):
        raise ValueError("camera focal lengths must be positive")

    forward = sampled
    lateral = -(uu[None].to(sampled.dtype) - cx) * sampled / fx
    # Upstream project_trajectory uses: v = H - 1 - fy*z/x - cy.
    height = (
        depth.shape[1] - 1 - cy - vv[None].to(sampled.dtype)
    ) * sampled / fy

    valid = (
        torch.isfinite(sampled)
        & (sampled >= config.minimum_depth)
        & (sampled <= config.maximum_depth)
        & (height >= config.minimum_height)
        & (height <= config.maximum_height)
    )

    per_batch: list[torch.Tensor] = []
    for batch_index in range(depth.shape[0]):
        points = torch.stack((forward[batch_index], lateral[batch_index]), dim=-1)
        points = points[valid[batch_index]]
        if points.shape[0] > config.maximum_points:
            # Evenly spaced deterministic sampling keeps the full field of view.
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
