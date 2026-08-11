from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .pixel_obstacles import PixelObstacleConfig, pixels_to_local_obstacles
from .s2diff_guidance import (
    S2DiffCandidateBatch,
    S2DiffGuidanceConfig,
    sample_gradient_pointgoal_candidates,
    sample_s2diff_pointgoal_candidates,
)


@dataclass
class S2DiffPointGoalAgent:
    """Drop-in PointGoal wrapper around the released ``NavDP_Agent``."""

    base_agent: Any
    guidance_config: S2DiffGuidanceConfig = S2DiffGuidanceConfig()
    pixel_obstacle_config: PixelObstacleConfig = PixelObstacleConfig()
    guidance_method: str = "particles"
    last_result: S2DiffCandidateBatch | None = None
    circulation_signs: np.ndarray | None = None

    def reset(self, batch_size: int, threshold: float) -> None:
        self.base_agent.reset(batch_size, threshold)
        self.last_result = None
        self.circulation_signs = np.zeros(batch_size, dtype=np.float32)

    def reset_env(self, index: int) -> None:
        self.base_agent.reset_env(index)
        if self.circulation_signs is not None:
            self.circulation_signs[index] = 0.0

    @property
    def batch_size(self) -> int:
        return self.base_agent.batch_size

    def step_pointgoal(
        self,
        goals: np.ndarray,
        images: np.ndarray,
        depths: np.ndarray,
        obstacle_pixels: Sequence[Sequence[Sequence[int | float]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Guide NavDP using only the externally supplied obstacle pixels."""

        process_images = self.base_agent.process_image(images)
        process_depths = self.base_agent.process_depth(depths.copy())
        input_images = []
        for index in range(len(self.base_agent.memory_queue)):
            queue = self.base_agent.memory_queue[index]
            if len(queue) < self.base_agent.memory_size:
                queue.append(process_images[index])
                history = np.asarray(queue)
                history = np.pad(
                    history,
                    (
                        (self.base_agent.memory_size - history.shape[0], 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                )
            else:
                del queue[0]
                queue.append(process_images[index])
                history = np.asarray(queue)
            input_images.append(history)

        input_goals = self.base_agent.process_pointgoal(np.asarray(goals).copy())
        obstacles = pixels_to_local_obstacles(
            depths,
            self.base_agent.image_intrinsic,
            obstacle_pixels,
            self.pixel_obstacle_config,
            device=self.base_agent.device,
        )
        if self.guidance_method not in {"particles", "gradient"}:
            raise ValueError("guidance_method must be particles or gradient")
        sampler = (
            sample_gradient_pointgoal_candidates
            if self.guidance_method == "gradient"
            else sample_s2diff_pointgoal_candidates
        )
        result = sampler(
            self.base_agent.navi_former,
            input_goals,
            np.asarray(input_images),
            process_depths,
            obstacles,
            self.guidance_config,
            preferred_circulation_signs=self.circulation_signs,
        )
        self.last_result = result
        selected_signs = np.asarray(
            result.diagnostics["selected_circulation_sign"], dtype=np.float32
        )
        update = (np.abs(selected_signs) > 0.5) & ~result.fallback_stop
        if self.circulation_signs is None:
            self.circulation_signs = np.zeros(self.batch_size, dtype=np.float32)
        self.circulation_signs[update] = selected_signs[update]
        trajectory_mask = self.base_agent.project_trajectory(
            images, result.all_trajectories, result.all_values
        )
        return (
            result.selected_trajectory,
            result.all_trajectories,
            result.all_values,
            trajectory_mask,
        )
