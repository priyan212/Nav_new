"""Paints the goal/obstacle semantic masks (render-only meshes registered via
MarsHabitatEnv.register_object_mask, tagged with core.goal_geometry's
MESH_GOAL_ID / MESH_OBST_ID) directly onto the RGB frame as color overlays,
so a VLM query can see them without a separate mask channel or input."""

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from sam_vla.core.goal_geometry import MESH_GOAL_ID, MESH_OBST_ID

GOAL_OVERLAY_COLOR = np.array([0, 255, 0], dtype=np.float32)
OBSTACLE_OVERLAY_COLOR = np.array([255, 0, 0], dtype=np.float32)


def overlay_semantic_masks(
    rgb: np.ndarray,
    semantic: np.ndarray,
    alpha: float = 0.45,
    goal_id: int = MESH_GOAL_ID,
    obstacle_id: int = MESH_OBST_ID,
    text: Optional[str] = None,
) -> np.ndarray:
    """Alpha-blend green over goal-mask pixels and red over obstacle-mask
    pixels; pixels with no registered mask are left untouched. `semantic` is
    the per-pixel semantic-id frame from MarsHabitatEnv.get_semantic_frame()
    and must be the same (H, W) shape as `rgb`. If `text` is given, it's drawn
    in a small banner in the top-left corner (e.g. distance-to-goal, action)."""
    semantic = np.asarray(semantic)
    overlaid = np.asarray(rgb, dtype=np.float32).copy()

    goal_mask = semantic == goal_id
    obstacle_mask = semantic == obstacle_id

    overlaid[goal_mask] = (1.0 - alpha) * overlaid[goal_mask] + alpha * GOAL_OVERLAY_COLOR
    overlaid[obstacle_mask] = (1.0 - alpha) * overlaid[obstacle_mask] + alpha * OBSTACLE_OVERLAY_COLOR

    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    if text:
        img = Image.fromarray(overlaid)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
        draw.text((4, 3), text, fill=(255, 255, 255))
        overlaid = np.asarray(img, dtype=np.uint8)

    return overlaid


if __name__ == "__main__":
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    semantic = np.zeros((8, 8), dtype=np.int32)
    semantic[0:3, 0:3] = MESH_GOAL_ID
    semantic[5:8, 5:8] = MESH_OBST_ID

    overlaid = overlay_semantic_masks(rgb, semantic)
    print("goal patch (should lean green):", overlaid[1, 1])
    print("obstacle patch (should lean red):", overlaid[6, 6])
    print("untouched patch (should stay gray):", overlaid[4, 4])

    assert overlaid[1, 1][1] > overlaid[1, 1][0]
    assert overlaid[6, 6][0] > overlaid[6, 6][1]
    assert tuple(overlaid[4, 4]) == (128, 128, 128)
    print("OK")
