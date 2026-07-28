# This module's entire job is chaining sam_segmenter -> sam_output_adapter ->
# qwen_client into a single first-frame goal resolution call, so importing all
# three (plus core.types) is intentional, not a layering violation.

import sys

import numpy as np
from PIL import Image

from sam_vla.core.types import Detection, GoalSpec
from sam_vla.perception import sam_output_adapter, sam_segmenter
from sam_vla.vlm import qwen_client


def _detect(rgb: np.ndarray) -> list[Detection]:
    raw_detections = sam_segmenter.segment_frame(rgb)
    detections: list[Detection] = sam_output_adapter.to_detections(
        raw_detections, rgb.shape[1], rgb.shape[0]
    )

    if not detections:
        raise RuntimeError(
            "no detections found on first frame — cannot resolve a goal"
        )
    return detections


def resolve(rgb: np.ndarray) -> GoalSpec:
    detections = _detect(rgb)
    return qwen_client.select_goal(rgb, detections)


def resolve_verbose(rgb: np.ndarray) -> tuple[GoalSpec, dict, list[Detection]]:
    """Same as resolve, but also returns the raw VLM goal-selection result dict and
    the SAM detections used to produce it, for logging."""
    detections = _detect(rgb)
    goal_spec, vlm_result = qwen_client.select_goal_verbose(rgb, detections)
    return goal_spec, vlm_result, detections


def resolve_from_path(image_path: str) -> GoalSpec:
    # PIL, to match qwen_client's own __main__ image loading convention.
    rgb = np.array(Image.open(image_path).convert("RGB"))
    return resolve(rgb)


if __name__ == "__main__":
    goal_spec = resolve_from_path(sys.argv[1])
    print("goal_bbox_norm:", goal_spec.goal_bbox_norm)
    print("num_obstacles:", len(goal_spec.obstacle_bboxes_norm))
    print("instruction_text:", goal_spec.instruction_text)
