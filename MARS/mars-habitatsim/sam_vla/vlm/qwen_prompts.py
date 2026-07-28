"""
Builds text prompts sent to Qwen2.5-VL for goal selection and driving.
"""


def build_select_goal_prompt(detections: list[dict]) -> str:
    lines = []
    for i, det in enumerate(detections):
        lines.append(
            f'{i}: class="{det["class_name"]}", bbox_norm={det["bbox_norm"]}, '
            f'confidence={det["confidence"]:.2f}'
        )
    detections_block = "\n".join(lines)

    return (
        "You are the vision system for a Mars rover. The image shows the "
        "rover's current camera view. Below is a list of detected rock "
        "instances, each with a normalized bounding box [x_min, y_min, x_max, "
        "y_max] in [0, 1] image coordinates.\n\n"
        f"{detections_block}\n\n"
        "Pick exactly ONE detection from the list above to serve as the "
        "rover's navigation goal (the rock the rover should drive to)."
        "Respond with ONLY a JSON object in this exact format, no other text:\n"
        '{"goal_index": <int, index into the list above>, '
        '"reasoning": <str, brief explanation>}'
    )


def build_drive_action_prompt(instruction_text: str, frame_idx: int) -> str:
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}).\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Output the rover's next action as ONLY a JSON object, no other "
        "text, in this exact format:\n"
        '{"v_fwd": <float in [0, 1]>, "v_lat": <float in [-1, 1]>, '
        '"yaw_rate": <float in [-1, 1]>, "reasoning": <str>}\n\n'
        "v_fwd is normalized forward speed (0 = stop, 1 = full speed), "
        "v_lat is normalized lateral speed (negative = left, positive = "
        "right), and yaw_rate is normalized turn rate (negative = turn "
        "left, positive = turn right). Steer away from any obstacles "
        "visible in the current frame while making progress toward the "
        "goal described in the instruction."
    )


def build_direction_prompt(instruction_text: str, frame_idx: int) -> str:
    """Same intent as build_drive_action_prompt, but constrains the model to a
    single discrete steering choice instead of a free continuous action. The
    goal region is overlaid in green and obstacle regions in red directly on
    the image (see perception.semantic_overlay), so the instruction leans on
    those overlays rather than the raw scene."""
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}). The navigation "
        "goal is highlighted with a GREEN overlay and known obstacles are "
        "highlighted with a RED overlay.\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Choose exactly ONE discrete direction for the rover's next move:\n"
        '  "forward"    - drive straight ahead\n'
        '  "turn_left"  - steer left\n'
        '  "turn_right" - steer right\n\n'
        "Pick whichever direction makes the most progress toward the green "
        "goal region while steering clear of any red obstacle regions. Do "
        "not output speeds or turn rates, only the discrete direction.\n\n"
        "Respond with ONLY a JSON object, no other text, in this exact "
        "format:\n"
        '{"direction": <"forward" | "turn_left" | "turn_right">, '
        '"reasoning": <str>}'
    )


if __name__ == "__main__":
    dummy_detections = [
        {"class_name": "rock", "bbox_norm": [0.12, 0.30, 0.28, 0.55], "confidence": 0.91},
        {"class_name": "rock", "bbox_norm": [0.60, 0.40, 0.80, 0.70], "confidence": 0.77},
        {"class_name": "obstacle", "bbox_norm": [0.40, 0.10, 0.55, 0.35], "confidence": 0.65},
    ]

    print("=== build_select_goal_prompt ===")
    print(build_select_goal_prompt(dummy_detections))
    print()
    print("=== build_drive_action_prompt ===")
    print(build_drive_action_prompt("Drive toward the large rock cluster ahead.", 42))
    print()
    print("=== build_direction_prompt ===")
    print(build_direction_prompt("Drive toward the large rock cluster ahead.", 42))
