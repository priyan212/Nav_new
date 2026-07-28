"""
Client for the persistent Qwen VLM server.
"""

import base64
import io
import json
import socket
import struct

import numpy as np

from sam_vla.core.types import Action, Detection, GoalSpec
from sam_vla.vlm.qwen_config import QWEN_SERVER_PORT

_HOST = "127.0.0.1"
_HEADER_SIZE = 4


def _encode_image(rgb: np.ndarray) -> str:
    # PNG is lossless and Pillow/numpy round-trip it without extra deps beyond what qwen_server already requires to decode it server-side.
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _send_request(mode: str, payload: dict, port: int = None, timeout: float = 30.0) -> dict:
    if port is None:
        port = QWEN_SERVER_PORT
    message = json.dumps({"mode": mode, "payload": payload}).encode("utf-8")
    try:
        with socket.create_connection((_HOST, port), timeout=timeout) as conn:
            conn.sendall(struct.pack(">I", len(message)) + message)
            header = _recv_exact(conn, _HEADER_SIZE)
            (body_len,) = struct.unpack(">I", header)
            body = _recv_exact(conn, body_len)
    except OSError as e:
        raise ConnectionError(
            f"could not reach qwen_server at {_HOST}:{port} (mode={mode!r}): {e}. "
            "Is the server running?"
        ) from e
    return json.loads(body.decode("utf-8"))


def _recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before expected bytes were received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _select_goal_result(rgb: np.ndarray, detections: list[Detection]) -> dict:
    detections_json = [
        {
            "class_name": d.class_name,
            "bbox_norm": list(d.bbox_norm),
            "confidence": d.confidence,
        }
        for d in detections
    ]
    response = _send_request(
        "select_goal",
        {"image_b64": _encode_image(rgb), "detections": detections_json},
    )
    if "error" in response:
        raise ValueError(f"select_goal failed: {response['error']}")
    return response["result"]


def _goal_spec_from_result(result: dict, detections: list[Detection]) -> GoalSpec:
    goal_index = result["goal_index"]
    reasoning = result.get("reasoning", "")

    goal_detection = detections[goal_index]
    obstacle_bboxes = [d.bbox_norm for i, d in enumerate(detections) if i != goal_index]
    instruction_text = f"Navigate to the {goal_detection.class_name} target while avoiding obstacles."
    if reasoning:
        instruction_text += f" ({reasoning})"

    return GoalSpec(
        goal_bbox_norm=goal_detection.bbox_norm,
        obstacle_bboxes_norm=obstacle_bboxes,
        instruction_text=instruction_text,
    )


def select_goal(rgb: np.ndarray, detections: list[Detection]) -> GoalSpec:
    result = _select_goal_result(rgb, detections)
    return _goal_spec_from_result(result, detections)


def select_goal_verbose(rgb: np.ndarray, detections: list[Detection]) -> tuple[GoalSpec, dict]:
    """Same as select_goal, but also returns the raw VLM result dict (goal_index, reasoning) for logging."""
    result = _select_goal_result(rgb, detections)
    return _goal_spec_from_result(result, detections), result


def _drive_action_result(rgb: np.ndarray, goal_spec: GoalSpec, frame_idx: int) -> dict:
    response = _send_request(
        "drive_action",
        {
            "image_b64": _encode_image(rgb),
            "instruction_text": goal_spec.instruction_text,
            "frame_idx": frame_idx,
        },
    )
    if "error" in response:
        raise ValueError(f"drive_action failed: {response['error']}")
    return response["result"]


def drive_action(rgb: np.ndarray, goal_spec: GoalSpec, frame_idx: int) -> Action:
    result = _drive_action_result(rgb, goal_spec, frame_idx)
    return Action(
        v_fwd=result["v_fwd"],
        v_lat=result["v_lat"],
        yaw_rate=result["yaw_rate"],
    )


def drive_action_verbose(rgb: np.ndarray, goal_spec: GoalSpec, frame_idx: int) -> tuple[Action, dict]:
    """Same as drive_action, but also returns the raw VLM result dict for logging."""
    result = _drive_action_result(rgb, goal_spec, frame_idx)
    action = Action(
        v_fwd=result["v_fwd"],
        v_lat=result["v_lat"],
        yaw_rate=result["yaw_rate"],
    )
    return action, result


def _drive_direction_result(rgb: np.ndarray, goal_spec: GoalSpec, frame_idx: int) -> dict:
    response = _send_request(
        "drive_direction",
        {
            "image_b64": _encode_image(rgb),
            "instruction_text": goal_spec.instruction_text,
            "frame_idx": frame_idx,
        },
    )
    if "error" in response:
        raise ValueError(f"drive_direction failed: {response['error']}")
    return response["result"]


def drive_direction_verbose(rgb: np.ndarray, goal_spec: GoalSpec, frame_idx: int) -> tuple[str, dict]:
    """Query the VLA for a single discrete direction (one of
    qwen_response_parser.DIRECTIONS) instead of a free continuous action, and
    return it alongside the raw VLM result dict for logging. `rgb` is expected
    to already carry the goal/obstacle semantic overlay (see
    perception.semantic_overlay.overlay_semantic_masks) so the model has the
    same visual grounding as drive_action would otherwise infer on its own."""
    result = _drive_direction_result(rgb, goal_spec, frame_idx)
    return result["direction"], result


if __name__ == "__main__":
    import sys

    from PIL import Image

    image_path = sys.argv[1] if len(sys.argv) > 1 else "marsyard2022_terrain_texture.png"
    rgb = np.array(Image.open(image_path).convert("RGB"))

    dummy_detections = [
        Detection(class_name="rover", bbox_norm=(0.1, 0.1, 0.3, 0.3), confidence=0.9),
        Detection(class_name="rock", bbox_norm=(0.4, 0.4, 0.6, 0.6), confidence=0.8),
        Detection(class_name="crater", bbox_norm=(0.7, 0.2, 0.9, 0.5), confidence=0.7),
    ]

    goal_spec = select_goal(rgb, dummy_detections)
    print("select_goal ->", goal_spec)

    action = drive_action(rgb, goal_spec, frame_idx=0)
    print("drive_action ->", action)

    direction, result = drive_direction_verbose(rgb, goal_spec, frame_idx=0)
    print("drive_direction_verbose ->", direction, result)
