"""
Persistent TCP server that keeps Qwen2.5-VL resident in memory.
"""

import base64
import io
import json
import socket
import struct

import numpy as np
from PIL import Image

from sam_vla.vlm.qwen_config import QWEN_SERVER_HOST, QWEN_SERVER_PORT
from sam_vla.vlm.qwen_model_runner import load_qwen_model, run_qwen_inference
from sam_vla.vlm.qwen_prompts import (
    build_direction_prompt,
    build_drive_action_prompt,
    build_select_goal_prompt,
)
from sam_vla.vlm.qwen_response_parser import (
    parse_direction_response,
    parse_drive_action_response,
    parse_select_goal_response,
)

_HEADER_SIZE = 4


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


def _recv_message(conn: socket.socket) -> dict:
    header = _recv_exact(conn, _HEADER_SIZE)
    (payload_len,) = struct.unpack(">I", header)
    payload = _recv_exact(conn, payload_len)
    return json.loads(payload.decode("utf-8"))


def _send_message(conn: socket.socket, message: dict) -> None:
    payload = json.dumps(message).encode("utf-8")
    header = struct.pack(">I", len(payload))
    conn.sendall(header + payload)


def _decode_image(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def _handle_select_goal(model, processor, payload: dict) -> dict:
    image = _decode_image(payload["image_b64"])
    detections = payload["detections"]
    prompt = build_select_goal_prompt(detections)
    raw_text = run_qwen_inference(model, processor, image, prompt)
    result = parse_select_goal_response(raw_text)
    return {"result": result}


def _handle_drive_action(model, processor, payload: dict) -> dict:
    image = _decode_image(payload["image_b64"])
    instruction_text = payload["instruction_text"]
    frame_idx = payload["frame_idx"]
    prompt = build_drive_action_prompt(instruction_text, frame_idx)
    raw_text = run_qwen_inference(model, processor, image, prompt)
    action = parse_drive_action_response(raw_text)
    return {
        "result": {
            "v_fwd": action.v_fwd,
            "v_lat": action.v_lat,
            "yaw_rate": action.yaw_rate,
        }
    }


def _handle_drive_direction(model, processor, payload: dict) -> dict:
    """Same shape as _handle_drive_action, but constrains the response to a
    single discrete direction (forward/turn_left/turn_right) instead of a
    free continuous action."""
    image = _decode_image(payload["image_b64"])
    instruction_text = payload["instruction_text"]
    frame_idx = payload["frame_idx"]
    prompt = build_direction_prompt(instruction_text, frame_idx)
    raw_text = run_qwen_inference(model, processor, image, prompt)
    result = parse_direction_response(raw_text)
    return {"result": result}


def _dispatch(model, processor, message: dict) -> dict:
    mode = message.get("mode")
    payload = message.get("payload", {})

    if mode == "ping":
        return {"status": "ok"}
    if mode == "select_goal":
        return _handle_select_goal(model, processor, payload)
    if mode == "drive_action":
        return _handle_drive_action(model, processor, payload)
    if mode == "drive_direction":
        return _handle_drive_direction(model, processor, payload)
    raise ValueError(f"unknown mode: {mode!r}")


def serve_forever(model, processor, host: str = QWEN_SERVER_HOST, port: int = QWEN_SERVER_PORT) -> None:
    # Single-threaded blocking accept loop: Qwen inference is not safe to run
    # concurrently across connections anyway, so requests are naturally
    # serialized rather than fanned out via ThreadingTCPServer.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    print(f"[!] qwen_server listening on {host}:{port}")

    while True:
        conn, _addr = listener.accept()
        with conn:
            try:
                message = _recv_message(conn)
                response = _dispatch(model, processor, message)
            except Exception as e:
                response = {"error": str(e)}
            _send_message(conn, response)


if __name__ == "__main__":
    print("[!] Loading Qwen model")
    _model, _processor = load_qwen_model()
    print("Model loaded, starting server.")
    serve_forever(_model, _processor)
