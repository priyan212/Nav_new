#!/usr/bin/env python3
"""Isaac-Sim-specific Zenoh I/O helpers for OmniVLA.

This module is intentionally isolated from the real-rover path.
It only handles:
- sensor_msgs/msg/Image (CDR) deserialization
- std_msgs/msg/String (CDR) deserialization
- geometry_msgs/msg/Twist (CDR) serialization
- Zenoh subscribers/publishers for Isaac camera + cmd_vel
"""

from __future__ import annotations

import re
import struct
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


class CDRReader:
    """Minimal CDR deserializer for ROS 2 wire messages."""

    def __init__(self, data: bytes):
        self.data = data
        self.le = data[1] in (0x01, 0x11)
        self.end = "<" if self.le else ">"
        self.offset = 4
        self.base = 4

    def _align(self, n: int):
        pos = self.offset - self.base
        rem = pos % n
        if rem:
            self.offset += n - rem

    def read_uint8(self) -> int:
        v = self.data[self.offset]
        self.offset += 1
        return v

    def read_int32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "i", self.data, self.offset)
        self.offset += 4
        return v

    def read_uint32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "I", self.data, self.offset)
        self.offset += 4
        return v

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8
        return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset : self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        out = self.data[self.offset : self.offset + count]
        self.offset += count
        return out


class CDRWriter:
    """Minimal CDR serializer (little-endian)."""

    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")
        self.base = 4

    def _align(self, n: int):
        pos = len(self.buf) - self.base
        rem = pos % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_float64(self, v: float):
        self._align(8)
        self.buf += struct.pack("<d", v)

    def write_uint32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self.buf += encoded

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/msg/Image CDR -> RGB ndarray [H,W,3]."""
    r = CDRReader(cdr_data)

    # std_msgs/Header
    r.read_int32()  # stamp.sec
    r.read_uint32()  # stamp.nanosec
    r.read_string()  # frame_id

    height = r.read_uint32()
    width = r.read_uint32()
    encoding = r.read_string().lower()
    r.read_uint8()  # is_bigendian
    r._align(4)
    r.read_uint32()  # step
    pixel_data = r.read_sequence_uint8()

    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None

    if encoding == "rgb8":
        return img[:, :, :3]
    if encoding == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    if encoding in ("rgba8", "bgra8") and img.shape[2] >= 4:
        out = img[:, :, :3]
        if encoding == "bgra8":
            out = out[:, :, ::-1]
        return out.copy()

    if img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/msg/CompressedImage CDR -> RGB ndarray [H,W,3].

    Real-rover WiFi Zenoh bridge only forwards the JPEG-compressed camera
    stream (rate-capped in zenoh_pi_bridge.json5); the raw image_raw topic
    never crosses the network, so this decoder is required on that path.
    """
    from io import BytesIO
    from PIL import Image as _PILImage

    r = CDRReader(cdr_data)
    r.read_int32()  # stamp.sec
    r.read_uint32()  # stamp.nanosec
    r.read_string()  # frame_id
    r.read_string()  # format, e.g. "jpeg"
    jpeg_bytes = r.read_sequence_uint8()
    try:
        img = _PILImage.open(BytesIO(bytes(jpeg_bytes))).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def parse_string(cdr_data: bytes) -> str:
    """std_msgs/msg/String CDR -> str."""
    return CDRReader(cdr_data).read_string()


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    """geometry_msgs/msg/Twist -> CDR bytes."""
    w = CDRWriter()
    w.write_float64(float(linear_x))
    w.write_float64(0.0)
    w.write_float64(0.0)
    w.write_float64(0.0)
    w.write_float64(0.0)
    w.write_float64(float(angular_z))
    return w.to_bytes()


def serialize_string(text: str) -> bytes:
    """std_msgs/msg/String -> CDR bytes."""
    w = CDRWriter()
    w.write_string(text)
    return w.to_bytes()


@dataclass
class IsaacZenohTopics:
    camera_keys: Sequence[str]
    cmd_keys: Sequence[str]
    goal_keys: Sequence[str]
    explanation_keys: Sequence[str]
    camera_compressed_keys: Sequence[str] = ()


@dataclass(frozen=True)
class TaskStep:
    kind: str
    target_id: str
    target_label: str
    instruction: str
    source_text: str
    turn_direction: str = ""
    turn_duration_s: float = 0.0


@dataclass(frozen=True)
class TaskPlan:
    original_command: str
    steps: List[TaskStep]

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1

    def summary(self) -> str:
        if not self.steps:
            return self.original_command
        return " -> ".join(step.target_label for step in self.steps)


_LANDMARKS = [
    (
        "blue_trash_bin",
        "blue trash bin",
        (
            "blue trash bin",
            "trash bin",
            "bin",
            "blue bin",
            "trash can",
            "blue trash can",
        ),
    ),
    (
        "door",
        "door",
        ("door", "entry door", "entrance", "exit"),
    ),
    (
        "yellow_wet_warning_sign",
        "yellow wet warning sign",
        (
            "yellow wet warning sign",
            "wet floor sign",
            "warning sign",
            "yellow sign",
            "wet warning sign",
            "caution sign",
        ),
    ),
    (
        "vending_machine",
        "vending machine",
        ("vending machine", "vending", "snack machine", "machine"),
    ),
    (
        "monitor",
        "monitor",
        ("monitor", "screen", "display"),
    ),
    (
        "chair",
        "chair",
        ("chair", "seat", "office chair"),
    ),
    (
        "stairs",
        "stairs",
        ("stairs", "staircase", "steps"),
    ),
]


def _clean_command(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_for_match(text: str) -> str:
    cleaned = _clean_command(text).lower()
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return " ".join(cleaned.split())


def _extract_clause_landmarks(clause: str) -> List[tuple[int, str, str]]:
    normalized = _normalize_for_match(clause)
    matches: List[tuple[int, str, str]] = []

    for target_id, target_label, aliases in _LANDMARKS:
        best_idx: Optional[int] = None
        for alias in aliases:
            idx = normalized.find(alias)
            if idx >= 0 and (best_idx is None or idx < best_idx):
                best_idx = idx
        if best_idx is not None:
            matches.append((best_idx, target_id, target_label))

    matches.sort(key=lambda item: item[0])
    return matches


def _instruction_for_clause(clause: str, target_label: str) -> str:
    normalized = _normalize_for_match(clause)

    if any(phrase in normalized for phrase in ("look for", "find", "locate", "inspect", "check")):
        return f"look for the {target_label}"
    if "avoid" in normalized or "go around" in normalized:
        return f"go towards the {target_label} carefully"
    if "pass" in normalized or "through" in normalized:
        return f"go towards the {target_label}"
    if "approach" in normalized or "head" in normalized:
        return f"go towards the {target_label}"
    return f"go to the {target_label}"


def _extract_turn_step(clause: str) -> Optional[TaskStep]:
    normalized = _normalize_for_match(clause)
    turn_specs = [
        ("turn around", "turn_around", "turn around", 4.8),
        ("u turn", "turn_around", "turn around", 4.8),
        ("turn left", "turn_left", "turn left", 1.4),
        ("take a left", "turn_left", "turn left", 1.4),
        ("turn right", "turn_right", "turn right", 1.4),
        ("take a right", "turn_right", "turn right", 1.4),
    ]
    for phrase, target_id, label, duration_s in turn_specs:
        if phrase in normalized:
            return TaskStep(
                kind="locomotion",
                target_id=target_id,
                target_label=label,
                instruction=label,
                source_text=clause,
                turn_direction=target_id,
                turn_duration_s=duration_s,
            )
    return None


def build_task_plan(command: str) -> TaskPlan:
    cleaned = _clean_command(command)
    if not cleaned:
        return TaskPlan(original_command="", steps=[])

    clauses = [
        chunk.strip()
        for chunk in re.split(
            r"\b(?:and then|then|after that|afterwards|next|before finally|finally)\b|[,;]",
            cleaned,
            flags=re.IGNORECASE,
        )
        if chunk.strip()
    ]
    if not clauses:
        clauses = [cleaned]

    steps: List[TaskStep] = []
    for clause in clauses:
        turn_step = _extract_turn_step(clause)
        if turn_step is not None:
            if not steps or steps[-1].target_id != turn_step.target_id:
                steps.append(turn_step)
        clause_landmarks = _extract_clause_landmarks(clause)
        if not clause_landmarks:
            continue
        for _, target_id, target_label in clause_landmarks:
            instruction = _instruction_for_clause(clause, target_label)
            if steps and steps[-1].target_id == target_id:
                continue
            steps.append(
                TaskStep(
                    kind="goal",
                    target_id=target_id,
                    target_label=target_label,
                    instruction=instruction,
                    source_text=clause,
                )
            )

    if not steps:
        fallback_target = _extract_clause_landmarks(cleaned)
        if fallback_target:
            _, target_id, target_label = fallback_target[0]
            steps = [
                TaskStep(
                    kind="goal",
                    target_id=target_id,
                    target_label=target_label,
                    instruction=f"go to the {target_label}",
                    source_text=cleaned,
                )
            ]
        else:
            steps = [
                TaskStep(
                    kind="goal",
                    target_id="freeform",
                    target_label=cleaned,
                    instruction=cleaned,
                    source_text=cleaned,
                )
            ]

    return TaskPlan(original_command=cleaned, steps=steps)


class IsaacZenohIO:
    """Zenoh session wrapper for Isaac-specific OmniVLA I/O."""

    # Real rover firmware (rover_6wd_complete.ino) zeroes velocity on its own
    # if no /cmd_vel arrives within CMD_VEL_TIMEOUT_MS=500ms -- a real,
    # measured inference tick can take up to ~1.5-1.7s (variable generation
    # length, look-down retries), so publishing only once per inference tick
    # left gaps well past that deadline: the firmware would force-stop the
    # rover mid-tick, then it would resume once the next result published --
    # observed live as "moves for ~2s, stops, moves again". Fixed by
    # decoupling the PUBLISH rate from the (slow, variable) INFERENCE rate: a
    # background heartbeat re-sends the last known command at a fast fixed
    # rate regardless of how long the next inference tick takes.
    HEARTBEAT_PERIOD_S = 0.15  # well under the firmware's 500ms cmd_vel timeout

    def __init__(
        self,
        session,
        topics: IsaacZenohTopics,
        initial_instruction: str,
        on_instruction: Optional[Callable[[str], None]] = None,
    ):
        self.session = session
        self.topics = topics
        self.lock = threading.Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_ts: float = 0.0
        self.frame_count: int = 0
        self.instruction = initial_instruction
        self.on_instruction = on_instruction
        self._last_cmd = (0.0, 0.0)

        self.pub_cmd = [session.declare_publisher(k) for k in topics.cmd_keys]
        self.pub_explanation = [session.declare_publisher(k) for k in topics.explanation_keys]

        self.sub_camera = [session.declare_subscriber(k, self._on_camera) for k in topics.camera_keys]
        self.sub_camera_compressed = [
            session.declare_subscriber(k, self._on_camera_compressed)
            for k in topics.camera_compressed_keys
        ]
        self.sub_goal = [session.declare_subscriber(k, self._on_goal) for k in topics.goal_keys]

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        import time
        while True:
            time.sleep(self.HEARTBEAT_PERIOD_S)
            with self.lock:
                lin, ang = self._last_cmd
            payload = serialize_twist(lin, ang)
            for pub in self.pub_cmd:
                pub.put(payload)

    def _on_camera(self, sample):
        try:
            rgb = parse_image(bytes(sample.payload))
            if rgb is None:
                return
            import time

            with self.lock:
                self.latest_rgb = rgb
                self.latest_rgb_ts = time.time()
                self.frame_count += 1
        except Exception:
            return

    def _on_camera_compressed(self, sample):
        """Fallback JPEG stream. Ignored while raw frames are still fresh
        (<0.5s old) so a live raw feed always takes priority -- same
        priority rule used by omnivla_custom.py."""
        try:
            import time

            with self.lock:
                raw_age = time.time() - self.latest_rgb_ts
            if self.latest_rgb_ts and raw_age < 0.5:
                return
            rgb = parse_compressed_image(bytes(sample.payload))
            if rgb is None:
                return
            with self.lock:
                self.latest_rgb = rgb
                self.latest_rgb_ts = time.time()
                self.frame_count += 1
        except Exception:
            return

    def _on_goal(self, sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if not text:
                return
            self.instruction = text
            if self.on_instruction is not None:
                self.on_instruction(text)
        except Exception:
            return

    def get_latest_rgb(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_rgb is None else self.latest_rgb.copy()

    def publish_cmd(self, linear_x: float, angular_z: float):
        with self.lock:
            self._last_cmd = (linear_x, angular_z)
        payload = serialize_twist(linear_x, angular_z)
        for pub in self.pub_cmd:
            pub.put(payload)

    def publish_explanation(self, text: str):
        payload = serialize_string(text)
        for pub in self.pub_explanation:
            pub.put(payload)
