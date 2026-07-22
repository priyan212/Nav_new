#!/usr/bin/env python3
"""Run InternVLA-N1 on Isaac Sim's camera stream via Zenoh.

Isaac-Sim sibling of ``internvla_zenoh_node.py``, mirroring how
``run_omnivla_isaac.py`` sits next to ``omnivla_zenoh_node.py``: same model
core, different transport (Isaac's ROS2 topic names/keys, multi-key
camera/cmd lookups) via the SHARED, model-agnostic ``isaac_zenoh_bridge.py``.

This is a separate integration path for Isaac Sim and does not modify or
reuse the real-rover runtime scripts.

Data flow
---------
Isaac Sim ROS 2 (/rover_camera) -> zenoh-bridge-ros2dds -> Zenoh key(s)
-> this script (InternVLA-N1 System-2 inference) -> Zenoh key(s) -> /cmd_vel.

Env: run under the `internnav` conda env with the transformers-4.51.0 shadow:
    PYTHONPATH=/home/i3d/internnav_n1_tf451 \
    /mnt/bigdisk/conda_envs/internnav/bin/python inference/run_internvla_n1_isaac.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh Python library not found. Install with: pip install eclipse-zenoh")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference.isaac_zenoh_bridge import IsaacZenohIO, IsaacZenohTopics
from inference.internvla_zenoh_node import (  # reused model + inference core only
    CFG as MODEL_CFG,
    load_model,
    infer_cmd,
)


@dataclass
class IsaacConfig:
    connect_endpoint: str
    camera_keys: List[str]
    cmd_keys: List[str]
    goal_keys: List[str]
    explanation_keys: List[str]
    instruction: str
    predict_hz: float
    max_linear: float
    max_angular: float
    model_path: str
    internnav_repo: str


DEFAULT_CONFIG: Dict[str, Any] = {
    "zenoh": {
        "connect_endpoint": "",
        "camera_keys": ["rover_camera", "/rover_camera", "rt/rover_camera"],
        "cmd_keys": ["cmd_vel", "/cmd_vel", "rt/cmd_vel"],
        "goal_keys": ["omnivla/goal_text"],
        "explanation_keys": ["omnivla/explanation"],
    },
    "internvla_n1": {
        "instruction": "go straight down the hallway and stop at the door",
        "predict_hz": 2.0,
        "max_linear": 0.15,
        "max_angular": 0.25,
        "model_path": MODEL_CFG.model_path,
        "internnav_repo": MODEL_CFG.internnav_repo,
    },
}


def _load_json_compatible_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    # File uses JSON syntax and a .yaml extension (JSON is valid YAML).
    loaded = json.loads(text)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for top_key in ("zenoh", "internvla_n1"):
        if top_key in loaded and isinstance(loaded[top_key], dict):
            merged[top_key].update(loaded[top_key])
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InternVLA-N1 Isaac-Sim inference path (Zenoh + ROS2 bridge, camera-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str,
        default=os.path.join(ROOT, "configs", "isaac_sim_internvla_n1.yaml"),
        help="Isaac integration config file (JSON-compatible YAML)",
    )
    p.add_argument("--instruction", type=str, default=None, help="Override instruction")
    p.add_argument("--connect-endpoint", type=str, default=None, help="Override Zenoh endpoint tcp/<ip>:7447")
    p.add_argument("--predict-hz", type=float, default=None, help="Override inference rate")
    p.add_argument("--max-linear", type=float, default=None, help="Override max linear speed")
    p.add_argument("--max-angular", type=float, default=None, help="Override max angular speed")
    p.add_argument("--model-path", type=str, default=None, help="Override InternVLA-N1 checkpoint path")
    p.add_argument("--internnav-repo", type=str, default=None, help="Override InternNav repo path")
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> IsaacConfig:
    if os.path.exists(args.config):
        cfg_raw = _load_json_compatible_yaml(args.config)
    else:
        cfg_raw = json.loads(json.dumps(DEFAULT_CONFIG))
    z = cfg_raw["zenoh"]
    n = cfg_raw["internvla_n1"]

    return IsaacConfig(
        connect_endpoint=args.connect_endpoint if args.connect_endpoint is not None else z["connect_endpoint"],
        camera_keys=list(z["camera_keys"]),
        cmd_keys=list(z["cmd_keys"]),
        goal_keys=list(z["goal_keys"]),
        explanation_keys=list(z["explanation_keys"]),
        instruction=args.instruction if args.instruction is not None else n["instruction"],
        predict_hz=float(args.predict_hz if args.predict_hz is not None else n["predict_hz"]),
        max_linear=float(args.max_linear if args.max_linear is not None else n["max_linear"]),
        max_angular=float(args.max_angular if args.max_angular is not None else n["max_angular"]),
        model_path=args.model_path if args.model_path is not None else n["model_path"],
        internnav_repo=args.internnav_repo if args.internnav_repo is not None else n["internnav_repo"],
    )


def main() -> int:
    args = parse_args()
    cfg = build_cfg(args)

    MODEL_CFG.predict_hz = cfg.predict_hz
    MODEL_CFG.max_linear = cfg.max_linear
    MODEL_CFG.max_angular = cfg.max_angular
    MODEL_CFG.model_path = cfg.model_path
    MODEL_CFG.internnav_repo = cfg.internnav_repo

    print("=" * 72)
    print("InternVLA-N1 Isaac-Sim path (camera-only, RGB, language-goal)")
    print(f"Config file      : {args.config}")
    print(f"Instruction      : {cfg.instruction}")
    print(f"Camera keys      : {cfg.camera_keys}")
    print(f"Cmd keys         : {cfg.cmd_keys}")
    print(f"Goal keys        : {cfg.goal_keys}")
    print(f"Predict rate     : {cfg.predict_hz} Hz")
    print(f"Model path       : {cfg.model_path}")
    if cfg.connect_endpoint:
        print(f"Zenoh endpoint   : {cfg.connect_endpoint}")
    else:
        print("Zenoh endpoint   : autodiscovery/scouting")
    print("=" * 72)

    print("[1/3] Loading InternVLA-N1 model...")
    load_model()

    print("[2/3] Opening Zenoh session...")
    zcfg = zenoh.Config()
    if cfg.connect_endpoint:
        zcfg.insert_json5("connect/endpoints", json.dumps([cfg.connect_endpoint]))
    session = zenoh.open(zcfg)

    instruction_box = {"text": cfg.instruction}

    def _on_instruction(text: str):
        instruction_box["text"] = text
        print(f"[GOAL] instruction updated: {text}")

    io = IsaacZenohIO(
        session=session,
        topics=IsaacZenohTopics(
            camera_keys=cfg.camera_keys,
            cmd_keys=cfg.cmd_keys,
            goal_keys=cfg.goal_keys,
            explanation_keys=cfg.explanation_keys,
        ),
        initial_instruction=cfg.instruction,
        on_instruction=_on_instruction,
    )

    stop_flag = {"stop": False}

    def _request_stop(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("[3/3] Running inference loop...")
    print("Waiting for Isaac camera frames...")

    period = 1.0 / cfg.predict_hz
    last_wait_log = 0.0
    infer_count = 0

    try:
        while not stop_flag["stop"]:
            t0 = time.time()
            rgb = io.get_latest_rgb()

            if rgb is None:
                now = time.time()
                if now - last_wait_log > 5.0:
                    print("[WAIT] no camera frame yet on configured camera keys")
                    last_wait_log = now
            else:
                instruction = instruction_box["text"]
                lin, ang, kind, detail = infer_cmd(rgb, instruction)

                io.publish_cmd(lin, ang)
                io.publish_explanation(
                    f"isaac path | InternVLA-N1 [{kind}:{detail}] | "
                    f"instruction='{instruction}' | lin={lin:.3f} ang={ang:.3f}"
                )

                infer_count += 1
                if infer_count <= 3 or infer_count % 20 == 0:
                    print(f"[PRED #{infer_count}] {kind}:{detail} lin={lin:.3f} ang={ang:.3f}")

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    finally:
        print("Stopping: publishing zero cmd_vel...")
        io.publish_cmd(0.0, 0.0)
        time.sleep(0.05)
        io.publish_cmd(0.0, 0.0)
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
