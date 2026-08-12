"""PointGoal NavDP server with S2Diff guidance from supplied obstacle pixels."""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from policy_agent import NavDP_Agent
from tube_planner.pixel_obstacles import PixelObstacleConfig
from tube_planner.s2diff_agent import S2DiffPointGoalAgent
from tube_planner.s2diff_guidance import S2DiffGuidanceConfig


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
)
parser.add_argument(
    "--remove-critic", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--candidates", type=int, default=16)
parser.add_argument("--particles", type=int, default=8)
parser.add_argument("--particle-std", type=float, default=0.22)
parser.add_argument("--guidance-strength", type=float, default=0.85)
parser.add_argument("--temperature", type=float, default=0.35)
parser.add_argument(
    "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--particle-energy-reweighting",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--safe-distance", type=float, default=0.42)
parser.add_argument("--hard-collision-distance", type=float, default=0.24)
parser.add_argument("--robot-radius", type=float, default=0.24)
parser.add_argument("--safety-weight", type=float, default=35.0)
parser.add_argument("--barrier-weight", type=float, default=25.0)
parser.add_argument("--barrier-rate", type=float, default=0.15)
parser.add_argument("--circulation-weight", type=float, default=18.0)
parser.add_argument("--circulation-activation-distance", type=float, default=1.50)
parser.add_argument("--circulation-activation-sharpness", type=float, default=0.20)
parser.add_argument("--minimum-circulation-progress", type=float, default=0.025)
parser.add_argument("--blocking-alignment-threshold", type=float, default=0.25)
parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
parser.add_argument("--escape-lateral-target", type=float, default=0.35)
parser.add_argument("--gradient-steps", type=int, default=3)
parser.add_argument("--gradient-step-size", type=float, default=0.04)
parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
parser.add_argument("--maximum-obstacle-depth", type=float, default=5.00)
parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

app = Flask(__name__)
navigator: S2DiffPointGoalAgent | None = None


def planner_name() -> str:
    if args.planner_mode == "pure-navdp":
        return "navdp-pure-critic"
    if args.planner_mode == "gradient":
        return (
            "navdp-hlc-gradient-no-critic"
            if args.remove_critic
            else "navdp-hlc-gradient"
        )
    return "navdp-hlc-s2diff-no-critic" if args.remove_critic else "navdp-hlc-s2diff"


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    global navigator
    payload = request.get_json()
    intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
    batch_size = int(payload["batch_size"])
    stop_threshold = float(payload["stop_threshold"])
    if navigator is None:
        base = NavDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=args.checkpoint,
            device=args.device,
        )
        if args.planner_mode == "pure-navdp":
            navigator = base
        else:
            # Load the released checkpoint before optionally deleting the unused
            # critic head, preserving checkpoint compatibility for both modes.
            if args.remove_critic and hasattr(base.navi_former, "critic_head"):
                delattr(base.navi_former, "critic_head")
            navigator = S2DiffPointGoalAgent(
                base,
                guidance_config=S2DiffGuidanceConfig(
                    candidate_count=args.candidates,
                    particles_per_candidate=args.particles,
                    particle_std=args.particle_std,
                    guidance_strength=args.guidance_strength,
                    temperature=args.temperature,
                    particle_anchor=args.particle_anchor,
                    particle_energy_reweighting=args.particle_energy_reweighting,
                    particle_collision_mask=args.particle_collision_mask,
                    particle_noise_schedule=args.particle_noise_schedule,
                    progressive_guidance=args.progressive_guidance,
                    safe_distance=args.safe_distance,
                    hard_collision_distance=args.hard_collision_distance,
                    robot_radius=args.robot_radius,
                    safety_weight=args.safety_weight,
                    barrier_weight=args.barrier_weight,
                    barrier_rate=args.barrier_rate,
                    circulation_weight=args.circulation_weight,
                    circulation_activation_distance=args.circulation_activation_distance,
                    circulation_activation_sharpness=args.circulation_activation_sharpness,
                    minimum_circulation_progress=args.minimum_circulation_progress,
                    blocking_alignment_threshold=args.blocking_alignment_threshold,
                    circulation_switch_weight=args.circulation_switch_weight,
                    gradient_steps=args.gradient_steps,
                    gradient_step_size=args.gradient_step_size,
                    escape_lateral_target=args.escape_lateral_target,
                ),
                guidance_method=("gradient" if args.planner_mode == "gradient" else "particles"),
                pixel_obstacle_config=PixelObstacleConfig(
                    minimum_depth=args.minimum_obstacle_depth,
                    maximum_depth=args.maximum_obstacle_depth,
                    maximum_points=args.maximum_obstacle_pixels,
                ),
            )
    navigator.reset(batch_size, stop_threshold)
    return jsonify({"algo": planner_name()})

@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    if navigator is None:
        return jsonify({"error": "call /navigator_reset first"}), 400
    navigator.reset_env(int(request.get_json()["env_id"]))
    return jsonify({"algo": planner_name()})


def _decode_request():
    if navigator is None:
        raise RuntimeError("call /navigator_reset first")
    goal_data = json.loads(request.form["goal_data"])
    if "obstacle_pixels" not in goal_data:
        raise ValueError(
            "goal_data must include obstacle_pixels as one list of [u,v] pixels per batch item"
        )
    goal_x = np.asarray(goal_data["goal_x"])
    goal_y = np.asarray(goal_data["goal_y"])
    goals = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)

    image = Image.open(request.files["image"].stream).convert("RGB")
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.reshape((navigator.batch_size, -1, image.shape[1], 3))

    depth = Image.open(request.files["depth"].stream).convert("I")
    depth = np.asarray(depth, dtype=np.float32)[..., None] / 10000.0
    depth = depth.reshape((navigator.batch_size, -1, depth.shape[1], 1))
    return goals, image, depth, goal_data["obstacle_pixels"]


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    if navigator is None:
        return jsonify({"error": "call /navigator_reset first"}), 400
    try:
        goals, images, depths, obstacle_pixels = _decode_request()
        if args.planner_mode == "pure-navdp":
            selected, all_trajectories, all_values, _ = navigator.step_pointgoal(
                goals, images, depths
            )
        else:
            selected, all_trajectories, all_values, _ = navigator.step_pointgoal(
                goals, images, depths, obstacle_pixels
            )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if args.planner_mode == "pure-navdp":
        best_index = np.argmax(all_values, axis=1).astype(np.int64)
        fallback_stop = np.max(all_values, axis=1) < navigator.stop_threshold
        best_index[fallback_stop] = -1
        batch_size, candidate_count = all_values.shape
        zeros_float = np.zeros(batch_size, dtype=np.float32)
        zeros_bool = np.zeros(batch_size, dtype=bool)
        s2diff_payload = {
            "selected_index": best_index.tolist(),
            "fallback_stop": fallback_stop.tolist(),
            "escape_turn": zeros_bool.tolist(),
            "minimum_clearance": np.full(
                (batch_size, candidate_count), -1.0, dtype=np.float32
            ).tolist(),
            "valid_obstacle_points": np.zeros(batch_size, dtype=np.int64).tolist(),
            "selected_circulation_sign": zeros_float.tolist(),
            "selected_barrier_energy": zeros_float.tolist(),
            "selected_circulation_energy": zeros_float.tolist(),
            "selected_minimum_clearance": np.full(
                batch_size, -1.0, dtype=np.float32
            ).tolist(),
            "mean_guidance_noise_correction": zeros_float.tolist(),
            "final_guidance_noise_correction": zeros_float.tolist(),
            "maximum_guidance_noise_correction": zeros_float.tolist(),
            "mean_final_effective_sample_size": zeros_float.tolist(),
        }
    else:
        result = navigator.last_result
        assert result is not None
        s2diff_payload = {
            "selected_index": result.selected_index.tolist(),
            "fallback_stop": result.fallback_stop.tolist(),
            "escape_turn": result.escape_turn.tolist(),
            "minimum_clearance": result.minimum_clearance.tolist(),
            "valid_obstacle_points": result.diagnostics[
                "valid_obstacle_points"
            ].tolist(),
            "selected_circulation_sign": result.diagnostics[
                "selected_circulation_sign"
            ].tolist(),
            "selected_barrier_energy": result.diagnostics[
                "selected_barrier_energy"
            ].tolist(),
            "selected_circulation_energy": result.diagnostics[
                "selected_circulation_energy"
            ].tolist(),
            "selected_minimum_clearance": result.diagnostics[
                "selected_minimum_clearance"
            ].tolist(),
            "mean_guidance_noise_correction": result.diagnostics[
                "mean_guidance_noise_correction"
            ].tolist(),
            "final_guidance_noise_correction": result.diagnostics[
                "final_guidance_noise_correction"
            ].tolist(),
            "maximum_guidance_noise_correction": result.diagnostics[
                "maximum_guidance_noise_correction"
            ].tolist(),
            "mean_final_effective_sample_size": np.asarray(
                result.diagnostics.get(
                    "mean_final_effective_sample_size",
                    np.zeros(result.selected_index.shape, dtype=np.float32),
                )
            ).tolist(),
        }

    return jsonify(
        {
            "trajectory": selected.tolist(),
            "all_trajectory": all_trajectories.tolist(),
            "all_values": all_values.tolist(),
            "s2diff": s2diff_payload,
        }
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=args.port)
