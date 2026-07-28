import argparse
import math
import time, datetime
from pathlib import Path

from sam_vla.env.habitat_env import HFOV_DEG, MarsHabitatEnv
from sam_vla.env.sim_utils import distance_to_goal
from sam_vla.vlm.qwen_server_manager import QwenServerManager
from sam_vla.goal_resolution import first_frame_resolver
from sam_vla.policy.navdp_policy import NavdpPolicy
from sam_vla.safety.safety_filter import filter as safety_filter_fn
from sam_vla.safety.cbf_avoidance import CbfObstacleAvoidance
from sam_vla.core.belief_tracking import BeliefGoalTracker, lost_goal_heading_assist
from sam_vla.core.goal_geometry import (
    MESH_GOAL_ID,
    MESH_OBST_ID,
    backproject_goal_position,
    bbox_to_world,
    intrinsics_from_hfov,
    mask_pixel_center,
)
from sam_vla.core.pose_integrator import integrate_mars
from sam_vla.core.types import Action
from sam_vla.logging.rollout_logger import RolloutLogger
from sam_vla.perception.semantic_overlay import overlay_semantic_masks


def register_goal_obstacle_masks(env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir):
    """Give the chosen goal object a goal-mask mesh and every other detected
    object an obstacle-mask mesh, each a disc of `obj_mask_radius` around its
    bbox's backprojected world coords. The rest of the scene is untouched."""
    if goal_position is not None:
        env.register_object_mask(goal_position, MESH_GOAL_ID, obj_mask_radius, out_dir, "goal")
    else:
        print("[WARN] goal bbox had no valid depth; skipping goal mask", flush=True)

    for i, obstacle_bbox in enumerate(goal_spec.obstacle_bboxes_norm):
        obstacle_position = bbox_to_world(obs0, obstacle_bbox, hfov_deg=HFOV_DEG)
        if obstacle_position is None:
            print(f"[WARN] obstacle[{i}] bbox had no valid depth; skipping obstacle mask", flush=True)
            continue
        env.register_object_mask(obstacle_position, MESH_OBST_ID, obj_mask_radius, out_dir, f"obstacle_{i}")


def run(
    scene_path: str,
    heightmap_path: str,
    ckpt_path: str,
    out_dir: str,
    navdp_root: str = None,
    device: str = "cuda",
    sample_steps: int = 20,
    max_steps: int = 500,
    dt: float = 0.1,
    save_video: bool = False,
    save_frames: bool = False,
    video_fps: int = 10,
    start_x: float = 0.0,
    start_z: float = 8.0,
    start_yaw_deg: float = 0.0,
    randomise_spawn: bool = False,
    rock_field_path: str = None,
    obj_mask_radius: float = 0.5,
    cbf: bool = False,
    cbf_d_safe: float = 0.75,
    cbf_gamma: float = 0.3,
    cbf_deadzone: float = 0.6,
    cbf_orbit_kr: float = 0.8,
    cbf_orbit_hyst: float = 0.4,
    cbf_pursuit_kp: float = 1.8,
    cbf_goaround_forward: float = 0.5,
    cbf_escape_yaw: bool = True,
    cbf_hard_gate: bool = True,
    robot_radius: float = 0.25,
    safety_margin: float = 0.15,
    obstacle_radius: float = 0.25,
    max_yaw_rate: float = 1.0,
    zero_lateral: bool = True,
    belief_goal_range: float = 8.0,
    belief_odom_noise: float = 0.0,
    lost_goal_min_px: int = 10,
    lost_goal_ghost: bool = False,
    lost_goal_turn_kp: float = 1.4,
    lost_goal_forward: float = 0.0,
    lost_goal_bearing_deg: float = 30.0,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Still needed for the one-shot first-frame goal selection below
    # (first_frame_resolver -> qwen_client.select_goal_verbose); the driving
    # loop itself no longer calls the VLM per frame -- NavdpPolicy drives.
    qwen_manager = QwenServerManager()
    logger = RolloutLogger()

    with MarsHabitatEnv(
        scene_path,
        heightmap_path,
        services=[qwen_manager],
        start_x=start_x,
        start_z=start_z,
        start_yaw=math.radians(start_yaw_deg),
        randomise_spawn=randomise_spawn,
        with_semantic=True,
        rock_field_path=rock_field_path,
    ) as env:
        obs0 = env.get_observation(frame_idx=0)
        goal_spec, goal_vlm_result, sam_detections = first_frame_resolver.resolve_verbose(obs0.rgb)
        goal_position = backproject_goal_position(obs0, goal_spec, hfov_deg=HFOV_DEG)
        logger.log_goal_resolution(goal_spec, goal_vlm_result, goal_position)
        logger.save_sam_first_frame(obs0.rgb, sam_detections, goal_spec, out_dir)
        print(f"resolved goal_spec: {goal_spec.instruction_text} | goal_position={goal_position}")
        register_goal_obstacle_masks(env, obs0, goal_spec, goal_position, obj_mask_radius, out_dir)

        # <-- policy plugged in here: NavdpPolicy replaces QwenDiscreteDirectionPolicy.
        # Same act_verbose(..., goal_spec, step) -> (Action, dict) shape as the VLA
        # policy it swaps out; see the loop below for the call site.
        policy = NavdpPolicy(
            ckpt_path=ckpt_path,
            navdp_root=navdp_root,
            device=device,
            sample_steps=sample_steps,
        )

        # Belief-goal tracking: re-seed a body-frame [forward, left] estimate of the
        # goal from the live rendered mask whenever it's visible, dead-reckon it by
        # odometry the rest of the time -- ported from rollout_navdp_policy.py's
        # mesh_tracking_mode. avoidance is None unless --cbf is passed (constructed
        # after NavdpPolicy so navdp.extensions is importable -- see its docstring).
        belief_tracker = BeliefGoalTracker(
            hfov_deg=HFOV_DEG, goal_range=belief_goal_range, min_px=lost_goal_min_px,
            odom_noise=belief_odom_noise,
        )
        avoidance = (
            CbfObstacleAvoidance(
                d_safe=cbf_d_safe, gamma=cbf_gamma, deadzone=cbf_deadzone,
                orbit_kr=cbf_orbit_kr, orbit_hyst=cbf_orbit_hyst, pursuit_kp=cbf_pursuit_kp,
                goaround_forward=cbf_goaround_forward, escape_yaw=cbf_escape_yaw,
                hard_gate=cbf_hard_gate, robot_radius=robot_radius, safety_margin=safety_margin,
                obstacle_radius=obstacle_radius, max_yaw_rate=max_yaw_rate,
            )
            if cbf else None
        )
        cbf_active_steps = 0
        hard_gate_fired_steps = 0

        for step in range(max_steps):
            obs = env.get_observation(frame_idx=step)
            semantic = env.get_semantic_frame()
            raw_action, vla_result = policy.act_verbose(obs, semantic, goal_spec, step)
            action = safety_filter_fn(raw_action, obs)

            goal_mask = (semantic == MESH_GOAL_ID).astype("uint8") * 255
            obstacle_mask = (semantic == MESH_OBST_ID).astype("uint8") * 255
            goal_visible = belief_tracker.observe(goal_mask, obs.depth)
            goal_bearing = belief_tracker.bearing()

            obstacle_point = None
            if avoidance is not None:
                height, width = obs.depth.shape[:2]
                intr = intrinsics_from_hfov(height, width, HFOV_DEG)
                obstacle_point = avoidance.nearest_obstacle(obstacle_mask, obs.depth, intr)

            blocked = avoidance.is_blocked(obstacle_point, goal_bearing) if avoidance is not None else False

            if lost_goal_ghost and not blocked and goal_bearing is not None:
                action = lost_goal_heading_assist(
                    action, goal_bearing, goal_lost=not goal_visible,
                    turn_kp=lost_goal_turn_kp, forward_floor=lost_goal_forward,
                    bearing_deg_thresh=lost_goal_bearing_deg, max_yaw_rate=max_yaw_rate,
                )

            if zero_lateral and avoidance is not None:
                action = Action(v_fwd=action.v_fwd, v_lat=0.0, yaw_rate=action.yaw_rate)

            cbf_info = {}
            if avoidance is not None:
                action, cbf_info = avoidance.apply(action, obstacle_point, goal_bearing)
                if cbf_info.get("blocked"):
                    cbf_active_steps += 1
                if cbf_info.get("hard_gate_fired"):
                    hard_gate_fired_steps += 1

            new_pose = integrate_mars(obs.pose, action, dt)
            env.step(new_pose)
            belief_tracker.propagate(action, dt)

            dist = (
                distance_to_goal(new_pose, goal_position)
                if goal_position is not None
                else None
            )
            dist_txt = f"{dist:.2f}m" if dist is not None else "n/a"
            overlay_text = (
                f"t={step} dist={dist_txt} "
                f"v=[{action.v_fwd:.2f},{action.v_lat:.2f}] yaw_rate={action.yaw_rate:.2f}"
            )
            vis_rgb = overlay_semantic_masks(obs.rgb, semantic, text=overlay_text)
            vla_result = {
                **vla_result,
                "belief_forward": None if belief_tracker.belief_g is None else float(belief_tracker.belief_g[0]),
                "belief_left": None if belief_tracker.belief_g is None else float(belief_tracker.belief_g[1]),
                "goal_visible": goal_visible,
                **cbf_info,
            }
            logger.log_step(obs, action, new_pose, vla_result=vla_result, vis_rgb=vis_rgb)

            if step % 10 == 0:
                goal_pixel = mask_pixel_center(goal_mask)
                print(
                    f"[traj] step={step} | distance_to_goal={dist} | "
                    f"goal_pixel={goal_pixel} | action={action}"
                )

        if avoidance is not None:
            print(f"[CBF diag] blocked_steps={cbf_active_steps} hard_gate_fired={hard_gate_fired_steps}", flush=True)
        logger.flush(out_dir)
        if save_frames:
            logger.save_frames(out_dir)
        if save_video:
            logger.save_video(out_dir, fps=video_fps)

    print("[inf] qwen_manager: stop confirmed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--heightmap-path", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to trained NavDP/S2DiT checkpoint")
    parser.add_argument("--navdp-root", default=None, help="Path to the navdp repo (default: ./navdp or $NAVDP_ROOT)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--out-dir", default=f"navdp_rollout{datetime.datetime.now().strftime('%d%m%y%H%M')}")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save-video", action="store_true", help="Save rollout.mp4 from logged RGB frames")
    parser.add_argument("--save-frames", action="store_true", help="Save individual PNG frames under out_dir/frames/")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--start-x", type=float, default=0.0, help="Rover spawn x coordinate")
    parser.add_argument("--start-z", type=float, default=8.0, help="Rover spawn z coordinate")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="Rover spawn yaw in degrees")
    parser.add_argument(
        "--randomise-spawn",
        action="store_true",
        help="Ignore --start-x/--start-z/--start-yaw and pick a random (x, z) spawn within the "
        "heightmap bounds, with height sampled from the heightmap",
    )
    parser.add_argument(
        "--rock-field",
        default=None,
        help="Path to a rock_field.json produced by generate_rock_env.py. Loads that fixed, "
        "already-placed rock layout into the scene instead of an empty terrain -- use the same "
        "path across ablation runs to keep the obstacle layout identical.",
    )
    parser.add_argument(
        "--obj-mask-radius",
        type=float,
        default=0.5,
        help="Radius (m) of the goal/obstacle mask mesh placed around each detected object's "
        "backprojected world coords",
    )
    parser.add_argument("--cbf", action="store_true", help="Enable cone-mode CBF obstacle avoidance (orbit controller + hard-gate backstop)")
    parser.add_argument("--cbf-d-safe", type=float, default=0.75)
    parser.add_argument("--cbf-gamma", type=float, default=0.3)
    parser.add_argument("--cbf-deadzone", type=float, default=0.6)
    parser.add_argument("--cbf-orbit-kr", type=float, default=0.8, help="radial pull-back gain (rad/m) onto the d_safe circle")
    parser.add_argument("--cbf-orbit-hyst", type=float, default=0.4, help="extra clearance (m) required to leave the orbit once committed")
    parser.add_argument("--cbf-pursuit-kp", type=float, default=1.8, help="gain from tangent heading error to yaw-rate")
    parser.add_argument("--cbf-goaround-forward", type=float, default=0.5, help="cruise speed (m/s) while orbiting")
    parser.add_argument("--cbf-escape-yaw", action=argparse.BooleanOptionalAction, default=True, help="orbit around a blocking obstacle instead of only braking")
    parser.add_argument("--cbf-hard-gate", action=argparse.BooleanOptionalAction, default=True, help="per-tick backstop: brake if the executed action would breach the collision radius")
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.15)
    parser.add_argument("--obstacle-radius", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--zero-lateral", action=argparse.BooleanOptionalAction, default=True, help="zero v_lat before CBF avoidance (only applied when --cbf is set)")
    parser.add_argument("--belief-goal-range", type=float, default=8.0, help="fallback range (m) for the goal belief when depth at the mask is invalid")
    parser.add_argument("--belief-odom-noise", type=float, default=0.0, help="Gaussian odom noise per step for belief dead-reckoning (0 = perfect)")
    parser.add_argument("--lost-goal-min-px", type=int, default=10, help="goal-mask pixels below this count means the goal is out of view")
    parser.add_argument("--lost-goal-ghost", action="store_true", help="proportional heading assist toward the tracked goal belief when it's off-centre or out of view")
    parser.add_argument("--lost-goal-turn-kp", type=float, default=1.4)
    parser.add_argument("--lost-goal-forward", type=float, default=0.0, help="forward speed floor while the goal is fully out of view (pivot recovery)")
    parser.add_argument("--lost-goal-bearing-deg", type=float, default=30.0, help="engage heading assist once |goal bearing| exceeds this angle; 0 disables the angle trigger")
    args = parser.parse_args()

    run(
        scene_path=args.scene_path,
        heightmap_path=args.heightmap_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        navdp_root=args.navdp_root,
        device=args.device,
        sample_steps=args.sample_steps,
        max_steps=args.max_steps,
        dt=args.dt,
        save_video=args.save_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
        start_x=args.start_x,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw,
        randomise_spawn=args.randomise_spawn,
        rock_field_path=args.rock_field,
        obj_mask_radius=args.obj_mask_radius,
        cbf=args.cbf,
        cbf_d_safe=args.cbf_d_safe,
        cbf_gamma=args.cbf_gamma,
        cbf_deadzone=args.cbf_deadzone,
        cbf_orbit_kr=args.cbf_orbit_kr,
        cbf_orbit_hyst=args.cbf_orbit_hyst,
        cbf_pursuit_kp=args.cbf_pursuit_kp,
        cbf_goaround_forward=args.cbf_goaround_forward,
        cbf_escape_yaw=args.cbf_escape_yaw,
        cbf_hard_gate=args.cbf_hard_gate,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        obstacle_radius=args.obstacle_radius,
        max_yaw_rate=args.max_yaw_rate,
        zero_lateral=args.zero_lateral,
        belief_goal_range=args.belief_goal_range,
        belief_odom_noise=args.belief_odom_noise,
        lost_goal_min_px=args.lost_goal_min_px,
        lost_goal_ghost=args.lost_goal_ghost,
        lost_goal_turn_kp=args.lost_goal_turn_kp,
        lost_goal_forward=args.lost_goal_forward,
        lost_goal_bearing_deg=args.lost_goal_bearing_deg,
    )
