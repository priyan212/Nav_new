from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .depth_obstacles import DepthObstacleBatch

# I have added a
@dataclass(frozen=True)
class S2DiffGuidanceConfig:
    """S2Diff-style sequential Monte Carlo guidance for NavDP actions."""

    candidate_count: int = 16 #This is the number of candidate trajectories to sample per batch. Each candidate will have its own set of particles for evaluation.
    particles_per_candidate: int = 8 #How each candidate trajectory is evaluated.
    particle_std: float = 0.22 # How much each particle explores around the clean action mean.
    guidance_strength: float = 0.85 # How strongly the S2Diff Gibbs factor influences the actor's noise prediction. A value of 0 means no guidance.
    temperature: float = 0.35 # My Gibbs free energy temperature for the S2Diff posterior. Lower values make the guidance more selective, as in it gives preference to higher energy particles.
    particle_anchor: bool = True
    particle_energy_reweighting: bool = True
    particle_collision_mask: bool = True
    particle_noise_schedule: bool = True
    progressive_guidance: bool = True
    safe_distance: float = 0.42 # Desired clearance beyond the rover body surface, in metres.
    hard_collision_distance: float = 0.24 # Hard-rejection margin beyond the rover body surface, in metres.
    robot_radius: float = 0.24 # Circular planar rover footprint radius in metres.
    safety_weight: float = 35.0 #Higher values make the robot begin avoiding obstacles earlier and more strongly.
    terminal_goal_weight: float = 1.5 #Controls how strongly the final predicted waypoint should approach the PointGoal.
    lyapunov_weight: float = 4.0 # Controls the importance of making consistent progress toward the goal throughout the trajectory.
    lyapunov_rate: float = 0.08 #Specifies the requested rate of goal-distance reduction per predicted step.
    lyapunov_buffer: float = 0.0
    nominal_weight: float = 0.35 #Keeps S2Diff particles close to the original NavDP clean-action prediction.
    smoothness_weight: float = 0.15 # Controls how strongly the planner avoids sudden changes in trajectory direction. The smoothness penalty uses the second difference of consecutive positions:
    maximum_step_length: float = 0.45 #Defines the preferred maximum distance between two consecutive predicted waypoints.
    step_weight: float = 20.0 #Controls the strength of the excessive-step penalty.
    hard_collision_penalty: float = 1.0e4 #Adds a large energy to trajectories that cross the hard collision distance:
    barrier_weight: float = 25.0
    barrier_rate: float = 0.15
    circulation_weight: float = 18.0
    circulation_activation_distance: float = 1.50
    circulation_activation_sharpness: float = 0.20
    minimum_circulation_progress: float = 0.025
    blocking_alignment_threshold: float = 0.25
    circulation_switch_weight: float = 2.0
    escape_lateral_target: float = 0.35
    gradient_steps: int = 3
    gradient_step_size: float = 0.04
    gradient_norm_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.candidate_count < 1 or self.particles_per_candidate < 1:
            raise ValueError("candidate and particle counts must be positive")
        if self.particle_std < 0.0 or self.temperature <= 0.0:
            raise ValueError("particle_std must be non-negative and temperature positive")
        if not 0.0 <= self.guidance_strength <= 1.0:
            raise ValueError("guidance_strength must be in [0,1]")
        if not 0.0 <= self.hard_collision_distance <= self.safe_distance:
            raise ValueError("collision distance must be in [0,safe_distance]")
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        if not 0.0 < self.barrier_rate <= 1.0:
            raise ValueError("barrier_rate must be in (0,1]")
        if self.circulation_activation_distance < self.safe_distance:
            raise ValueError("circulation activation distance must cover safe_distance")
        if self.circulation_activation_sharpness <= 0.0:
            raise ValueError("circulation activation sharpness must be positive")
        if not -1.0 < self.blocking_alignment_threshold < 1.0:
            raise ValueError("blocking alignment threshold must be in (-1,1)")
        if self.gradient_steps < 1:
            raise ValueError("gradient_steps must be positive")
        if self.gradient_step_size <= 0.0 or self.gradient_norm_epsilon <= 0.0:
            raise ValueError("gradient step size and norm epsilon must be positive")



@dataclass(frozen=True)
class TrajectoryEnergy:
    total: torch.Tensor
    soft_total: torch.Tensor
    minimum_clearance: torch.Tensor
    collision: torch.Tensor
    safety: torch.Tensor
    lyapunov: torch.Tensor
    terminal_goal: torch.Tensor
    barrier: torch.Tensor
    circulation: torch.Tensor
    mode_switch: torch.Tensor


@dataclass(frozen=True)
class S2DiffCandidateBatch:
    selected_trajectory: np.ndarray
    all_trajectories: np.ndarray
    all_values: np.ndarray
    action_deltas: np.ndarray
    energy: np.ndarray
    minimum_clearance: np.ndarray
    selected_index: np.ndarray
    fallback_stop: np.ndarray
    escape_turn: np.ndarray
    diagnostics: dict[str, np.ndarray | float | int]


def integrate_actions(actions: torch.Tensor) -> torch.Tensor:
    """Use the released NavDP action scaling and integration convention."""

    return torch.cumsum(actions / 4.0, dim=-2)


def trajectory_energy(
    trajectories: torch.Tensor,
    goals: torch.Tensor,
    obstacles: DepthObstacleBatch,
    config: S2DiffGuidanceConfig,
    nominal_actions: torch.Tensor | None = None,
    circulation_signs: torch.Tensor | None = None,
    preferred_circulation_signs: torch.Tensor | None = None,
) -> TrajectoryEnergy:
    """Evaluate safety, stability, and homotopy-conditioned liveness energy.

    ``circulation_signs`` assigns every candidate a persistent left/right
    homotopy mode. The discrete barrier discourages loss of clearance, while
    circulation makes a stationary head-on solution expensive near a blocking
    obstacle. Shapes are ``trajectories=[B,N,T,3]`` and ``goals=[B,3]``.
    """

    if trajectories.ndim != 4 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [B,N,T,>=2]")
    batch, count, horizon = trajectories.shape[:3]
    if goals.shape[0] != batch or goals.shape[-1] < 2:
        raise ValueError("goals must have shape [B,>=2]")
    if obstacles.points_xy.shape[0] != batch:
        raise ValueError("obstacle batch does not match trajectory batch")

    xy = trajectories[..., :2]
    flat_xy = xy.reshape(batch, count * horizon, 2)
    distances = torch.cdist(flat_xy, obstacles.points_xy)
    distances = distances.masked_fill(~obstacles.valid_mask[:, None, :], float("inf"))
    flat_clearance, closest_indices = distances.min(dim=-1)
    # Convert center-to-obstacle distance into rover-body surface clearance.
    # The configured hard and safe distances are margins outside the footprint.
    clearance = (
        flat_clearance.reshape(batch, count, horizon) - config.robot_radius
    )
    closest_points = obstacles.points_xy.gather(
        1, closest_indices[..., None].expand(-1, -1, 2)
    ).reshape(batch, count, horizon, 2)
    obstacle_present = obstacles.valid_mask.any(dim=-1)[:, None, None]

    minimum_clearance = clearance.amin(dim=-1)
    collision = minimum_clearance < config.hard_collision_distance
    safety = F.relu(config.safe_distance - clearance).square().mean(dim=-1)

    finite_clearance = torch.where(
        torch.isfinite(clearance),
        clearance.clamp_min(0.0),
        torch.full_like(
            clearance,
            config.circulation_activation_distance + config.safe_distance + 1.0,
        ),
    )
    barrier_value = finite_clearance.square() - config.safe_distance**2
    if horizon >= 2:
        barrier_residual = (
            (1.0 - config.barrier_rate) * barrier_value[..., :-1]
            - barrier_value[..., 1:]
        )
        barrier = F.relu(barrier_residual).square().mean(dim=-1)
    else:
        barrier = torch.zeros_like(minimum_clearance)
    barrier = barrier * obstacle_present[..., 0].to(barrier.dtype)

    goal_xy = goals[:, None, None, :2]
    squared_goal_distance = (xy - goal_xy).square().sum(dim=-1)
    start_value = goals[:, :2].square().sum(dim=-1)[:, None, None]
    start_value = start_value.expand(-1, count, -1)
    values = torch.cat((start_value, squared_goal_distance), dim=-1)
    lyapunov_residual = (
        values[..., 1:]
        - values[..., :-1]
        + config.lyapunov_rate * values[..., :-1]
        + config.lyapunov_buffer
    )
    lyapunov = F.relu(lyapunov_residual).square().mean(dim=-1)
    terminal_goal = squared_goal_distance[..., -1]

    displacements = torch.diff(
        xy,
        dim=-2,
        prepend=torch.zeros_like(xy[..., :1, :]),
    )
    circulation = torch.zeros_like(terminal_goal)
    mode_switch = torch.zeros_like(terminal_goal)
    if circulation_signs is not None:
        signs = torch.as_tensor(
            circulation_signs, dtype=xy.dtype, device=xy.device
        )
        if signs.shape != (batch, count):
            raise ValueError("circulation_signs must have shape [B,N]")

        relative = xy - closest_points
        radial = relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)

        to_goal = goal_xy - xy
        to_obstacle = closest_points - xy
        alignment = (to_goal * to_obstacle).sum(dim=-1) / (
            torch.linalg.vector_norm(to_goal, dim=-1)
            * torch.linalg.vector_norm(to_obstacle, dim=-1)
        ).clamp_min(1.0e-6)
        blocking = F.relu(
            (alignment - config.blocking_alignment_threshold)
            / (1.0 - config.blocking_alignment_threshold)
        ).clamp_max(1.0)
        distance_gate = torch.sigmoid(
            (config.circulation_activation_distance - finite_clearance)
            / config.circulation_activation_sharpness
        )
        activation = (
            distance_gate
            * blocking
            * obstacle_present.to(distance_gate.dtype)
        )

        tangential_progress = (displacements * tangent).sum(dim=-1)
        signed_progress = signs[..., None] * tangential_progress
        circulation_residual = F.relu(
            config.minimum_circulation_progress - signed_progress
        ).square()
        circulation = (activation * circulation_residual).sum(dim=-1) / (
            activation.sum(dim=-1).clamp_min(1.0)
        )

        if preferred_circulation_signs is not None:
            preferred = torch.as_tensor(
                preferred_circulation_signs, dtype=xy.dtype, device=xy.device
            ).reshape(-1)
            if preferred.shape != (batch,):
                raise ValueError("preferred circulation signs must have shape [B]")
            preference_active = preferred.abs() > 0.5
            mismatch = signs * preferred[:, None] < 0.0
            mode_switch = (
                mismatch.to(xy.dtype)
                * preference_active[:, None].to(xy.dtype)
                * activation.amax(dim=-1)
            )

    actions = torch.diff(
        trajectories,
        dim=-2,
        prepend=torch.zeros_like(trajectories[..., :1, :]),
    ) * 4.0
    if nominal_actions is None:
        nominal = torch.zeros_like(terminal_goal)
    else:
        if nominal_actions.shape != actions.shape:
            raise ValueError("nominal_actions must match reconstructed action shape")
        nominal = (actions - nominal_actions).square().mean(dim=(-2, -1))
    if horizon >= 3:
        smoothness = torch.diff(xy, n=2, dim=-2).square().mean(dim=(-2, -1))
    else:
        smoothness = torch.zeros_like(terminal_goal)
    step_length = torch.linalg.vector_norm(actions[..., :2] / 4.0, dim=-1)
    step_penalty = F.relu(
        step_length - config.maximum_step_length
    ).square().mean(dim=-1)

    soft_total = (
        config.safety_weight * safety
        + config.terminal_goal_weight * terminal_goal
        + config.lyapunov_weight * lyapunov
        + config.nominal_weight * nominal
        + config.smoothness_weight * smoothness
        + config.step_weight * step_penalty
        + config.barrier_weight * barrier
        + config.circulation_weight * circulation
        + config.circulation_switch_weight * mode_switch
    )
    total = soft_total + collision.to(soft_total.dtype) * config.hard_collision_penalty
    return TrajectoryEnergy(
        total=total,
        soft_total=soft_total,
        minimum_clearance=minimum_clearance,
        collision=collision,
        safety=safety,
        lyapunov=lyapunov,
        terminal_goal=terminal_goal,
        barrier=barrier,
        circulation=circulation,
        mode_switch=mode_switch,
    )

def smc_particle_mean(
    clean_action_particles: torch.Tensor,
    energy: TrajectoryEnergy,
    temperature: float,
    *,
    use_energy_reweighting: bool = True,
    use_collision_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute E[U0|Ui] with the hard-safe Gibbs factor used by S2Diff."""

    if clean_action_particles.ndim != 5:
        raise ValueError("particles must have shape [B,K,P,T,A]")
    logits = (
        -energy.soft_total / temperature
        if use_energy_reweighting
        else torch.zeros_like(energy.soft_total)
    )
    if use_collision_mask:
        safe_logits = logits.masked_fill(energy.collision, -torch.inf)
        all_collide = energy.collision.all(dim=-1, keepdim=True)
        logits = torch.where(all_collide, logits, safe_logits)
    weights = torch.softmax(logits, dim=-1)
    mean = (weights[..., None, None] * clean_action_particles).sum(dim=2)
    return mean, weights


def _reshape_particle_energy(
    energy: TrajectoryEnergy, batch: int, candidates: int, particles: int
) -> TrajectoryEnergy:
    return TrajectoryEnergy(
        **{
            name: getattr(energy, name).reshape(batch, candidates, particles)
            for name in TrajectoryEnergy.__dataclass_fields__
        }
    )


def sample_s2diff_pointgoal_candidates(
    policy: Any,
    goal_point: np.ndarray,
    input_images: np.ndarray,
    input_depths: np.ndarray,
    obstacles: DepthObstacleBatch,
    config: S2DiffGuidanceConfig = S2DiffGuidanceConfig(),
    preferred_circulation_signs: np.ndarray | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> S2DiffCandidateBatch:
    """Run NavDP denoising with S2Diff-style safe/stable posterior guidance.

    NavDP supplies the learned diffusion proposal. At every DDPM step, local
    clean-action particles are reweighted by the S2Diff Gibbs target and their
    posterior mean modifies the actor score. Candidate modes stay separate.
    """

    goals_np = np.asarray(goal_point, dtype=np.float32)
    if goals_np.ndim != 2 or goals_np.shape[1] != 3:
        raise ValueError("goal_point must have shape [B,3]")
    batch = goals_np.shape[0]
    device = torch.device(policy.device)
    obstacles = obstacles.to(device)
    goals = torch.as_tensor(goals_np, dtype=torch.float32, device=device)
    candidates = config.candidate_count
    particles = config.particles_per_candidate
    if preferred_circulation_signs is None:
        preferred_signs = torch.zeros(batch, dtype=torch.float32, device=device)
    else:
        preferred_signs = torch.as_tensor(
            preferred_circulation_signs, dtype=torch.float32, device=device
        ).reshape(-1)
        if preferred_signs.shape != (batch,):
            raise ValueError("preferred circulation signs must have shape [B]")
        preferred_signs = torch.sign(preferred_signs)
    base_signs = torch.where(
        torch.arange(candidates, device=device) % 2 == 0,
        torch.ones(candidates, device=device),
        -torch.ones(candidates, device=device),
    )
    orientation = torch.where(
        preferred_signs.abs() > 0.5,
        preferred_signs,
        torch.ones_like(preferred_signs),
    )
    candidate_signs = orientation[:, None] * base_signs[None, :]
    particle_signs = candidate_signs[:, :, None].expand(-1, -1, particles).reshape(
        batch, candidates * particles
    )

    with torch.inference_mode():
        rgbd = policy.rgbd_encoder(input_images, input_depths)
        pointgoal = policy.point_encoder(goals).unsqueeze(1)
        repeated_rgbd = torch.repeat_interleave(rgbd, candidates, dim=0)
        repeated_goal = torch.repeat_interleave(pointgoal, candidates, dim=0)
        noisy_actions = torch.randn(
            (batch * candidates, policy.predict_size, 3),
            device=device,
            generator=generator,
        )

        scheduler = policy.noise_scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        timesteps = scheduler.timesteps
        final_weights = None
        guidance_corrections = []
        for step_index, timestep in enumerate(timesteps):
            actor_noise = policy.predict_noise(
                noisy_actions, timestep.unsqueeze(0), repeated_goal, repeated_rgbd
            )
            alpha_bar = torch.as_tensor(
                scheduler.alphas_cumprod[int(timestep.item())],
                dtype=noisy_actions.dtype,
                device=device,
            )
            clean_mean = (
                noisy_actions - torch.sqrt(1.0 - alpha_bar) * actor_noise
            ) / torch.sqrt(alpha_bar)
            clean_mean = clean_mean.clamp(-1.0, 1.0).reshape(
                batch, candidates, policy.predict_size, 3
            )

            noise_scale = config.particle_std
            if config.particle_noise_schedule:
                noise_scale = noise_scale * torch.sqrt(1.0 - alpha_bar)
            particle_noise = torch.randn(
                (batch, candidates, particles, policy.predict_size, 3),
                dtype=clean_mean.dtype,
                device=device,
                generator=generator,
            )
            clean_particles = clean_mean[:, :, None] + noise_scale * particle_noise
            if config.particle_anchor:
                clean_particles[:, :, 0] = clean_mean
            clean_particles = clean_particles.clamp(-1.0, 1.0)
            particle_trajectories = integrate_actions(clean_particles)
            flattened_trajectories = particle_trajectories.reshape(
                batch, candidates * particles, policy.predict_size, 3
            )
            nominal = clean_mean[:, :, None].expand_as(clean_particles).reshape_as(
                flattened_trajectories
            )
            particle_energy = _reshape_particle_energy(
                trajectory_energy(
                    flattened_trajectories,
                    goals,
                    obstacles,
                    config,
                    nominal_actions=nominal,
                    circulation_signs=particle_signs,
                    preferred_circulation_signs=preferred_signs,
                ),
                batch,
                candidates,
                particles,
            )
            posterior_clean, final_weights = smc_particle_mean(
                clean_particles,
                particle_energy,
                config.temperature,
                use_energy_reweighting=config.particle_energy_reweighting,
                use_collision_mask=config.particle_collision_mask,
            )
            posterior_clean = posterior_clean.reshape_as(noisy_actions)
            posterior_noise = (
                noisy_actions - torch.sqrt(alpha_bar) * posterior_clean
            ) / torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-8))
            strength = config.guidance_strength
            if config.progressive_guidance:
                strength = strength * (step_index + 1) / len(timesteps)
            guided_noise = torch.lerp(actor_noise, posterior_noise, strength)
            correction = (guided_noise - actor_noise).square().mean(dim=(-2, -1)).sqrt()
            guidance_corrections.append(
                correction.reshape(batch, candidates).mean(dim=1)
            )
            noisy_actions = scheduler.step(
                model_output=guided_noise,
                timestep=timestep,
                sample=noisy_actions,
            ).prev_sample

        action_deltas = noisy_actions.reshape(batch, candidates, policy.predict_size, 3)
        trajectories = integrate_actions(action_deltas)
        final_energy = trajectory_energy(
            trajectories,
            goals,
            obstacles,
            config,
            nominal_actions=action_deltas,
            circulation_signs=candidate_signs,
            preferred_circulation_signs=preferred_signs,
        )
        selectable_energy = final_energy.total.masked_fill(final_energy.collision, torch.inf)
        all_collide = final_energy.collision.all(dim=1)
        safe_index = selectable_energy.argmin(dim=1)
        least_violating_index = final_energy.soft_total.argmin(dim=1)
        selected_index = torch.where(all_collide, least_violating_index, safe_index)
        batch_indices = torch.arange(batch, device=device)
        selected = trajectories[batch_indices, selected_index].clone()
        selected_sign = candidate_signs.gather(1, selected_index[:, None])[:, 0]

        origin_distance = torch.linalg.vector_norm(obstacles.points_xy, dim=-1)
        origin_distance = origin_distance.masked_fill(~obstacles.valid_mask, torch.inf)
        origin_clearance = origin_distance.amin(dim=-1) - config.robot_radius
        escape_turn = all_collide & (
            origin_clearance >= config.hard_collision_distance
        )
        fallback_stop = all_collide & ~escape_turn
        if escape_turn.any():
            escape_rows = torch.where(escape_turn)[0]
            ramp = torch.linspace(
                1.0 / policy.predict_size,
                1.0,
                policy.predict_size,
                dtype=selected.dtype,
                device=device,
            )
            selected[escape_rows] = 0.0
            selected[escape_rows, :, 1] = (
                selected_sign[escape_rows, None]
                * config.escape_lateral_target
                * ramp[None, :]
            )
        selected[fallback_stop] = 0.0
        selected_index = selected_index.masked_fill(fallback_stop, -1)

        energy_np = final_energy.total.cpu().numpy()
        assert final_weights is not None
        correction_history = torch.stack(guidance_corrections, dim=1)
        gather_index = selected_index.clamp_min(0)[:, None]
        selected_clearance = final_energy.minimum_clearance.gather(
            1, gather_index
        )[:, 0]
        # Escape executes a rotation at the current position, so report its
        # real positional clearance instead of the rejected path's clearance.
        selected_clearance = torch.where(
            escape_turn | fallback_stop, origin_clearance, selected_clearance
        )
        selected_barrier = final_energy.barrier.gather(1, gather_index)[:, 0]
        selected_circulation = final_energy.circulation.gather(1, gather_index)[:, 0]
        selected_sign = torch.where(
            obstacles.valid_mask.any(dim=-1),
            selected_sign,
            torch.zeros_like(selected_sign),
        )
        return S2DiffCandidateBatch(
            selected_trajectory=selected.cpu().numpy(),
            all_trajectories=trajectories.cpu().numpy(),
            all_values=-energy_np,
            action_deltas=action_deltas.cpu().numpy(),
            energy=energy_np,
            minimum_clearance=final_energy.minimum_clearance.cpu().numpy(),
            selected_index=selected_index.cpu().numpy(),
            fallback_stop=fallback_stop.cpu().numpy(),
            escape_turn=escape_turn.cpu().numpy(),
            diagnostics={
                "particles_per_candidate": particles,
                "guidance_strength": config.guidance_strength,
                "valid_obstacle_points": obstacles.valid_mask.sum(dim=-1).cpu().numpy(),
                "robot_radius": config.robot_radius,
                "selected_circulation_sign": selected_sign.cpu().numpy(),
                "selected_barrier_energy": selected_barrier.cpu().numpy(),
                "selected_circulation_energy": selected_circulation.cpu().numpy(),
                "mean_guidance_noise_correction": correction_history.mean(dim=1)
                .cpu()
                .numpy(),
                "final_guidance_noise_correction": correction_history[:, -1]
                .cpu()
                .numpy(),
                "maximum_guidance_noise_correction": correction_history.amax(dim=1)
                .cpu()
                .numpy(),
                "selected_minimum_clearance": selected_clearance.cpu().numpy(),
                "final_effective_sample_size": (
                    1.0 / final_weights.square().sum(dim=-1).clamp_min(1.0e-8)
                ).cpu().numpy(),
                "mean_final_effective_sample_size": (
                    1.0
                    / final_weights.square().sum(dim=-1).clamp_min(1.0e-8)
                )
                .mean(dim=1)
                .cpu()
                .numpy(),
            },
        )


def sample_gradient_pointgoal_candidates(
    policy: Any,
    goal_point: np.ndarray,
    input_images: np.ndarray,
    input_depths: np.ndarray,
    obstacles: DepthObstacleBatch,
    config: S2DiffGuidanceConfig = S2DiffGuidanceConfig(),
    preferred_circulation_signs: np.ndarray | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> S2DiffCandidateBatch:
    """Guide NavDP by descending trajectory energy in clean-action space.

    Unlike the particle sampler, this method differentiates the soft trajectory
    energy with respect to each candidate clean action. The hard collision mask
    remains a terminal rejection rule because a Boolean threshold has no useful
    gradient.
    """

    goals_np = np.asarray(goal_point, dtype=np.float32)
    if goals_np.ndim != 2 or goals_np.shape[1] != 3:
        raise ValueError("goal_point must have shape [B,3]")
    batch = goals_np.shape[0]
    device = torch.device(policy.device)
    obstacles = obstacles.to(device)
    goals = torch.as_tensor(goals_np, dtype=torch.float32, device=device)
    candidates = config.candidate_count

    if preferred_circulation_signs is None:
        preferred_signs = torch.zeros(batch, dtype=torch.float32, device=device)
    else:
        preferred_signs = torch.as_tensor(
            preferred_circulation_signs, dtype=torch.float32, device=device
        ).reshape(-1)
        if preferred_signs.shape != (batch,):
            raise ValueError("preferred_circulation_signs must have shape [B]")
        preferred_signs = torch.sign(preferred_signs)

    base_signs = torch.where(
        torch.arange(candidates, device=device) % 2 == 0,
        torch.ones(candidates, device=device),
        -torch.ones(candidates, device=device),
    )
    orientation = torch.where(
        preferred_signs.abs() > 0.5,
        preferred_signs,
        torch.ones_like(preferred_signs),
    )
    candidate_signs = orientation[:, None] * base_signs[None, :]

    with torch.inference_mode():
        rgbd = policy.rgbd_encoder(input_images, input_depths)
        pointgoal = policy.point_encoder(goals).unsqueeze(1)
        repeated_rgbd = torch.repeat_interleave(rgbd, candidates, dim=0)
        repeated_goal = torch.repeat_interleave(pointgoal, candidates, dim=0)
        noisy_actions = torch.randn(
            (batch * candidates, policy.predict_size, 3),
            device=device,
            generator=generator,
        )
        scheduler = policy.noise_scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        timesteps = scheduler.timesteps

    guidance_corrections = []
    gradient_rms_history = []
    for step_index, timestep in enumerate(timesteps):
        with torch.inference_mode():
            actor_noise = policy.predict_noise(
                noisy_actions, timestep.unsqueeze(0), repeated_goal, repeated_rgbd
            )
            alpha_bar = torch.as_tensor(
                scheduler.alphas_cumprod[int(timestep.item())],
                dtype=noisy_actions.dtype,
                device=device,
            )
            clean_mean = (
                noisy_actions - torch.sqrt(1.0 - alpha_bar) * actor_noise
            ) / torch.sqrt(alpha_bar)
            clean_mean = clean_mean.clamp(-1.0, 1.0).reshape(
                batch, candidates, policy.predict_size, 3
            )

        # Clone outside inference mode so autograd may treat it as an ordinary
        # leaf during the inner energy-descent iterations.
        guided_clean = clean_mean.detach().clone()
        step_gradient_rms = []
        for _ in range(config.gradient_steps):
            clean_variable = guided_clean.detach().requires_grad_(True)
            trajectories = integrate_actions(clean_variable)
            energy = trajectory_energy(
                trajectories,
                goals,
                obstacles,
                config,
                nominal_actions=clean_mean.detach(),
                circulation_signs=candidate_signs,
                preferred_circulation_signs=preferred_signs,
            )
            gradient = torch.autograd.grad(
                energy.soft_total.sum(), clean_variable, only_inputs=True
            )[0]
            gradient_rms = gradient.square().mean(dim=(-2, -1), keepdim=True).sqrt()
            normalized_gradient = gradient / gradient_rms.clamp_min(
                config.gradient_norm_epsilon
            )
            guided_clean = (
                clean_variable - config.gradient_step_size * normalized_gradient
            ).clamp(-1.0, 1.0).detach()
            step_gradient_rms.append(gradient_rms.detach())

        posterior_clean = guided_clean.reshape_as(noisy_actions)
        with torch.inference_mode():
            posterior_noise = (
                noisy_actions - torch.sqrt(alpha_bar) * posterior_clean
            ) / torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-8))
            strength = config.guidance_strength * (step_index + 1) / len(timesteps)
            guided_noise = torch.lerp(actor_noise, posterior_noise, strength)
            correction = (
                (guided_noise - actor_noise).square().mean(dim=(-2, -1)).sqrt()
            )
            guidance_corrections.append(
                correction.reshape(batch, candidates).mean(dim=1)
            )
            mean_step_gradient = torch.stack(step_gradient_rms, dim=0).mean(dim=0)
            gradient_rms_history.append(
                mean_step_gradient.reshape(batch, candidates).mean(dim=1)
            )
            noisy_actions = scheduler.step(
                model_output=guided_noise,
                timestep=timestep,
                sample=noisy_actions,
            ).prev_sample

    with torch.inference_mode():
        action_deltas = noisy_actions.reshape(
            batch, candidates, policy.predict_size, 3
        )
        trajectories = integrate_actions(action_deltas)
        final_energy = trajectory_energy(
            trajectories,
            goals,
            obstacles,
            config,
            nominal_actions=action_deltas,
            circulation_signs=candidate_signs,
            preferred_circulation_signs=preferred_signs,
        )
        selectable_energy = final_energy.total.masked_fill(
            final_energy.collision, torch.inf
        )
        all_collide = final_energy.collision.all(dim=1)
        safe_index = selectable_energy.argmin(dim=1)
        least_violating_index = final_energy.soft_total.argmin(dim=1)
        selected_index = torch.where(all_collide, least_violating_index, safe_index)
        batch_indices = torch.arange(batch, device=device)
        selected = trajectories[batch_indices, selected_index].clone()
        selected_sign = candidate_signs.gather(1, selected_index[:, None])[:, 0]

        origin_distance = torch.linalg.vector_norm(obstacles.points_xy, dim=-1)
        origin_distance = origin_distance.masked_fill(
            ~obstacles.valid_mask, torch.inf
        )
        origin_clearance = origin_distance.amin(dim=-1) - config.robot_radius
        escape_turn = all_collide & (
            origin_clearance >= config.hard_collision_distance
        )
        fallback_stop = all_collide & ~escape_turn
        if escape_turn.any():
            escape_rows = torch.where(escape_turn)[0]
            ramp = torch.linspace(
                1.0 / policy.predict_size,
                1.0,
                policy.predict_size,
                dtype=selected.dtype,
                device=device,
            )
            selected[escape_rows] = 0.0
            selected[escape_rows, :, 1] = (
                selected_sign[escape_rows, None]
                * config.escape_lateral_target
                * ramp[None, :]
            )
        selected[fallback_stop] = 0.0
        selected_index = selected_index.masked_fill(fallback_stop, -1)

        energy_np = final_energy.total.cpu().numpy()
        correction_history = torch.stack(guidance_corrections, dim=1)
        gradient_history = torch.stack(gradient_rms_history, dim=1)
        gather_index = selected_index.clamp_min(0)[:, None]
        selected_clearance = final_energy.minimum_clearance.gather(
            1, gather_index
        )[:, 0]
        selected_clearance = torch.where(
            escape_turn | fallback_stop, origin_clearance, selected_clearance
        )
        selected_barrier = final_energy.barrier.gather(1, gather_index)[:, 0]
        selected_circulation = final_energy.circulation.gather(
            1, gather_index
        )[:, 0]
        selected_sign = torch.where(
            obstacles.valid_mask.any(dim=-1),
            selected_sign,
            torch.zeros_like(selected_sign),
        )
        return S2DiffCandidateBatch(
            selected_trajectory=selected.cpu().numpy(),
            all_trajectories=trajectories.cpu().numpy(),
            all_values=-energy_np,
            action_deltas=action_deltas.cpu().numpy(),
            energy=energy_np,
            minimum_clearance=final_energy.minimum_clearance.cpu().numpy(),
            selected_index=selected_index.cpu().numpy(),
            fallback_stop=fallback_stop.cpu().numpy(),
            escape_turn=escape_turn.cpu().numpy(),
            diagnostics={
                "particles_per_candidate": 0,
                "gradient_steps": config.gradient_steps,
                "gradient_step_size": config.gradient_step_size,
                "guidance_strength": config.guidance_strength,
                "valid_obstacle_points": obstacles.valid_mask.sum(
                    dim=-1
                ).cpu().numpy(),
                "robot_radius": config.robot_radius,
                "selected_circulation_sign": selected_sign.cpu().numpy(),
                "selected_barrier_energy": selected_barrier.cpu().numpy(),
                "selected_circulation_energy": selected_circulation.cpu().numpy(),
                "mean_guidance_noise_correction": correction_history.mean(
                    dim=1
                ).cpu().numpy(),
                "final_guidance_noise_correction": correction_history[
                    :, -1
                ].cpu().numpy(),
                "maximum_guidance_noise_correction": correction_history.amax(
                    dim=1
                ).cpu().numpy(),
                "mean_energy_gradient_rms": gradient_history.mean(
                    dim=1
                ).cpu().numpy(),
                "selected_minimum_clearance": selected_clearance.cpu().numpy(),
            },
        )
