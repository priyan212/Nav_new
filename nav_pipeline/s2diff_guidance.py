"""S2Diff-style safety energy for guiding NavDP's DDPM action sampler.

Ported and trimmed from tryout/s2diff_guidance.py for this repo's actual
NavDPStandalone (nav_pipeline/navdp_net.py), which always runs batch=1 on the
real rover. Dropped relative to the tryout version: the "circulation"
homotopy-mode bookkeeping (left/right persistence across ticks) and the
gradient-descent guidance variant -- this repo already has its own
oscillation guard at the steering-command level (obstacle_guard.py's
apply_avoid_cooldown), so duplicating a second, differently-mechanised
anti-oscillation system inside the diffusion loop is unneeded complexity for
a first integration.

This module has no dependency on pipeline.py or navdp_net.py and does not
modify either -- see nav_pipeline/s2diff_navdp.py for the NavDPStandalone
integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class S2DiffConfig:
    """S2Diff-style particle guidance for a single-episode (batch=1) NavDP call."""

    candidate_count: int = 12          # number of candidate action sequences kept through the whole DDPM loop
    particles_per_candidate: int = 6   # particles sampled per candidate per step, used only to estimate the guided posterior
    particle_std: float = 0.22         # exploration noise around each candidate's predicted clean action
    guidance_strength: float = 0.85    # 0 = no guidance, 1 = fully replace the actor's noise with the guided posterior
    temperature: float = 0.35          # Gibbs temperature; lower = more selective toward low-energy particles
    particle_anchor: bool = True       # keep particle 0 exactly at the unperturbed clean-action mean
    particle_energy_reweighting: bool = True
    particle_collision_mask: bool = True
    particle_noise_schedule: bool = True   # shrink particle noise as denoising converges (sqrt(1-alpha_bar))
    progressive_guidance: bool = True      # ramp guidance_strength from 0 -> guidance_strength over the DDPM steps

    safe_distance: float = 0.42        # desired clearance beyond the robot footprint, meters
    hard_collision_distance: float = 0.24  # clearance below this marks a particle as colliding (near-infinite energy)
    robot_radius: float = 0.24         # single-circle footprint approximation for the in-loop soft guidance
    safety_weight: float = 35.0

    terminal_goal_weight: float = 1.5
    lyapunov_weight: float = 4.0       # penalizes trajectories that don't make steady progress toward the goal
    lyapunov_rate: float = 0.08
    lyapunov_buffer: float = 0.0
    nominal_weight: float = 0.35       # keeps guided particles close to NavDP's own unguided prediction
    smoothness_weight: float = 0.15
    maximum_step_length: float = 0.45
    step_weight: float = 20.0
    hard_collision_penalty: float = 1.0e4
    barrier_weight: float = 25.0       # discrete control-barrier-style term: clearance must not shrink too fast step to step
    barrier_rate: float = 0.15

    def __post_init__(self) -> None:
        if self.candidate_count < 1 or self.particles_per_candidate < 1:
            raise ValueError("candidate and particle counts must be positive")
        if self.particle_std < 0.0 or self.temperature <= 0.0:
            raise ValueError("particle_std must be non-negative and temperature positive")
        if not 0.0 <= self.guidance_strength <= 1.0:
            raise ValueError("guidance_strength must be in [0,1]")
        if not 0.0 <= self.hard_collision_distance <= self.safe_distance:
            raise ValueError("hard_collision_distance must be in [0,safe_distance]")
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        if not 0.0 < self.barrier_rate <= 1.0:
            raise ValueError("barrier_rate must be in (0,1]")


@dataclass(frozen=True)
class TrajectoryEnergy:
    total: torch.Tensor
    soft_total: torch.Tensor
    minimum_clearance: torch.Tensor
    collision: torch.Tensor


def integrate_actions(actions: torch.Tensor) -> torch.Tensor:
    """Match NavDPStandalone's own action -> trajectory convention (cumsum/4)."""

    return torch.cumsum(actions / 4.0, dim=-2)


def trajectory_energy(
    trajectories: torch.Tensor,
    goals: torch.Tensor,
    points_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    config: S2DiffConfig,
    nominal_actions: torch.Tensor | None = None,
) -> TrajectoryEnergy:
    """Safety + goal-progress + smoothness energy.

    Shapes: trajectories=[B,N,T,>=2], goals=[B,>=2], points_xy=[B,M,2],
    valid_mask=[B,M]. All obstacle points and the goal must already be in the
    same local frame as the trajectories (x forward, y left).
    """

    if trajectories.ndim != 4 or trajectories.shape[-1] < 2:
        raise ValueError("trajectories must have shape [B,N,T,>=2]")
    batch, count, horizon = trajectories.shape[:3]
    if goals.shape[0] != batch or goals.shape[-1] < 2:
        raise ValueError("goals must have shape [B,>=2]")
    if points_xy.shape[0] != batch or valid_mask.shape[0] != batch:
        raise ValueError("obstacle batch does not match trajectory batch")

    xy = trajectories[..., :2]
    flat_xy = xy.reshape(batch, count * horizon, 2)
    distances = torch.cdist(flat_xy, points_xy)
    distances = distances.masked_fill(~valid_mask[:, None, :], float("inf"))
    flat_clearance, _ = distances.min(dim=-1)
    clearance = flat_clearance.reshape(batch, count, horizon) - config.robot_radius
    obstacle_present = valid_mask.any(dim=-1)[:, None, None]

    minimum_clearance = clearance.amin(dim=-1)
    collision = minimum_clearance < config.hard_collision_distance
    safety = F.relu(config.safe_distance - clearance).square().mean(dim=-1)

    finite_clearance = torch.where(
        torch.isfinite(clearance),
        clearance.clamp_min(0.0),
        torch.full_like(clearance, config.safe_distance + 1.0),
    )
    barrier_value = finite_clearance.square() - config.safe_distance**2
    if horizon >= 2:
        barrier_residual = (
            (1.0 - config.barrier_rate) * barrier_value[..., :-1] - barrier_value[..., 1:]
        )
        barrier = F.relu(barrier_residual).square().mean(dim=-1)
    else:
        barrier = torch.zeros_like(minimum_clearance)
    barrier = barrier * obstacle_present[..., 0].to(barrier.dtype)

    goal_xy = goals[:, None, None, :2]
    squared_goal_distance = (xy - goal_xy).square().sum(dim=-1)
    start_value = goals[:, :2].square().sum(dim=-1)[:, None, None].expand(-1, count, -1)
    values = torch.cat((start_value, squared_goal_distance), dim=-1)
    lyapunov_residual = (
        values[..., 1:] - values[..., :-1] + config.lyapunov_rate * values[..., :-1] + config.lyapunov_buffer
    )
    lyapunov = F.relu(lyapunov_residual).square().mean(dim=-1)
    terminal_goal = squared_goal_distance[..., -1]

    actions = torch.diff(trajectories, dim=-2, prepend=torch.zeros_like(trajectories[..., :1, :])) * 4.0
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
    step_penalty = F.relu(step_length - config.maximum_step_length).square().mean(dim=-1)

    soft_total = (
        config.safety_weight * safety
        + config.terminal_goal_weight * terminal_goal
        + config.lyapunov_weight * lyapunov
        + config.nominal_weight * nominal
        + config.smoothness_weight * smoothness
        + config.step_weight * step_penalty
        + config.barrier_weight * barrier
    )
    total = soft_total + collision.to(soft_total.dtype) * config.hard_collision_penalty
    return TrajectoryEnergy(
        total=total, soft_total=soft_total, minimum_clearance=minimum_clearance, collision=collision
    )


def reshape_particle_energy(energy: TrajectoryEnergy, candidates: int, particles: int) -> TrajectoryEnergy:
    return TrajectoryEnergy(
        **{
            name: getattr(energy, name).reshape(1, candidates, particles)
            for name in TrajectoryEnergy.__dataclass_fields__
        }
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
        -energy.soft_total / temperature if use_energy_reweighting else torch.zeros_like(energy.soft_total)
    )
    if use_collision_mask:
        safe_logits = logits.masked_fill(energy.collision, -torch.inf)
        all_collide = energy.collision.all(dim=-1, keepdim=True)
        logits = torch.where(all_collide, logits, safe_logits)
    weights = torch.softmax(logits, dim=-1)
    mean = (weights[..., None, None] * clean_action_particles).sum(dim=2)
    return mean, weights
