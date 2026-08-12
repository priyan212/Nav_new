"""Guided replacement for NavDPStandalone.sample_pointgoal -- additive only.

Nothing in navdp_net.py or pipeline.py is edited by this module. It builds a
drop-in `sample_pointgoal(self, goal_point, images, depths, sample_num=32)`
function that steers NavDP's DDPM denoising loop away from depth-observed
obstacles using the S2Diff particle-guidance idea (see
tryout/S2DIFF_GUIDANCE.md and s2diff_guidance.py), then installs it via a
class-level monkeypatch of NavDPStandalone -- see patch_navdp_standalone()
and nav_pipeline/s2diff_runner.py, the new opt-in entry point that applies
the patch before starting the existing nav_pipeline.isaac_gui.main().

Reuses this repo's own obstacle_guard.depth_to_obstacle_points (the same
local-slope ground filter pipeline.py's forward_guard/swept_clearance
already rely on) to build the obstacle point cloud, so the guidance energy
sees the same notion of "obstacle" the rest of the stack does. Everything
downstream of sample_pointgoal in pipeline.py -- _select_trajectory,
swept_clearance, forward_guard, apply_avoid_cooldown -- is untouched and
still has final say over the executed command; this module only changes
which candidate trajectories NavDP proposes.

Known deviations from pipeline.py's own obstacle extraction:
  * pipeline.py's depth_to_obstacle_points call uses the full-resolution
    camera depth image and excludes the target's SAM mask. This module only
    has access to the 224x224 depth tensor already handed to
    sample_pointgoal, and has no SAM mask, so intrinsics are approximated
    from that smaller resolution and the goal object itself may be softly
    penalized as an obstacle near approach. Guidance strength is capped and
    ramped (S2DiffConfig.progressive_guidance), and the terminal-goal energy
    term still pulls toward the goal, so this only softens the final
    approach rather than blocking it -- the precise, SAM-excluded veto still
    happens downstream in pipeline.py, unchanged.
"""

from __future__ import annotations

import torch

from .goal_utils import intrinsics_from_fov
from .obstacle_guard import GuardConfig, depth_to_obstacle_points
from .s2diff_guidance import (
    S2DiffConfig,
    integrate_actions,
    reshape_particle_energy,
    smc_particle_mean,
    trajectory_energy,
)


def robot_radius_from_guard(guard: GuardConfig) -> float:
    """Single-circle footprint approximation for the in-loop soft guidance.

    swept_clearance()'s precise two-circle hull sweep still runs unchanged
    downstream in pipeline.py for the actual veto; this radius only shapes
    the soft nudge during diffusion sampling.
    """

    return float(torch.hypot(torch.tensor(guard.footprint_length / 2.0), torch.tensor(guard.footprint_width / 2.0)))


def _current_frame_obstacles(depths: torch.Tensor, fov_deg: float, guard: GuardConfig):
    """depths: (1,M,H,W,1) as passed to sample_pointgoal -> (Ni,2) obstacle points."""

    depth_np = depths[0, -1, :, :, 0].detach().cpu().float().numpy()
    h, w = depth_np.shape
    fx, fy, cx, cy = intrinsics_from_fov(w, h, fov_deg)
    return depth_to_obstacle_points(depth_np, fx, fy, cx, cy, guard)


def make_guided_sample_pointgoal(
    fov_deg: float = 90.0,
    guard: GuardConfig | None = None,
    config: S2DiffConfig | None = None,
):
    """Build a sample_pointgoal(self, goal_point, images, depths, sample_num=32) replacement.

    `self` is expected to be a NavDPStandalone instance (or anything exposing
    the same rgbd_encoder/point_encoder/_predict_noise/_predict_critic/
    noise_scheduler/predict_size/_device/input_dtype surface).
    """

    guard = guard or GuardConfig()
    config = config or S2DiffConfig(robot_radius=robot_radius_from_guard(guard))
    candidates = config.candidate_count
    particles = config.particles_per_candidate

    @torch.no_grad()
    def sample_pointgoal(self, goal_point, images, depths, sample_num=32):
        device = self._device
        goal = torch.as_tensor(goal_point, dtype=self.input_dtype, device=device).reshape(1, 3)
        rgbd_embed = self.rgbd_encoder(images.to(device), depths.to(device))
        goal_embed = self.point_encoder(goal).unsqueeze(1)

        obstacle_np = _current_frame_obstacles(depths, fov_deg, guard)
        if obstacle_np.shape[0] > 0:
            points_xy = torch.as_tensor(obstacle_np, dtype=torch.float32, device=device).unsqueeze(0)
            valid_mask = torch.ones(points_xy.shape[:2], dtype=torch.bool, device=device)
        else:
            points_xy = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
            valid_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

        # goal_point here is the checkpoint-native [x fwd, y RIGHT, z] convention
        # (pipeline.py already flips y before calling sample_pointgoal); obstacle
        # points and trajectories are [x fwd, y LEFT] -- flip back so the energy
        # compares goal and obstacles in the same frame as the trajectories.
        goal_local = torch.stack([goal[:, 0], -goal[:, 1]], dim=-1).to(torch.float32)

        # _predict_noise internally repeats goal_embed/rgbd_embed (batch=1) to
        # match the action batch (navdp_net.py's cond_embedding.repeat(...)),
        # so they're passed through unrepeated here -- same as the original
        # _denoise loop.
        noisy_actions = torch.randn((candidates, self.predict_size, 3), dtype=self.input_dtype, device=device)

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        timesteps = scheduler.timesteps

        for step_index, timestep in enumerate(timesteps):
            actor_noise = self._predict_noise(noisy_actions, timestep.unsqueeze(0), goal_embed, rgbd_embed)
            alpha_bar = torch.as_tensor(
                scheduler.alphas_cumprod[int(timestep.item())], dtype=noisy_actions.dtype, device=device
            )
            clean_mean = (noisy_actions - torch.sqrt(1.0 - alpha_bar) * actor_noise) / torch.sqrt(alpha_bar)
            clean_mean = clean_mean.clamp(-1.0, 1.0)

            noise_scale = config.particle_std
            if config.particle_noise_schedule:
                noise_scale = noise_scale * torch.sqrt(1.0 - alpha_bar)
            particle_noise = torch.randn(
                (candidates, particles, self.predict_size, 3), dtype=clean_mean.dtype, device=device
            )
            clean_particles = clean_mean[:, None] + noise_scale * particle_noise
            if config.particle_anchor:
                clean_particles[:, 0] = clean_mean
            clean_particles = clean_particles.clamp(-1.0, 1.0)

            flattened = clean_particles.reshape(candidates * particles, self.predict_size, 3)
            flat_traj = integrate_actions(flattened.float()).unsqueeze(0)
            nominal = clean_mean[:, None].expand_as(clean_particles).reshape_as(flattened).float().unsqueeze(0)

            energy = trajectory_energy(flat_traj, goal_local, points_xy, valid_mask, config, nominal_actions=nominal)
            energy = reshape_particle_energy(energy, candidates, particles)

            posterior_clean, _weights = smc_particle_mean(
                clean_particles.unsqueeze(0),
                energy,
                config.temperature,
                use_energy_reweighting=config.particle_energy_reweighting,
                use_collision_mask=config.particle_collision_mask,
            )
            posterior_clean = posterior_clean.reshape_as(noisy_actions).to(noisy_actions.dtype)
            posterior_noise = (noisy_actions - torch.sqrt(alpha_bar) * posterior_clean) / torch.sqrt(
                torch.clamp(1.0 - alpha_bar, min=1.0e-8)
            )
            strength = config.guidance_strength
            if config.progressive_guidance:
                strength = strength * (step_index + 1) / len(timesteps)
            guided_noise = torch.lerp(actor_noise, posterior_noise, strength)
            noisy_actions = scheduler.step(model_output=guided_noise, timestep=timestep, sample=noisy_actions).prev_sample

        critic = self._predict_critic(noisy_actions, rgbd_embed).float()
        trajs = torch.cumsum(noisy_actions.float() / 4.0, dim=1)
        return trajs, critic

    return sample_pointgoal


def patch_navdp_standalone(fov_deg: float = 90.0, guard: GuardConfig | None = None, config: S2DiffConfig | None = None) -> None:
    """Install the guided sampler onto the NavDPStandalone class.

    Must run before any DinoNavDPPipeline is constructed (pipeline.py's
    self.policy.sample_pointgoal(...) call resolves the method on the class
    dynamically, so patching the class attribute before construction is
    sufficient -- no edits to navdp_net.py or pipeline.py needed).
    """

    from . import navdp_net

    navdp_net.NavDPStandalone.sample_pointgoal = make_guided_sample_pointgoal(fov_deg=fov_deg, guard=guard, config=config)
