from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import habitat_sim
import numpy as np
import quaternion
from habitat_sim.agent import AgentConfiguration

from sam_vla.core.goal_geometry import GoalPosition, terrain_patch_mesh
from sam_vla.core.lifecycle import ServiceRegistry
from sam_vla.core.types import Observation, Pose
from sam_vla.env.terrain import SIZE_X, SIZE_Y, SIZE_Z, HeightmapGrid, Terrain
from sam_vla.env.sim_utils import make_sensor, register_semantic_mesh, rgb_depth, save_obj, set_agent_pose
from sam_vla.env.rock_generation import RockSpec, load_rock_field, register_rocks

RGB_HEIGHT = 480
RGB_WIDTH = 640
HFOV_DEG = 90.0
DEPTH_MAX_RANGE_M = 10.0

SPAWN_CLEARANCE_M = 1.8
SPAWN_TERRAIN_RADIUS_M = 0.8


class MarsHabitatEnv:
    def __init__(
        self,
        scene_path: str,
        heightmap_path: str,
        services: list = None,
        start_x: float = 0.0,
        start_z: float = 8.0,
        start_yaw: float = 0.0,
        randomise_spawn: bool = False,
        spawn_clearance: float = SPAWN_CLEARANCE_M,
        spawn_terrain_radius: float = SPAWN_TERRAIN_RADIUS_M,
        with_semantic: bool = False,
        rock_field_path: Optional[str] = None,
    ):
        self._scene_path = Path(scene_path)
        self._heightmap_path = Path(heightmap_path)
        self._registry = ServiceRegistry()
        for service in services or []:
            self._registry.register(service)
        self._sim = None
        self._agent = None
        self._terrain = None
        self._start_x = start_x
        self._start_z = start_z
        self._start_yaw = start_yaw
        self._randomise_spawn = randomise_spawn
        self._spawn_clearance = spawn_clearance
        self._spawn_terrain_radius = spawn_terrain_radius
        self._with_semantic = with_semantic
        self._rock_field_path = Path(rock_field_path) if rock_field_path else None
        self.rocks: List[RockSpec] = []

    def __enter__(self) -> "MarsHabitatEnv":
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self._scene_path.expanduser().resolve())
        sim_cfg.enable_physics = False

        rgb_spec = make_sensor("rgb", habitat_sim.SensorType.COLOR, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG)
        depth_spec = make_sensor("depth", habitat_sim.SensorType.DEPTH, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG)
        # make_sensor doesn't set a far clip; cap depth range here to a sane sensor spec.
        depth_spec.far = DEPTH_MAX_RANGE_M

        sensor_specs = [rgb_spec, depth_spec]
        if self._with_semantic:
            sensor_specs.append(
                make_sensor("semantic", habitat_sim.SensorType.SEMANTIC, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG)
            )

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self._agent = self._sim.initialize_agent(0)

        heightmap_grid = HeightmapGrid(
            self._heightmap_path.expanduser().resolve(),
            size_x=SIZE_X,
            size_z=SIZE_Z,
            size_y=SIZE_Y,
            flip_x=False,
            flip_z=True,
            swap_xz=False,
        )
        self._terrain = Terrain(heightmap_grid, flip_x=False, flip_z=False, swap_xz=False)

        if self._randomise_spawn:
            x = random.uniform(-SIZE_X / 2.0, SIZE_X / 2.0)
            z = random.uniform(-SIZE_Z / 2.0, SIZE_Z / 2.0)
            yaw = random.uniform(0.0, 2.0 * 3.141592653589793)
        else:
            x, z, yaw = self._start_x, self._start_z, self._start_yaw

        y = self.get_height_at_xz(x, z)
        set_agent_pose(self._agent, x, y, z, yaw)

        if self._rock_field_path is not None:
            self.rocks, _rock_config = load_rock_field(self._rock_field_path)
            register_rocks(self._sim, self.rocks)

        self._registry.start_all()
        return self

    def get_height_at_xz(self, x: float, z: float) -> float:
        """Terrain height at (x, z), plus rover clearance, sampled from the heightmap."""
        return self._terrain.local_height_max(x, z, self._spawn_terrain_radius) + self._spawn_clearance

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._registry.stop_all()
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        return False

    def get_observation(self, frame_idx: int) -> Observation:
        obs = self._sim.get_sensor_observations()
        rgb, depth = rgb_depth(obs)
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        yaw = float(quaternion.as_rotation_vector(state.rotation)[1])
        pose = Pose(x=x, y=float(state.position[1]), z=z, yaw=yaw)
        return Observation(rgb=rgb, depth=depth, pose=pose, frame_idx=frame_idx)

    def get_semantic_frame(self) -> np.ndarray:
        """Raw per-pixel semantic-id image (H, W) from the semantic sensor:
        each pixel holds the semantic_id of whatever registered mask mesh is
        rendered there (see register_object_mask / goal_geometry.MESH_GOAL_ID
        / MESH_OBST_ID), 0 elsewhere. Requires with_semantic=True."""
        obs = self._sim.get_sensor_observations()
        return np.asarray(obs["semantic"])

    def step(self, pose: Pose) -> None:
        # Match spawn's ground offset (local-max + clearance), not a raw single-point
        # sample, or the agent snaps to bare terrain height every step and clips into
        # the surface.
        y = self.get_height_at_xz(pose.x, pose.z)
        set_agent_pose(self._agent, pose.x, y, pose.z, pose.yaw)

    def register_object_mask(
        self,
        world_pos: GoalPosition,
        semantic_id: int,
        radius: float,
        out_dir: str,
        name: str,
    ):
        """Register a small terrain-following patch mesh at world_pos as a
        render-only, non-collidable object carrying `semantic_id`, so the
        semantic sensor renders a goal/obstacle mask around that point.
        Seeded from an already-backprojected world point
        (goal_geometry.bbox_to_world) rather than a raw pixel + depth patch;
        the patch itself is resampled from the terrain heightmap (self._terrain)
        so it hugs the ground instead of sitting on one flat plane. Requires
        with_semantic=True.
        """
        verts, faces = terrain_patch_mesh(world_pos, radius, self._terrain)
        mesh_dir = Path(out_dir) / "masks"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = str(mesh_dir / f"{name}.obj")
        save_obj(mesh_path, verts, faces)
        return register_semantic_mesh(self._sim, mesh_path, semantic_id)


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent.parent.parent
    scene = HERE / "assets" / "marsyard2022.glb"
    heightmap = HERE / "marsyard2022_terrain_hm.png"

    with MarsHabitatEnv(str(scene), str(heightmap), services=[]) as env:
        obs = env.get_observation(0)
        print(f"rgb shape={obs.rgb.shape} dtype={obs.rgb.dtype}")
        print(f"pose={obs.pose}")
