"""Survey the Indian Bend & Pima terrain: overhead tiles + a ground-level
grid, so we can catalog what's actually in the scene before picking GUI
target presets.

Run inside `mars_habitat`:  python EARTH/scripts/survey.py
Writes EARTH/out/survey/*.png
"""
import math
import os

import numpy as np
import quaternion  # noqa: F401
from PIL import Image

import habitat_sim

EARTH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(EARTH_DIR, "data", "indian_bend_and_pima_zup.glb")
OUT_DIR = os.path.join(EARTH_DIR, "out", "survey")


def ground_height(sim, x, z):
    ray = habitat_sim.geo.Ray()
    ray.origin = np.array([x, 300.0, z], dtype=np.float32)
    ray.direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    hits = sim.cast_ray(ray, max_distance=600.0)
    if hits.has_hits():
        return float(hits.hits[0].point[1])
    return None


def make_sim():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE
    backend_cfg.enable_physics = True

    def cam(uuid):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = habitat_sim.SensorType.COLOR
        s.resolution = [720, 960]
        s.hfov = 100.0
        return s

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam("rgb")]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


def save(sim, name, pos, rot):
    state = habitat_sim.AgentState()
    state.position = np.array(pos, dtype=np.float32)
    state.rotation = rot
    sim.get_agent(0).set_state(state, reset_sensors=False)
    obs = sim.get_sensor_observations()
    Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(OUT_DIR, name))
    print(f"saved {name}")


def looking_down_rotation():
    # pitch -90 deg about local X (camera forward -Z rotated to -Y)
    return np.quaternion(math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0)


def yaw_rotation(yaw):
    return np.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = make_sim()

    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    xmin, xmax = bb.min[0], bb.max[0]
    zmin, zmax = bb.min[2], bb.max[2]
    print(f"world bounds x [{xmin:.1f}, {xmax:.1f}]  z [{zmin:.1f}, {zmax:.1f}]")

    # --- overhead tiles: split the long (z) axis into 3 segments -------- #
    xc = (xmin + xmax) / 2.0
    seg = (zmax - zmin) / 3.0
    for i in range(3):
        zc = zmin + seg * (i + 0.5)
        gy = ground_height(sim, xc, zc)
        height = (gy if gy is not None else 0.0) + 140.0
        save(sim, f"overhead_{i}.png", [xc, height, zc], looking_down_rotation())

    # --- ground-level grid along the long axis, a few lateral offsets -- #
    n_along = 8
    lateral_offsets = [-40.0, 0.0, 40.0]
    for i in range(n_along):
        zc = zmin + 15 + (zmax - zmin - 30) * i / (n_along - 1)
        for j, dx in enumerate(lateral_offsets):
            xc2 = min(max(xc + dx, xmin + 5), xmax - 5)
            gy = ground_height(sim, xc2, zc)
            if gy is None:
                continue
            save(sim, f"grid_{i}_{j}.png", [xc2, gy + 1.2, zc], yaw_rotation(0.0))

    sim.close()


if __name__ == "__main__":
    main()
