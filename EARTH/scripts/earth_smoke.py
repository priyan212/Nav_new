"""Render RGB + depth from the (Z-up-corrected) Indian Bend & Pima terrain GLB.

Run inside `mars_habitat`:  python EARTH/scripts/earth_smoke.py
Writes EARTH/out/earth_rgb.png and earth_depth.png.
"""
import os

import numpy as np

import habitat_sim

EARTH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(EARTH_DIR, "data", "indian_bend_and_pima_zup.glb")
OUT_DIR = os.path.join(EARTH_DIR, "out")


def ground_height(sim, x: float, z: float):
    ray = habitat_sim.geo.Ray()
    ray.origin = np.array([x, 300.0, z], dtype=np.float32)
    ray.direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    hits = sim.cast_ray(ray, max_distance=600.0)
    if hits.has_hits():
        return float(hits.hits[0].point[1])
    return None


def main() -> None:
    assert os.path.exists(SCENE), f"terrain missing: {SCENE}"
    os.makedirs(OUT_DIR, exist_ok=True)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE
    backend_cfg.enable_physics = True

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 0.0, 0.0]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [480, 640]
    depth.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    scene_bounds = sim.get_active_scene_graph().get_root_node().cumulative_bb
    print(f"world-space scene bounds: {scene_bounds.min} .. {scene_bounds.max}")

    # Sample ground height near the middle of the footprint.
    cx = (scene_bounds.min[0] + scene_bounds.max[0]) / 2.0
    cz = (scene_bounds.min[2] + scene_bounds.max[2]) / 2.0
    gy = ground_height(sim, cx, cz)
    print(f"ground_height at center ({cx:.1f}, {cz:.1f}) = {gy}")
    if gy is None:
        gy = (scene_bounds.min[1] + scene_bounds.max[1]) / 2.0
        print(f"no raycast hit, falling back to bbox mid-height {gy:.2f}")

    state = habitat_sim.AgentState()
    state.position = np.array([cx, gy + 1.2, cz], dtype=np.float32)
    sim.get_agent(0).set_state(state)

    obs = sim.get_sensor_observations()

    from PIL import Image

    Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(OUT_DIR, "earth_rgb.png"))
    d = obs["depth"]
    d_vis = (np.clip(d / 25.0, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(d_vis).save(os.path.join(OUT_DIR, "earth_depth.png"))

    print(f"depth range [{d.min():.2f}, {d.max():.2f}] m")
    print(f"saved {OUT_DIR}/earth_rgb.png and earth_depth.png")
    sim.close()


if __name__ == "__main__":
    main()
