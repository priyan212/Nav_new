"""Render RGB + depth from the generated Marsyard terrain GLB.

Run inside `mars_habitat`:  python MARS/scripts/mars_smoke.py
Writes MARS/out/mars_rgb.png and mars_depth.png.
"""
import os

import numpy as np

import habitat_sim

MARS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(MARS_DIR, "mars-habitatsim", "assets", "marsyard2022.glb")
OUT_DIR = os.path.join(MARS_DIR, "out")


def main() -> None:
    assert os.path.exists(SCENE), f"terrain missing: {SCENE}"
    os.makedirs(OUT_DIR, exist_ok=True)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE
    backend_cfg.enable_physics = False

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

    # Rover-height camera near the center of the yard.
    state = habitat_sim.AgentState()
    state.position = np.array([0.0, 1.2, 0.0], dtype=np.float32)
    sim.get_agent(0).set_state(state)

    obs = sim.get_sensor_observations()

    from PIL import Image

    Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(OUT_DIR, "mars_rgb.png"))
    d = obs["depth"]
    d_vis = (np.clip(d / 25.0, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(d_vis).save(os.path.join(OUT_DIR, "mars_depth.png"))

    print(f"depth range [{d.min():.2f}, {d.max():.2f}] m")
    print(f"saved {OUT_DIR}/mars_rgb.png and mars_depth.png")
    sim.close()


if __name__ == "__main__":
    main()
