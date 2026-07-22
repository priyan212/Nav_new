"""Headless Habitat-Sim smoke test.

Loads a test scene, renders RGB + depth from an agent camera, and saves
PNGs to MARS/out/. Run inside the `mars_habitat` conda env:

    python MARS/scripts/smoke_test.py
"""
import os

import numpy as np

import habitat_sim

MARS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(
    MARS_DIR, "data", "scene_datasets", "habitat-test-scenes", "skokloster-castle.glb"
)
OUT_DIR = os.path.join(MARS_DIR, "out")


def make_sim(scene_path: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = True

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 1.5, 0.0]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [480, 640]
    depth.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


def main() -> None:
    assert os.path.exists(SCENE), (
        f"Test scene missing: {SCENE}\n"
        "Download it first (see MARS/README.md)."
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    sim = make_sim(SCENE)
    obs = sim.get_sensor_observations()

    from PIL import Image

    rgb = obs["rgb"][..., :3]
    Image.fromarray(rgb).save(os.path.join(OUT_DIR, "smoke_rgb.png"))

    d = obs["depth"]
    d_vis = (np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(d_vis).save(os.path.join(OUT_DIR, "smoke_depth.png"))

    print(f"habitat_sim {habitat_sim.__version__}")
    print(f"rgb {rgb.shape} depth {d.shape} range [{d.min():.2f}, {d.max():.2f}] m")
    print(f"saved {OUT_DIR}/smoke_rgb.png and smoke_depth.png")
    sim.close()


if __name__ == "__main__":
    main()
