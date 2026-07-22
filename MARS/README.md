# MARS — Habitat simulation for NavDP evaluation

Evaluate the Nav_new DINO+NavDP navigation stack in a Habitat-Sim Mars
environment (ERC Marsyard 2022 terrain). Everything for this sub-project
lives under this folder.

## Environment

Dedicated conda env (`internnav` stays untouched — Habitat-Sim's conda builds
need Python 3.9):

```bash
conda activate mars_habitat
```

Installed: `habitat-sim` 0.3.3 (headless + bullet physics, `aihabitat`
channel), `habitat-lab` 0.3.3 (editable install from `MARS/habitat-lab`,
tag v0.3.4), plus `trimesh`, `numpy-quaternion`, `matplotlib`, `tqdm`,
`pillow==10.4.0` (habitat-sim pin — do not upgrade).

## Layout

```
MARS/
├── habitat-lab/          # habitat-lab source (editable install)
├── mars-habitatsim/      # clone of github.com/saad-mh/mars-habitatsim
│   ├── assets/marsyard2022.glb   # GENERATED terrain (see below)
│   ├── rock_envs/        # generated rock obstacle fields
│   ├── navdp/, sam_vla/  # that repo's NavDP-variant + SAM/VLM policy code
│   └── kb_teleop.py, rgbd_drive.py, ...   # sim drivers (expect assets/marsyard2022.glb)
├── data/                 # habitat-test-scenes (smoke test only)
├── scripts/
│   ├── smoke_test.py         # test-scene render sanity check
│   ├── obj2glb_trimesh.py    # OBJ->GLB without Blender (see axis note!)
│   └── mars_smoke.py         # Marsyard terrain render sanity check
└── out/                  # rendered outputs
```

## Marsyard terrain generation

The repo ships only the heightmap + texture; the mesh is generated:

```bash
cd MARS/mars-habitatsim
python hm2obj.py --heightmap marsyard2022_terrain_hm.png \
    --texture marsyard2022_terrain_texture.png \
    --size-x 50 --size-y 50 --size-z 4.820803273566 \
    --out assets/marsyard2022.obj --stride 4
python ../scripts/obj2glb_trimesh.py assets/marsyard2022.obj assets/marsyard2022.glb
```

Dimensions come from the repo's own `sam_vla/env/terrain.py` / `kb_teleop.py`
(SIZE_X=50, SIZE_Z=50, SIZE_Y=4.820803273566).

**Axis gotcha:** habitat-sim treats stage meshes as Z-up (front +Y) and
rotates to its internal Y-up world on load. The GLB must therefore keep
hm2obj's Z-up coordinates — a "correct" Y-up glTF renders standing on its
side in habitat.

## Quick checks

```bash
conda activate mars_habitat
python MARS/scripts/smoke_test.py   # indoor test scene -> MARS/out/smoke_*.png
python MARS/scripts/mars_smoke.py   # Marsyard terrain  -> MARS/out/mars_*.png
```

## Running the nav stack on Mars

```bash
./MARS/launch_mars.sh --rocks    # sim node (mars_habitat) + GUI (internnav)
```

Two processes over Zenoh (same contract as the Isaac Sim setup):

- `scripts/habitat_sim_node.py` (mars_habitat env) — headless habitat sim of
  the Marsyard. Publishes `image_raw/compressed` (JPEG 10 Hz), `depth_raw`
  (32FC1, perfect sim depth) and `mars/pose` (JSON ground truth); subscribes
  `cmd_vel` (ROS Twist, +angular = left) with a rover-style 0.5 s watchdog,
  and `mars/reset`. Ground height via bullet raycast; kinematic agent.
  Verified: fwd/turn signs match ROS, 10 Hz feed, DINO detects the rocks
  (~0.5-0.6 score for "big stone"/"rock"/"boulder").
- `scripts/mars_gui.py` (internnav env) — the Isaac GUI adapted for Mars:
  same camera + trajectory panels, Mars presets ("big stone", "boulder", ...),
  **Random goal** (random world point 4-8 m ahead, NavDP point-goal using
  mars/pose ground truth — no detection), **Go home** (point-goal back to
  spawn), **Reset rover**, STOP. `MarsPipeline.step_point` adds the
  detection-free point-goal path (guard + NavDP + servo, same shaping).

Never run mars_gui.py and the other GUIs/zenoh_node at the same time — all
publish `cmd_vel`.
