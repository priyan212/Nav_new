# EARTH — Real-world photogrammetry terrain in Habitat-Sim for NavDP evaluation

Evaluate the Nav_new DINO+NavDP navigation stack on a real-world photogrammetry
scan instead of a generated heightmap terrain like [MARS](../MARS/README.md).
First (and so far only) asset: a Sketchfab scan called "Indian Bend and Pima"
(Indian Bend Rd & Pima Rd, Scottsdale AZ). Despite the name, survey renders
show it's a construction/retail-intersection site, not a wash or nature
trail: paved road + roundabout, parking lots with cars, a tan stucco
building, a red "Target" pylon sign, an orange "Home Depot"-style sign, a
yellow excavator, active dirt/construction mounds, small site sheds,
landscaping shrubs, and a pink/rusted wrecked-car prop.

## Environment

Same shared conda env as MARS (`internnav` stays untouched — Habitat-Sim's
conda builds need Python 3.9):

```bash
conda activate mars_habitat
```

## Layout

```
EARTH/
├── data/
│   ├── indian_bend_and_pima.glb       # raw Sketchfab download (Y-up, 99 MB, 4k textures, 16 sub-meshes)
│   ├── indian_bend_and_pima_zup.glb   # Z-up-corrected terrain habitat-sim actually loads
│   └── sky_dome.glb                   # generated sky (see below)
├── scripts/
│   ├── glb_yup_to_zup.py     # Y-up -> Z-up axis fix (see gotcha below)
│   ├── survey.py             # overhead + ground-level grid renders -> out/survey/*.png, used to catalog the scene
│   ├── earth_smoke.py        # terrain render sanity check -> out/earth_rgb.png, earth_depth.png
│   ├── make_skydome.py       # generates data/sky_dome.glb (gradient + sun disc, KHR_materials_unlit patch)
│   ├── habitat_sim_node.py   # headless sim node (mars_habitat env)
│   └── earth_gui.py          # DINO+NavDP GUI (internnav env; port of MARS's mars_gui.py)
├── out/                      # rendered outputs, survey tiles, GUI screenshots
└── launch_earth.sh           # one-command bringup (sim node + GUI)
```

## Terrain: Sketchfab photogrammetry scan

**Axis gotcha (opposite cause from MARS, same symptom):** a Sketchfab GLB is
proper Y-up per the glTF spec (verified via trimesh bounds: Y extent ~19 m vs
X~148 m / Z~330 m footprint). habitat-sim's raw `backend.scene_id = <path>`
loader (used directly — no scene-dataset/stage_config JSON in this install)
always treats input as Z-up and rotates it into its own Y-up world, so a
spec-correct Y-up GLB renders standing on its edge. MARS's hand-generated
terrain was built Z-up on purpose to dodge this; a Sketchfab download isn't,
so it needs an explicit fix:

```bash
python EARTH/scripts/glb_yup_to_zup.py \
    EARTH/data/indian_bend_and_pima.glb \
    EARTH/data/indian_bend_and_pima_zup.glb
```

The transform is +90° about X, `(x, y, z) -> (x, -z, y)`, confirmed
empirically by rendering and checking the mound/tire-tracks/Target-sign scene
is right-side up. The "textbook" Y-up→Z-up rotation (-90° about X) renders it
upside down — don't re-derive this from first principles, the matrix in
`glb_yup_to_zup.py` is already correct. Applied to the whole trimesh `Scene`
(not `force="mesh"`, which would flatten the 16 materials into one).

Ground height comes from a bullet raycast straight down
(`enable_physics=True`), same as MARS's `ground_height()` — no heightmap PNG
needed since there's no heightmap-generation step for a photogrammetry scan.

**License note:** the Sketchfab asset's CC license/status has not yet been
confirmed — check before any external distribution.

## Quick checks

```bash
conda activate mars_habitat
python EARTH/scripts/earth_smoke.py   # render sanity check -> EARTH/out/earth_{rgb,depth}.png
python EARTH/scripts/survey.py        # overhead + ground grid -> EARTH/out/survey/*.png
```

## Running the nav stack on Earth

```bash
./EARTH/launch_earth.sh
./EARTH/launch_earth.sh --target "target sign"
./EARTH/launch_earth.sh --max-climb-deg 25   # rover balking at climbing curbs/mounds? raise this
```

Two processes over Zenoh (same contract as Isaac Sim / MARS), extra args
forwarded to the GUI unchanged:

- `scripts/habitat_sim_node.py` (mars_habitat env) — headless habitat sim,
  kinematic rover agent, bullet ground raycast. Publishes
  `image_raw/compressed` (JPEG, 10 Hz default), `depth_raw` (32FC1, perfect
  sim depth), `earth/pose` (JSON `{"x","z","yaw","t"}` ground truth);
  subscribes `cmd_vel`/`rt/cmd_vel` (Twist, 0.5 s watchdog) and `earth/reset`
  (`"x,z,yaw"`, empty = default start). Topics are namespaced `earth/*` so
  MARS and EARTH nodes could in principle run side by side. Shares MARS's
  `RoverAgent` (same file layout/logic, see [MARS/README.md](../MARS/README.md#running-the-nav-stack-on-mars)),
  including the cosmetic terrain-slope pitch/roll/bounce camera shake —
  `earth/pose`'s x/z/yaw ground truth is unaffected.
- `scripts/earth_gui.py` (internnav env) — the DINO+NavDP GUI, presets
  `["yellow building", "target sign", "parked car", "sand mound",
  "excavator", "bush"]` chosen from the survey renders, plus an asymmetric
  world clamp for Random-goal (`WORLD_X_LIMIT=(-47.5, 90.5)`,
  `WORLD_Z_LIMIT=(-152.8, 161.1)`) — the footprint isn't an
  origin-centered square like Marsyard's.

**Default start pose is survey-verified, not guessed** (`--start-x 21.5
--start-z 25.9`): an earlier guess sat the rover nose-first against a
construction mound (camera buried in dirt, DINO returned near-full-frame
low-info boxes, false STOP triggers everywhere). This terrain's ground
height isn't smooth like Marsyard's — pick start/reset poses from
`survey.py`'s confirmed-clear grid, don't interpolate a "nearby" point.

**Sky:** the scene ships with zero configured lights, so the plain
vertex-colored mesh would render pitch black. `make_skydome.py` builds a
large (450 m radius) inverted-icosphere dome — gradient + sun disc, patched
with a `KHR_materials_unlit` material since glTF's default material is fully
metallic — loaded by default via `--sky` (pass `--sky ""` to disable).

Never run `earth_gui.py` alongside another `cmd_vel` publisher
(`mars_gui.py`, `isaac_gui.py`, `zenoh_node.py`).

**Occlusion handling:** like `MarsPipeline`, `earth_gui.py`'s pipeline feeds
the DINO/SAM goal point into a persistent `SubgoalBeliefBank`
(`navdp.extensions.belief_bank`) instead of freezing it at the last-seen
position — while the target is out of view its estimate is propagated by
the rover's own ego-motion (`earth/pose`) and its confidence decays, so
`SEARCH` only kicks in once confidence drops below `belief_confidence_min`.
See [MARS/README.md](../MARS/README.md#running-the-nav-stack-on-mars) for
more detail (including the `--belief-only` distractor-gated variant, which
EARTH doesn't currently have its own copy of) and the top-level
[README's "Goal belief" section](../README.md#goal-belief-surviving-occlusion)
for the real-rover/Isaac port of the same idea.
