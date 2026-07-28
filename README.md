```
Hello new user of this repo, I'm Priyan an intern at this lab.
If you need any help contact through GitHub: priyan212

```

# Nav_new — 6WD Rover Navigation: Grounding DINO + NavDP

Point a 6-wheeled rover at anything you can name — "trash bin", "red chair",
"boulder" — and it drives there on its own, swerving around whatever's in the
way. No LiDAR, a single RGB camera for perception — the only other sensing
is the ESP32's signed wheel-encoder RPM feed (`/rover/rpm`), which the GPU
side dead-reckons into a logged pose for diagnostics/safety (see
[Odometry logging](#odometry-logging)), not for navigation itself. The whole
stack (detection, depth, driving policy) runs in **≈1.3 GB of GPU memory**,
about 1/13th of the full InternVLA-N1 dual-system it was distilled from.

It runs unmodified against four different bodies — the real rover, an Isaac
Sim digital twin, a Habitat-Sim Mars yard, and a Habitat-Sim real-world
photogrammetry scan — because every one of them speaks the same Zenoh/ROS 2
wire contract described below.

## How a frame becomes a motor command

```
camera RGB ──► Grounding DINO ──► bbox ──► SAM 2.1 small ──► instance mask ──► CLIP
    │                                        (throttled ~1 Hz;         crop vs. target text;
    │                                    cached mask reused           below threshold ->
    │                                    between refreshes)            treated as no detection
    │                      mask centroid + mask-median depth + intrinsics
    ▼                                                          ▼
depth (sensor, or Depth Anything V2 ─────────────► 3D point goal (robot frame)
       metric monocular fallback)                              │
    │                                                          ▼
    ├──► obstacle guard: depth → obstacle points (target mask excluded)
    │       • hard stop + escape turn if forward corridor blocked
    │       • per-trajectory footprint-swept clearance veto
    │       • linear slow-down near obstacles
    │                                                          │
    └────────────► NavDP System-1: 32 diffusion trajectories + critic
                                                               │
              clearance veto → critic top-half gate → goal-progress argmax
                                                               │
                                                               ▼
                       look-ahead waypoint → cmd_vel (v, ω) → Zenoh → rover
```

**Why two networks instead of one big one:** Grounding DINO turns an
open-vocabulary text prompt into "where is it in this image" — something a
diffusion driving policy is bad at. NavDP turns "here's a 3D point, drive
there without hitting anything" into a smooth trajectory — something a
detector can't do. Neither model was trained on the other's job; gluing them
together with a goal-progress + safety-critic selection layer (rather than
trusting either one blindly) is what makes the combination reliable. See
`nav_pipeline/pipeline.py` for the exact per-frame logic.

**SAM+CLIP verification:** SAM re-segments the DINO box at most once per
second (`PipelineConfig.sam_period_s`) — the mask is cached and reused on
ticks in between, as long as the current DINO box still overlaps the one SAM
last saw. Every fresh SAM pass is paired with a CLIP check: the masked crop
is scored against the target phrase (softmax vs. generic "background/wall"
negatives), and a crop scoring below `clip_min_similarity` is rejected —
treated as no detection that tick — instead of letting the rover commit to a
confident DINO false positive.

**State machine:** `TRACK` (target detected → drive) → `SEARCH` (target lost
→ rotate to re-acquire) → `AVOID` (obstacle inside the hard-stop corridor →
stop + escape turn) → `STOP` (goal close enough, or its mask fills the view).

## Odometry logging

The real rover has no LiDAR and no dedicated odometry node, but the ESP32
firmware (`esp32/rover_6wd_complete.ino`) does have real quadrature encoders
on the mid wheel of each side, publishing signed L/R RPM on `/rover/rpm` (10
Hz, +ve = drives forward). `nav_pipeline/odometry_logger.py` dead-reckons
that into a differential-drive pose (x, y, θ) and appends a CSV row per
sample — no GPS/SLAM involved, so it drifts over long runs, but it's good
enough for per-attempt diagnostics.

- **One file per goal, not per run.** Every time the target text changes
  (GUI Send/preset button, or a new `omnivla/goal_text` message), the logger
  closes the current file and starts a fresh one — `odometry_log/odom_
  <slugified-target>_<timestamp>.csv` — resetting x/y/θ back to the origin,
  so each file is a self-contained record of "how did the rover move while
  pursuing this specific goal."
- **Spin-stall watchdog** (`isaac_gui.py`): a generic, multi-instance target
  (e.g. "chair" in a room full of chairs) can make Grounding DINO's
  re-acquire-on-loss hop between different physical objects each time the
  tracked one scrolls out of frame — the rover keeps turning the same way
  chasing "whichever one is in view now" without ever closing distance on
  any of them. If the dead-reckoned pose racks up more than a full rotation
  within 15 s (`SPIN_WINDOW_S`) while translating less than 0.3 m
  (`SPIN_DIST_THRESH_M`), the GUI force-stops and shows `SPIN STALL` instead
  of spinning indefinitely — latched until a new target is sent. Caught a
  real ~145 s / 17-turn incident during testing; see the CSV history for how
  to recognize the signature (θ monotonically running away while x, y barely
  move).

## The four worlds it runs in

The same `nav_pipeline` code drives all four — only the transport peer and
the speed caps change:

| Target | Launcher | Camera/depth source | Notes |
|---|---|---|---|
| **Real rover** (6WD, ESP32 + micro-ROS) | `./launch_rover.sh` | Pi camera (compressed JPEG only — raw RGB saturates the rover's Wi-Fi) | Brings the Pi's systemd services up, waits for camera + ESP32 heartbeat, then starts the GUI at real-world speed caps |
| **Isaac Sim** digital twin | `./launch_gui.sh` | Isaac's simulated camera + depth over Zenoh | Needs the Isaac scene playing and its ROS 2 bridge scripts running first (see script header) |
| **Mars habitat sim** (Habitat-Sim, ERC Marsyard terrain) | `./MARS/launch_mars.sh --rocks` | Habitat's simulated camera + perfect depth | Separate `mars_habitat` conda env for the sim node; see [MARS/README.md](MARS/README.md) |
| **Earth habitat sim** (Habitat-Sim, real-world photogrammetry scan) | `./EARTH/launch_earth.sh` | Habitat's simulated camera + perfect depth | Same `mars_habitat` conda env, a Sketchfab scan instead of generated terrain; see [EARTH/README.md](EARTH/README.md) |

All four publish/subscribe the identical Zenoh topics, so a policy change in
`nav_pipeline/` is tested once offline, once in Isaac/Mars/Earth, then run on
the real rover with nothing else touched.

## Repo layout

```
Nav_new/
├── nav_pipeline/         the package: perception + policy + transport
├── checkpoints/          model weights (see below)
├── scripts/              offline tests, diagnostics, Pi provisioning
├── configs/               yaml configs (isaac / real rover)
├── data/                 saved test frames + pipeline output snapshots
├── odometry_log/         dead-reckoned pose CSVs, one per goal (own README)
├── reference/            prior OmniVLA/InternVLA-N1 nodes, kept for reference
├── third_party/InternNav/  vendored InternNav source (checkpoints symlinked in)
├── esp32/                rover firmware + Pi-side serial/handshake helpers
├── MARS/                 Habitat-Sim Mars sub-project (own README)
├── EARTH/                Habitat-Sim real-world photogrammetry sub-project (own README)
└── launch_*.sh           one-command entry points for each of the 4 worlds
```

### `nav_pipeline/` — the package

- `dino_detector.py` — Grounding DINO wrapper (`IDEA-Research/grounding-dino-base`, local HF cache)
- `sam_segmenter.py` — SAM 2.1 hiera-small, box-prompted → instance mask, throttled ~1 Hz (`PipelineConfig.sam_period_s`)
- `clip_verifier.py` — CLIP (`openai/clip-vit-base-patch32`) scores the SAM crop against the target phrase; below-threshold crops are rejected as false positives (paired 1:1 with each SAM refresh)
- `obstacle_guard.py` — depth → obstacle points, footprint-swept trajectory clearance veto, forward hard-stop corridor
- `navdp_crossmodal.py` — wrapper for the official standalone NavDP checkpoint (**default policy**)
- `navdp_net.py`, `navdp_backbone.py`, `depth_anything/` — standalone/extracted NavDP + Depth Anything V2, vendored from InternNav with imports fixed (InternNav's own package breaks under this repo's transformers version)
- `depth_estimator.py` — Depth Anything V2 metric monocular depth for the RGB-only real rover
- `goal_utils.py` — preprocessing + bbox→3D-goal math
- `odometry_logger.py` — dead-reckons `/rover/rpm` (real wheel-encoder RPM) into a pose, one CSV per goal under `odometry_log/`; also backs the GUI's spin-stall watchdog (see [Odometry logging](#odometry-logging))
- `pipeline.py` — the full perception→policy step, state machine, trajectory selection (the diagram above, as code)
- `isaac_gui.py` — the control panel: camera + mask/bbox overlay, top-down trajectories + obstacles, live target entry
- `zenoh_node.py` — headless transport node, same Zenoh/CDR contract as the old OmniVLA node

### `checkpoints/`

Symlinks into `OmniVLA_safe/third_party/InternNav/checkpoints` for the
shared Depth Anything V2 weights, plus `navdp_extracted.pth` (198 MB,
extracted from `InternVLA-N1-w-NavDP` via `scripts/extract_navdp_weights.py`
— the full source checkpoint is a 16 GB Qwen2.5-VL + NavDP bundle we don't
need most of). The official standalone `navdp-cross-modal.ckpt` (543 MB, not
symlinked, obtained via the InternRobotics request form) is the default
policy at repo root.

### `scripts/`

Everything here runs without a robot or GPU-simulator attached, except
where noted:

- `test_pipeline_offline.py` / `smoke_test.py` — full DINO→NavDP chain on a saved frame
- `test_zenoh_loopback.py` — exercises a *running* node over Zenoh with a fake camera
- `diag_goal_conditioning.py` — does the policy actually steer toward the goal?
- `test_footprint_guard.py` — unit test for the footprint-swept clearance math
- `extract_navdp_weights.py` — pulls the 198 MB NavDP head out of the 16 GB InternVLA-N1 checkpoint
- `pi_install_services.sh` / `pi_auto_handshake.sh` — run **on the Pi**: install the camera/agent/zenoh systemd services and recover a wedged ESP32 micro-ROS session without a physical reset button

### `esp32/`

Micro-ROS firmware for the 6-wheel differential drive base
(`rover_6wd_complete.ino`, subscribes `/cmd_vel`, publishes `/rover/rpm` —
signed L/R wheel RPM from the real quadrature encoders on each side's mid
wheel, 10 Hz, consumed by `nav_pipeline/odometry_logger.py`; `WHEEL_RADIUS_M`
/ `TRACK_WIDTH_M` must stay in sync between the two), plus Pi-side helpers
(`rover_handshake_manager.py`, `serial_bridge.py`) for talking to the ESP32
over `/dev/ttyUSB0`.

### `MARS/`

A self-contained sub-project: a Habitat-Sim recreation of the ERC Marsyard
2022 terrain (heightmap → generated mesh, since the source repo ships no
mesh) used to stress-test the same nav stack with GPS-quality ground truth
and a rock obstacle field. Full detail in [MARS/README.md](MARS/README.md).

### `EARTH/`

A second self-contained Habitat-Sim sub-project, sharing MARS's conda env
and node/GUI pattern but loading a real-world Sketchfab photogrammetry scan
("Indian Bend and Pima", Scottsdale AZ — a construction/retail-intersection
site) instead of generated terrain. Needs its own Y-up→Z-up axis fix (the
opposite gotcha from MARS's hand-built Z-up mesh) and a generated sky dome,
since the scan ships with no lights. Full detail in
[EARTH/README.md](EARTH/README.md).

## Running it

```bash
# Real rover (Pi bringup + GUI, real-world speed caps)
./launch_rover.sh

# Isaac Sim digital twin (needs the Isaac scene + ROS 2 bridge already running)
./launch_gui.sh --target "cardboard box"

# Mars habitat sim (spins up both the sim node and the GUI for you)
./MARS/launch_mars.sh --rocks

# Earth habitat sim (real-world photogrammetry scan, same conda env as Mars)
./EARTH/launch_earth.sh --target "target sign"

# Headless node only (no GUI), auto-discovers the rover/Isaac peer
./launch_dino_navdp.sh --target "trash bin"
./launch_dino_navdp.sh --pi-ip <PI_IP> --target "door"
```

Change the target at runtime by publishing `std_msgs/String` on
`omnivla/goal_text` — no restart needed.

Only ever run **one** controller against a given peer at a time
(`isaac_gui.py`, `mars_gui.py`, `zenoh_node.py` all publish `cmd_vel` and will
fight each other for it).

### Zenoh transport contract (mostly unchanged from the OmniVLA project this was distilled from)

- **in:** `image_raw` / `rt/image_raw` / `rover_camera` (+ `/compressed`),
  optional `depth_raw` (32FC1 metres or 16UC1 millimetres), `omnivla/goal_text`,
  `rover/rpm` / `rt/rover/rpm` (`std_msgs/Float32MultiArray [left_rpm,
  right_rpm]`, real rover only — logged, not used for control)
- **out:** `cmd_vel` (Twist), `omnivla/explanation`, `omnivla/waypoints` (`nav_msgs/Path`)

All CDR-encoded ROS 2 messages, so any of the three worlds — or a brand new
one — just needs a Zenoh bridge that speaks this contract to plug in.

## Offline tests (no robot needed)

```bash
conda activate internnav
python scripts/test_pipeline_offline.py --target "trash bin"   # full pipeline on a saved frame
python scripts/test_zenoh_loopback.py                          # against a running node
python scripts/diag_goal_conditioning.py                       # goal-conditioning diagnostics
```

## Policy backends

Selected via `PipelineConfig.policy_type`:

1. **`crossmodal` (default)** — official standalone NavDP
   (`navdp-cross-modal.ckpt`, 543 MB fp32, obtained via the InternRobotics
   request form; wrapper: `nav_pipeline/navdp_crossmodal.py`). memory=8,
   predict=24, DDPM-10. **Strong, accurate point-goal conditioning** —
   endpoints track the goal, standard ROS **y-left** convention, informative
   critic. All 1066 tensors load.
2. **`extracted`** — 198 MB NavDP weights pulled out of `InternVLA-N1-w-NavDP`
   (`checkpoints/navdp_extracted.pth`; wrapper: `nav_pipeline/navdp_net.py`).
   Kept as fallback. Quirks: goal convention is y-RIGHT (pipeline flips the
   sign internally), conditioning is weaker, the pixel-goal head is
   effectively untrained, trajectories saturate around ~2.9 m regardless of
   goal distance.

Both run under the same goal-progress + critic **trajectory selection**
layer, and stopping is always enforced externally (goal distance /
bbox-size thresholds) — neither checkpoint learned to stop on its own.

## Key facts

- Latency on an RTX 3090 Ti: DINO ~220 ms + depth ~30 ms + NavDP-CM ~250 ms
  ≈ 2 Hz end-to-end (extracted backend: ~4 Hz). A node heartbeat keeps
  `cmd_vel` flowing at ~7–10 Hz regardless, since the rover firmware zeros
  velocity if it doesn't hear from the node within ~500 ms.
- `navdp-cross-modal/` (an unzipped copy of the `.ckpt` torch archive) is
  redundant with `navdp-cross-modal.ckpt` and can be deleted to save ~540 MB.
- CLIP (`openai/clip-vit-base-patch32`, ~600 MB, downloaded once into
  `HF_HOME` on first run) is loaded with `use_safetensors=True` —
  `transformers` refuses `torch.load`-based checkpoint loading below torch
  2.6, which this env (torch 2.5.1) is under.
