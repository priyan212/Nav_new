```
Hello new user of this repo, I'm Priyan an intern at this lab.
If you need any help contact through GitHub: priyan212
```

# Nav_new — 6WD Rover Navigation: Grounding DINO + NavDP

Point a 6-wheeled rover at anything you can name — "trash bin", "red chair",
"boulder" — and it drives there on its own, swerving around whatever's in the
way. No LiDAR, no wheel odometry, a single RGB camera. The whole stack
(detection, depth, driving policy) runs in **≈1.3 GB of GPU memory**, about
1/13th of the full InternVLA-N1 dual-system it was distilled from.

It runs unmodified against three different bodies — the real rover, an Isaac
Sim digital twin, and a Habitat-Sim Mars yard — because every one of them
speaks the same Zenoh/ROS 2 wire contract described below.

## How a frame becomes a motor command

```
camera RGB ──► Grounding DINO ──► bbox ──► SAM 2.1 small ──► instance mask
    │                                                          │
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

**State machine:** `TRACK` (target detected → drive) → `SEARCH` (target lost
→ rotate to re-acquire) → `AVOID` (obstacle inside the hard-stop corridor →
stop + escape turn) → `STOP` (goal close enough, or its mask fills the view).

## The three worlds it runs in

The same `nav_pipeline` code drives all three — only the transport peer and
the speed caps change:

| Target | Launcher | Camera/depth source | Notes |
|---|---|---|---|
| **Real rover** (6WD, ESP32 + micro-ROS) | `./launch_rover.sh` | Pi camera (compressed JPEG only — raw RGB saturates the rover's Wi-Fi) | Brings the Pi's systemd services up, waits for camera + ESP32 heartbeat, then starts the GUI at real-world speed caps |
| **Isaac Sim** digital twin | `./launch_gui.sh` | Isaac's simulated camera + depth over Zenoh | Needs the Isaac scene playing and its ROS 2 bridge scripts running first (see script header) |
| **Mars habitat sim** (Habitat-Sim, ERC Marsyard terrain) | `./MARS/launch_mars.sh --rocks` | Habitat's simulated camera + perfect depth | Separate `mars_habitat` conda env for the sim node; see [MARS/README.md](MARS/README.md) |

All three publish/subscribe the identical Zenoh topics, so a policy change in
`nav_pipeline/` is tested once offline, once in Isaac or Mars, then run on
the real rover with nothing else touched.

## Repo layout

```
Nav_new/
├── nav_pipeline/         the package: perception + policy + transport
├── checkpoints/          model weights (see below)
├── scripts/              offline tests, diagnostics, Pi provisioning
├── configs/               yaml configs (isaac / real rover)
├── data/                 saved test frames + pipeline output snapshots
├── reference/            prior OmniVLA/InternVLA-N1 nodes, kept for reference
├── third_party/InternNav/  vendored InternNav source (checkpoints symlinked in)
├── esp32/                rover firmware + Pi-side serial/handshake helpers
├── MARS/                 Habitat-Sim Mars sub-project (own README)
└── launch_*.sh           one-command entry points for each of the 3 worlds
```

### `nav_pipeline/` — the package

- `dino_detector.py` — Grounding DINO wrapper (`IDEA-Research/grounding-dino-base`, local HF cache)
- `sam_segmenter.py` — SAM 2.1 hiera-small, box-prompted → instance mask
- `obstacle_guard.py` — depth → obstacle points, footprint-swept trajectory clearance veto, forward hard-stop corridor
- `navdp_crossmodal.py` — wrapper for the official standalone NavDP checkpoint (**default policy**)
- `navdp_net.py`, `navdp_backbone.py`, `depth_anything/` — standalone/extracted NavDP + Depth Anything V2, vendored from InternNav with imports fixed (InternNav's own package breaks under this repo's transformers version)
- `depth_estimator.py` — Depth Anything V2 metric monocular depth for the RGB-only real rover
- `goal_utils.py` — preprocessing + bbox→3D-goal math
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
(`rover_6wd_complete.ino`, subscribes `/cmd_vel`, publishes `/rover/rpm`),
plus Pi-side helpers (`rover_handshake_manager.py`, `serial_bridge.py`) for
talking to the ESP32 over `/dev/ttyUSB0`.

### `MARS/`

A self-contained sub-project: a Habitat-Sim recreation of the ERC Marsyard
2022 terrain (heightmap → generated mesh, since the source repo ships no
mesh) used to stress-test the same nav stack with GPS-quality ground truth
and a rock obstacle field. Full detail in [MARS/README.md](MARS/README.md).

## Running it

```bash
# Real rover (Pi bringup + GUI, real-world speed caps)
./launch_rover.sh

# Isaac Sim digital twin (needs the Isaac scene + ROS 2 bridge already running)
./launch_gui.sh --target "cardboard box"

# Mars habitat sim (spins up both the sim node and the GUI for you)
./MARS/launch_mars.sh --rocks

# Headless node only (no GUI), auto-discovers the rover/Isaac peer
./launch_dino_navdp.sh --target "trash bin"
./launch_dino_navdp.sh --pi-ip <PI_IP> --target "door"
```

Change the target at runtime by publishing `std_msgs/String` on
`omnivla/goal_text` — no restart needed.

Only ever run **one** controller against a given peer at a time
(`isaac_gui.py`, `mars_gui.py`, `zenoh_node.py` all publish `cmd_vel` and will
fight each other for it).

### Zenoh transport contract (unchanged from the OmniVLA project this was distilled from)

- **in:** `image_raw` / `rt/image_raw` / `rover_camera` (+ `/compressed`),
  optional `depth_raw` (32FC1 metres or 16UC1 millimetres), `omnivla/goal_text`
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
