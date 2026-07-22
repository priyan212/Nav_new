# Nav_new — 6WD Rover Navigation: Grounding DINO + NavDP

Self-navigation for the 6-wheeled rover using **Grounding DINO** (open-vocabulary
object detection) to pick the goal and **NavDP** (InternVLA-N1 System-1
trajectory-diffusion policy) to drive toward it while avoiding obstacles.
Total GPU footprint ≈ 1.3 GB (vs ~17 GB for the full InternVLA-N1 dual-system).

## Architecture

```
camera RGB ──► Grounding DINO ──► bbox ──► SAM 2.1 small ──► instance mask
    │                                                          │
    │                      mask centroid + mask-median depth + intrinsics
    ▼                                                          ▼
depth (sensor, or Depth Anything V2 ─────────────► 3D point goal (robot frame)
       metric monocular fallback)                              │
    │                                                          ▼
    ├──► obstacle guard: depth → obstacle points (target mask excluded)
    │       • hard stop + escape turn if forward corridor < 0.45 m
    │       • per-trajectory clearance veto (< 0.32 m discarded)
    │       • linear slow-down below 1.2 m
    │                                                          │
    └────────────► NavDP System-1: 32 diffusion trajectories + critic
                                                               │
              clearance veto → critic top-half gate → goal-progress argmax
                                                               │
                                                               ▼
                       look-ahead waypoint → cmd_vel (v, ω) → Zenoh → rover
```

States: `TRACK` (target detected → drive), `SEARCH` (target lost → rotate to
re-acquire), `AVOID` (obstacle inside hard-stop corridor → stop + escape turn),
`STOP` (goal within `stop_distance` or mask fills `mask_stop_frac` of view).

## Layout

- `nav_pipeline/` — the package
  - `dino_detector.py` — Grounding DINO wrapper (`IDEA-Research/grounding-dino-base`, local HF cache)
  - `sam_segmenter.py` — SAM 2.1 hiera-small box-prompted segmentation (`facebook/sam2.1-hiera-small`)
  - `obstacle_guard.py` — depth→obstacle-points, trajectory clearance veto, forward hard-stop corridor
  - `isaac_gui.py` — Nav_new control panel (camera + mask/bbox overlay, top-down trajectories + obstacles, target entry)
  - `navdp_net.py` — standalone NavDP policy (no Qwen VLM, no `diffusion_policy` dep)
  - `navdp_backbone.py`, `depth_anything/` — vendored from InternNav (imports fixed)
  - `depth_estimator.py` — Depth Anything V2 metric monocular depth (RGB-only mode)
  - `goal_utils.py` — preprocessing + bbox→3D-goal math
  - `pipeline.py` — full perception→policy step with trajectory selection
  - `zenoh_node.py` — transport node (same Zenoh/CDR contract as the OmniVLA node)
- `checkpoints/` — symlinks to weights in `OmniVLA_safe/third_party/InternNav/checkpoints`
  plus `navdp_extracted.pth` (198 MB NavDP weights extracted from InternVLA-N1-w-NavDP)
- `scripts/` — extraction, smoke tests, diagnostics, loopback test
- `third_party/InternNav/` — InternNav code copy (checkpoints symlinked)
- `reference/` — prior OmniVLA/InternVLA nodes kept for reference
- `configs/`, `data/` — configs and test frames/outputs

## Run

```bash
./launch_dino_navdp.sh --target "trash bin"              # auto-discover rover/Isaac
./launch_dino_navdp.sh --pi-ip <PI_IP> --target "door"   # explicit peer
```

Change target at runtime by publishing `std_msgs/String` on `omnivla/goal_text`.

Zenoh topics (CDR, ROS 2 messages — unchanged from the OmniVLA contract):
- in: `image_raw` / `rt/image_raw` / `rover_camera` (+ `/compressed`), optional
  `depth_raw` (32FC1 m or 16UC1 mm), `omnivla/goal_text`
- out: `cmd_vel`, `omnivla/explanation`, `omnivla/waypoints` (nav_msgs/Path)

## Tests (no robot needed)

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
   predict=24, DDPM-10. Verified 2026-07-16: **strong, accurate point-goal
   conditioning** — endpoints track the goal (LEFT `[1,+3]` → `[+1.5,+2.5]`,
   near goals hit within ~0.1 m, trajectories stop at goal distance), standard
   ROS **y-left** convention, informative critic. All 1066 tensors load.
2. **`extracted`** — 198 MB NavDP weights pulled out of `InternVLA-N1-w-NavDP`
   (`checkpoints/navdp_extracted.pth`; wrapper: `nav_pipeline/navdp_net.py`).
   Kept as fallback. Quirks (verified): goal convention y-RIGHT (pipeline flips
   sign), weak conditioning, dead pixel-goal head, trajectories saturate ~2.9 m.

Both run under the goal-progress + critic **trajectory selection** layer, and
stopping is always enforced externally (goal distance / bbox-size thresholds).

## Key facts

- Latency on RTX 3090 Ti: DINO ~220 ms + depth ~30 ms + NavDP-CM ~250 ms
  ≈ 2 Hz (extracted backend: ~4 Hz). Node heartbeat keeps cmd_vel at ~7-10 Hz.
- `navdp-cross-modal/` (unzipped copy of the .ckpt torch archive) is redundant
  with `navdp-cross-modal.ckpt` and can be deleted to save ~540 MB.
