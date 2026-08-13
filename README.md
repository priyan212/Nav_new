```
Hello new user of this repo, I'm Priyan an intern at this lab.
If you need any help contact through GitHub: priyan212
****PLEASE KEEP THE READMES UPDATED****
```

# Nav_new — 6WD Rover Navigation: Grounding DINO + NavDP

Point a 6-wheeled rover at anything you can name — "trash bin", "red chair",
"boulder" — and it drives there on its own, swerving around whatever's in the
way. No LiDAR, a single RGB camera for perception — the only other sensing
is the ESP32's signed wheel-encoder RPM feed plus a BNO055 IMU's fused
heading (`/rover/rpm`), which the GPU side dead-reckons into a logged pose
for diagnostics/safety (see [Odometry logging](#odometry-logging)) and for
the standalone [manual control + Go Home](#manual-control--go-home) GUI, not
for the DINO+NavDP navigation stack itself. The whole stack (detection,
depth, driving policy) runs in **≈1.3 GB of GPU memory**, about 1/13th of the
full InternVLA-N1 dual-system it was distilled from.

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
Hz, +ve = drives forward), plus a BNO055 IMU (I2C, raw register access, no
Adafruit library) fused into the same message as `imu_heading_deg` (absolute
compass heading, NaN if the sensor never acked at boot) and `imu_calib`
(packed `SYS*1000+GYR*100+ACC*10+MAG` calibration digits, 0-3 each).
`nav_pipeline/odometry_logger.py` dead-reckons that into a differential-drive
pose (x, y, θ) and appends a CSV row per sample — x/y always come from
encoder speed, but θ is taken directly from the IMU heading once the
magnetometer digit reaches `imu_min_mag_calib` (default 3), instead of being
integrated from the wheel differential. That switch matters because
skid-steer wheel-diff heading drifts sharply past ~135-165° of rotation
(measured, `odometry_log/odom_accuracy_results.csv`) — wheel slip during
in-place turns is exactly what the encoders can't see, while the BNO055's
onboard sensor fusion has no such accumulating drift once its magnetometer
is calibrated. No GPS/SLAM involved either way, so it still drifts over long
runs without a calibrated IMU, but it's good enough for per-attempt
diagnostics.

**"Calibrated" is not the same as accurate at low thresholds.** The
BNO055's magnetometer sub-score only needs to reach 1 (out of 0-3) to be
trusted by the BNO055's own internal fusion state machine, which is a low
bar — with real magnetic interference nearby (motors, wiring, metal
furniture), the reported heading can still swing 100+ degrees while sitting
completely still at that level (reproduced 2026-08-06: a stationary rover's
heading jumped ~176° at `mag=2`). `imu_min_mag_calib` therefore now
**defaults to 3**, Bosch's own bar for a trustworthy absolute heading,
rather than requiring the flag to be passed manually; if it never reaches 3
in a given room, that's a real environment/mounting issue; the odometry
falls back to wheel-diff dead reckoning (with its own, better-understood
drift) until it does, rather than trusting a low-confidence heading. The
firmware also now persists the calibration offset profile to flash the
first time it reaches full calibration each boot (see the IMU note in
`rover_6wd_complete.ino`'s header, confirmed surviving a real reset
2026-08-06), so the manual wave-around calibration dance is only needed
once per physical environment, not every power-cycle.

- **One file per goal, not per run.** Every time the target text changes
  (GUI Send/preset button, or a new `omnivla/goal_text` message), the logger
  closes the current file and starts a fresh one — `odometry_log/odom_
  <slugified-target>_<timestamp>.csv` — so each file is a self-contained
  record of "how did the rover move while pursuing this specific goal."
  **Pose itself (x, y, θ) is continuous across goals by default** — a new
  goal is just a new CSV file in the same running world frame, it does
  *not* re-zero the origin. This is what makes
  [object-location memory](#object-persistent-targeting-remind) meaningful
  across goals/rooms instead of resetting every time the target changes;
  call `reset_pose()` (or `start_new_goal(..., reset_pose=True)`) for the
  old per-goal-origin behavior back, e.g. an explicit operator "reset map"
  action.
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

## Goal belief (surviving occlusion)

By default (`PipelineConfig.use_belief_goal = True`) the pipeline doesn't
just freeze the goal at its last-seen position while the target is out of
view — `nav_pipeline/goal_belief.py`'s `GoalBelief` propagates it by the
rover's own dead-reckoned ego-motion instead (same idea as
`SubgoalBeliefBank` in `MARS/mars-habitatsim/navdp/navdp/extensions/belief_bank.py`,
which the MARS/EARTH sim GUIs use directly — this is a slimmed-down port for
the single-goal real-rover/Isaac case). A scalar uncertainty (`sigma`) grows
each occluded tick — a flat per-tick term plus a rotation-proportional term,
since turning is where dead reckoning drifts fastest — and once it crosses
`belief_max_sigma` the pipeline gives up on the propagated estimate and
drops to `SEARCH`, overriding the older frame-count `lost_patience` cutoff
while belief is enabled.

The rotation-noise tuning came from real spin-accuracy trials
(`scripts/odom_accuracy_gui.py`, launched via `./LAUNCH/launch_odom_test.sh
[PI_IP]` — a standalone GUI with no DINO/SAM/NavDP/depth models that just
compares dead-reckoned odometry against hand-measured ground truth), logged
to `odometry_log/odom_accuracy_results.csv`: heading error stays within a
few degrees through ~90-135° of rotation, then grows sharply by 165-270°.
See `nav_pipeline/goal_belief.py`'s module docstring and
`belief_eval_20260730/RESULTS.md` for the full comparison against the old
frozen-goal behavior (belief cuts occlusion-while-moving error ~3-11x, at
the cost of depending on odometry quality). `scripts/test_belief_vs_frozen.py`
reruns that comparison (~5 s, CPU only); `scripts/test_belief_integration.py`
exercises the belief wiring inside a real `DinoNavDPPipeline` end to end.

## Object-persistent targeting (REMIND)

`nav_pipeline/remind_gui.py` (launched via `./LAUNCH/launch_rover_remind.sh
[PI_IP]`, which brings up the real rover exactly like `launch_rover.sh`
*and* starts REMIND's own live-tracking server as a background process in
its own conda env — see [REMIND/remind-reid-tracker/README.md](REMIND/remind-reid-tracker/README.md))
replaces DINO's per-frame open-vocabulary detection with REMIND's
persistent per-object identity: every camera frame is sent to that server
(SAM automatic mask generation, class-agnostic — no fixed vocabulary), which
hands back every currently-tracked object as a stable `object_id`, overlaid
on the video feed as `ID <n>`. Targeting is **ID-only** — type or click a
number back (`nav_pipeline/remind_target.py`) — because REMIND's own
BLIP/InternVL caption per object is unstable frame to frame and stays
internal bookkeeping only (`object_map.py`), never shown or matched against
directly. Same control panel otherwise (camera feed, top-down NavDP
trajectory plot, state/velocity readout, manual drive, STOP) as
`isaac_gui.py`, same belief/AVOID/STOP state machine and 1.5 m default
`stop_distance`, and RGB-only Depth Anything V2 **ViT-B** by default (more
accurate than the ViT-S used elsewhere, since depth error feeds directly
into the stop decision).

- **Free-text targeting** ("go to the black chair", "chair near the
  window") layers on top via `nav_pipeline/object_query.py`: a CLIP image
  embedding is cached once per object — the first time it's seen, never
  overwritten — and matched against a CLIP text embedding of the query,
  reusing `relational_target.py`'s "X near Y" / "leftmost X" parsers but
  ranking by remembered *world* position instead of pixel position.
  Resolves to one `object_id` in a background thread, then behaves exactly
  as if that ID had been typed directly — so it stays locked onto the one
  instance the query picked even with several same-class objects in frame.
- **Object-location memory** (`nav_pipeline/object_map.py`, persisted to
  `object_map/object_map.json`): every tick, every REMIND object currently
  in view — not just the driving target — gets its world-frame location
  folded into a running per-ID estimate, using the rover's own continuous
  odometry pose (see [Odometry logging](#odometry-logging)). That's why
  pose no longer resets per goal: an object seen while chasing one target
  needs to still mean something once the target changes. Survives GUI
  restarts within a room/building, but is **not** safe to trust across a
  power cycle or a physical pick-up-and-move of the rover — there's no way
  to detect the odometry origin went stale, so clear it with the GUI's
  **Forget locations** button (or delete the JSON) if that happened.
- **Navigate-back (`GOTO`)**: if the selected object isn't currently
  visible but a remembered world location exists for it, `pipeline.py`
  gains a new `GOTO` state — drives toward that remembered point blind
  (odometry-only waypoint) through the *same* obstacle-guard/NavDP
  trajectory-selection machinery as a live `TRACK` (so depth-based
  collision avoidance stays active the whole leg), but never self-declares
  `STOP` from proximity alone, since dead-reckoning drift over a
  cross-room walk makes trusting that unsafe. The instant REMIND matches
  the object again it drops straight back to normal camera-based
  `TRACK`/`STOP`; if it reaches the remembered spot without reacquiring it
  visually, it falls back to an ordinary `SEARCH` spin there instead of
  declaring arrival. Escalation to `GOTO` only happens once `pipeline.py`'s
  own short-horizon [goal belief](#goal-belief-surviving-occlusion) has
  genuinely given up (`sigma` past `belief_max_sigma`) — a bare drop in
  REMIND's per-tick match (SAM's grid-point automatic masking has no
  cross-frame memory, so a real object can miss a tick or two from pure
  grid-sampling noise) is first coasted through via a
  `--match-grace-period` window, so belief gets first crack at it instead
  of the rover flickering into `GOTO`/`SEARCH` on every brief drop-out.

```bash
./LAUNCH/launch_rover_remind.sh [PI_IP]
```

**VLM-confirmed arrival variant:** `./launch_rover_remind_vlm.sh [PI_IP]`
(repo root, not `LAUNCH/`) is identical bring-up but runs
`nav_pipeline/remind_gui_vlm.py` instead of `remind_gui.py` — the same 1.5 m
depth-based `stop_distance` trigger still zeroes velocity every tick, but it
no longer *declares* arrival by itself: once it fires, the GUI asks REMIND's
already-loaded InternVL model (over the live server's `/confirm_arrival`
endpoint) whether the current camera frame actually shows the target
reached, and only reports **GOAL REACHED** once the VLM agrees
(`VLMArrivalGate`). Falls back to the plain metric-only behavior
automatically if that endpoint is unavailable; `--no-vlm-confirm` forces
that fallback deliberately, for an A/B comparison against
`launch_rover_remind.sh`.

## Manual control + Go Home

`nav_pipeline/home_gui.py` (launched via `./LAUNCH/launch_rover_home.sh
[PI_IP]`, or `./LAUNCH/launch_bot.sh [PI_IP]` directly — `launch_rover_home.sh`
is a thin backward-compatible wrapper around it) is
a separate, lightweight control panel for the real rover: arrow-key/hold-button
manual driving, plus a **GO HOME** button that drives back to wherever the
rover was when the GUI launched (or wherever **Set Home Here** was last
pressed). By default it's still deliberately independent of `pipeline.py` /
`isaac_gui.py` / `zenoh_node.py` — none of DINO/NavDP/SAM/CLIP/depth are
imported, so it starts in under a second with no GPU, using only the fused
encoder+IMU pose described above (`OdometryLogger`, small CDR bits
duplicated from `zenoh_node.py` to avoid the heavy import chain).

Go Home is closed-loop, not an open-loop "turn 180°, drive N meters" — it
recomputes bearing/distance from the live fused pose every tick, so drift or
a bump mid-return gets corrected on the fly:

- **ROTATE** (heading error > ~20°) → turn in place only.
- **DRIVE** (heading error < ~8°) → drive forward with steering correction
  folded in (hysteresis between the two thresholds stops phase-chattering).
- **FACE** (within `--home-dist-tol`, default 10 cm) → stop translating and
  rotate in place to match the heading recorded at "Set Home Here" (or
  θ=0, the launch heading, if home was never re-set). Only then does it
  report **ARRIVED** — the rover ends up facing the same way it started, not
  just standing in the same spot.

**Optional obstacle avoidance while homing** (`--enable-obstacle-avoidance`,
off by default): starts the Pi camera and loads the full
`DinoNavDPPipeline` (DINO + NavDP + Depth Anything V2 — a real model-load
pause comparable to `launch_rover.sh`'s, plus continuous camera Wi-Fi/CPU
traffic the plain camera-free Go Home never had), and drives the long
cross-room leg through `pipeline.py`'s `GOTO` state (the same mechanism
[REMIND navigate-back](#object-persistent-targeting-remind) uses, just
pointed at the fixed home point via `object_map.world_to_local` instead of
a remembered object) instead of the plain bearing-servo above. NavDP scores
candidate trajectories by both goal progress and a footprint-aware
obstacle-clearance veto, so it naturally keeps making progress toward home
after steering around something rather than resuming a fixed heading and
re-hitting whatever it just avoided. `GOTO` never self-declares arrival (see
above), so `navdp_home_loop` owns that check itself: once within
`--home-arrival-radius` (default 1 m) of home, it hands off to the same
vision-free ROTATE/DRIVE/FACE/ARRIVED servo above for the precise final
approach and heading match. Manual drive is never gated by any of this,
flag on or off.

## The five bodies it runs on

The same `nav_pipeline` code drives all five — only the transport peer and
the speed caps change:

| Target | Launcher | Camera/depth source | Notes |
|---|---|---|---|
| **Real rover** (6WD, ESP32 + micro-ROS) | `./LAUNCH/launch_rover.sh` | Pi camera (compressed JPEG only — raw RGB saturates the rover's Wi-Fi) | Brings the Pi's systemd services up, waits for camera + ESP32 heartbeat, then starts the GUI at real-world speed caps |
| **Hiwonder LanderPi** (Mecanum chassis) | `./LAUNCH/launch_rover.sh --hiwonder` | LanderPi's `usb_cam` (compressed JPEG) | Same GUI/pipeline, different Pi bring-up (`landerpi/deploy_bridge.sh` instead of systemd services) and lower speed caps — see [The `--rover` / `--hiwonder` backend flag](#the---rover----hiwonder-backend-flag) and [landerpi/README.md](landerpi/README.md) |
| **Isaac Sim** digital twin | `./LAUNCH/launch_gui.sh` | Isaac's simulated camera + depth over Zenoh | Needs the Isaac scene playing and its ROS 2 bridge scripts running first (see script header) |
| **Mars habitat sim** (Habitat-Sim, ERC Marsyard terrain) | `./MARS/launch_mars.sh --rocks` | Habitat's simulated camera + perfect depth | Separate `mars_habitat` conda env for the sim node; see [MARS/README.md](MARS/README.md) |
| **Earth habitat sim** (Habitat-Sim, real-world photogrammetry scan) | `./EARTH/launch_earth.sh` | Habitat's simulated camera + perfect depth | Same `mars_habitat` conda env, a Sketchfab scan instead of generated terrain; see [EARTH/README.md](EARTH/README.md) |

All five publish/subscribe the identical Zenoh topics, so a policy change in
`nav_pipeline/` is tested once offline, once in Isaac/Mars/Earth, then run on
the real rover (or the LanderPi) with nothing else touched.

## The `--rover` / `--hiwonder` backend flag

Every script in `LAUNCH/` (except `launch_dino_navdp.sh`, `launch_gui.sh`)
sources `LAUNCH/_backend.sh`, which accepts `--rover` (default) or
`--hiwonder` as their first flag and fills in the right Pi-bringup method,
default Pi IP/SSH password, camera FOV, footprint, and steering-shape
constants (`max-angular`, `search-angular`, `angular-slew-max`) for whichever
body you're pointing at — `pipeline.py` / `obstacle_guard.py` /
`odometry_logger.py` are completely unchanged either way, only what runs on
the Pi differs:

- **`--rover`** (default) — the old 6WD rover: ESP32 micro-ROS +
  `zenoh-bridge-ros2dds`, brought up via systemd services
  (`rover-camera`/`rover-agent`/`rover-zenoh`). Every steering-shape constant
  here is real, live-measured tuning for this exact chassis/firmware (see
  `LAUNCH/_backend.sh`'s comments for the derivations).
- **`--hiwonder`** — the newer Hiwonder LanderPi (Mecanum): stock ROS1
  Noetic stack in Hiwonder's own `armpi_pro` Docker container (untouched),
  brought up via `landerpi/deploy_bridge.sh` running `landerpi/bridge.py`
  inside that container. Its own real encoder+IMU odometry, but its
  steering-shape constants are carried over from the rover **unvalidated**
  for this chassis (`backend_bringup` prints an explicit warning) — see
  [landerpi/README.md](landerpi/README.md) for the full integration story
  and known caveats.

```bash
./LAUNCH/launch_rover.sh --hiwonder --target "trash bin"
./LAUNCH/launch_rover.sh --hiwonder 10.47.234.228 --target "trash bin"
```

## Repo layout

```
Nav_new/
├── nav_pipeline/         the package: perception + policy + transport
├── LAUNCH/               one-command entry points for the real rover / LanderPi (own section below)
├── checkpoints/          model weights (see below)
├── scripts/              offline tests, diagnostics, Pi provisioning
├── configs/               yaml configs (isaac / real rover)
├── data/                 saved test frames + pipeline output snapshots
├── odometry_log/         dead-reckoned pose CSVs, one per goal (own README)
├── scene_log/            passive open-vocabulary object-inventory JSONL, one file per process run (own README)
├── object_map/           persistent id -> world-location JSON (object_map.py, see REMIND targeting)
├── belief_eval_*/        dated write-ups of the belief-vs-frozen occlusion eval (see Goal belief)
├── reference/            prior OmniVLA/InternVLA-N1 nodes, plus InternVLA-N1's native DualVLN
│                        dual-system node (internvla_dualvln_zenoh_node.py, see LAUNCH/launch_hiwonder_dualvln.sh)
├── third_party/InternNav/  vendored InternNav source (checkpoints symlinked in)
├── esp32/                rover firmware + Pi-side serial/handshake helpers (see esp32/FLASHING.md
│                        for how to compile + flash it — the ESP32 is wired to the Pi, not the
│                        GPU machine, so this is a compile-locally/scp/flash-via-SSH workflow)
├── landerpi/             Hiwonder LanderPi bridge (bridge.py + deploy_bridge.sh) -- the second
│                        robot backend selected via --hiwonder, see the backend-flag section above (own README)
├── MARS/                 Habitat-Sim Mars sub-project (own README)
├── EARTH/                Habitat-Sim real-world photogrammetry sub-project (own README)
├── REMIND/remind-reid-tracker/  persistent per-object re-ID tracker, live-served for
│                        object-persistent targeting (own README, separate conda env)
├── tryout/               S2Diff in-loop obstacle guidance for NavDP's DDPM sampler -- experimental,
│                        opt-in alternative trajectory sampler (see tryout/S2DIFF_GUIDANCE.md and
│                        LAUNCH/launch_rover_s2diff*.sh below)
└── launch_rover_remind_vlm.sh  REMIND targeting + VLM-confirmed arrival (repo root, not LAUNCH/ --
                         see Object-persistent targeting below)
```

### `LAUNCH/` — real-rover / LanderPi one-command entry points

Every script here (except `launch_dino_navdp.sh`) sources `LAUNCH/_backend.sh`
for the shared `--rover`/`--hiwonder` bring-up described above. Full flag
combinations for every one of these live in [priyan.md](priyan.md).

| Script | What it starts |
|---|---|
| `launch_rover.sh` | Pi bring-up + `isaac_gui.py` (DINO+SAM+NavDP), real-world speed caps — the default real-body launcher |
| `launch_rover_vitb.sh` | `launch_rover.sh` + `--depth-encoder vitb` (more accurate monocular depth, ~2x slower) |
| `launch_rover_s2diff.sh` | `launch_rover.sh`'s same GUI/bring-up, but `nav_pipeline.s2diff_runner` — NavDP sampling routed through in-process S2Diff obstacle guidance instead of `isaac_gui.py`'s plain DDPM sampler |
| `launch_rover_s2diff_http.sh` | Same GUI again, but NavDP sampling routed over HTTP to a separately-started `tryout/navdp_s2diff_server.py` instead of running in-process — needs that server already running |
| `launch_rover_vitb_s2diff.sh` | `launch_rover_s2diff.sh` + `--depth-encoder vitb` (both experimental changes at once) |
| `launch_rover_remind.sh` | Pi bring-up + REMIND live server (own conda env) + `remind_gui.py` — persistent per-object ID targeting instead of a bare DINO phrase |
| `launch_rover_home.sh` | Thin wrapper around `launch_bot.sh` — manual control + Go Home, kept for backward-compatible muscle memory |
| `launch_bot.sh` | Manual control + Go Home (`home_gui.py`) on either backend; `--enable-obstacle-avoidance` opts into full NavDP-guided homing |
| `launch_odom_test.sh` | Pi bring-up (no camera) + `scripts/odom_accuracy_gui.py` — dead-reckoned odometry vs. hand-measured ground truth, no perception/policy models loaded |
| `launch_dino_navdp.sh` | Headless `zenoh_node.py` only — does **not** bring the Pi up itself, use this when the Pi/Isaac/LanderPi side is already running separately |
| `launch_gui.sh` | Isaac Sim GUI (`isaac_gui.py`, sim speed caps) — no Pi bring-up, needs the Isaac scene + its own ROS 2 bridge already running |
| `launch_hiwonder_dualvln.sh` | LanderPi bring-up + InternVLA-N1's native DualVLN dual-system (`reference/internvla_dualvln_zenoh_node.py`) instead of the DINO+NavDP stack — suited to long, compound, multi-landmark instructions; see its own header comment |
| `_backend.sh` | Not a launcher — sourced by all of the above for `--rover`/`--hiwonder` parsing and Pi bring-up |

### `nav_pipeline/` — the package

- `dino_detector.py` — Grounding DINO wrapper (`IDEA-Research/grounding-dino-base`, local HF cache)
- `sam_segmenter.py` — SAM 2.1 hiera-small, box-prompted → instance mask, throttled ~1 Hz (`PipelineConfig.sam_period_s`)
- `clip_verifier.py` — CLIP (`openai/clip-vit-base-patch32`) scores the SAM crop against the target phrase; below-threshold crops are rejected as false positives (paired 1:1 with each SAM refresh)
- `obstacle_guard.py` — depth → obstacle points, footprint-swept trajectory clearance veto, forward hard-stop corridor
- `navdp_crossmodal.py` — wrapper for the official standalone NavDP checkpoint (**default policy**)
- `navdp_net.py`, `navdp_backbone.py`, `depth_anything/` — standalone/extracted NavDP + Depth Anything V2, vendored from InternNav with imports fixed (InternNav's own package breaks under this repo's transformers version)
- `depth_estimator.py` — Depth Anything V2 metric monocular depth for the RGB-only real rover
- `goal_utils.py` — preprocessing + bbox→3D-goal math
- `goal_belief.py` — ego-motion propagation of the tracked goal point while it's out of view, instead of freezing it (see [Goal belief](#goal-belief-surviving-occlusion))
- `odometry_logger.py` — dead-reckons `/rover/rpm` (real wheel-encoder RPM + BNO055 IMU heading) into a pose, continuous across goals (one CSV per goal under `odometry_log/`, but the pose itself doesn't re-zero); also backs the GUI's spin-stall watchdog (see [Odometry logging](#odometry-logging))
- `object_map.py` — persistent per-object-ID world-location memory (`local_to_world`/`world_to_local` against the continuous odometry pose), backing REMIND [navigate-back](#object-persistent-targeting-remind); persisted to `object_map/object_map.json`
- `object_query.py` — resolves free-text ("go to the chair near the window") to a specific `object_id` via cached CLIP image embeddings + `relational_target.py`'s parsers, ranked by remembered world position (see [Object-persistent targeting](#object-persistent-targeting-remind))
- `pipeline.py` — the full perception→policy step, state machine, trajectory selection (the diagram above, as code); also owns the `GOTO` blind-navigate-back state (`external_goal` argument to `step()`)
- `isaac_gui.py` — the control panel: camera + mask/bbox overlay, top-down trajectories + obstacles, live target entry
- `remind_gui.py` — REMIND-backed variant of `isaac_gui.py`: ID-only persistent-object targeting instead of a bare DINO phrase, launched via `./LAUNCH/launch_rover_remind.sh` (see [Object-persistent targeting](#object-persistent-targeting-remind))
- `remind_gui_vlm.py` — `remind_gui.py` plus a VLM (InternVL) arrival confirmation gate on top of the metric `stop_distance` trigger (`VLMArrivalGate`), launched via `./launch_rover_remind_vlm.sh` (repo root)
- `remind_client.py` / `remind_target.py` — HTTP client for the REMIND live server (`REMIND/remind-reid-tracker/scripts/live_server.py`) and its "ID <n>" target-text parser
- `home_gui.py` — manual control + Go Home panel; camera-free/no-GPU by default (fused encoder+IMU pose only), or opt into NavDP obstacle avoidance for the homing leg via `--enable-obstacle-avoidance`, launched via `./LAUNCH/launch_bot.sh` / `./LAUNCH/launch_rover_home.sh` (see [Manual control + Go Home](#manual-control--go-home))
- `s2diff_navdp.py`, `s2diff_runner.py` — in-process S2Diff obstacle-guided NavDP sampling and its `isaac_gui.py`-equivalent runner, launched via `LAUNCH/launch_rover_s2diff.sh` (see `tryout/S2DIFF_GUIDANCE.md`)
- `s2diff_http_client.py`, `s2diff_http_runner.py` — same S2Diff guidance, but sampling delegated over HTTP to `tryout/navdp_s2diff_server.py` instead of running in-process, launched via `LAUNCH/launch_rover_s2diff_http.sh`
- `scene_tagger.py` — passive open-vocabulary object-inventory tagging (broad Grounding DINO vocabulary, ~1 Hz, independent of the current goal), called from `pipeline.py`, logged to `scene_log/` (own README)
- `dinov2_embedder.py` — DINOv2 appearance descriptors used by REMIND-adjacent re-identification/appearance-similarity checks
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
- `test_belief_vs_frozen.py` — synthetic comparison of belief vs. frozen-goal occlusion handling (see [Goal belief](#goal-belief-surviving-occlusion))
- `test_belief_integration.py` — exercises the belief wiring inside a real `DinoNavDPPipeline`
- `odom_accuracy_gui.py` — standalone GUI (no perception/policy models) for measuring real-rover dead-reckoned odometry drift against hand-measured ground truth; launched via `./LAUNCH/launch_odom_test.sh`
- `demo_odom_accuracy_gui.py` — offline/no-hardware demo of the odometry-accuracy GUI's plotting
- `extract_navdp_weights.py` — pulls the 198 MB NavDP head out of the 16 GB InternVLA-N1 checkpoint
- `pi_install_services.sh` / `pi_auto_handshake.sh` — run **on the Pi**: install the camera/agent/zenoh systemd services and recover a wedged ESP32 micro-ROS session without a physical reset button

### `esp32/`

Micro-ROS firmware for the 6-wheel differential drive base
(`rover_6wd_complete.ino`, subscribes `/cmd_vel`, publishes `/rover/rpm` —
signed L/R wheel RPM from the real quadrature encoders on each side's mid
wheel, plus BNO055 IMU fused heading (`imu_heading_deg`) and calibration
status (`imu_calib`), 10 Hz, consumed by `nav_pipeline/odometry_logger.py`;
`WHEEL_RADIUS_M` / `TRACK_WIDTH_M` must stay in sync between the two), plus
Pi-side helpers (`rover_handshake_manager.py`, `serial_bridge.py`) for
talking to the ESP32 over `/dev/ttyUSB0`. The BNO055 talks raw I2C register
access (GPIO21 SDA / GPIO22 SCL, no Adafruit library dependency) — `NaN` in
`imu_heading_deg` means the sensor never acked its chip ID at boot (check
wiring), not a runtime fault.

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

### `REMIND/`

A self-contained multi-object re-identification tracker (own conda env —
its torch/transformers pins are incompatible with this project's `internnav`
env) that gives every detected object a persistent identity across a live
camera stream, instead of DINO's stateless per-frame open-vocab detection.
`remind-reid-tracker/scripts/live_server.py` serves it over HTTP for
`nav_pipeline/remind_client.py` to poll. Two READMEs cover it:
[REMIND/README.md](REMIND/README.md) for how it plugs into Nav_new's
navigation (object-location memory, navigate-back, free-text targeting —
also summarized in [Object-persistent targeting](#object-persistent-targeting-remind)
above), and [REMIND/remind-reid-tracker/README.md](REMIND/remind-reid-tracker/README.md)
for the tracker itself (SAM detection backend, BLIP/InternVL captioning,
the live server's API, and the `/confirm_arrival` VLM endpoint used by
`remind_gui_vlm.py`).

### `landerpi/`

The Hiwonder LanderPi integration: `bridge.py` (the only file added to the
robot, running inside Hiwonder's own untouched `armpi_pro` Docker container)
and `deploy_bridge.sh` (copies it over and starts it). Selected via
`--hiwonder` on any `LAUNCH/*.sh` script (see
[The `--rover`/`--hiwonder` backend flag](#the---rover----hiwonder-backend-flag)
above). Full detail — including the real per-wheel-encoder + BNO055 odometry
story and known caveats around the carried-over steering constants and the
spec-sheet (not tape-measured) footprint — in
[landerpi/README.md](landerpi/README.md).

### `tryout/`

Experimental, opt-in S2Diff in-loop obstacle guidance for NavDP's diffusion
sampler: instead of sampling 32 unguided DDPM trajectories and picking one
via the clearance-veto + critic gate (the default path, see the top diagram),
the caller supplies obstacle pixels and every one of NavDP's 10 DDPM
denoising steps is nudged by a safety/stability/cost energy computed from
them. `pipeline.py`'s own obstacle veto/hard-stop/anti-oscillation logic is
unchanged and still has final say either way. Two ways to run it —
in-process (`nav_pipeline/s2diff_navdp.py`, via `LAUNCH/launch_rover_s2diff.sh`)
or over HTTP against a standalone `tryout/navdp_s2diff_server.py`
(via `LAUNCH/launch_rover_s2diff_http.sh`) — neither touches the plain
`launch_rover.sh` path. Full mechanics in `tryout/S2DIFF_GUIDANCE.md`.

## Running it

```bash
# Real rover (Pi bringup + GUI, real-world speed caps)
./LAUNCH/launch_rover.sh

# Hiwonder LanderPi -- same GUI/pipeline, different Pi bring-up + speed caps
./LAUNCH/launch_rover.sh --hiwonder

# Isaac Sim digital twin (needs the Isaac scene + ROS 2 bridge already running)
./LAUNCH/launch_gui.sh --target "cardboard box"

# Mars habitat sim (spins up both the sim node and the GUI for you)
./MARS/launch_mars.sh --rocks

# Earth habitat sim (real-world photogrammetry scan, same conda env as Mars)
./EARTH/launch_earth.sh --target "target sign"

# Headless node only (no GUI), auto-discovers the rover/Isaac peer
./LAUNCH/launch_dino_navdp.sh --target "trash bin"
./LAUNCH/launch_dino_navdp.sh --pi-ip <PI_IP> --target "door"

# Real-rover odometry accuracy check (no camera/DINO/SAM/NavDP -- see
# "Goal belief" below for why this matters)
./LAUNCH/launch_odom_test.sh <PI_IP>

# Real-rover manual control + Go Home (no camera/DINO/SAM/NavDP by default -- see
# "Manual control + Go Home" below; add --enable-obstacle-avoidance to opt in)
./LAUNCH/launch_rover_home.sh <PI_IP>

# Real-rover object-persistent targeting via REMIND (brings up its own live
# server too -- see "Object-persistent targeting (REMIND)" below)
./LAUNCH/launch_rover_remind.sh <PI_IP>

# Same, but with a VLM (InternVL) confirming arrival on top of the metric trigger
./launch_rover_remind_vlm.sh <PI_IP>

# S2Diff obstacle-guided NavDP sampling (experimental, see tryout/ above)
./LAUNCH/launch_rover_s2diff.sh <PI_IP>

# LanderPi + InternVLA-N1's native DualVLN dual-system, long compound instructions
./LAUNCH/launch_hiwonder_dualvln.sh --instruction "go through the doorway and stop by the closet"
```

Full flag-by-flag combinations for every `LAUNCH/*.sh` script are in
[priyan.md](priyan.md).

Change the target at runtime by publishing `std_msgs/String` on
`omnivla/goal_text` — no restart needed.

Only ever run **one** controller against a given peer at a time
(`isaac_gui.py`, `mars_gui.py`, `earth_gui.py`, `zenoh_node.py`, `home_gui.py`,
`remind_gui.py`, `remind_gui_vlm.py`, `s2diff_runner.py`, `s2diff_http_runner.py`
all publish `cmd_vel` and will fight each other for it).

### Zenoh transport contract (mostly unchanged from the OmniVLA project this was distilled from)

- **in:** `image_raw` / `rt/image_raw` / `rover_camera` (+ `/compressed`),
  optional `depth_raw` (32FC1 metres or 16UC1 millimetres), `omnivla/goal_text`,
  `rover/rpm` / `rt/rover/rpm` (`std_msgs/Float32MultiArray [left_rpm,
  right_rpm, imu_heading_deg, imu_calib]`, real rover only — logged and
  fused into heading, not otherwise used for control; the last two fields
  are additive, so any consumer still checking `len(data) >= 2` is
  unaffected)
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
