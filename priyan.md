```
Hello new user of this repo, I'm Priyan an intern at this lab.
If you need any help contact through GitHub: priyan212
****PLEASE KEEP THE READMES UPDATED****
```

# Priyan's Notes — Launch File Cheatsheet

> Personal scratch notes — see [README.md](README.md) for the full narrative
> docs (pipeline diagram, state machine, goal belief, REMIND, etc). This file
> is just "what do I type to make the rover move" for every launcher in the
> repo, plus every flag each one accepts.

Almost all real-body launchers live in [LAUNCH/](LAUNCH/) now (they used to
sit at repo root — if a command in an old note/screenshot doesn't have
`LAUNCH/` in front of it, add it). One launcher is still at repo root on
purpose: `launch_rover_remind_vlm.sh` (#7 below) — new enough that it was
never at repo root in the first place, so there's no old muscle memory to
preserve either way.

```bash
cd /mnt/bigdisk/Priyan/Nav_new
```
(run everything below from the repo root unless noted)

## The `--rover` / `--hiwonder` backend flag (applies to almost every LAUNCH/ script)

Every script in `LAUNCH/` except `launch_dino_navdp.sh`/`launch_gui.sh`
sources `LAUNCH/_backend.sh`. Pass `--rover` or `--hiwonder` as any argument
(order doesn't matter, last one wins) to pick the body; omit it and you get
`--rover`. This one flag silently changes a pile of other defaults:

| | `--rover` (default) | `--hiwonder` |
|---|---|---|
| Body | old 6WD ESP32 rover | Hiwonder LanderPi (Mecanum) |
| Pi bring-up | systemd services (`rover-camera`/`rover-agent`/`rover-zenoh`) | `landerpi/deploy_bridge.sh` (docker exec into `armpi_pro`) |
| Default Pi IP | `192.168.21.125` (churns — verify if unreachable) | `10.47.234.228` |
| Default SSH password | `hri` | `raspberrypi` |
| Camera FOV | `60°` (Logitech webcam) | `64.6°` (usb_cam intrinsics) |
| `--max-angular` | `1.2` rad/s (matches ESP32 firmware normalization) | `0.5` rad/s (conservative, carried over) |
| `--search-angular` | `0.18` rad/s (tuned 2026-08-12, ESP32 `VEL_DEADBAND_MS` floor) | `0.13` rad/s (old rover's carried-over value, unvalidated on this chassis) |
| `--angular-slew-max` | `0.10` rad/s/tick (pipeline.py default) | `0.05` rad/s/tick (halved 2026-08-10, re-ID continuity fix) |
| `--footprint-length` / `--footprint-width` | not passed (uses `pipeline.py` default `0.482 x 0.380` m) | `0.298 x 0.256` m (ArmPi Pro spec sheet, **not tape-measured**) |

`--hiwonder` prints an explicit warning that the steering-shape constants
(search-angular/servo-ramp-deg/max-angular normalization) are carried over
from the rover unvalidated for this chassis — see
[landerpi/README.md](landerpi/README.md)'s "Known caveats".

`PI_IP` can also be passed positionally as the first non-flag argument to
any of these scripts, overriding the table above:

```bash
./LAUNCH/launch_rover.sh 192.168.21.99 --target "trash bin"
./LAUNCH/launch_rover.sh --hiwonder 10.47.234.228 --target "trash bin"
```

---

## 1. `LAUNCH/launch_rover.sh` — the default real-body launcher

Brings up the Pi (camera + backend), then starts `nav_pipeline.isaac_gui`
(DINO + SAM + NavDP) at real-world speed caps.

```
./LAUNCH/launch_rover.sh [--rover|--hiwonder] [PI_IP] [isaac_gui.py flags...]
```

Hardcoded by the script itself (not overridable via the flags below, they're
baked into the `exec python ...` call): `--max-linear 0.15`,
`--max-angular $BACKEND_MAX_ANGULAR`, `--fov $BACKEND_FOV`,
`--search-angular $BACKEND_SEARCH_ANGULAR`, `--servo-ramp-deg 70`,
`--angular-slew-max $BACKEND_ANGULAR_SLEW_MAX`, `--compressed-only`, plus
`--footprint-length/--footprint-width` when `--hiwonder`.

Everything else forwards straight to `nav_pipeline/isaac_gui.py`'s own
argparse (full flag list — these also apply to `launch_rover_vitb.sh`,
`launch_rover_s2diff.sh`, `launch_rover_s2diff_http.sh`,
`launch_rover_vitb_s2diff.sh`, and `launch_rover_remind.sh`/
`launch_rover_remind_vlm.sh` where noted):

| flag | default | meaning |
|---|---|---|
| `--target TEXT` | `""` | open-vocabulary text goal (DINO) |
| `--avoid TEXT` | `""` | text to actively steer away from |
| `--policy-type {crossmodal,extracted}` | `crossmodal` | NavDP checkpoint — see README's "Policy backends" |
| `--pi-ip IP` | `None` | overridden by the script |
| `--predict-hz N` | `2.5` | pipeline tick rate |
| `--fov DEG` | `90.0` | overridden by the script (`$BACKEND_FOV`) |
| `--device STR` | `cuda:0` | torch device |
| `--max-linear M/S` | `0.5` | overridden by the script (`0.15`) |
| `--max-angular RAD/S` | `0.4` | overridden by the script (`$BACKEND_MAX_ANGULAR`) |
| `--search-angular RAD/S` | `0.15` | overridden by the script (`$BACKEND_SEARCH_ANGULAR`) |
| `--servo-ramp-deg DEG` | `35.0` | overridden by the script (`70`) |
| `--angular-slew-max RAD/S` | `0.10` | overridden by the script (`$BACKEND_ANGULAR_SLEW_MAX`) |
| `--invert-angular` | off | flip turn direction (real-rover wiring escape hatch) |
| `--no-belief-goal` | off | disable [goal belief](README.md#goal-belief-surviving-occlusion), old frozen-goal behavior |
| `--depth-encoder {vits,vitb}` | `vits` | monocular Depth-Anything-V2 size; `vitb` = more accurate, ~2x slower |
| `--compressed-only` | off | forced on by the script (real Pi Wi-Fi can't take raw RGB) |
| `--odometry-log-dir DIR` | `odometry_log` | where per-goal CSVs land |
| `--imu-min-mag-calib INT` | `3` | IMU calibration digit (0-3) gating theta onto the IMU heading vs. wheel-diff dead reckoning (added 2026-08-14 — previously only `launch_bot.sh`/`home_gui.py` exposed this; every `OdometryLogger` caller does now) |
| `--footprint-length M` | `0.482` | obstacle-guard footprint length |
| `--footprint-width M` | `0.380` | obstacle-guard footprint width |

Examples:

```bash
./LAUNCH/launch_rover.sh                                    # rover, default IP, no target
./LAUNCH/launch_rover.sh --target "trash bin"
./LAUNCH/launch_rover.sh 192.168.21.99 --target "red chair"
./LAUNCH/launch_rover.sh --hiwonder --target "trash bin"
./LAUNCH/launch_rover.sh --hiwonder 10.47.234.228 --target "door"
./LAUNCH/launch_rover.sh --target "chair" --no-belief-goal   # A/B: old frozen-goal behavior
./LAUNCH/launch_rover.sh --target "person" --avoid "cat"
./LAUNCH/launch_rover.sh --target "trash bin" --policy-type extracted
./LAUNCH/launch_rover.sh --target "box" --invert-angular     # if the rover turns backwards
./LAUNCH/launch_rover.sh --hiwonder --target "box" --footprint-length 0.30 --footprint-width 0.27
```

Supervised A/B belief test (both variants at same tuning otherwise):

```bash
./LAUNCH/launch_rover.sh                        # belief ON (default)
./LAUNCH/launch_rover.sh --no-belief-goal        # old frozen-goal behavior
```

---

## 2. `LAUNCH/launch_rover_vitb.sh` — `launch_rover.sh` + accurate depth

Identical wrapper — `exec ./launch_rover.sh "$@" --depth-encoder vitb`. Same
flags as above apply (all forwarded), `--depth-encoder` is just pre-appended.

```bash
./LAUNCH/launch_rover_vitb.sh                                  # default Pi IP
./LAUNCH/launch_rover_vitb.sh 10.47.234.125 --target "trash bin"
./LAUNCH/launch_rover_vitb.sh --hiwonder --target "trash bin"
```

---

## 3. `LAUNCH/launch_rover_s2diff.sh` — S2Diff-guided NavDP, in-process

Same GUI/bring-up as `launch_rover.sh`, but runs `nav_pipeline.s2diff_runner`
instead of `nav_pipeline.isaac_gui` directly — NavDP's DDPM sampling is
guided by obstacle pixels every denoising step (see
[tryout/S2DIFF_GUIDANCE.md](tryout/S2DIFF_GUIDANCE.md)). Accepts the exact
same flags as `launch_rover.sh` (table above) — `s2diff_runner.py`
pre-parses `--fov`/`--footprint-length`/`--footprint-width` for the guidance
module's geometry, then hands everything to `isaac_gui.main()` unchanged.

```bash
./LAUNCH/launch_rover_s2diff.sh                          # default rover, default Pi IP
./LAUNCH/launch_rover_s2diff.sh 192.168.21.125 --target "trash bin"
./LAUNCH/launch_rover_s2diff.sh --hiwonder --target "trash bin"
```

---

## 4. `LAUNCH/launch_rover_s2diff_http.sh` — S2Diff-guided NavDP, over HTTP

Same GUI again, but NavDP sampling is delegated over HTTP to a **separately
started** server instead of running in-process — useful for exercising the
server itself, or letting something else talk to it too.

⚠ **Start the server first, in its own terminal:**

```bash
source /home/i3d/exit/etc/profile.d/conda.sh && conda activate internnav
cd tryout && python navdp_s2diff_server.py --checkpoint ../checkpoints/navdp_extracted.pth --port 8888
```

Then, in a second terminal:

```bash
./LAUNCH/launch_rover_s2diff_http.sh                          # server on localhost:8888
./LAUNCH/launch_rover_s2diff_http.sh 192.168.21.125 --target "trash bin"
./LAUNCH/launch_rover_s2diff_http.sh 192.168.21.125 --server-url http://127.0.0.1:9000
```

Extra flags on top of the `launch_rover.sh` table (handled by
`s2diff_http_runner.py`, stripped before the rest reach `isaac_gui.py`):

| flag | default | meaning |
|---|---|---|
| `--server-url URL` | `http://127.0.0.1:8888` | where `navdp_s2diff_server.py` is listening |
| `--stop-threshold M` | `0.3` | passed to `patch_navdp_standalone_http` |

Note the script forces `--policy-type extracted` itself (the HTTP server
only serves the extracted checkpoint).

---

## 5. `LAUNCH/launch_rover_vitb_s2diff.sh` — both experimental changes at once

`exec ./launch_rover_s2diff.sh "$@" --depth-encoder vitb`. Combines #2 and #3.

```bash
./LAUNCH/launch_rover_vitb_s2diff.sh                          # default Pi IP
./LAUNCH/launch_rover_vitb_s2diff.sh 192.168.21.125 --target "trash bin"
```

---

## 6. `LAUNCH/launch_rover_remind.sh` — REMIND persistent-object targeting

Pi bring-up (camera + backend) → REMIND live server (own conda env,
background, port `8765` by default, up to 90s model-load wait) →
`nav_pipeline.remind_gui`. Type/click an object ID instead of a text phrase.

```
./LAUNCH/launch_rover_remind.sh [--rover|--hiwonder] [PI_IP] [remind_gui.py flags...]
```

Hardcoded by the script: `--max-linear 0.15`, `--max-angular
$BACKEND_MAX_ANGULAR`, `--fov $BACKEND_FOV`, `--search-angular
$BACKEND_SEARCH_ANGULAR`, `--servo-ramp-deg 70`, `--angular-slew-max
$BACKEND_ANGULAR_SLEW_MAX`, `--compressed-only`,
`--remind-server http://127.0.0.1:$REMIND_PORT`, plus
`--footprint-length/--footprint-width` when `--hiwonder`.

`remind_gui.py`'s own flags (forwarded — also apply to
`launch_rover_remind_vlm.sh` below, which shares the same GUI base):

| flag | default | meaning |
|---|---|---|
| `--target TEXT` | `""` | `"ID 3"` / `"3"` / free text (resolved via `object_query.py`) |
| `--pi-ip IP` | `None` | overridden by the script |
| `--remind-server URL` | `http://127.0.0.1:8765` | overridden by the script |
| `--remind-period S` | `0.4` | min. interval between REMIND `/infer` calls |
| `--predict-hz N` | `2.5` | pipeline tick rate |
| `--fov DEG` | `90.0` | overridden by the script |
| `--device STR` | `cuda:0` | torch device |
| `--max-linear M/S` | `0.5` | overridden by the script (`0.15`) |
| `--max-angular RAD/S` | `0.4` | overridden by the script |
| `--search-angular RAD/S` | `0.15` | overridden by the script |
| `--servo-ramp-deg DEG` | `35.0` | overridden by the script (`70`) |
| `--angular-slew-max RAD/S` | `0.10` | overridden by the script |
| `--invert-angular` | off | flip turn direction |
| `--no-belief-goal` | off | disable goal belief |
| `--stop-distance M` | `1.5` | arrival distance |
| `--depth-encoder {vits,vitb}` | **`vitb`** (differs from `isaac_gui.py`'s `vits` default — accuracy feeds directly into STOP distance) | monocular depth size |
| `--compressed-only` | off | forced on by the script |
| `--odometry-log-dir DIR` | `odometry_log` | — |
| `--imu-min-mag-calib INT` | `3` | IMU calibration digit (0-3) gating theta onto the IMU heading (added 2026-08-14, see launcher #1's row) — matters more here than most: `object_map.py`'s world-frame recall depends directly on theta accuracy across turns |
| `--object-map-path PATH` | `object_map/object_map.json` | persistent world-location store |
| `--goto-arrival-radius M` | `1.0` | blind `GOTO` giving-up radius |
| `--match-grace-period S` | `1.2` | REMIND per-tick match-drop coasting window |
| `--object-map-update-period S` | `1.0` | passive object-memory update interval |
| `--footprint-length M` | `0.482` | — |
| `--footprint-width M` | `0.380` | — |

Environment variable: `REMIND_PORT` (default `8765`) picks the REMIND
server's port for both this script and the VLM variant.

```bash
./LAUNCH/launch_rover_remind.sh                          # default rover, default Pi IP
./LAUNCH/launch_rover_remind.sh 192.168.21.125 --target "chair id 1"
./LAUNCH/launch_rover_remind.sh --hiwonder --target "chair id 1"
./LAUNCH/launch_rover_remind.sh --target "chair near the window"   # free-text -> object_query.py
REMIND_PORT=9001 ./LAUNCH/launch_rover_remind.sh
```

---

## 7. `launch_rover_remind_vlm.sh` (repo root, not `LAUNCH/`) — REMIND + VLM-confirmed arrival

Identical bring-up to #6 (REMIND live server stays InternVL-enabled — this
launcher's whole point is the `/confirm_arrival` endpoint it powers), but
runs `nav_pipeline.remind_gui_vlm` instead of `remind_gui`. Same flag table
as #6, **plus**:

| flag | default | meaning |
|---|---|---|
| `--no-vlm-confirm` | off | A/B: skip the VLM gate, pure metric-distance arrival — same behavior as `launch_rover_remind.sh` |
| `--vlm-confirm-period S` | `1.5` | min. interval between `/confirm_arrival` calls once in range |
| `--vlm-confirm-timeout S` | `6.0` | HTTP timeout on the confirm call |

Note: this script hardcodes `--max-angular 1.2` and `--search-angular 0.13`
itself (not `$BACKEND_MAX_ANGULAR`/`$BACKEND_SEARCH_ANGULAR`) regardless of
`--rover`/`--hiwonder` — check the script header if that matters for your run.

```bash
./launch_rover_remind_vlm.sh                          # default rover, default Pi IP
./launch_rover_remind_vlm.sh 192.168.21.125 --target "chair id 1"
./launch_rover_remind_vlm.sh --hiwonder --target "chair id 1"
./launch_rover_remind_vlm.sh --no-vlm-confirm          # A/B: pure metric, same as launch_rover_remind.sh
```

---

## 8. `LAUNCH/launch_bot.sh` — manual control + Go Home (either backend)

```
./LAUNCH/launch_bot.sh [--rover|--hiwonder] [PI_IP] [--enable-obstacle-avoidance] [home_gui.py flags...]
```

No camera bring-up unless `--enable-obstacle-avoidance` is passed (then it's
identical to `launch_rover.sh`'s camera bring-up — a real model-load pause).
`--home-max-linear 0.15 --home-max-angular 0.5` hardcoded by the script;
with `--enable-obstacle-avoidance` also adds `--fov $BACKEND_FOV
--compressed-only` (+ footprint flags for `--hiwonder`).

`home_gui.py`'s full flag set (also applies to `launch_rover_home.sh`, #9,
which is a thin wrapper around this script):

| flag | default | meaning |
|---|---|---|
| `--pi-ip IP` | `None` | overridden by the script |
| `--max-linear M/S` | `0.15` | manual-drive cap |
| `--max-angular RAD/S` | `0.5` | manual-drive cap |
| `--home-max-linear M/S` | `0.15` | Go Home drive cap |
| `--home-max-angular RAD/S` | `0.5` | Go Home turn cap |
| `--home-ang-min-cmd RAD/S` | `0.12` | rotate-stiction floor |
| `--home-lin-min-cmd M/S` | `0.08` | drive-stiction floor |
| `--home-kp-lin` | `0.15` | linear speed = clip(kp*dist, floor, max) |
| `--home-deadband RAD` | `0.05` | matches `pipeline.py`'s `servo_deadband` |
| `--home-ramp-deg DEG` | `45.0` | servo ramp |
| `--home-angular-slew-max` | `0.6` | rad/s² cap, 10 Hz loop |
| `--home-dist-tol M` | `0.10` | stop-within distance for "at home" |
| `--home-heading-tol DEG` | `5.0` | heading match tolerance for FACE→ARRIVED |
| `--imu-min-mag-calib INT` | `3` | IMU calibration gate (0-3) — BNO055 magnetometer sub-score on `--hiwonder`, BNO085/BNO08x combined accuracy status on `--rover` (chip changed 2026-08-14, same 0-3 scale/gate) |
| `--odometry-log-dir DIR` | `odometry_log` | — |
| `--enable-obstacle-avoidance` | off | load full DINO+NavDP for the long homing leg (see README) |
| `--compressed-only` | off | — |
| `--fov DEG` | `60.0` | — |
| `--device STR` | `cuda:0` | — |
| `--depth-encoder {vits,vitb}` | `vits` | — |
| `--hard-stop-dist M` | `0.60` | forward obstacle → AVOID |
| `--reverse-dist M` | `0.35` | — |
| `--slow-dist M` | `2.5` | — |
| `--corridor-half-width M` | `0.35` | forward-corridor half-width |
| `--max-range M` | `4.0` | ignore obstacle points beyond this |
| `--avoid-confirm-ticks N` | `2` | — |
| `--avoid-cooldown-ticks N` | `8` | — |
| `--avoid-bias-gain` | `0.15` | rad/s added during cooldown |
| `--home-arrival-radius M` | `1.0` | `navdp_home_loop`'s own arrival check (GOTO never self-declares) |
| `--home-navdp-timeout-s S` | `240.0` | — |
| `--home-navdp-predict-hz N` | `2.5` | — |
| `--home-navdp-sample-num N` | `32` | NavDP candidate trajectories per tick |
| `--home-navdp-policy-type {crossmodal,extracted}` | `crossmodal` | — |
| `--home-navdp-angular-slew-max RAD/S` | `0.10` | per-tick delta at ~2.5 Hz (not the same units as `--home-angular-slew-max`) |
| `--home-navdp-invert-angular` | off | — |
| `--footprint-length M` | `0.482` | — |
| `--footprint-width M` | `0.380` | — |

```bash
./LAUNCH/launch_bot.sh --rover
./LAUNCH/launch_bot.sh --hiwonder
./LAUNCH/launch_bot.sh --hiwonder 10.47.234.228 --enable-obstacle-avoidance
./LAUNCH/launch_bot.sh --home-dist-tol 0.05 --home-heading-tol 3
./LAUNCH/launch_bot.sh --enable-obstacle-avoidance --home-navdp-policy-type extracted
```

---

## 9. `LAUNCH/launch_rover_home.sh` — alias for #8

`exec ./launch_bot.sh "$@"` — kept for backward-compatible muscle memory.
Same flags, same table as #8.

```bash
./LAUNCH/launch_rover_home.sh                          # default rover, default Pi IP
./LAUNCH/launch_rover_home.sh 192.168.21.125 --home-dist-tol 0.05
./LAUNCH/launch_rover_home.sh --hiwonder --enable-obstacle-avoidance
```

---

## 10. `LAUNCH/launch_odom_test.sh` — odometry-accuracy GUI

Pi bring-up **without** camera, then `scripts/odom_accuracy_gui.py` — no
DINO/SAM/NavDP/depth models loaded at all.

| flag | default | meaning |
|---|---|---|
| `--pi-ip IP` | `None` | overridden by the script |
| `--max-linear M/S` | `0.15` | overridden by the script |
| `--max-angular RAD/S` | `1.2` | overridden by the script (`$BACKEND_MAX_ANGULAR`) |
| `--odometry-log-dir DIR` | `odometry_log` | — |
| `--imu-min-mag-calib INT` | `3` | IMU calibration digit (0-3) gating theta onto the IMU heading (added 2026-08-14, see launcher #1's row) — this tool exists specifically to measure odometry accuracy, so it's a natural place to A/B the gate itself |

```bash
./LAUNCH/launch_odom_test.sh                          # default rover, default Pi IP
./LAUNCH/launch_odom_test.sh 192.168.21.125 --max-angular 1.0
./LAUNCH/launch_odom_test.sh --hiwonder
```

---

## 11. `LAUNCH/launch_dino_navdp.sh` — headless node only, no bring-up

Does **not** bring the Pi up itself — the rover/Isaac/LanderPi side must
already be running separately (systemd services / Isaac's zenoh bridge /
`landerpi/deploy_bridge.sh`). `--rover`/`--hiwonder` here ONLY fills in
`--pi-ip`/`--fov`/footprint defaults (still overridable) — no bring-up, no
camera/rpm verification. Omit both flags entirely and you get the original
behavior: no `--pi-ip` at all → Zenoh multicast discovery.

`zenoh_node.py`'s full flag set:

| flag | default | meaning |
|---|---|---|
| `--pi-ip IP` | `None` | omit for multicast discovery |
| `--target TEXT` | `"trash bin"` | — |
| `--predict-hz N` | `3.0` | — |
| `--max-linear M/S` | `0.15` | — |
| `--max-angular RAD/S` | `0.25` | — |
| `--fov DEG` | `90.0` | — |
| `--stop-distance M` | `1.5` | — |
| `--angular-slew-max RAD/S` | `0.10` | — |
| `--invert-angular` | off | — |
| `--no-belief-goal` | off | — |
| `--device STR` | `cuda:0` | — |
| `--odometry-log-dir DIR` | `odometry_log` | — |
| `--imu-min-mag-calib INT` | `3` | IMU calibration digit (0-3) gating theta onto the IMU heading (added 2026-08-14, see launcher #1's row) |
| `--footprint-length M` | `0.482` | — |
| `--footprint-width M` | `0.380` | — |

```bash
./LAUNCH/launch_dino_navdp.sh                                    # multicast discovery, default target
./LAUNCH/launch_dino_navdp.sh --target "red chair"
./LAUNCH/launch_dino_navdp.sh --pi-ip 192.168.1.42 --target "trash bin"
./LAUNCH/launch_dino_navdp.sh --hiwonder --target "trash bin"    # LanderPi defaults, no bring-up
```

---

## 12. `LAUNCH/launch_gui.sh` — Isaac Sim GUI, no bring-up

No `--rover`/`--hiwonder` support (doesn't source `_backend.sh` at all) — the
Isaac scene + its own ROS 2 bridge scripts must already be running (see
`Isaac_omniVLA_readme.txt` for the Isaac-side setup: activate `isaacsim`,
run the `.usd`, press Play, run `isaac_cmdvel_bridge.py` +
`isaac_camera_ros2_pub.py` in the script editor, then a
`zenoh-bridge-ros2dds` terminal). Hardcodes `--max-linear 0.5 --max-angular
0.6` (sim caps). Accepts the same `isaac_gui.py` flags as `launch_rover.sh`'s
table above (all of them — `--fov`, `--target`, `--policy-type`, etc.).

```bash
./LAUNCH/launch_gui.sh                                   # sim defaults (0.5 m/s, 0.6 rad/s)
./LAUNCH/launch_gui.sh --target "cardboard box"
./LAUNCH/launch_gui.sh --max-linear 0.15 --max-angular 0.25   # real-rover-like caps in sim
./LAUNCH/launch_gui.sh --pi-ip 192.168.1.42              # explicit Zenoh peer
```

---

## 13. `LAUNCH/launch_hiwonder_dualvln.sh` — InternVLA-N1 native DualVLN

Defaults the backend to `--hiwonder` (a later `--rover` in the args still
overrides — last flag wins, same as `_backend.sh` everywhere else). Skips
the DINO+SAM+NavDP stack entirely: runs InternVLA-N1's own System-2
(QwenVL-2.5-7B pixel+latent goal) + System-1 (Diffusion Transformer)
end-to-end, as released ("Ground Slow, Move Fast", arXiv 2512.08186) — good
for long, compound, multi-landmark instructions.

```
./LAUNCH/launch_hiwonder_dualvln.sh [--rover|--hiwonder] [--no-gui] [PI_IP] [node args...]
```

| flag | default | meaning |
|---|---|---|
| `--no-gui` | off | this script's own flag (not `_backend.sh`'s) — skip the Tkinter path-viewer helper (needs a display; useful over SSH without X11) |
| `--instruction TEXT` | module default | the compound navigation instruction |
| `--predict-hz N` | module default | — |
| `--max-linear M/S` | module default (`0.15`) | **not** overridden from `$BACKEND_MAX_ANGULAR` — this node has its own validated defaults, deliberately more conservative than the backend ceiling |
| `--max-angular RAD/S` | module default (`0.25`) | see above |
| `--model-path PATH` | module default | checkpoint path |
| `--internnav-repo PATH` | module default | — |

```bash
./LAUNCH/launch_hiwonder_dualvln.sh --instruction "walk through the opening \
    between the kitchen and the dining room, turn right, go through \
    the doorway and stop next to the closet"
./LAUNCH/launch_hiwonder_dualvln.sh 10.47.234.228 --instruction "..."
./LAUNCH/launch_hiwonder_dualvln.sh --rover --instruction "..."   # override backend
./LAUNCH/launch_hiwonder_dualvln.sh --no-gui --instruction "..."  # headless
```

---

## 14. `MARS/launch_mars.sh` — Habitat Mars yard

No `--rover`/`--hiwonder` (sim-only, `mars_habitat` conda env for the sim
node + `internnav` for the GUI). Launcher-level flags are consumed first (any
order), everything left over forwards to the GUI script:

| flag | meaning |
|---|---|
| `--rocks` | load `rock_envs/run1/rock_field.json` obstacle field |
| `--no-rocks` | accepted for back-compat; no-rocks is the default anyway |
| `--belief-only` | switch GUI script to `scripts/mars_belief_only_gui.py` (goal locked to belief, distractor-gated) |

`scripts/mars_gui.py` flags (default GUI script):

| flag | default |
|---|---|
| `--target TEXT` | `"big stone"` |
| `--predict-hz N` | `2.5` |
| `--fov DEG` | `90.0` |
| `--device STR` | `cuda:0` |
| `--max-linear M/S` | `0.5` |
| `--max-angular RAD/S` | `0.4` |
| `--invert-angular` | off |
| `--belief-confidence-min` | `0.15` |
| `--max-climb-deg DEG` | `20.0` |

`scripts/mars_belief_only_gui.py` flags (with `--belief-only`) — same as
above minus `--belief-confidence-min`, plus:

| flag | default | meaning |
|---|---|---|
| `--distractor-gate M` | `1.5` | fresh detections farther than this from the belief's predicted position are ignored as distractors |

```bash
./MARS/launch_mars.sh                                    # empty yard, no rocks
./MARS/launch_mars.sh --rocks
./MARS/launch_mars.sh --belief-only --rocks
./MARS/launch_mars.sh --belief-only --distractor-gate 2.0 --target "boulder"
./MARS/launch_mars.sh --max-climb-deg 25                 # balking at slopes/hills? raise this
./MARS/launch_mars.sh --rocks --target "boulder" --max-linear 0.3
```

---

## 15. `EARTH/launch_earth.sh` — Habitat real-world photogrammetry scan

Same `mars_habitat`/`internnav` env split as MARS, no launcher-level flags of
its own — everything forwards to `scripts/earth_gui.py`:

| flag | default |
|---|---|
| `--target TEXT` | `"yellow building"` |
| `--predict-hz N` | `2.5` |
| `--fov DEG` | `90.0` |
| `--device STR` | `cuda:0` |
| `--max-linear M/S` | `0.5` |
| `--max-angular RAD/S` | `0.4` |
| `--invert-angular` | off |
| `--belief-confidence-min` | `0.15` |
| `--max-climb-deg DEG` | `20.0` |

Presets: `"yellow building"`, `"target sign"`, `"parked car"`, `"sand
mound"`, `"excavator"`, `"bush"`.

```bash
./EARTH/launch_earth.sh
./EARTH/launch_earth.sh --target "target sign"
./EARTH/launch_earth.sh --max-climb-deg 25    # balking at climbing curbs/mounds? raise this
```

---

## Not currently runnable

`ISAAC/launch_isaac_topo_repeat.sh` (topological-repeat-navigation
experiment) doesn't exist anymore — `ISAAC/scripts/` and `ISAAC/topo_nav/`
are down to `__pycache__` leftovers only, the source was removed at some
point. Don't chase this one; if the topo-repeat experiment gets revived,
document it here again with real flags.

---

## S2Diff server (used by `launch_rover_s2diff_http.sh`, #4)

```bash
# terminal 1 — the server
source /home/i3d/exit/etc/profile.d/conda.sh
conda activate internnav
cd tryout && python navdp_s2diff_server.py --checkpoint ../checkpoints/navdp_extracted.pth --port 8888

# terminal 2 — the rover GUI, talking to it
./LAUNCH/launch_rover_s2diff_http.sh 10.93.142.125
```

---

## Indoor-relevant targets (COCO vocabulary, DINO/CLIP presets)

chair, couch, dining table, bed, potted plant, tv, laptop, mouse, keyboard,
remote, cell phone, book, clock, vase, scissors, teddy bear, backpack,
handbag, suitcase, umbrella, bottle, wine glass, cup, bowl, microwave, oven,
toaster, sink, refrigerator, toilet, bench, person.

Good bye.