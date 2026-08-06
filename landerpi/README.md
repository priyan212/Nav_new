# LanderPi integration

Adds a second robot backend to Nav_new (Hiwonder LanderPi, Mecanum chassis)
alongside the existing 6WD ESP32 rover. **Purely additive** — nothing in
Hiwonder's stock software (the `armpi_pro` package/container/services) is
modified. The only thing installed on the robot is `bridge.py` (this dir)
plus one pip package (`eclipse-zenoh`, built from source — see below).

Switch between robots with `LAUNCH/launch_bot.sh --rover` /
`LAUNCH/launch_bot.sh --hiwonder`. Both drive the exact same
`nav_pipeline.home_gui` / `pipeline.py` / `obstacle_guard.py` — nothing on
the GPU side needed a code change, only a new bridge process on the Pi.

## What's actually on this robot

SSH: `pi` / `raspberrypi` (Debian 12 bookworm on a Pi 5, **not** a blank
image — Hiwonder pre-loads their demo stack). IP is DHCP and has changed
several times already; if `192.168.0.8` doesn't respond, check the robot
directly (`hostname -I` on its console) rather than guessing.

Everything robot-specific runs inside a long-lived Docker container:

```
docker ps -a   ->   armpi_pro   (image: ros:noetic, --network host)
```

Inside it: ROS1 Noetic, Hiwonder's "ArmPi Pro" bringup stack (`~/armpi_pro/
src/...`), started by the host's `start_node.service` (`~/armpi_pro/
start_node.sh` → `docker exec ... armpi_pro_bringup/scripts/start_node.sh`).
This is Hiwonder's shared base image across several of their Pi kit lines —
"armpi_pro" branding, not "landerpi" — but the chassis under it really is
Mecanum (confirmed in `chassis_control_node.py`'s `MecanumChassis` class),
matching the kit that was actually purchased.

Key ROS1 topics (all pre-existing, unmodified):

| Topic | Type | Notes |
|---|---|---|
| `/usb_cam/image_raw/compressed` | `sensor_msgs/CompressedImage` | 640x480 JPEG. `camera_info`: fx=507.22 fy=507.37 cx=311.81 cy=242.91 → horizontal FOV ≈ 64.6° |
| `/chassis_control/set_velocity` | `chassis_control/SetVelocity` (custom: `float64 velocity` mm/s, `float64 direction` deg, `float64 angular` rad/s) | Polar drive command. `direction`: 90°=fwd (+y), 270°=reverse, per `MecanumChassis.translation()`'s own convention. **No cmd timeout** — drives forever on the last command if nothing re-sends (see bridge's watchdog). |
| `/ros_robot_controller/imu_raw` | `sensor_msgs/Imu` | **Confirmed dead on the STOCK hardware/firmware**, not just unwired. Traced conclusively (2026-08-06): the UART link, packet framing, and checksum are all provably good (battery telemetry flows reliably over the identical link/parser at 9.09Hz), `enable_reception()` is called and enabled at node startup, but the SDK has no command to *request* IMU streaming — it's purely passive. Zero IMU packets were ever observed, including under physical motion. Either the IMU chip isn't populated on this board revision, or the firmware doesn't implement/enable that reporting — not fixable from the Pi side without new STM32 firmware. **Superseded**: a separate BNO055 breakout was wired directly to the Pi's own I2C-1 bus (addr `0x28`, alongside the motor driver at `0x34`) — see "IMU: separate BNO055" below. This topic itself is still dead; the replacement bypasses it entirely. |
| `/ros_robot_controller/battery` | `std_msgs/UInt16`(ish) | mV, e.g. `7020` = 7.02V |

**No `/odom`, no `/tf`, and `chassis_control_node.py`'s I2C motor driver
(`EncoderMotorController` @ addr `0x34`) is write-only** despite being
"encoder motors" — Hiwonder's own ROS code never reads counts back.
**However, real per-wheel encoder feedback does exist on this chip** at
register `0x3C` ("total pulse value of 4 encoder motors", 4x int32 LE) —
confirmed via Hiwonder's own public [PX4 driver
source](https://github.com/PX4/PX4-Autopilot/tree/main/src/drivers/hiwonder_emm),
which had to document it to build a real closed-loop driver. `bridge.py`
reads this directly (read-only I2C traffic alongside `chassis_control_node.py`'s
writes — Linux i2c-dev serializes bus transactions, safe to run
concurrently) for **real, closed-loop odometry** — not synthesized from
commanded velocity. See `bridge.py`'s docstring and the live-verified
section below for the empirical channel-mixing calibration this needed
(the raw channel order doesn't match the write-side motor indices).

## IMU: separate BNO055 (added 2026-08-06)

Since the stock `/ros_robot_controller/imu_raw` path is confirmed dead (see
topic table above), a standalone BNO055 breakout was wired directly to the
Pi's own I2C-1 bus (**SDA→SDA, SCL→SCL, straight-through — NOT a crossover
like UART TX/RX**; a first attempt had these swapped, which is exactly why
the chip didn't show up on an `i2cdetect` scan at first). Confirmed via
`CHIP_ID` register `0x00` reading `0xA0` (genuine BNO055).

`bridge.py`'s `init_bno055()` brings it up into NDOF (9-DOF fusion) mode at
startup, and `read_bno055()` reads Euler heading + calibration status each
tick, packed into the exact same `sys*1000+gyr*100+acc*10+mag` byte the old
ESP32 rover's firmware already uses — so `odometry_logger.py`'s existing
IMU heading-fusion path (gated on the MAG sub-score ≥3) picks it up with
**zero GPU-side code changes**. If the chip isn't detected at startup (e.g.
still being wired up), the bridge sends `0.0/0.0` for heading/calib — always
below the trust gate, so it cleanly falls back to encoder-only heading — and
retries the I2C probe every 5s in the background, so a later reconnect is
picked up automatically without a bridge restart.

**Mounting note**: heading (Euler yaw) is only numerically stable once the
chip is reasonably flat/level. Verified live while hand-held before
mounting: the raw **quaternion** output stayed smooth and small-varying,
while the derived **heading** jumped ~90° a few times — the textbook
signature of gimbal lock in Euler-angle extraction near ±90° pitch/roll, not
a wiring or calibration problem. Once mounted flat and stationary, heading
tracked cleanly (stable to ~0.1° while idle). **Not yet validated while
actually rigidly attached to the driving chassis** — a live turn test while
still hand-held (not mounted) showed a heading delta measurably larger than
the encoder-only estimate for the same command, which is expected/
meaningless if the IMU wasn't moving with the robot at that moment, not
evidence of inaccuracy. Re-validate once permanently mounted.

No depth camera / LiDAR in this ROS1 package set — RGB only, same as the old
rover. `pipeline.py`'s monocular Depth-Anything-V2 fallback applies
unchanged.

## Bridge dependencies (one-time setup)

`bridge.py` needs `eclipse-zenoh` **1.9.0** (matching the GPU side's version
— `pip show eclipse-zenoh` in the `internnav` conda env — since zenoh's wire
protocol is not compatible across major versions) plus `rospy`.

**The container's system Python is 3.8, and current `eclipse-zenoh` cannot
run on it at all** — its PyO3 bindings reference `PyCMethod_New`, a CPython
C-API symbol that doesn't exist before 3.9 (confirmed by actually building
zenoh from source for cp38 with a full Rust toolchain — it built fine, then
segfaulted-on-import with `undefined symbol: PyCMethod_New`; not a packaging
fluke, a real MSRV bump). The fix is a **separate Python 3.10 venv**
alongside the system Python 3.8 Hiwonder's own stack uses — nothing in
`armpi_pro` touched, just one more interpreter:

```bash
# 1. container DNS was broken out of the box (two separate bugs, both
#    runtime-only fixes, neither touches any Hiwonder file) -- needed for
#    anything below to reach the internet at all:
docker exec -u root armpi_pro bash -c \
  'sed -i "s/mdns4_minimal \[NOTFOUND=return\] //" /etc/nsswitch.conf'
    # ^ nsswitch had mdns4_minimal with [NOTFOUND=return] BEFORE dns, which
    #   aborts resolution for any non-.local name before ever trying DNS —
    #   classic Debian/Avahi misconfig, unrelated to Hiwonder's own code.
docker exec -u root armpi_pro bash -c \
  'printf "nameserver 192.168.0.1\nnameserver 8.8.8.8\n" > /etc/resolv.conf'
    # ^ Docker's host-network resolv.conf generation left this empty;
    #   192.168.0.1 (the LAN router) resolves fine, 8.8.8.8 alone did not
    #   (router likely blocks direct external DNS from this interface).
    #   Both entries are runtime state — reset on container restart, redo
    #   if the bridge stops resolving anything after a reboot.

# 2. install `uv` (single static binary, fetches prebuilt CPython -- no
#    compiling) and a Python 3.10 interpreter:
docker exec -u ubuntu -w /home/ubuntu armpi_pro bash -c \
  'curl -LsSf https://astral.sh/uv/install.sh | sh'
docker exec -u ubuntu -w /home/ubuntu armpi_pro bash -c \
  '~/.local/bin/uv python install 3.10'

# 3. venv + eclipse-zenoh (prebuilt cp310 wheel exists, no Rust needed) +
#    rospy's small pure-Python deps (rospy itself is pure Python, so it
#    works fine under 3.10 even though ROS Noetic's system Python is 3.8 --
#    only PyYAML/rospkg/netifaces/defusedxml were actually missing):
docker exec -u ubuntu -w /home/ubuntu armpi_pro bash -c \
  '~/.local/bin/uv venv --python 3.10 ~/nav_new_bridge/venv310'
docker exec -u ubuntu -w /home/ubuntu armpi_pro bash -c \
  '~/.local/bin/uv pip install --python ~/nav_new_bridge/venv310/bin/python3.10 \
     eclipse-zenoh==1.9.0 pyyaml rospkg netifaces defusedxml'
```

Verify: `docker exec -u ubuntu armpi_pro ~/nav_new_bridge/venv310/bin/python3.10
-c 'import zenoh, rospy; print("ok")'` (run after sourcing
`source_env.bash` if testing `rospy`/message-package imports specifically —
see `deploy_bridge.sh` for the exact invocation pattern).

If the GPU side's `eclipse-zenoh` version ever changes, rerun step 3's
`uv pip install` with the new version pinned.

## Deploying / running the bridge

```bash
cd landerpi && ./deploy_bridge.sh [PI_IP]
```

Copies `bridge.py` to `~/nav_new_bridge/` on the Pi and starts it via
`docker exec -d` inside the existing `armpi_pro` container (sourcing
Hiwonder's own `source_env.bash` for ROS env, exactly like their
`start_node.sh` does) — runs alongside their stack, doesn't replace or
restart anything of theirs. `LAUNCH/launch_bot.sh --hiwonder` calls this
automatically. Logs: `~/nav_new_bridge/bridge.log` on the Pi (inside the
container's filesystem view).

Not yet wired into a systemd service for boot persistence (unlike the old
rover's `rover-agent`/`rover-zenoh`) — re-run `deploy_bridge.sh` after any
Pi power cycle. Worth adding once the basic integration is validated live.

## Live-verified (2026-08-06)

Tested directly over Zenoh against the real robot (bridge deployed via
`deploy_bridge.sh`, no GUI/pipeline involved — raw `cmd_vel` pulses):

- **Camera round-trip**: `image_raw/compressed` decoded via
  `zenoh_node.py`'s actual `parse_compressed_image()` into a real
  (480, 640, 3) RGB frame.
- **Forward direction**: `linear_x > 0` → confirmed forward. No
  `--invert-linear`-equivalent needed.
- **Turn direction**: `angular_z > 0` → confirmed left (counterclockwise),
  matching standard ROS `Twist` convention and what `pipeline.py` already
  assumes. No `--invert-angular` needed (unlike the old rover, which did).
- **cmd_vel watchdog**: sent one non-zero command then abruptly closed the
  Zenoh session (no stop command, simulating a crash) — robot stopped on
  its own within ~0.5s via `bridge.py`'s independent watchdog thread.
- **IMU: confirmed non-functional**, not just unwired — see the topic
  table above for the full evidence trail.
- **Real encoder odometry** (register `0x3C`, replacing the original
  open-loop synthetic version — see "Odometry" below): round-tripped
  through the actual `OdometryLogger` class end to end, both before and
  after adding lateral tracking (see below), same order of magnitude both
  times: forward test (0.05 m/s x 2s) landed at x≈0.09-0.10m (ideal ≈0.10m
  accounting for ramp-up lag), y/theta ≈ 0. Turn test (+0.3 rad/s x 2s)
  landed at theta≈0.62-0.67rad (ideal 0.6rad).

Not yet tested: sustained driving over longer distances/many turns
(cumulative real-world drift, as opposed to these short controlled tests),
obstacle avoidance with the spec-sheet footprint (see caveat below),
behavior under `--enable-obstacle-avoidance`/full NavDP autonomy.

## Odometry: real encoder feedback, then real *holonomic* feedback (2026-08-06)

**Problem**: a full `launch_bot.sh --hiwonder` Go-Home session produced a
visibly "bogus" route both on-screen and physically — the robot drove
wrong, and the displayed path didn't reflect reality.

**First root cause (fixed)**: the *original* `bridge.py` synthesized
`rover/rpm` purely from the last *commanded* velocity (open-loop, no real
feedback at all), so any mismatch between commanded and actually-achieved
motion (chassis_control_node.py ramps velocity changes over ~250ms rather
than achieving them instantly, plus ordinary slip) became permanent,
uncorrected error in the dead-reckoned pose — heading error compounds
through `cos(theta)`/`sin(theta)` every tick. One logged example: a 258s
session accumulated **4.7 full turns worth of |dtheta|** with 26 sign
flips, ending at a nonsensical pose. Fix: `chassis_control_node.py`'s I2C
motor driver chip (addr `0x34`) is write-only in Hiwonder's own code, but
the chip itself exposes real per-wheel encoder totals at register `0x3C`
— confirmed via Hiwonder's own public [PX4 driver
source](https://github.com/PX4/PX4-Autopilot/tree/main/src/drivers/hiwonder_emm).
`bridge.py` reads this directly and computes real `(v, w)` from actual
measured wheel motion.

**Second root cause (fixed)**: even with real encoder feedback, the route
was *still* reported bogus. Reason: this is a **Mecanum** chassis (4 DOF
of wheel actuation, 3 DOF of real planar motion — forward, turn, AND
lateral/strafe), but `odometry_logger.py`'s pose model was inherited
byte-for-byte from the old **differential-drive** rover, which only has 2
real DOF (forward, turn) and has no representation of lateral motion at
all. The first fix's encoder decoding only ever extracted 2 of the 3 real
signals (forward, turn) from the 4 raw channels — any genuine sideways
component (Mecanum wheels are inherently prone to real lateral
slip/scrub, far more than regular wheels) was measured-then-discarded,
not just unmeasured. Confirmed via a 3rd isolated live test (pure
lateral/strafe, commanded via `direction=0` directly on
`/chassis_control/set_velocity`): a completely distinct 4th linear
combination of the raw encoder channels (the plain sum of all 4 deltas)
lit up strongly, and the same combination showed real (if smaller)
leakage during the "pure forward" test too — proving the 3rd DOF is
real and was being lost, not just theoretically missing.

Fix, in two additive, backward-compatible parts:
- `odometry_logger.py`'s `update()` gained an optional `lateral_m_s`
  parameter (default `None` → every existing call site, e.g. the ESP32
  rover, is byte-for-byte unaffected — verified via a direct regression
  check). When given, pose integration becomes a proper body-frame
  rotation `x += (v*cos(theta) - lateral*sin(theta))*dt`, `y += (v*sin(theta)
  + lateral*cos(theta))*dt` instead of the old unicycle-only formula.
- `bridge.py` now extracts all 3 real DOF from the encoder channels (see
  `read_encoder_totals()`/`_rpm_loop()` docstrings for the exact empirical
  derivation) and sends `lateral_m_s` as a 5th element on `rover/rpm`;
  `zenoh_node.py`'s and `home_gui.py`'s `on_rpm` handlers parse it
  (`data[4]` if present) and pass it through. The lateral sign convention
  (+left, REP103-style) was confirmed via a live visual test — commanding
  `direction=0` (the robot's own "+x" axis) produced a visually-confirmed
  rightward slide, matching the derived geometry (0° is 90° clockwise from
  the confirmed forward=90° axis) with no extra inversion needed.

This does not eliminate drift entirely (still dead-reckoning, no absolute
position reference, and IMU heading fusion is unavailable — see above), but
the pose model can now actually represent everything this chassis can
physically do. **Needs a real-world retest** (the original bogus-report
scenario, e.g. a longer Go-Home session) to confirm the practical
improvement — the isolated test pulses above validate the mechanism, not
long-run cumulative accuracy.

## Known caveats / what to verify before more involved driving

1. Velocity caps start at the same conservative real-rover defaults
   (0.15 m/s / 0.5 rad/s via `home_gui`) — per project convention, tune ONE
   constant at a time on real hardware, don't bundle changes.
2. **Footprint (obstacle-guard) is from a retailer spec sheet, not a tape
   measure.** `obstacle_guard.py`'s `swept_clearance()` was hardcoded to the
   old rover's 0.482x0.380m footprint as a module constant — now
   parameterized via `GuardConfig.footprint_length/width` (default
   unchanged, verified byte-identical via `scripts/test_footprint_guard.py`).
   `LAUNCH/launch_bot.sh --hiwonder --enable-obstacle-avoidance` passes
   0.298 x 0.256m (ArmPi Pro spec sheet, thinkrobotics.com listing — 521mm
   height in that same listing includes the raised arm, irrelevant here).
   Single online source, not independently measured on this physical unit —
   confirm with a tape measure before trusting obstacle clearance for real
   driving; override with `--footprint-length`/`--footprint-width` if wrong.
