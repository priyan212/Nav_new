# `odometry_log/`

Dead-reckoned rover pose, logged by `nav_pipeline/odometry_logger.py`. The
real rover has no LiDAR or dedicated odometry node, but the ESP32 firmware
(`esp32/rover_6wd_complete.ino`) does have real quadrature encoders on each
side's mid wheel, publishing signed L/R RPM on `/rover/rpm` (10 Hz), plus a
BNO055 IMU's fused absolute heading (`imu_heading_deg`) and calibration
status (`imu_calib`) in the same message. This directory holds the
integration of that feed into a differential-drive pose — useful for
diagnostics (did the rover actually move, or just spin?), not a substitute
for real localization: `x`/`y` still drift with no external correction, and
the whole pose resets to `(0, 0, 0)` at the start of every goal.

`theta` is taken directly from the IMU heading once the magnetometer
calibration digit reaches `imu_min_mag_calib` (default 1) — the BNO055's
onboard sensor fusion has no accumulating drift, unlike wheel-differential
heading, which drifts sharply past ~135-165° of rotation on this skid-steer
chassis (wheel slip during in-place turns, invisible to the encoders; see
`odom_accuracy_results.csv` below). Until the IMU is calibrated (or if it
never acked at boot), `theta` falls back to integrating the wheel
differential, same as before the IMU existed. See `theta_src` in the columns
below to tell which source produced a given row.

## One file per goal

A new CSV starts every time the target text changes — GUI Send/preset
button, or a new `omnivla/goal_text` message — named:

```
odom_<slugified-target>_<YYYYMMDD_HHMMSS>.csv
```

e.g. `odom_trash_bin_20260728_130406.csv`. Each file's `x, y, theta` start
at the origin, so it's a self-contained record of "how did the rover move
while pursuing this one goal" — no need to know the pose at the end of the
previous goal to interpret it. `nav_pipeline/home_gui.py`'s manual
control + Go Home GUI (see the top-level [README's "Manual control + Go
Home" section](../README.md#manual-control--go-home)) opens one file for its
whole session, named `odom_home_session_<YYYYMMDD_HHMMSS>.csv`, since it
isn't goal-text-driven.

## Columns

| column | meaning |
|---|---|
| `t` | wall-clock time (`time.time()`) the `/rover/rpm` sample was received |
| `dt` | seconds since the previous sample (0 on the first row of a file) |
| `left_rpm`, `right_rpm` | signed wheel RPM as published by the firmware (+ve = drives the robot forward) |
| `v` | robot linear velocity (m/s), `(v_left + v_right) / 2` |
| `w` | robot angular velocity (rad/s), `(v_right - v_left) / TRACK_WIDTH_M` |
| `x`, `y`, `theta` | integrated pose in the goal's local frame (metres, radians) |
| `imu_heading_deg` | raw BNO055 fused compass heading for this row, or blank if the firmware reported NaN (IMU absent/unread) |
| `theta_src` | `"imu"` if `theta` this row came from the calibrated IMU heading, `"enc"` if it fell back to integrating the wheel differential |

`v`/`x`/`y` use `WHEEL_RADIUS_M = 0.056`, `w`/`x`/`y` use
`TRACK_WIDTH_M = 0.345` — both must stay in sync with the same-named
`#define`s in `esp32/rover_6wd_complete.ino` (a bench recalibration there
should be mirrored here).

## `odom_accuracy_results.csv` and `odom_spin_*.csv` — a different kind of file

These aren't per-goal navigation logs from `odometry_logger.py` — they're
output from `scripts/odom_accuracy_gui.py` (launched via
`./launch_odom_test.sh`), a standalone GUI that drives the real rover
through a known motion (e.g. a controlled in-place spin) and compares the
same dead-reckoned odometry against hand-measured ground truth, to answer
"how much do we trust this odometry?" `odom_accuracy_results.csv` holds the
summary (one row per trial); `odom_spin_<angle>deg_<timestamp>.csv` holds
the raw per-sample trace for each individual spin trial. This is the
empirical basis for `nav_pipeline/goal_belief.py`'s rotation-noise tuning —
see the top-level [README's "Goal belief" section](../README.md#goal-belief-surviving-occlusion)
— and it's also what motivated fusing in the BNO055 IMU heading described
above: the sharp wheel-diff drift past ~135-165° visible in these trials is
exactly the failure mode the IMU heading doesn't have. Note these files
predate the IMU fields and only have the original wheel-diff-only columns.

## Reading the signature of a stuck rover

`theta` running away monotonically while `x, y` barely move is the signature
of the rover spinning in place without making progress — usually a generic,
multi-instance target (e.g. "chair" in a room full of chairs) causing
Grounding DINO's re-acquire-on-loss to hop between different physical
objects each time the tracked one scrolls out of frame. `isaac_gui.py`'s
spin-stall watchdog (`OdometryLogger.spin_delta`) catches this live: more
than a full rotation within 15 s while translating under 0.3 m force-stops
the rover instead of letting it spin indefinitely.

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('odom_trash_bin_20260728_130406.csv')))
print('elapsed s:', float(rows[-1]['t']) - float(rows[0]['t']))
print('theta start/end:', rows[0]['theta'], rows[-1]['theta'])
print('x/y start/end:', (rows[0]['x'], rows[0]['y']), (rows[-1]['x'], rows[-1]['y']))
"
```
