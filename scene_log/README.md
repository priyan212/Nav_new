# `scene_log/`

Open-vocabulary object inventory, logged by `nav_pipeline/scene_tagger.py`
via `nav_pipeline/pipeline.py`. Once per second (`PipelineConfig.scene_tag_period_s`),
the pipeline runs Grounding DINO against a broad vocabulary
(`scene_tagger.DEFAULT_VOCAB`, configurable via `PipelineConfig.scene_vocab`)
against the current frame and counts detections per label — independent of
whatever goal the rover is currently pursuing. Meant for a supervisor to see
what the rover has observed, and later for route-finding logic to query.

Each entry is also printed to stdout as it's captured:

```
[scene] 14:32:07  bottle:1, chair:2, monitor:1  pose=(1.34, -0.20, +0.05)
```

## One file per run

A new JSONL file starts every time a `DinoNavDPPipeline` is constructed,
named:

```
scene_<YYYYMMDD_HHMMSS>.jsonl
```

Unlike `odometry_log/` this is **not** reset per-goal — the log spans the
whole process lifetime, since the object inventory isn't tied to any one
target.

## Format

One JSON object per line:

```json
{"t": 1784037127.4, "objects": {"bottle": 1, "chair": 2, "monitor": 1}, "pose": {"x": 1.34, "y": -0.20, "theta": 0.05}}
```

| field | meaning |
|---|---|
| `t` | wall-clock time (`time.time()`) the frame was tagged |
| `objects` | `{label: count}` for every vocabulary label Grounding DINO found in that frame |
| `pose` | rover's dead-reckoned `(x, y, theta)` at capture time, from `nav_pipeline/odometry_logger.py` -- **only present when the caller passes `pose=` into `pipeline.step()`** (both `isaac_gui.py` and `zenoh_node.py` do; offline/sim scripts that call `step()` without a `pose` argument won't have this field) |

### The pose is a continuous world frame, not per-goal-local

`OdometryLogger` keeps `(x, y, theta)` continuous across goals by default —
it does **not** reset when the target text changes (see the top-level
[README's Odometry logging section](../README.md#odometry-logging)), so two
entries with `pose.x=1.0` from *different* goals genuinely are the same
physical spot (modulo ordinary dead-reckoning drift, which is unbounded over
a long enough session either way — no GPS/SLAM correction happens here).
That's what makes this log usable as a single accumulating inventory across
a whole session instead of needing per-goal origin-stitching. It's still
just dead reckoning, though: a `reset_pose()` call (an explicit operator
"reset map" action, or `start_new_goal(..., reset_pose=True)`) starts a new
local frame, and entries logged before vs. after that call are **not**
comparable without knowing where that reset happened.

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('scene_20260728_143000.jsonl')]
print('entries:', len(rows))
print('latest:', rows[-1])
"
```
