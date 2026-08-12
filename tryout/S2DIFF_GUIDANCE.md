# S2Diff obstacle guidance for NavDP PointGoal

This path uses only the S2Diff idea. It does not use a tube head, conformal
calibration, or corridor optimization.

The caller supplies the pixels that represent obstacles. The server reads metric
depth only at those pixels and back-projects them into NavDP's local
x-forward/y-left frame. Every pixel not supplied by the caller is ignored by the
safety energy.

## Obstacle-pixel request format

Add `obstacle_pixels` to the existing `goal_data` JSON. Coordinates are `[u,v]`,
meaning `[column,row]`, in each individual depth image—not in the vertically
combined batch image. There must be one pixel list per batch item.

```json
{
  "goal_x": [2.0],
  "goal_y": [0.0],
  "obstacle_pixels": [
    [[119, 83], [120, 83], [121, 83], [119, 84]]
  ]
}
```

For batch size two:

```json
{
  "goal_x": [2.0, 1.5],
  "goal_y": [0.0, -0.5],
  "obstacle_pixels": [
    [[119, 83], [120, 83]],
    [[42, 91], [43, 91]]
  ]
}
```

An empty list means no obstacle pixels for that batch item:

```json
"obstacle_pixels": [[]]
```

## Guided diffusion

At each of NavDP's 10 DDPM denoising steps the implementation predicts the clean
NavDP action sequence, samples particles around each candidate, integrates them
with `cumsum(action / 4)`, scores them with the S2Diff safety/stability/cost Gibbs
target, and uses the weighted posterior clean action to modify the DDPM score.

Hard-colliding particles have zero SMC weight. Candidate modes remain separate.
If every final candidate violates the hard collision distance, the server returns
a zero trajectory so the robot stops and re-observes.

## Files

- `tube_planner/pixel_obstacles.py`: supplied `[u,v]` pixels plus depth to local obstacle points.
- `tube_planner/s2diff_guidance.py`: Gibbs energy, SMC reweighting, guided DDPM and selection.
- `tube_planner/s2diff_agent.py`: released `NavDP_Agent` preprocessing/history wrapper.
- `tube_planner/depth_obstacles.py`: dense whole-frame depth -> obstacle points (alternate
  to pixel_obstacles.py; not used by policy_agent.py's server path below).
- `navdp_s2diff_server.py`: API-compatible PointGoal server with the additional
  required `obstacle_pixels` field.
- `policy_agent.py`: **not from the released NavDP baseline** -- this repo never had
  the official `NavDP_Agent`/its `cross-waic-*.ckpt` checkpoint, so this adapter wraps
  `nav_pipeline/navdp_net.py`'s `NavDPStandalone` + `checkpoints/navdp_extracted.pth`
  (the same weights the real rover pipeline runs) behind the interface
  `tube_planner/s2diff_agent.py`/`s2diff_guidance.py` expect. See its module docstring
  for the exact deviations (forced memory_size=2/predict_size=32, batch_size=1 only,
  goal y-sign flip). This is a *separate* integration path from
  `nav_pipeline/s2diff_navdp.py` (used by `LAUNCH/launch_rover_s2diff.sh`), which runs
  guided sampling in-process instead of over HTTP -- pick one, they don't share state.

## Run

From `tryout/` (so `tube_planner` and `policy_agent` resolve as top-level imports):

```bash
cd tryout
python navdp_s2diff_server.py \
  --checkpoint ../checkpoints/navdp_extracted.pth \
  --port 8888 \
  --particles 8 \
  --safe-distance 0.42 \
  --hard-collision-distance 0.24
```

`--memory-size`/`--predict-size` aren't exposed as flags (the server hardcodes 8/24 for
the released checkpoint); `policy_agent.py` ignores those and forces (2, 32) to match
`navdp_extracted.pth`, printing a warning when it does.

This remains an S2Diff-style adaptation rather than the paper's formal guarantee:
safety depends on the accuracy, coverage and freshness of the supplied pixels and
their depth values, and the current energy checks discrete predicted waypoints.
