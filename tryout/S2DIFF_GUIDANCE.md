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

- `pixel_obstacles.py`: supplied `[u,v]` pixels plus depth to local obstacle points.
- `s2diff_guidance.py`: Gibbs energy, SMC reweighting, guided DDPM and selection.
- `s2diff_agent.py`: released `NavDP_Agent` preprocessing/history wrapper.
- `../navdp_s2diff_server.py`: API-compatible PointGoal server with the additional
  required `obstacle_pixels` field.

## Run

From `baselines/navdp`:

```bash
python navdp_s2diff_server.py \
  --checkpoint checkpoints/cross-waic-final4-125.ckpt \
  --port 8888 \
  --particles 8 \
  --safe-distance 0.42 \
  --hard-collision-distance 0.24
```

This remains an S2Diff-style adaptation rather than the paper's formal guarantee:
safety depends on the accuracy, coverage and freshness of the supplied pixels and
their depth values, and the current energy checks discrete predicted waypoints.
