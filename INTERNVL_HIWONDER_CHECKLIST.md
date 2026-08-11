# Checklist — InternVL-driven pipeline for the Hiwonder bot

Originally a general planning checklist for standing up a navigation
pipeline that uses InternVL for goal grounding instead of the Grounding
DINO + CLIP + DINOv2 stack. Superseded by a more specific, better-fit
target: **InternVLA-N1's native DualVLN dual-system** ("Ground Slow, Move
Fast", arXiv 2512.08186) — System-2 (QwenVL-2.5-7B) predicts a pixel+latent
goal from RGB + a language instruction, System-1 (a Diffusion Transformer)
turns that into a smooth local trajectory, both jointly trained and shipped
in one checkpoint. Well suited to the long, compound, multi-landmark
instructions this project actually needs ("walk through the opening between
the kitchen and the dining room, turn right, go through the doorway and stop
next to the closet"), since that's exactly the R2R/RxR-style data it was
trained on.

Implemented as two new, purely additive files (nothing existing modified):
`reference/internvla_dualvln_zenoh_node.py` and
`LAUNCH/launch_hiwonder_dualvln.sh`.

## 1. Environment / model
- [x] Grounding output confirmed: System-2 emits a pixel goal (Qwen-VL
      0-1000 grounding scale) + a compact latent goal; System-1 (DiT)
      turns the goal into a 32-waypoint local trajectory — not just
      captioning.
- [x] Separate env confirmed: runs in the `internnav` conda env with
      `transformers` shadowed to 4.51.0 via `PYTHONPATH` (required —
      the base env's own transformers, 5.9.0, is incompatible).
- [x] Runs on the GPU workstation, not the Hiwonder's onboard compute.
      Checkpoint (~16.7GB, System-2 + System-1 weights together) and the
      `internnav` package are already present locally — nothing to
      download.

## 2. Transport
- [x] Reuses the existing Zenoh contract (`image_raw` in, `cmd_vel` +
      `omnivla/explanation` out) — same as every other launcher.
- [x] No depth/intrinsics needed from the bot: the model is RGB-only
      (confirmed — its native System-1 branch accepts but never reads a
      depth argument). A monocular depth estimate is still computed
      locally on the GPU side, purely for the independent safety net.
- [x] Command channel: `cmd_vel` publisher, same as all other launchers,
      with the same 0.15s heartbeat pattern (real firmware cmd_vel
      timeout is 500ms).

## 3. Grounding
- [x] Output format confirmed via source: discrete action OR pixel goal
      (0-1000 normalized) + latent goal.
- [x] Parsing reused as-is from the validated sibling node
      (`internvla_zenoh_node.py`'s regex/coordinate handling).
- [x] Fallback: "no output" holds position (lin=ang=0); STOP is the
      model's own arrival signal, confirmed over `stop_confirm_count`
      consecutive ticks before latching goal-reached.

## 4. 3D lift
- [x] N/A by design: System-1's trajectory is already egocentric-metric
      (meters, re-anchored to the live camera frame every call) — no
      depth-based 2D→3D lift step needed, unlike the DINO+NavDP pipeline.
- [x] Sanity handled by the existing `_trajectory_to_cmd` windowed
      lookahead + cross-tick damping (reused unchanged).

## 5. Motion
- [x] Reuses System-1's own DiT trajectory output — the "real" model
      path, not NavDP or a hand-rolled controller.
- [x] Trajectory → velocity conversion reused unchanged from the sibling
      node (`_trajectory_to_cmd`, majority-cluster reduction over the 32
      sampled trajectories, heading-based steer + speed taper).
- [x] Velocity limits: node's own defaults (max_linear=0.15,
      max_angular=0.25) — already below the Hiwonder ceiling
      (`BACKEND_MAX_ANGULAR=0.5`) established for the other pipeline.

## 6. Two-speed loop (the critical part)
- [x] Already implemented natively by the checkpoint's own async design:
      System-2 (~2Hz, ~0.7-1.1s/tick per the paper) predicts a goal;
      System-1 re-derives a fresh trajectory from the *current* frame
      every tick using the last (possibly stale) latent goal — no
      custom tracker needed, this is how the model was trained to run.
- [x] Track-loss / re-query condition: built into `agent.step()`
      (`PLAN_STEP_GAP`), reused unchanged.
- [ ] Confirm control loop rate stays stable on THIS machine's GPU under
      real load — needs an actual run (cannot verify without hardware).

## 7. Safety
- [x] Independent, model-agnostic depth-based hard stop (30cm) +
      reactive steer-around (0.9m) + stall/wedged-robot recovery — all
      reused unchanged from the sibling node, which already validated
      this live on real hardware.
- [x] STOP/hold-position fallback on no-output ticks.

## 8. Test — none of this can be done from here; needs the physical bot
- [ ] Bench-test grounding accuracy on static frames before closing the
      loop.
- [ ] Dry-run with bot stationary / wheels off the ground, commands
      logged but not sent.
- [ ] Live test at reduced max speed first.
- [ ] Try the actual target instruction style (multi-landmark, e.g. the
      kitchen/dining-room/closet example) and confirm the model tracks
      progress through it correctly, not just single-step instructions.
- [ ] Compare against the sibling node's System-2-only /
      `--use-navdp` modes to see whether the native System-1 is in fact
      smoother/more accurate on this hardware.
