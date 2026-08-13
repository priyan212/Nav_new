# `REMIND/` — object-persistent targeting for Nav_new

This directory has two parts:

- **`remind-reid-tracker/`** — the tracker itself: a multi-object
  re-identification model that gives every object a persistent identity
  across a live camera stream (SAM segmentation + DINOv3 appearance
  descriptors + relational/memory-based re-ID, BLIP/InternVL captioning for
  a human-readable label). Self-contained, its own conda env, own paper —
  full detail in [remind-reid-tracker/README.md](remind-reid-tracker/README.md).
- **This README** — how Nav_new's rover pipeline (the repo root, one level
  up) uses that tracker to let an operator target a *specific* object
  ("that chair, not the other one") instead of a bare open-vocabulary text
  phrase, and to send the rover back to an object it isn't currently
  looking at.

## Why this exists

`nav_pipeline/pipeline.py`'s default targeting (Grounding DINO) is
stateless per frame — it re-detects "a chair" fresh on every tick, with no
notion that the chair in view now is or isn't the same one it saw a minute
ago. That's fine for "drive to *a* trash bin" but can't express "drive to
*that specific* chair" once there's more than one chair in the room, and it
has no memory once the target scrolls out of frame for good. REMIND
supplies both: a persistent `object_id` that survives occlusion/re-framing,
and (via the integration layer in this repo) a running memory of *where*
each object was last seen in the world, so the rover can be sent back to
one that isn't even in the current camera view.

## Two processes, two conda envs

REMIND's own torch/transformers/ultralytics pins are newer than (and
incompatible with) this project's `internnav` env (see
`remind-reid-tracker/SETUP.md`), so it can't be imported in-process. Instead
it runs as a standalone HTTP service inside its own env, polled over
`localhost` by the rover process:

```
nav_pipeline.remind_gui  --(JPEG frame, HTTP POST /infer)-->  remind-reid-tracker/scripts/live_server.py
                          <--(object_id, bbox, mask, class_name)--
```

`LAUNCH/launch_rover_remind.sh` brings both up together: Pi camera/ESP32/Zenoh
bring-up identical to `LAUNCH/launch_rover.sh` (real rover or, via
`--hiwonder`, the Hiwonder LanderPi — see the root README's
[backend-flag section](../README.md#the---rover----hiwonder-backend-flag)),
then the REMIND live server as a background process in
`REMIND/remind-reid-tracker/.venv`, then `nav_pipeline.remind_gui`.

```bash
./LAUNCH/launch_rover_remind.sh [PI_IP]
```

Real rover / LanderPi only (needs a live camera stream) — there's no
Isaac/MARS/EARTH equivalent of this launcher.

**VLM-confirmed arrival variant:** `./launch_rover_remind_vlm.sh` (repo
root) is identical bring-up but launches `nav_pipeline.remind_gui_vlm`
instead of `remind_gui`: the same 1.5 m depth-based `stop_distance` still
zeroes velocity every tick, but "arrived" is only declared once REMIND's
already-loaded InternVL model confirms it from the live camera frame, via
the live server's `/confirm_arrival` endpoint (see
[remind-reid-tracker/README.md](remind-reid-tracker/README.md)). Falls back
to the plain metric-only behavior if that endpoint is unavailable;
`--no-vlm-confirm` forces the same fallback deliberately, for an A/B
comparison against `launch_rover_remind.sh`.

## `nav_pipeline/remind_client.py` / `remind_target.py`

`RemindClient` is the HTTP client: posts each camera frame to `/infer`,
gets back a list of `RemindObject` (id, bbox, mask, confidence, class_name).
`RemindObject.label` is deliberately **ID-only** ("ID 3", not "CHAIR ID
3") — REMIND's own BLIP/InternVL caption per object is a live model output
and can be unstable frame to frame (a chair captioned "black chair" one
tick, something else the next), so trusting it as part of the identity an
operator types back would inherit that instability. The caption is kept
purely as internal bookkeeping (see `object_map.py` below); the operator
only ever needs the number, read straight off the video overlay or the
"known objects" list. `remind_target.py`'s `parse_object_target` parses
that back: a bare number, `"ID 3"`, `"id3"`, or `"ID 3 (visible)"` (the
known-objects list's visibility suffix) all resolve to `3`.

## `nav_pipeline/object_map.py` — world-location memory

REMIND's identity is purely camera-relative — it says nothing about *where*
an object is in the world. `object_map.py` fills that gap: every tick, every
currently-visible REMIND object's local-frame goal point (same
mask-centroid + mask-median-depth math `pipeline.py` uses to drive toward a
live detection) is transformed into the rover's world frame
(`local_to_world`, using the live odometry pose) and folded into a running
per-`object_id` estimate (EMA-smoothed position, last-seen timestamp,
observation count, cached CLIP embedding — see below).

This requires **odometry pose to be continuous across goals**, not reset
each time the target changes — see the root README's
[Odometry logging](../README.md#odometry-logging) section. An object
remembered while chasing one target has to still mean something once the
target switches to a different object; a per-goal-reset origin would make
every remembered world coordinate meaningless the moment the goal changed.

Persisted to `object_map/object_map.json` (repo root) so it survives GUI
restarts within the same room/building. **Not** safe to trust across a
power cycle or a physical pick-up-and-move of the rover — there's no way to
detect that the odometry origin is no longer valid, so a stale map should
be discarded (the GUI's **Forget locations** button, or delete the file) if
the rover was moved by hand since it was last written.

## `nav_pipeline/object_query.py` — free-text targeting

Layered on top of ID-only targeting: type "go to the black chair" or "chair
near the window" instead of a number. A CLIP image embedding is cached once
per object — the first time it's seen, via `object_map.set_embedding`,
never overwritten — and matched against a CLIP text embedding of the query
(not against the noisy BLIP/InternVL caption text). Relational ("X near Y")
and positional ("leftmost X") phrasing reuse `relational_target.py`'s
existing parsers, just ranked by remembered *world* position (relational)
or current local position (positional) instead of pixel position. A leading
imperative ("go to the ...", "find the ...", "navigate to ...") is stripped
before parsing, since the underlying parsers were built for a bare category
phrase. Resolves to exactly one `object_id` in a background thread (CLIP
inference off the GUI's redraw loop), then behaves exactly as if that ID
had been typed directly — REMIND's per-tick matching is already
`object_id == target_id`, so once resolved the rover stays locked onto that
one instance even with several same-class objects in frame.

## Navigate-back: `pipeline.py`'s `GOTO` state

If the selected object isn't currently visible but `object_map.py` has a
remembered world location for it, `remind_gui.py` hands `pipeline.step()`
an `external_goal` (the remembered point, transformed into the rover's
*current* local frame via `object_map.world_to_local`) instead of a live
detection. This puts the pipeline into a new `GOTO` state: it drives toward
that point using the exact same obstacle-guard/NavDP trajectory-selection
machinery as a live `TRACK` (so depth-based collision avoidance stays
active for the whole blind leg), but it **never self-declares `STOP`** from
proximity alone — dead-reckoning drift accumulated crossing rooms makes
trusting an odometric "arrived" unsafe. Two exit conditions instead:

- REMIND matches the object again mid-leg → drops straight back to normal
  camera-based `TRACK`/`STOP`.
- The rover reaches the remembered point (within `--goto-arrival-radius`,
  default 1 m) without reacquiring it visually → falls back to an ordinary
  `SEARCH` spin there, rather than trusting the stale point as "arrived."

Escalation into `GOTO` isn't immediate on the first missed detection —
`pipeline.py`'s own short-horizon [goal belief](../README.md#goal-belief-surviving-occlusion)
gets first crack at a dropped match (it already knows how much to still
trust the last live goal, decaying smoothly over a few seconds), and a
`--match-grace-period` window (default 1.2 s) absorbs SAM's frame-to-frame
detection flicker (its automatic point-grid masking has no cross-frame
memory of its own, so a real object can miss a tick or two from pure
grid-sampling noise) before either of those even get consulted. Only once
belief has genuinely given up (`sigma` past `belief_max_sigma`) does a
remembered `object_map` location get used at all.

## The same mechanism, reused for Go Home

`nav_pipeline/home_gui.py`'s optional `--enable-obstacle-avoidance` flag
drives its "Go Home" leg through this exact same `GOTO` state — just
pointed at a fixed home coordinate (via `object_map.world_to_local`)
instead of a remembered object. See the root README's
[Manual control + Go Home](../README.md#manual-control--go-home) section.

## Key flags (`nav_pipeline/remind_gui.py`)

| flag | default | meaning |
|---|---|---|
| `--remind-server` | `http://127.0.0.1:8765` | REMIND live server base URL |
| `--remind-period` | `0.4` s | min. interval between REMIND `/infer` calls (reused between polls otherwise) |
| `--object-map-path` | `object_map/object_map.json` | persistent world-location store |
| `--goto-arrival-radius` | `1.0` m | odometry distance counted as "arrived" before giving up on blind `GOTO` and falling back to `SEARCH` |
| `--match-grace-period` | `1.2` s | how long to coast on the last-known detection after REMIND stops matching, before treating the target as truly not visible |
| `--object-map-update-period` | `1.0` s | min. interval between passive object-memory updates (world location + CLIP embedding for every visible object, not just the driving target) |
| `--depth-encoder` | `vitb` | RGB-only metric depth model (more accurate than the `vits` default used elsewhere — depth error feeds directly into the stop decision) |
