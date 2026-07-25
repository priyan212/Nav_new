# Claude Code Prompt: Topological Teach/Repeat Navigation for Earth Sim (DINOv2 + SAM + NavDP)

Copy everything below into Claude Code as your task prompt. This is a rewrite
of a prompt originally drafted in Claude web with no repo context — it
assumed a ROS2 package, a from-scratch NavDP integration, and no prior
place-recognition/route work. None of that matches Nav_new. Read the
"Ground truth about this repo" section before writing any code — several
things this task wants already exist in some form, under a directory name
that will look off-limits at first glance.

## Task

Implement a **monocular, camera-only topological teach/repeat navigation
mode for the EARTH Habitat-Sim environment only**, based on the "view-
sequenced route representation" idea (Matsumoto et al.), adapted to this
project's existing DINO+NavDP stack:

1. **Teach phase** — drive the rover (manually, or via the existing
   `earth_gui.py` controls) along a route in the Indian Bend & Pima Habitat
   scan; build a topological graph ("tunnel") of place nodes from the camera
   feed.
2. **Repeat phase** — autonomously re-drive the route by localizing the live
   frame against the graph and generating local motion via NavDP conditioned
   on the next node's image as a subgoal.

No LiDAR, no wheel odometry dependency for mapping (the rover has none in
reality; EARTH sim's `earth/pose` ground truth exists only for debugging/eval,
never as a mapping input), no metric SLAM.

## Ground truth about this repo (verified before writing this prompt)

**Transport is Zenoh, not ROS2 nodes.** Nav_new does not use `rclpy`, ROS2
launch files, or a ROS2 node graph anywhere in its own code. It publishes/
subscribes CDR-serialized ROS message types (Twist, Image, CompressedImage,
String) directly over `eclipse-zenoh` — see [nav_pipeline/zenoh_node.py](../nav_pipeline/zenoh_node.py).
`zenoh-bridge-ros2dds` only enters the picture on the real rover, to talk to
the ESP32/micro-ROS side — irrelevant here since EARTH sim scope means no
real rover involved. **Do not build a ROS2 package, `rclpy` nodes, or launch
files.** New code should be plain Python, structured like the existing
`EARTH/scripts/habitat_sim_node.py` (sim backend) and `EARTH/scripts/earth_gui.py`
(control loop + Tkinter/PIL visualization) — optionally using the same Zenoh
topics for consistency, but a single-process/in-process design is fine too
since EARTH sim runs entirely on one workstation (no Pi/ESP32 split).

**Two existing modules are NOT what their names suggest — check before reusing:**
- `nav_pipeline/dino_detector.py` wraps **Grounding DINO** (`IDEA-Research/grounding-dino-base`),
  an open-vocabulary text→bbox detector. It is **not** DINOv2. DINOv2 (the
  self-supervised ViT used for place-recognition embeddings, which this task
  actually needs) is not vendored or installed anywhere in this repo yet —
  it's new. Name new code unambiguously (e.g. `dinov2_descriptor.py`, never
  reuse the token "dino" alone) so it isn't confused with `dino_detector.py`
  in imports, logs, or config keys.
- `nav_pipeline/sam_segmenter.py` wraps SAM2 (`facebook/sam2.1-hiera-small`)
  in **box-prompted** mode only (`segment_box(rgb, box)` — needs a bbox from
  Grounding DINO first). It does not do automatic/free whole-scene
  segmentation, so it cannot directly produce a "ground/free-space segment"
  the way the original prompt assumed. Getting a free-space mask means either
  (a) adding SAM's automatic-mask-generator mode, or (b) just reusing
  `nav_pipeline/obstacle_guard.py`'s existing depth-based footprint/swept-clearance
  guard (already does "reject a trajectory that hits a non-floor obstacle,"
  driven off Depth Anything depth, no SAM involved). (b) is very likely the
  right call — flag this and confirm before building a parallel SAM-based
  free-space check.

**A shared NavDP + belief/route library lives *inside* `MARS/`, but is
generic and already imported by EARTH.** `MARS/mars-habitatsim/navdp/navdp/`
is a vendored NavDP package with an `extensions/` folder including
`belief_bank.py` (`SubgoalBeliefBank` — persistent Gaussian belief per named
subgoal, decayed/updated from detections) and `route_manager.py`
(`RouteManager` — ordered, repeatable route pointer over named subgoals,
`route = ["A", "B", "A"]`-style). `EARTH/scripts/earth_gui.py` already
imports `SubgoalBeliefBank` from this path with the comment "cloned for MARS
... it's generic, not Mars-specific." **This means the physical location
under `MARS/` is not a reliable signal of what's MARS-only** — treat
`MARS/mars-habitatsim/navdp/` as a shared, read-only dependency, same as
`nav_pipeline/`. Per the Earth-only constraint below: do not edit anything
under `MARS/`, including this subtree, even to extend it for this feature —
if the topological graph/localizer needs something added there, stop and
report it as a shared-module change to confirm first, don't just do it.

**`RouteManager`/`SubgoalBeliefBank` already solve a version of "repeatable
route following."** It's belief-tracking over *named, semantically detected*
subgoals (via Grounding DINO + odometry-free Gaussian decay), not appearance-
embedding place recognition over a sequence of images — genuinely different
mechanism from the DINOv2-graph idea here, but overlapping purpose. Before
building a new graph/localizer/route-pointer from scratch, check how (or
whether) `RouteManager` is actually used today (grep callers — `earth_gui.py`
only imports the belief bank, not the route manager) and report whether this
new topological system should sit alongside it, replace it, or be layered on
top of it (e.g. topological localization feeding a `RouteManager`-style
pointer). Don't silently duplicate it.

**NavDP already has a trained image-goal path — but it isn't wired up for
inference yet, and it lives in a shared module.** The active checkpoint
(`policy_type="crossmodal"`, `navdp-cross-modal.ckpt`, wrapper
`nav_pipeline/navdp_crossmodal.py`) was trained with point/pixel/**image**
goal encoders (per its own docstring: "pixel goal = 4ch RGB+mask, image goal
= 6ch") and `NavDPCrossModal.__init__` already loads an `ImageGoalBackbone`
(`self.image_encoder`) from the full 1066-tensor state dict. But the only
inference methods actually exposed are `sample_pointgoal` and `sample_nogoal`
([nav_pipeline/navdp_crossmodal.py:172-185](../nav_pipeline/navdp_crossmodal.py#L172-L185))
— there is no `sample_imagegoal` yet. This is the natural hook for feeding
the topological graph's "next node" image as NavDP's subgoal instead of a
3D point, and the trained weights for it already exist — but adding that
method means editing `nav_pipeline/navdp_crossmodal.py`, which is shared
across MARS/EARTH/the real rover. **Flag this specific change and get
confirmation before editing it** — don't route around it by duplicating the
wrapper class inside EARTH.

**Depth is already solved, reuse as-is (this part of the original prompt was
right):** `nav_pipeline/depth_estimator.py` → `MetricDepthEstimator` wraps
Depth Anything V2 (ViT-S, metric/hypersim checkpoint). EARTH's sim backend
(`habitat_sim_node.py`) also publishes perfect sim depth directly
(`depth_raw`), so for EARTH specifically you may not even need the monocular
estimator in the loop — check which one `earth_gui.py`/`pipeline.py` prefer
today and stay consistent. Do not add a second depth estimator.

**Environments are split and matter for where new code runs:** `internnav`
conda env has torch 2.5.1/transformers 5.9/eclipse-zenoh and is what
`nav_pipeline` and `earth_gui.py` run in. `mars_habitat` is a separate,
constrained Python 3.9 env that only runs habitat-sim itself
(`habitat_sim_node.py`, `survey.py`) — it does not have `transformers`.
DINOv2 loading and any new torch-heavy descriptor/graph code must live in a
process running under `internnav`, not inside the `habitat_sim_node.py`
process.

## Architecture

**Node representation (per keyframe):**
- `dinov2_descriptor`: global appearance embedding from DINOv2
  (`facebookresearch/dinov2` via `torch.hub` or `transformers`), used for
  place retrieval / loop closure / localization. CLS token or patch-averaged,
  L2-normalized. New to this repo (see naming-collision note above).
- `rgb_image`: the raw keyframe (path reference into `EARTH/out/` or a new
  graph-data dir, not embedded in the JSON/SQLite blob), used as the NavDP
  subgoal image once image-goal conditioning exists (see NavDP note above).
- `sam_segment_summary` (optional, only if the automatic-mask-generator route
  from the SAM note above is chosen): lightweight labels/centroids, not full
  masks.
- `pose_hint` (optional, EARTH-only, debug/eval use only): `earth/pose`
  ground truth from the sim, purely as a retrieval tie-breaker/edge-ordering
  aid during development — never treat this as available on the real rover,
  and don't let any core logic depend on it being present.

**Graph construction (teach phase):**
- Process the teach-run frames (from `earth_gui.py`'s live camera feed, or a
  recorded sequence) at a modest rate (2–5 Hz is fine for Habitat sim).
- New node when cosine similarity to the last node drops below `T_sim`, or
  heading change exceeds `T_angle` (heading from `earth/pose` yaw in sim —
  acceptable here since EARTH sim scope permits it, but call this out
  explicitly as sim-only in code comments/config, since the real rover has no
  heading source).
- Sequence-order nodes as the "tunnel"; directed edges between consecutive
  nodes; loop-closure edges when a new node's descriptor matches an earlier
  non-adjacent node above a high threshold.
- Persist as JSON (SQLite is overkill at this graph size — keep it simple):
  node id, descriptor vector, image path, adjacent edges, edge direction.

**Localization + control (repeat phase):**
- Per live frame: DINOv2 descriptor → top-k nearest nodes (brute-force torch
  cosine similarity is plenty at this scale — no need for `faiss`).
- Short temporal window (SeqSLAM-style) to disambiguate perceptual aliasing,
  since the scan has repeated structure (parking lot cars, similar dirt
  mounds — see `EARTH/scripts/survey.py` output for what's actually in-scene).
- Next node ahead on the planned path = subgoal.
- Feed `{current RGB frame, subgoal image}` to NavDP once image-goal
  inference exists (see flagged NavDP gap above); until/unless that's
  approved, the fallback is: use the subgoal node's stored `pose_hint`
  (sim-only) as a point-goal through the existing `sample_pointgoal` path —
  make this fallback explicit in code and config so it's obvious it's a
  sim-only crutch, not the real design.
- Cross-check the chosen trajectory against the free-space signal (per the
  SAM/obstacle_guard decision above).
- Convert to `cmd_vel`, publish over the existing Zenoh contract
  (`nav_pipeline/zenoh_node.py`'s `serialize_twist`), same as `earth_gui.py`
  does today — do not invent a new command channel.
- Advance to the next node once live-descriptor similarity to the current
  subgoal exceeds an arrival threshold, or (once wired) NavDP's critic says
  reached.

## Deliverables

1. New Python module(s) under `EARTH/` (e.g. `EARTH/topo_nav/` or
   `EARTH/scripts/topo_*.py`, matching the flat-script style already used by
   `EARTH/scripts/`) — not a ROS2 package:
   - a graph builder (teach phase),
   - a DINOv2 descriptor wrapper (`dinov2_descriptor.py`),
   - a localizer (retrieval + sequence matching + path-over-graph, reusing
     plain BFS/Dijkstra — no new dependency needed for a graph this size),
   - a repeat-phase control loop that plugs into the existing
     `DinoNavDPPipeline`/`EarthPipeline` (`nav_pipeline/pipeline.py`,
     `EARTH/scripts/earth_gui.py`) rather than reimplementing detection/guard/
     command-shaping from scratch.
2. Two CLI modes (`--mode teach` / `--mode repeat`) on a single entry script,
   following `earth_gui.py`'s `argparse` pattern — no launch files.
3. Config for checkpoint paths, thresholds (`T_sim`, `T_angle`, arrival
   threshold, sequence-window length), following the existing style in
   `configs/*.yaml`.
4. Visualization: extend `earth_gui.py`'s existing Tkinter/PIL overlay
   (it already renders a live camera view + goal star) to also show the
   graph, current localization match, and retrieved subgoal image — no RViz2,
   there's no ROS2 desktop tooling in this stack.
5. Smoke tests following the existing convention in `scripts/` (plain
   Python/pytest scripts against recorded frame sequences, e.g. modeled on
   `scripts/test_pipeline_offline.py`) — this repo has no rosbag recording
   infrastructure, so don't introduce one for this.

## Constraints & environment

- **Scope: EARTH sim only.** New packages/configs/scripts go under `EARTH/`.
  Do not modify `MARS/` (including `MARS/mars-habitatsim/navdp/`, which is
  imported by EARTH but physically lives there — see landmine above) or
  anything hardware-facing (`esp32/`, `scripts/pi_*.sh`, `launch_rover.sh`),
  even where it looks incidentally related.
- `nav_pipeline/` is shared across MARS, EARTH, and the real rover. Reuse its
  classes (`DinoNavDPPipeline`, `MetricDepthEstimator`, `Sam2Segmenter`,
  `NavDPStandalone`/`NavDPCrossModal`, `obstacle_guard`, `zenoh_node`)
  read-only. If a change inside `nav_pipeline/` genuinely seems required
  (the NavDP image-goal method is the known candidate), stop and flag it with
  the specific file/function instead of editing it — get that confirmed
  before touching a module the other two environments depend on.
- Depth: reuse the existing pipeline's depth source as-is (see note above) —
  do not add a second depth estimator.
- Runs entirely on the workstation (no Pi/ESP32 in this loop) — target the
  `internnav` conda env for anything torch/transformers-based; `mars_habitat`
  only runs the sim backend itself.
- Keep DINOv2 (and SAM, if the automatic-mask route is chosen) loaded once,
  not per frame.
- Log per-stage timing (descriptor extraction, NavDP inference, control loop)
  the same way the existing pipeline already logs stage timing, so latency
  is comparable across the two.

## What to do first

1. Confirm the `MARS`/EARTH boundary as described above — in particular,
   confirm current usage (or non-usage) of `RouteManager` across the repo
   (`grep -rn RouteManager`), and report back whether this task should
   integrate with it rather than building a parallel route/graph mechanism.
2. Inspect `nav_pipeline/navdp_crossmodal.py`'s `ImageGoalBackbone` /
   `sample_pointgoal` to confirm exactly what would be needed to add
   image-goal inference, and report back the concrete diff before making it
   — this is the one part of this task that touches a shared module.
3. Decide and report on the free-space check: automatic SAM segmentation
   (new) vs. reusing `obstacle_guard.py`'s existing depth-based guard —
   don't build both.

Only after these are confirmed, start on the teach-phase graph builder.
