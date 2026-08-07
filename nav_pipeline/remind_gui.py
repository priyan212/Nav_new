#!/usr/bin/env python3
"""REMIND + NavDP rover GUI -- Nav_new's isaac_gui.py, retargeted to drive by
persistent object ID instead of a bare text phrase.

Same control panel as nav_pipeline.isaac_gui (camera feed, top-down NavDP
trajectory plot, state/velocity readout, manual drive, STOP), but:

- Target selection is ID-ONLY: this sends every camera frame to a REMIND
  live-tracking server (REMIND/remind-reid-tracker, a separate process/
  conda env -- see remind_client.py and launch_rover_remind.sh) and
  overlays every currently-tracked object with just "ID <n>" (REMIND's own
  BLIP caption is kept internally -- see object_map.py -- but never shown;
  the operator only ever needs the number). Type an ID back (or
  double-click it in the "known objects" list, which lists every object
  ever remembered, not just ones currently in frame) -- see
  remind_target.parse_object_target.
- Object-location memory: every tick, every currently-visible REMIND object
  (not just the driving target) gets its world-frame location updated in
  object_map.py, using the rover's own continuous odometry pose (see
  odometry_logger.py -- pose no longer resets per goal, precisely so this
  means something across goals/rooms).
- Navigate-back: if the selected target ID isn't currently visible but a
  remembered world location exists for it, the rover drives there BLIND
  (odometry-only waypoint, live depth-based obstacle avoidance still
  active) via pipeline.py's GOTO state, switching back to normal
  camera-based TRACK/STOP the moment REMIND matches it again. If it arrives
  at the remembered spot without reacquiring it visually, falls back to an
  ordinary search spin there rather than trusting dead-reckoning as
  "arrived."
- Depth: RGB-only metric depth via Depth Anything V2 ViT-B by default (more
  accurate than the vits default used elsewhere; depth error feeds directly
  into the STOP distance decision -- see depth_estimator.py).

Run (from Nav_new root, after the REMIND live server is up -- see
launch_rover_remind.sh, which brings up both):
    conda activate internnav
    python -m nav_pipeline.remind_gui --pi-ip <IP> --remind-server http://127.0.0.1:8765
"""

import argparse
import colorsys
import os
import signal
import sys
import time
from threading import Lock, Thread
from typing import List, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.dino_detector import Detection  # noqa: E402
from nav_pipeline.goal_utils import intrinsics_from_fov, pixel_depth_to_point  # noqa: E402
from nav_pipeline.isaac_gui import (  # noqa: E402
    DEPTH_STALE_S,
    HEARTBEAT_PERIOD_S,
    SPIN_DIST_THRESH_M,
    SPIN_ROT_THRESH_RAD,
    SPIN_WINDOW_S,
    heartbeat_loop,
    zenoh_setup,
)
from nav_pipeline.object_map import ObjectMap, local_to_world, world_to_local  # noqa: E402
from nav_pipeline.object_query import ClipObjectMatcher, resolve_object_query  # noqa: E402
from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.remind_client import RemindClient, RemindObject  # noqa: E402
from nav_pipeline.remind_target import parse_object_target  # noqa: E402
from nav_pipeline.sam_segmenter import mask_centroid, mask_median_depth  # noqa: E402
from nav_pipeline.zenoh_node import serialize_path, serialize_string, serialize_twist  # noqa: E402


def _color_for_id(object_id: Optional[int]) -> tuple:
    """Deterministic per-ID color (RGB) so the same object keeps the same
    box color tick to tick -- matches REMIND's own render_frame convention
    (REMIND/remind-reid-tracker/scripts/run_video_tracking.py's
    _color_for_id) enough to feel consistent, without importing across the
    two separate environments."""
    if object_id is None:
        return (150, 150, 150)
    hue = (int(object_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


class SharedState:
    def __init__(self, target: str):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.frame_count = 0
        self.mode = "manual"                    # "text" | "manual" -- starts inert
        self.target = target                     # display string, e.g. "ID 1"
        self.target_id: Optional[int] = None
        self.stopped = False
        self.goal_reached = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.5
        self.max_angular = 0.6
        # for display
        self.display_rgb: Optional[np.ndarray] = None
        self.detection = None
        self.mask: Optional[np.ndarray] = None
        self.remind_objects: List[RemindObject] = []
        self.remind_ok = True
        self.state_text = "waiting for camera"
        self.vel_text = "lin 0.000  ang +0.000"
        self.lat_text = ""
        self.trajs = None
        self.chosen = None
        self.goal_pt = None
        self.obstacles = None
        self.min_forward = float("inf")
        self.infer_count = 0

        if target:
            parsed = parse_object_target(target)
            if parsed is not None:
                self.target_id = parsed
                self.mode = "text"


def remind_poll_loop(remind: RemindClient, st: SharedState, running, remind_period_s: float = 0.4):
    """Runs independently of the nav control tick, in its own thread.

    REMIND's own latency (~0.2-0.5s measured on an RTX 3090 Ti; see
    REMIND_METHOD.md) is heavier than the nav loop's tick budget -- calling
    it synchronously from remind_inference_loop put that latency directly
    on the control loop's critical path (measured: dropped the effective
    loop rate from the requested ~2.5 Hz to ~1.5-1.9 Hz, eroding the
    obstacle-guard confirm-tick timing margin). This thread just keeps
    st.remind_objects fresh at REMIND's own achievable rate; the nav loop
    always reads whatever's latest instead of waiting on it.
    """
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
        if rgb is not None:
            try:
                objects = remind.infer(rgb)
                with st.lock:
                    st.remind_objects = objects
                    st.remind_ok = True
            except Exception as e:
                print(f"[WARN] REMIND inference failed: {e}")
                with st.lock:
                    st.remind_ok = False
        dt = remind_period_s - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def _update_object_map(objects: List[RemindObject], depth: Optional[np.ndarray],
                       pose: Optional[tuple], object_map: Optional[ObjectMap], fov_deg: float,
                       rgb: Optional[np.ndarray] = None,
                       clip_matcher: Optional[ClipObjectMatcher] = None) -> None:
    """Fold this tick's REMIND observations into the persistent world-location
    map -- every object currently in view with a usable mask+depth, not just
    the one being driven to, so the "known objects" list accumulates
    locations for objects the operator never explicitly targeted.

    Also caches a CLIP image embedding the first time each object_id is seen
    (object_map.set_embedding is itself a no-op past the first call) --
    that's what free-text queries (object_query.py) get matched against
    instead of BLIP's caption text."""
    if object_map is None or depth is None or pose is None:
        return
    H, W = depth.shape[:2]
    fx, fy, cx, cy = intrinsics_from_fov(W, H, fov_deg)
    now = time.time()
    for o in objects:
        if o.object_id is None or o.mask is None:
            continue
        if o.mask.shape[:2] != (H, W):
            continue
        d = mask_median_depth(depth, o.mask)
        if d is None:
            continue
        u, v = mask_centroid(o.mask)
        local = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
        world_xy = local_to_world((float(local[0]), float(local[1])), pose)
        object_map.update(o.object_id, o.class_name, world_xy, now)
        if (clip_matcher is not None and rgb is not None
                and not object_map.has_embedding(o.object_id)):
            try:
                emb = clip_matcher.embed_crop(rgb, o.bbox, mask=o.mask)
                if emb is not None:
                    object_map.set_embedding(o.object_id, emb)
            except Exception as e:
                print(f"[WARN] CLIP embedding failed for object {o.object_id}: {e}")


def remind_inference_loop(pipe: DinoNavDPPipeline, st: SharedState, pubs,
                          running, predict_hz: float,
                          stop_confirm: int = 3, odom: Optional[OdometryLogger] = None,
                          object_map: Optional[ObjectMap] = None,
                          goto_arrival_radius: float = 1.0,
                          match_grace_period_s: float = 1.2,
                          clip_matcher: Optional[ClipObjectMatcher] = None,
                          object_map_update_period_s: float = 1.0):
    period = 1.0 / predict_hz
    stop_streak = 0
    last_target_id: Optional[int] = None
    last_map_update_t = 0.0
    # (timestamp, RemindObject) of the last tick REMIND actually matched the
    # current target -- SAM's grid-point automatic mask generation has no
    # cross-frame memory of its own (see detection/sam_segmenter.py), so a
    # real object can drop out for a tick or two purely from grid-sampling
    # noise (camera jitter, a borderline confidence/texture score) and
    # reappear immediately after. Without this, that single missed tick
    # flipped "matched" to empty and kicked the pipeline into GOTO/SEARCH
    # every time -- visibly flickering on screen and, worse, cycling the
    # rover's driving mode on every drop-out instead of just coasting
    # through it (pipeline.py's own belief mechanism already smooths the
    # DRIVING COMMAND during a loss, but that's downstream of this -- it
    # never got the chance to run because `matched` itself was flapping).
    last_seen_target: Optional[tuple] = None

    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            mode = st.mode
            target_id, target_text = st.target_id, st.target
            paused = st.stopped or st.goal_reached
            last_objects = st.remind_objects  # kept fresh by remind_poll_loop

        if target_id is not None and target_id != last_target_id:
            if last_target_id is not None:
                # new goal: don't let tracked-box/goal-belief state from the
                # PREVIOUS target leak into this one
                pipe.reset()
            if odom is not None:
                # reset_pose deliberately NOT passed (defaults False): pose
                # stays continuous across goals -- see odometry_logger.py's
                # module docstring and object_map.py, both of which depend
                # on it not resetting here.
                odom.start_new_goal(target_text)
            last_target_id = target_id
            last_seen_target = None  # don't let the PREVIOUS target's grace window leak into this one

        if rgb is None:
            time.sleep(0.1)
            continue

        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None

        # Passive object-memory building runs on every tick a frame exists,
        # regardless of mode/target/paused state below -- deliberately not
        # inside any of those branches' gates, so an operator who just drives
        # around manually (or hasn't sent a target yet) still builds up the
        # map free-text targeting depends on, instead of it only ever running
        # once a target was already active (a chicken-and-egg deadlock: you
        # can't resolve "go to the chair" until something has an embedding
        # cached, but that used to only get cached AFTER a target was sent).
        #
        # Throttled to object_map_update_period_s (objects don't move, so
        # sub-second freshness buys nothing) rather than running every loop
        # iteration: MANUAL mode's branch below only sleeps 0.05s (~20Hz),
        # and unthrottled this was running full RGB-only depth estimation
        # (Depth Anything V2 -- a real model pass, not cheap) plus a CLIP
        # embed_crop() per visible object on every one of those ~20
        # iterations/sec, which is what actually caused the latency spike --
        # manual driving used to reach its cheap branch almost immediately,
        # before any of this existed.
        if (t0 - last_map_update_t) >= object_map_update_period_s:
            last_map_update_t = t0
            map_depth = depth
            if map_depth is None and pipe.depther is not None:
                # No physical depth sensor on the real rover -- pipe.step()
                # below already falls back to this same RGB-only estimate
                # internally (pipeline.py's _step_inner) whenever it's
                # handed depth=None. Compute it here and hand the SAME array
                # to both that call and _update_object_map (assigned back
                # into `depth` below) so they stay consistent and it's not
                # estimated twice on ticks where navigation ALSO needs it.
                map_depth = pipe.depther.estimate(rgb)
                depth = map_depth
            pose = (odom.x, odom.y, odom.theta) if odom is not None else None
            _update_object_map(last_objects, map_depth, pose, object_map, pipe.cfg.horizontal_fov_deg,
                              rgb=rgb, clip_matcher=clip_matcher)
        else:
            pose = (odom.x, odom.y, odom.theta) if odom is not None else None

        if mode == "manual":
            with st.lock:
                if st.stopped:
                    st.last_cmd = (0.0, 0.0)
                lin, ang = st.last_cmd
                st.display_rgb = rgb
                st.state_text = "MANUAL (stopped)" if st.stopped else "MANUAL DRIVE"
                st.vel_text = f"lin {lin:.3f}  ang {ang:+.3f}"
            time.sleep(0.05)
            continue

        if paused or target_id is None:
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.vel_text = "lin 0.000  ang +0.000"
                if target_id is None:
                    st.state_text = "waiting for target, e.g. 'ID 1'"
            time.sleep(0.1)
            continue

        matched = [o for o in last_objects if o.object_id == target_id]
        if matched:
            last_seen_target = (t0, matched[0])
        elif last_seen_target is not None and (t0 - last_seen_target[0]) <= match_grace_period_s:
            # brief drop-out within the grace window -- coast on the frozen
            # last-known detection instead of treating this tick as "not
            # currently visible" (see the loop's docstring comment above).
            matched = [last_seen_target[1]]
        external_dets: Optional[list] = []
        external_goal = None
        goto_dist = None
        if matched:
            det = Detection(box=matched[0].bbox, score=max(matched[0].confidence, 0.01), label=matched[0].label)
            # REMIND already segmented this object (its own SAM
            # backend) -- attach the mask so pipeline.py's external_dets
            # branch reuses it instead of running a second SAM2 pass (see
            # pipeline.py's goal computation, `getattr(det, "mask", None)`).
            det.mask = matched[0].mask
            external_dets = [det]
        else:
            # Not currently visible past the grace window. Give pipeline.py's
            # own short-horizon belief (GoalBelief -- ego-motion-propagated
            # 3D goal, see pipeline.py's module docstring/PipelineConfig.
            # use_belief_goal) first crack at this: it already tracks exactly
            # "how much do I still trust the last live LOCAL goal", decaying
            # smoothly over a few seconds of occlusion/turning. Committing to
            # a blind cross-room GOTO leg the instant REMIND drops the match
            # for one tick past the grace window skipped belief entirely --
            # it never got a chance to run, so it looked "dead" even though
            # nothing was wrong with it. Only escalate to GOTO once belief
            # has genuinely given up (sigma blew past belief_max_sigma, or
            # this goal never got a live detection at all) -- read pipe's
            # OWN belief state from the previous tick (pipe.step() below is
            # what mutates it) rather than duplicating its math here.
            belief_confident = (pipe.cfg.use_belief_goal and pipe.belief.initialized
                                and pipe.belief.sigma <= pipe.cfg.belief_max_sigma)
            if not belief_confident and object_map is not None and pose is not None:
                entry = object_map.get(target_id)
                if entry is not None:
                    lx, ly = world_to_local((entry["world_x"], entry["world_y"]), pose)
                    goto_dist = float(np.hypot(lx, ly))
                    if goto_dist > goto_arrival_radius:
                        # belief already gave up AND we remember roughly
                        # where this object is (possibly from a previous
                        # room) -- drive there blind (obstacle guard stays
                        # active the whole way; see pipeline.py's GOTO
                        # state). external_dets=None (not []) is what
                        # selects this path -- see pipeline.step()'s
                        # docstring.
                        external_dets = None
                        external_goal = np.array([lx, ly, 0.0], dtype=np.float32)
                    # else: arrived at the remembered spot but can't see it --
                    # external_dets stays [] (its default above), which puts
                    # pipeline.py into its normal SEARCH spin so the rover
                    # looks around here instead of trusting a possibly
                    # drift-stale point as "close enough."
            # else: belief_confident, or no remembered location yet --
            # external_dets stays [] (its default above), letting pipe.step()
            # run its normal belief-coast/SEARCH path below.

        try:
            res = pipe.step(rgb, f"remembered object {target_id}", depth=depth, pose=pose,
                            external_dets=external_dets, external_goal=external_goal)
        except Exception as e:
            print(f"[ERROR] pipeline step: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = f"ERROR: {e}"
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.5)
            continue

        spin = odom.spin_delta(SPIN_WINDOW_S) if odom is not None else None
        if spin is not None and spin[0] > SPIN_ROT_THRESH_RAD and spin[1] < SPIN_DIST_THRESH_M:
            print(f"[WARN] spin-stall watchdog: turned {spin[0]:.1f}rad in {SPIN_WINDOW_S:.0f}s, "
                  f"only {spin[1]:.2f}m net travel -- forcing stop until a new target is sent")
            with st.lock:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = (f"SPIN STALL: {np.degrees(spin[0]):.0f}deg turned, "
                                 f"{spin[1]:.2f}m travel -- send a new target")
                st.vel_text = "lin 0.000  ang +0.000"
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            continue

        if res.state == "STOP":
            stop_streak += 1
        else:
            stop_streak = 0
        reached = stop_streak >= stop_confirm

        with st.lock:
            st.display_rgb = rgb
            st.detection = res.detection
            st.mask = res.mask
            st.trajs = res.all_trajectories
            st.chosen = res.trajectory
            st.goal_pt = res.goal_point
            st.obstacles = res.obstacle_points
            st.min_forward = res.min_forward
            st.infer_count += 1
            if reached:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.state_text = f"GOAL REACHED: '{target_text}' (stopped at {pipe.cfg.stop_distance:.1f}m)"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                if matched:
                    st.state_text = res.state
                elif res.state == "GOTO" and goto_dist is not None:
                    st.state_text = f"GOTO -- driving to remembered location of ID {target_id} ({goto_dist:.1f}m)"
                elif res.state == "TRACK":
                    st.state_text = (f"TRACK (coasting on belief, 'ID {target_id}' not currently visible, "
                                     f"sigma={pipe.belief.sigma:.2f})")
                else:
                    st.state_text = f"{res.state} ('ID {target_id}' not currently visible)"
                st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v * 1000:.0f}ms" for k, v in res.timing.items())

        if reached:
            pubs["explain"].put(serialize_string(f"GOAL REACHED: '{target_text}'. Stopping."))
        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        pubs["explain"].put(serialize_string(
            f"REMIND+NavDP [{res.state}] target='{target_text}' -> lin={res.linear:.3f} ang={res.angular:.3f}"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448
    PLOT_SIZE = 448
    PLOT_RANGE = 3.5

    def __init__(self, root: tk.Tk, st: SharedState, remind: RemindClient, object_map: Optional[ObjectMap] = None,
                 clip_matcher: Optional[ClipObjectMatcher] = None, odom: Optional[OdometryLogger] = None):
        self.root = root
        self.st = st
        self.remind = remind
        self.object_map = object_map
        self.clip_matcher = clip_matcher
        self.odom = odom
        root.title("Nav_new — REMIND + NavDP")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)
        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        known = ttk.Frame(main)
        known.grid(row=0, column=2, padx=4, pady=4, sticky="ns")
        ttk.Label(known, text="Known objects (double-click to target):").pack(anchor="w")
        self.known_list = tk.Listbox(known, width=28, height=22)
        self.known_list.pack(fill="y", expand=True)
        self.known_list.bind("<Double-Button-1>", self._on_known_double_click)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="Target (ID or 'go to the chair near the window'):").pack(side="left")
        self.id_entry = ttk.Entry(bar, width=32)
        if st.target_id is not None:
            self.id_entry.insert(0, str(st.target_id))
        self.id_entry.pack(side="left", padx=4)
        self.id_entry.bind("<Return>", lambda e: self.send_target())

        ttk.Button(bar, text="Send", command=self.send_target).pack(side="left", padx=(8, 2))
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)
        ttk.Button(bar, text="Reset REMIND memory", command=self.reset_memory).pack(side="left", padx=10)
        ttk.Button(bar, text="Forget locations", command=self.forget_locations).pack(side="left", padx=10)

        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                width=110, anchor="w")
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=110, anchor="w")
        self.info.grid(row=4, column=0, columnspan=3, sticky="w")
        self.imu_status = ttk.Label(main, text="", width=110, anchor="w")
        self.imu_status.grid(row=5, column=0, columnspan=3, sticky="w")

        self._photo = None
        self._known_ids: List[str] = []
        self.root.after(66, self.refresh)

    def send_target(self):
        """Reads the entry field. A bare number sets st.target_id directly
        (instant, no model call -- unchanged from before). Anything else is
        a free-text query ("go to the black chair", "chair near the
        window") resolved against object_map's cached CLIP embeddings in a
        background thread (object_query.resolve_object_query) -- CLIP
        inference shouldn't block the GUI's redraw loop. Once resolved, the
        result is applied exactly like typing the ID directly: from then on
        REMIND only matches that one object_id (see remind_inference_loop),
        so it stays locked onto the one instance the query picked even if
        other same-class objects are in frame."""
        text = self.id_entry.get().strip()
        if not text:
            with self.st.lock:
                self.st.state_text = "enter a target ID or a description"
            return
        try:
            target_id = int(text)
        except ValueError:
            target_id = None
        if target_id is not None:
            self._apply_target(target_id, f"ID {target_id}")
            return
        if self.clip_matcher is None or self.object_map is None:
            with self.st.lock:
                self.st.state_text = "text targeting unavailable (CLIP matcher not loaded)"
            return
        with self.st.lock:
            self.st.state_text = f"resolving '{text}'..."

        def _do():
            pose = (self.odom.x, self.odom.y, self.odom.theta) if self.odom is not None else None
            try:
                result = resolve_object_query(text, self.object_map, self.clip_matcher, pose=pose)
            except Exception as e:
                with self.st.lock:
                    self.st.state_text = f"query resolution failed: {e}"
                return
            if result.object_id is None:
                with self.st.lock:
                    self.st.state_text = result.message
                return
            self.root.after(0, lambda: self._apply_target(
                result.object_id, f"ID {result.object_id}",
                note=result.message + (" -- ambiguous, picked closest/best match" if result.ambiguous else "")))
        Thread(target=_do, daemon=True).start()

    def _apply_target(self, target_id: int, canonical: str, note: str = ""):
        self._manual_held.clear()
        self._set_fields(target_id)
        with self.st.lock:
            self.st.mode = "text"
            self.st.target_id = target_id
            self.st.target = canonical
            self.st.stopped = False
            self.st.goal_reached = False
            if note:
                self.st.state_text = note

    def _set_fields(self, object_id: int):
        self.id_entry.delete(0, "end")
        self.id_entry.insert(0, str(object_id))

    def _on_known_double_click(self, _event):
        sel = self.known_list.curselection()
        if not sel:
            return
        label = self._known_ids[sel[0]]  # e.g. "ID 1"
        parsed = parse_object_target(label)
        if parsed is None:
            return
        self._set_fields(parsed)
        self.send_target()

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

    def reset_memory(self):
        def _do():
            try:
                self.remind.reset()
                with self.st.lock:
                    self.st.state_text = "REMIND memory reset -- re-explore to rebuild the catalogue"
            except Exception as e:
                with self.st.lock:
                    self.st.state_text = f"REMIND reset failed: {e}"
        Thread(target=_do, daemon=True).start()

    def forget_locations(self):
        """Wipe the remembered world-location map (object_map.py) -- use
        this if the rover has been physically picked up/moved since the map
        was written, since there's no automatic way to detect that the
        stored world coordinates are now stale relative to a discontinuous
        pose jump. Does NOT touch REMIND's own identity catalogue (that's
        reset_memory above) or the current odometry pose."""
        if self.object_map is not None:
            self.object_map.clear()
        with self.st.lock:
            self.st.state_text = "forgot all remembered object locations"

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str):
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str):
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self):
        with self.st.lock:
            lin = 0.0
            ang = 0.0
            if "fwd" in self._manual_held:
                lin += self.st.max_linear
            if "back" in self._manual_held:
                lin -= 0.5 * self.st.max_linear
            if "left" in self._manual_held:
                ang += self.st.max_angular
            if "right" in self._manual_held:
                ang -= self.st.max_angular
            self.st.mode = "manual"
            self.st.stopped = False
            self.st.goal_reached = False
            self.st.last_cmd = (lin, ang)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    # ------------------------------------------------------------------ #
    def refresh(self):
        if self.closed:
            return
        with self.st.lock:
            rgb = self.st.latest_rgb if self.st.latest_rgb is not None else self.st.display_rgb
            det = self.st.detection
            mask = self.st.mask
            objects = list(self.st.remind_objects)
            remind_ok = self.st.remind_ok
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers, target = self.st.frame_count, self.st.infer_count, self.st.target
            drive_mode = self.st.mode
            stopped = self.st.stopped
            target_id = self.st.target_id

        if rgb is not None:
            frame = rgb
            if mask is not None and mask.shape[:2] == rgb.shape[:2]:
                frame = rgb.copy()
                frame[mask] = (0.55 * frame[mask] + 0.45 * np.array([0, 255, 60])).astype(np.uint8)
            img = Image.fromarray(frame).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            d = ImageDraw.Draw(img)
            for o in objects:
                is_target = (o.object_id == target_id)
                color = (0, 255, 60) if is_target else _color_for_id(o.object_id)
                x0, y0, x1, y1 = o.bbox
                d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=color, width=3 if is_target else 2)
                d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), o.label, fill=color)
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        # known-objects list: every object_id ever remembered (object_map.py),
        # NOT just ones currently in frame -- that's what makes navigate-back
        # possible: an object seen in a previous room is still selectable
        # here after the rover has since left it. "(visible)" marks IDs
        # REMIND is matching this tick; that's a status flag, not a name --
        # the label is always just the ID (see remind_gui.py's module
        # docstring: BLIP captions are internal bookkeeping only). Only
        # rebuild when the content actually changes, and restore the
        # selection afterward -- refresh() runs every 66ms, well inside a
        # double-click gesture's ~300-500ms window, so an unconditional
        # delete()+insert() every tick wiped selection state and could shift
        # indices mid-click if the list's order shifted between polls
        # (double-click then acted on a different row than the one clicked).
        # The "REMIND server unreachable" state is shown in the info line
        # below instead of as a fake extra row here -- inserting it into this
        # listbox at index 0 previously shifted every real row's index by
        # one relative to self._known_ids, an off-by-one on top of the above.
        visible_ids = {o.object_id for o in objects if o.object_id is not None}
        known_ids_all = list(self.object_map.all_ids()) if self.object_map is not None else []
        for oid in visible_ids:
            if oid not in known_ids_all:
                known_ids_all.append(oid)
        known_ids_all.sort()
        new_known_ids = [f"ID {oid}" + (" (visible)" if oid in visible_ids else "") for oid in known_ids_all]
        if new_known_ids != self._known_ids:
            sel = self.known_list.curselection()
            selected_label = self._known_ids[sel[0]] if sel and sel[0] < len(self._known_ids) else None
            self.known_list.delete(0, "end")
            for label in new_known_ids:
                self.known_list.insert("end", label)
            self._known_ids = new_known_ids
            if selected_label is not None and selected_label in new_known_ids:
                self.known_list.selection_set(new_known_ids.index(selected_label))

        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

        def to_px(x, y):
            return S / 2 - (y / R) * (S / 2), S - (x / R) * S * 0.92 - 20

        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")
        if obstacles is not None and len(obstacles):
            for ox, oy in obstacles[:: max(1, len(obstacles) // 400)]:
                px, py = to_px(ox, oy)
                self.plot.create_rectangle(px - 1, py - 1, px + 1, py + 1, fill="#8a8a8a", outline="")
        if trajs is not None:
            for t in trajs:
                pts = [to_px(p[0], p[1]) for p in t[::2]]
                self.plot.create_line(*[c for xy in pts for c in xy], fill="#cccccc")
        if chosen is not None:
            pts = [to_px(p[0], p[1]) for p in chosen]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)
        if goal is not None:
            gx, gy = to_px(goal[0], goal[1])
            self.plot.create_text(gx, gy, text="★", fill="#d4a017", font=("TkDefaultFont", 22))

        mode_txt = "STOPPED" if stopped else state_text
        target_txt = "manual drive" if drive_mode == "manual" else f"'{target}'"
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  target: {target_txt}   {vel_text}{fwd}")
        remind_txt = "" if remind_ok else "  [REMIND UNREACHABLE]"
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}{remind_txt}")

        if self.odom is not None:
            heading = self.odom.last_imu_heading_deg
            heading_txt = f"{heading:.1f}deg" if heading is not None else "n/a"
            calib_txt = OdometryLogger.decode_calib(self.odom.last_imu_calib)
            if self.odom.last_imu_calib is None:
                self.imu_status.configure(text="IMU: no data received yet (check /rover/rpm carries a 4th value)")
            elif self.odom.is_imu_calibrated():
                self.imu_status.configure(
                    text=f"IMU: calibrated [{calib_txt}]  heading {heading_txt}  (theta source: {self.odom.theta_source})")
            else:
                self.imu_status.configure(
                    text=f"⚠ IMU: NOT calibrated [{calib_txt}] -- tilt/rotate the rover until MAG >= "
                         f"{self.odom.imu_min_mag_calib} (currently driving theta from wheel encoders only)")

        self.root.after(66, self.refresh)


def main():
    ap = argparse.ArgumentParser(description="Nav_new REMIND+NavDP rover GUI")
    ap.add_argument("--target", default="",
                    help="initial target as 'ID N', e.g. 'ID 1' -- starts empty "
                         "(manual drive) until a target is sent from the GUI")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--remind-server", default="http://127.0.0.1:8765",
                    help="base URL of the REMIND live server (see launch_rover_remind.sh)")
    ap.add_argument("--remind-period", type=float, default=0.4,
                    help="minimum seconds between REMIND inference calls; the nav loop reuses "
                         "the last response in between instead of blocking on every tick")
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 1.2 on the real rover)")
    ap.add_argument("--search-angular", type=float, default=0.15)
    ap.add_argument("--servo-ramp-deg", type=float, default=35.0)
    ap.add_argument("--angular-slew-max", type=float, default=0.10)
    ap.add_argument("--invert-angular", action="store_true")
    ap.add_argument("--no-belief-goal", action="store_true")
    ap.add_argument("--stop-distance", type=float, default=1.5,
                    help="meters from the object at which to stop (depth-based)")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vitb",
                    help="RGB-only metric depth model (no depth sensor on the real rover); "
                         "defaults to vitb here since depth error feeds directly into the "
                         "STOP distance decision -- needs checkpoints/depth_anything_v2_"
                         "metric_hypersim_vitb.pth (scripts/download_models.py --depth-encoder vitb)")
    ap.add_argument("--compressed-only", action="store_true")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    ap.add_argument("--object-map-path", type=str, default="object_map/object_map.json",
                    help="persistent id->world-location store (object_map.py); survives GUI "
                         "restarts within the same room/building, but is NOT safe to trust "
                         "across a power cycle or hand-carrying the rover -- delete it (or use "
                         "the GUI's 'Forget locations' button) if that happened")
    ap.add_argument("--goto-arrival-radius", type=float, default=1.0,
                    help="meters: how close (by odometry) counts as 'arrived' at a remembered "
                         "but not-currently-visible object's location before giving up on blind "
                         "GOTO driving and falling back to a search spin (see pipeline.py's GOTO "
                         "state)")
    ap.add_argument("--match-grace-period", type=float, default=1.2,
                    help="seconds: how long to keep coasting on the last-known detection after "
                         "REMIND stops matching the target, before treating it as truly not "
                         "visible -- smooths over SAM's frame-to-frame detection dropouts (see "
                         "remind_inference_loop's docstring comment) instead of flickering "
                         "between visible/not-visible and cycling the rover's driving mode")
    ap.add_argument("--object-map-update-period", type=float, default=1.0,
                    help="seconds: minimum interval between passive object-memory updates "
                         "(world location + CLIP embedding caching for every visible object, not "
                         "just the driving target). Runs regardless of mode/target, so on the real "
                         "rover (no depth sensor) it pays for an RGB-only depth pass each time -- "
                         "throttled well below the control loop's own rate since objects don't move "
                         "fast enough for sub-second freshness to matter, and MANUAL mode's tighter "
                         "loop would otherwise pay this cost ~20x/sec")
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length,
                    help="robot length (m) for obstacle_guard's swept-footprint clearance -- "
                         "defaults to the ESP32 rover's real size, override for a different robot "
                         "(e.g. the LanderPi, see landerpi/README.md) before trusting obstacle avoidance")
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width,
                    help="robot width (m), see --footprint-length")
    args = ap.parse_args()

    print(f"[INFO] checking REMIND server at {args.remind_server} ...")
    remind = RemindClient(args.remind_server)
    if not remind.health():
        print(f"[ERROR] REMIND server not reachable at {args.remind_server} -- "
              f"start it first (see launch_rover_remind.sh)")
        sys.exit(1)
    print("[INFO] REMIND server OK")

    print("[INFO] loading CLIP object-query matcher...")
    clip_matcher = ClipObjectMatcher(device=args.device)

    print("[INFO] loading navigation models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(args.search_angular, args.max_angular),
        servo_ramp_deg=args.servo_ramp_deg,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        use_belief_goal=not args.no_belief_goal,
        depth_encoder=args.depth_encoder,
        stop_distance=args.stop_distance,
        guard=GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width),
        # REMIND already provides persistent per-object identity; the
        # pipeline's own single-target DINOv2 appearance re-lock (tuned for
        # raw multi-candidate DINO streams) is redundant here and is fully
        # bypassed anyway by the external_dets hook (see pipeline.py).
        use_appearance_reid=False,
        # Same reasoning: REMIND's own SAM backend already segments every
        # detection as part of its own pipeline, and the mask is forwarded
        # through external_dets (see remind_client.RemindObject.mask) --
        # a second SAM2 pass here would just recompute the same thing. CLIP
        # verification is dropped too: identity here is already established
        # by REMIND's own DINOv3-based re-ID, not by target_text (which is
        # a BLIP caption, not a fixed class label) -- CLIP would only be
        # re-checking a weaker, class-based notion of "is this the target"
        # against a system that no longer works that way.
        use_sam=False,
        use_clip=False,
        # The periodic scene inventory (scene_log/) runs a separate
        # Grounding DINO pass over a large vocabulary purely for offline
        # logging -- it's not consulted by any navigation decision, and
        # REMIND's own catalogue already captures a strictly richer version
        # of the same information. Measured cost: this was the single
        # remaining latency spike on the nav loop (~170-285ms once a
        # second, pushing p95 to ~420ms); disabling it flattens the loop to
        # a steady ~222ms/tick with no navigation-relevant loss.
        use_scene_tagger=False,
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    odom = OdometryLogger(args.odometry_log_dir)
    object_map = ObjectMap(args.object_map_path)
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=remind_poll_loop, args=(remind, st, running),
           kwargs={"remind_period_s": args.remind_period}, daemon=True).start()
    Thread(target=remind_inference_loop,
           args=(pipe, st, pubs, running, args.predict_hz),
           kwargs={"odom": odom, "object_map": object_map,
                   "goto_arrival_radius": args.goto_arrival_radius,
                   "match_grace_period_s": args.match_grace_period,
                   "clip_matcher": clip_matcher,
                   "object_map_update_period_s": args.object_map_update_period},
           daemon=True).start()

    root = tk.Tk()
    App(root, st, remind, object_map=object_map, clip_matcher=clip_matcher, odom=odom)

    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, root.destroy))

    def _tick():
        root.after(200, _tick)

    _tick()
    try:
        root.mainloop()
    finally:
        running["on"] = False
        time.sleep(0.2)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        time.sleep(0.1)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        try:
            session.close()
        except zenoh.ZError as e:
            print(f"[WARN] zenoh session close timed out/failed: {e}")
        odom.close()
        object_map.save(force=True)
        print("[INFO] zero velocity sent, session closed")


if __name__ == "__main__":
    main()
