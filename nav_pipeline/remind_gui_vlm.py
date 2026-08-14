#!/usr/bin/env python3
"""REMIND + NavDP rover GUI, VLM-confirmed arrival variant.

Identical to nav_pipeline.remind_gui in every respect (same control panel,
same ID-only REMIND targeting, same object-location memory, same
navigate-back GOTO behavior -- see that module's docstring for the full
description) except for ONE thing: the STOP decision.

remind_gui.py trusts a pure metric depth threshold (pipeline.py's
stop_distance + a stop_streak hysteresis counter) to declare "arrived".
NavDP/InternVLA-derived checkpoints don't learn to stop on their own (see
README.md) -- stopping has always been enforced externally by that
threshold. This variant treats the metric threshold as NECESSARY but not
SUFFICIENT: once it fires, it asks REMIND's already-loaded InternVL model
(see live_server.py's /confirm_arrival) "has the robot actually arrived?"
on the current full frame, and only latches goal_reached once the VLM
agrees -- see VLMArrivalGate below.

Why gate rather than replace the metric check: a VLM call is a
~0.3-0.8s HTTP round-trip to a separate process -- too slow and too
fragile (server down, model hiccup, occasional bad answer) to be the
thing that actually keeps the rover from driving into the target. The
metric check still runs every tick and is what zeroes the velocity
command the instant the rover gets close (pipeline.py's STOP state,
unchanged); the VLM only decides whether that already-stopped moment
counts as a CONFIRMED goal. If the endpoint is unavailable (older REMIND
server, or started with --no-internvl/--use-blip), VLMArrivalGate detects
that on the first failed call and permanently falls back to the exact
pure-metric behavior of remind_gui.py, so this never strands the rover
short of goal because a model server hiccupped.

Run (from Nav_new root, after the REMIND live server is up -- see
launch_rover_remind_vlm.sh, which brings up both):
    conda activate internnav
    python -m nav_pipeline.remind_gui_vlm --pi-ip <IP> --remind-server http://127.0.0.1:8765
"""

import argparse
import os
import signal
import sys
import time
from threading import Lock, Thread
from typing import Optional

import numpy as np

import tkinter as tk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.dino_detector import Detection  # noqa: E402
from nav_pipeline.isaac_gui import (  # noqa: E402
    DEPTH_STALE_S,
    SPIN_DIST_THRESH_M,
    SPIN_ROT_THRESH_RAD,
    SPIN_WINDOW_S,
    heartbeat_loop,
    zenoh_setup,
)
from nav_pipeline.object_map import ObjectMap, world_to_local  # noqa: E402
from nav_pipeline.object_query import ClipObjectMatcher  # noqa: E402
from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.remind_client import RemindClient  # noqa: E402
from nav_pipeline.remind_gui import App, SharedState, _update_object_map, remind_poll_loop  # noqa: E402
from nav_pipeline.zenoh_node import serialize_path, serialize_string, serialize_twist  # noqa: E402


class VLMArrivalGate:
    """Confirms a metric-triggered STOP is a real arrival by asking
    REMIND's already-loaded InternVL model (live_server.py's
    /confirm_arrival) instead of trusting depth-threshold proximity alone
    -- see this module's docstring for why this GATES the metric check
    rather than replacing it.

    Call check() every tick with the current frame and whether the metric
    threshold is confirmed this tick; it returns whether "reached" should
    be declared. Internally throttles VLM calls (period_s between
    attempts, and only ever one in flight) and runs them in a background
    thread so a slow HTTP round-trip never blocks the control loop.
    """

    def __init__(self, remind: RemindClient, period_s: float = 1.5, call_timeout_s: float = 6.0):
        self.remind = remind
        self.period_s = period_s
        self.call_timeout_s = call_timeout_s
        self.available = True  # flips False permanently once the endpoint proves unusable
        self.lock = Lock()
        self._pending = False
        self._pending_since = 0.0
        self._result: Optional[bool] = None
        self._last_attempt_t = 0.0

    def reset(self):
        """Call when the target changes (or tracking is lost) so a stale
        confirmation/pending call from the PREVIOUS target can't leak into
        the new one's arrival decision."""
        with self.lock:
            self._pending = False
            self._result = None

    def check(self, rgb: np.ndarray, target_desc: str, metric_confirmed: bool, now: float) -> bool:
        if not metric_confirmed:
            self.reset()
            return False
        if not self.available:
            return True  # degrade to pure-metric behavior, exactly like remind_gui.py

        with self.lock:
            if self._pending and (now - self._pending_since) > self.call_timeout_s:
                self._pending = False  # call hung/dropped -- allow a fresh attempt below
            result, pending = self._result, self._pending

        if result is True:
            return True
        if result is False and (now - self._last_attempt_t) < self.period_s:
            return False  # cooling down after a "no" before retrying

        if not pending and (now - self._last_attempt_t) >= self.period_s:
            with self.lock:
                self._pending = True
                self._pending_since = now
                self._result = None
            self._last_attempt_t = now
            Thread(target=self._worker, args=(rgb.copy(), target_desc), daemon=True).start()
        return False

    def _worker(self, rgb: np.ndarray, target_desc: str):
        try:
            answer = self.remind.confirm_arrival(rgb, target_desc)
        except NotImplementedError as e:
            print(f"[WARN] {e} -- falling back to pure metric stop-distance confirmation "
                  f"for the rest of this run")
            with self.lock:
                self.available = False
                self._pending = False
            return
        except Exception as e:
            print(f"[WARN] VLM arrival confirm failed: {e} -- will retry")
            with self.lock:
                self._pending = False
            return
        with self.lock:
            self._result = answer
            self._pending = False


def remind_inference_loop_vlm(pipe: DinoNavDPPipeline, st: SharedState, pubs,
                              running, predict_hz: float,
                              stop_confirm: int = 3, odom: Optional[OdometryLogger] = None,
                              object_map: Optional[ObjectMap] = None,
                              goto_arrival_radius: float = 1.0,
                              match_grace_period_s: float = 1.2,
                              clip_matcher: Optional[ClipObjectMatcher] = None,
                              object_map_update_period_s: float = 1.0,
                              vlm_gate: Optional[VLMArrivalGate] = None):
    """Identical to remind_gui.remind_inference_loop except for the
    reached decision at the bottom (metric stop_streak confirmation ANDed
    with vlm_gate.check() when a gate is given) -- see this module's
    docstring and VLMArrivalGate above."""
    period = 1.0 / predict_hz
    stop_streak = 0
    last_target_id: Optional[int] = None
    last_map_update_t = 0.0
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
            last_objects = st.remind_objects

        if target_id is not None and target_id != last_target_id:
            if last_target_id is not None:
                pipe.reset()
            if vlm_gate is not None:
                vlm_gate.reset()
            if odom is not None:
                odom.start_new_goal(target_text)
            last_target_id = target_id
            last_seen_target = None

        if rgb is None:
            time.sleep(0.1)
            continue

        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None

        if (t0 - last_map_update_t) >= object_map_update_period_s:
            last_map_update_t = t0
            map_depth = depth
            if map_depth is None and pipe.depther is not None:
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
            matched = [last_seen_target[1]]
        external_dets: Optional[list] = []
        external_goal = None
        goto_dist = None
        if matched:
            det = Detection(box=matched[0].bbox, score=max(matched[0].confidence, 0.01), label=matched[0].label)
            det.mask = matched[0].mask
            external_dets = [det]
        else:
            belief_confident = (pipe.cfg.use_belief_goal and pipe.belief.initialized
                                and pipe.belief.sigma <= pipe.cfg.belief_max_sigma)
            if not belief_confident and object_map is not None and pose is not None:
                entry = object_map.get(target_id)
                if entry is not None:
                    lx, ly = world_to_local((entry["world_x"], entry["world_y"]), pose)
                    goto_dist = float(np.hypot(lx, ly))
                    if goto_dist > goto_arrival_radius:
                        external_dets = None
                        external_goal = np.array([lx, ly, 0.0], dtype=np.float32)

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
        metric_confirmed = stop_streak >= stop_confirm

        if vlm_gate is not None:
            desc = matched[0].class_name if (matched and matched[0].class_name) else target_text
            reached = vlm_gate.check(rgb, desc, metric_confirmed, t0)
        else:
            reached = metric_confirmed

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
                confirm_txt = " (VLM-confirmed)" if vlm_gate is not None and vlm_gate.available else ""
                st.state_text = f"GOAL REACHED{confirm_txt}: '{target_text}' (stopped at {pipe.cfg.stop_distance:.1f}m)"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                if matched and res.state == "STOP" and vlm_gate is not None and metric_confirmed and vlm_gate.available:
                    st.state_text = "STOP -- metric threshold reached, confirming arrival with VLM..."
                elif matched:
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


def main():
    ap = argparse.ArgumentParser(description="Nav_new REMIND+NavDP rover GUI (VLM-confirmed arrival)")
    ap.add_argument("--target", default="",
                    help="initial target as 'ID N', e.g. 'ID 1' -- starts empty "
                         "(manual drive) until a target is sent from the GUI")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--remind-server", default="http://127.0.0.1:8765",
                    help="base URL of the REMIND live server (see launch_rover_remind_vlm.sh)")
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
                    help="meters from the object at which to stop (depth-based) -- this is the "
                         "NECESSARY trigger the VLM confirmation gate sits on top of, see "
                         "--no-vlm-confirm")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vitb",
                    help="RGB-only metric depth model (no depth sensor on the real rover); "
                         "defaults to vitb here since depth error feeds directly into the "
                         "STOP distance decision -- needs checkpoints/depth_anything_v2_"
                         "metric_hypersim_vitb.pth (scripts/download_models.py --depth-encoder vitb)")
    ap.add_argument("--compressed-only", action="store_true")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3,
                    help="IMU calibration digit (0-3) required before theta rides the IMU "
                         "heading instead of wheel-diff dead reckoning -- see OdometryLogger. "
                         "Matters more here than most callers: object_map.py's world-frame "
                         "recall directly depends on theta accuracy across turns.")
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
                         "REMIND stops matching the target before treating it as truly not "
                         "visible")
    ap.add_argument("--object-map-update-period", type=float, default=1.0,
                    help="seconds: minimum interval between passive object-memory updates")
    ap.add_argument("--no-vlm-confirm", action="store_true",
                    help="disable the VLM arrival-confirmation gate and fall back to pure "
                         "metric stop-distance confirmation (identical behavior to plain "
                         "remind_gui.py) -- for A/B comparison, or if the REMIND server was "
                         "started with --no-internvl/--use-blip")
    ap.add_argument("--vlm-confirm-period", type=float, default=1.5,
                    help="seconds between VLM arrival-confirmation attempts once the metric "
                         "stop threshold has fired -- a 'no' answer waits this long before "
                         "retrying so a single bad frame/answer doesn't spam the VLM every tick")
    ap.add_argument("--vlm-confirm-timeout", type=float, default=6.0,
                    help="seconds to wait for a pending VLM confirmation call before treating "
                         "it as failed and allowing a retry (InternVL measured ~0.5s/call "
                         "locally -- this is a generous ceiling for the extra HTTP hop)")
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
              f"start it first (see launch_rover_remind_vlm.sh)")
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
        use_appearance_reid=False,
        use_sam=False,
        use_clip=False,
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
    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)
    object_map = ObjectMap(args.object_map_path)
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)
    running = {"on": True}

    vlm_gate = None if args.no_vlm_confirm else VLMArrivalGate(
        remind, period_s=args.vlm_confirm_period, call_timeout_s=args.vlm_confirm_timeout)
    if vlm_gate is not None:
        print("[INFO] VLM arrival-confirmation gate ENABLED (metric stop threshold + InternVL "
              "confirmation on the REMIND server; falls back to pure-metric automatically if "
              "the endpoint is unavailable)")
    else:
        print("[INFO] VLM arrival-confirmation gate disabled (--no-vlm-confirm) -- pure metric "
              "stop-distance behavior, same as remind_gui.py")

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=remind_poll_loop, args=(remind, st, running),
           kwargs={"remind_period_s": args.remind_period}, daemon=True).start()
    Thread(target=remind_inference_loop_vlm,
           args=(pipe, st, pubs, running, args.predict_hz),
           kwargs={"odom": odom, "object_map": object_map,
                   "goto_arrival_radius": args.goto_arrival_radius,
                   "match_grace_period_s": args.match_grace_period,
                   "clip_matcher": clip_matcher,
                   "object_map_update_period_s": args.object_map_update_period,
                   "vlm_gate": vlm_gate},
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
