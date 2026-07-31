"""Exercises the real DinoNavDPPipeline._step_inner belief wiring end to end,
with detector output and pose faked so no camera/rover/DINO weights need to
match anything real -- just proves the plumbing (observe/propagate/SEARCH
fallback/use_belief_goal toggle) behaves as designed inside the actual
pipeline object, not just the isolated GoalBelief class.
"""
import sys
import numpy as np

sys.path.insert(0, "/mnt/bigdisk/Priyan/Nav_new")
from nav_pipeline.pipeline import PipelineConfig, DinoNavDPPipeline
from nav_pipeline.dino_detector import Detection

RGB = np.zeros((240, 320, 3), dtype=np.uint8)
DEPTH = np.full((240, 320), 3.0, dtype=np.float32)
BOX = np.array([140.0, 100.0, 180.0, 140.0], dtype=np.float32)  # centered box


def run(use_belief_goal: bool, dtheta_per_tick: float = 0.5, n_ticks: int = 16):
    cfg = PipelineConfig(device="cuda:0", use_sam=False, use_clip=False,
                         use_scene_tagger=False, use_appearance_reid=False,
                         lost_patience=5, use_belief_goal=use_belief_goal,
                         belief_max_sigma=1.0, belief_rot_noise_gain=0.35)
    pipe = DinoNavDPPipeline(cfg, use_depth_estimator=False)

    print(f"\n=== use_belief_goal={use_belief_goal} dtheta/tick={dtheta_per_tick} lost_patience=5 ===")
    det = Detection(box=BOX.copy(), score=0.9, label="chair")
    pipe.detector.detect = lambda rgb, text: [det]
    x = y = theta = 0.0
    res = pipe.step(RGB, "chair", depth=DEPTH, pose=(x, y, theta))
    print(f"tick 0 (seen):    state={res.state:6s} goal={np.round(res.goal_point[:2], 3)}")

    pipe.detector.detect = lambda rgb, text: []
    search_sides = []
    for i in range(1, n_ticks):
        theta += dtheta_per_tick
        res = pipe.step(RGB, "chair", depth=DEPTH, pose=(x, y, theta))
        sigma = f"{pipe.belief.sigma:.3f}" if use_belief_goal else "n/a"
        gp = np.round(res.goal_point[:2], 3) if res.goal_point is not None else None
        print(f"tick {i:2d} (lost):    state={res.state:6s} sigma={sigma:>6s} "
              f"angular={res.angular:+.3f} goal={gp}")
        if res.state == "SEARCH":
            search_sides.append(res.angular)

    # ignore the first few SEARCH ticks -- angular_slew_max ramps the command
    # smoothly from whatever TRACK was commanding, so it can cross zero once
    # on the state transition alone; only the STEADY-STATE side (once the
    # ramp settles at +-search_angular) reflects the actual side decision
    steady = [s for s in search_sides if abs(s) > 0.9 * cfg.search_angular]
    if steady:
        flips = sum(1 for a, b in zip(steady, steady[1:]) if np.sign(a) != np.sign(b))
        print(f"  steady-state SEARCH side flips across {len(steady)} ticks: {flips} "
              f"({'STABLE - OK' if flips == 0 else 'FLIPPING - BUG'})")


run(use_belief_goal=True, dtheta_per_tick=0.5, n_ticks=16)   # fast rotation -- should still give up reasonably soon
run(use_belief_goal=True, dtheta_per_tick=0.02, n_ticks=16)  # gentle -- should coast PAST tick 5 now (was capped before)
run(use_belief_goal=False, dtheta_per_tick=0.5, n_ticks=16)
