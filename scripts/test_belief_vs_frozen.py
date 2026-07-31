"""Synthetic offline test: would SubgoalBeliefBank (MARS/EARTH sim belief
system) improve real-rover goal tracking over the current mechanism?

The real rover (nav_pipeline/pipeline.py) has NO belief system: when the
target detection is lost, it freezes `self._last_goal` verbatim -- the exact
robot-frame (x, y) point from the last tick it saw the target -- and reuses
it unchanged for up to `lost_patience` (default 5) ticks before giving up
and entering SEARCH. That frozen point is never corrected for the rover's
own motion during the occlusion, so if the rover keeps driving/turning while
the target is briefly lost, the frozen goal silently drifts out of date.

SubgoalBeliefBank (MARS/mars-habitatsim/navdp/navdp/extensions/belief_bank.py,
used only by the MARS/EARTH Habitat-sim GUIs today, never by the real rover)
instead ego-motion-propagates its belief mean every tick via odometry
(`ego_motion_update`), so an occluded target's estimated position keeps
tracking the rover's own turning/driving, and its confidence decays smoothly
(decay_factor=0.95/tick) instead of a hard frame-count cutoff.

This script is a controlled SYNTHETIC simulation, not a live-rover or
live-Habitat-sim test: a stationary world-frame target, a rover that moves
with a known (v, w) during a synthetic occlusion window, comparing both
goal-estimate rules against ground truth. It imports and uses the REAL
SubgoalBeliefBank / ego_motion_update code (not a reimplementation), so the
belief-side numbers reflect the actual sim code, not an idealized model.
No detector, depth model, or GPU is used, so it runs in seconds on CPU.

Simplification: the rover is held stationary except during the occlusion
window itself, isolating the effect of motion-during-occlusion on goal
accuracy. In reality the rover also keeps driving on the frozen bearing
while TRACK-ing between detections, which would compound frozen's error
further -- so the "stationary" scenario here is closer to real slow/careful
driving, and "turn"/"spin" are the cases the real rover actually reports
hitting (steering at the stiction floor while re-acquiring).

Run: python scripts/test_belief_vs_frozen.py
Outputs -> belief_eval_<date>/ (3 PNGs + results.csv)
"""

import csv
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "MARS", "mars-habitatsim", "navdp")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navdp.extensions.belief_bank import SubgoalBeliefBank  # real sim code, not reimplemented

OUTDIR = os.path.join(os.path.dirname(__file__), "..", f"belief_eval_{date.today():%Y%m%d}")
os.makedirs(OUTDIR, exist_ok=True)

# --- constants matched to real config -------------------------------- #
DT = 1.0 / 3.0                 # predict_hz=3.0, real-rover default (isaac_gui.py/zenoh_node.py)
DETECT_NOISE_STD = 0.05        # m; matches sigma_visible used by mars_gui.py / mars_belief_only_gui.py
ODOM_NOISE = 0.02              # matches mars_gui.py's SubgoalBeliefBank(odom_noise=0.02)
DECAY_FACTOR = 0.95            # SubgoalBeliefBank default, as used in mars_gui.py
LOST_PATIENCE = 5              # nav_pipeline/pipeline.py PipelineConfig default
BELIEF_CONF_MIN = 0.15         # mars_gui.py BELIEF_CONFIDENCE_MIN default
MAX_LINEAR = 0.15              # real-rover PipelineConfig default
MAX_ANGULAR = 0.25             # real-rover PipelineConfig default
N_TRIALS = 200


def rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def simulate_episode(occlusion_ticks, v, w, target_dist=2.0, target_bearing=0.0,
                      pre_ticks=5, post_ticks=5, seed=0):
    """Returns a list of per-tick dicts: t, occluded, err_frozen, err_belief, belief_conf."""
    rng = np.random.default_rng(seed)
    pose = np.array([0.0, 0.0, 0.0])  # x, y, theta (world frame)
    target_world = target_dist * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    bank = SubgoalBeliefBank(["target"], dim=2, sigma_visible=DETECT_NOISE_STD,
                              odom_noise=ODOM_NOISE, decay_factor=DECAY_FACTOR)
    frozen_goal = None
    odom_delta = [0.0, 0.0, 0.0]
    total = pre_ticks + occlusion_ticks + post_ticks
    rows = []
    for t in range(total):
        occluded = pre_ticks <= t < pre_ticks + occlusion_ticks
        theta = pose[2]
        true_goal = rotate(target_world - pose[:2], -theta)

        if not occluded:
            meas = true_goal + rng.normal(0, DETECT_NOISE_STD, size=2)
            frozen_goal = meas.copy()
            obs = {"target": {"visible": True, "position": meas, "confidence": 0.9}}
        else:
            obs = {"target": {"visible": False}}

        slots = bank.update(obs, odom_delta=odom_delta, step=t)
        slot = slots["target"]

        err_frozen = float(np.linalg.norm(frozen_goal - true_goal)) if frozen_goal is not None else np.nan
        err_belief = float(np.linalg.norm(slot.mu - true_goal)) if slot.initialized else np.nan
        rows.append(dict(t=t, tick_in_occlusion=t - pre_ticks, occluded=occluded,
                          err_frozen=err_frozen, err_belief=err_belief, belief_conf=slot.confidence))

        # move only during the occlusion window, to isolate its effect (see docstring)
        if occluded:
            dtheta = w * DT
            if abs(w) > 1e-6:
                dx, dy = v / w * np.sin(dtheta), v / w * (1 - np.cos(dtheta))
            else:
                dx, dy = v * DT, 0.0
        else:
            dtheta = dx = dy = 0.0
        pose = pose + np.array([np.cos(theta) * dx - np.sin(theta) * dy,
                                 np.sin(theta) * dx + np.cos(theta) * dy, dtheta])
        odom_delta = [dx, dy, dtheta]
    return rows


MOTIONS = {
    "stationary": (0.0, 0.0),
    "straight":   (MAX_LINEAR, 0.0),
    "turn":       (MAX_LINEAR, MAX_ANGULAR),
    "spin":       (0.0, MAX_ANGULAR),
}
DURATIONS = [1, 2, 3, 5, 8, 12, 20, 30]


def run_grid():
    """end-of-occlusion error (the value actually used for the next
    STOP/TRACK decision) for every (motion, duration), averaged over N_TRIALS
    noise realizations."""
    results = {}
    for mname, (v, w) in MOTIONS.items():
        for dur in DURATIONS:
            fe, be = [], []
            for trial in range(N_TRIALS):
                rows = simulate_episode(dur, v, w, seed=trial)
                last_occ = [r for r in rows if r["occluded"]][-1]
                fe.append(last_occ["err_frozen"])
                be.append(last_occ["err_belief"])
            results[(mname, dur)] = dict(
                mean_frozen=float(np.mean(fe)), std_frozen=float(np.std(fe)),
                mean_belief=float(np.mean(be)), std_belief=float(np.std(be)),
            )
    return results


def plot_example_episode():
    rows = simulate_episode(occlusion_ticks=8, v=MAX_LINEAR, w=MAX_ANGULAR, seed=0)
    ticks = [r["tick_in_occlusion"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ticks, [r["err_frozen"] for r in rows], "o-", color="tab:red", label="frozen (current real-rover)")
    ax.plot(ticks, [r["err_belief"] for r in rows], "o-", color="tab:blue", label="belief (SubgoalBeliefBank)")
    ax.axvspan(0, 8, color="gray", alpha=0.15, label="occluded window")
    ax.axvline(LOST_PATIENCE, color="k", linestyle="--", linewidth=1,
               label=f"lost_patience={LOST_PATIENCE} (real rover gives up here)")
    ax.set_xlabel("tick (0 = occlusion start)")
    ax.set_ylabel("goal-point error vs ground truth (m)")
    ax.set_title("Example occlusion episode: rover driving + turning (v=0.15 m/s, w=0.25 rad/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "error_vs_time_example.png"), dpi=130)
    plt.close(fig)


def plot_duration_grid(results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, mname in zip(axes.flat, MOTIONS):
        mf = [results[(mname, d)]["mean_frozen"] for d in DURATIONS]
        sf = [results[(mname, d)]["std_frozen"] for d in DURATIONS]
        mb = [results[(mname, d)]["mean_belief"] for d in DURATIONS]
        sb = [results[(mname, d)]["std_belief"] for d in DURATIONS]
        ax.errorbar(DURATIONS, mf, yerr=sf, fmt="o-", color="tab:red", capsize=3, label="frozen")
        ax.errorbar(DURATIONS, mb, yerr=sb, fmt="o-", color="tab:blue", capsize=3, label="belief")
        ax.axvline(LOST_PATIENCE, color="k", linestyle="--", linewidth=1)
        v, w = MOTIONS[mname]
        ax.set_title(f"{mname} (v={v:.2f} m/s, w={w:.2f} rad/s)")
        ax.set_xlabel("occlusion duration (ticks)")
        ax.set_ylabel("end-of-occlusion error (m)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Goal-point error at end of occlusion, mean ± std over {N_TRIALS} trials")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "error_vs_occlusion_duration.png"), dpi=130)
    plt.close(fig)


def plot_confidence_decay():
    rows = simulate_episode(occlusion_ticks=60, v=0.0, w=0.0, pre_ticks=3, post_ticks=0, seed=0)
    occ_rows = [r for r in rows if r["occluded"]]
    ticks = [r["tick_in_occlusion"] for r in occ_rows]
    conf = [r["belief_conf"] for r in occ_rows]
    cross = next((t for t, c in zip(ticks, conf) if c < BELIEF_CONF_MIN), None)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ticks, conf, "o-", color="tab:blue")
    ax.axhline(BELIEF_CONF_MIN, color="k", linestyle="--", linewidth=1,
               label=f"belief_confidence_min={BELIEF_CONF_MIN}")
    ax.axvline(LOST_PATIENCE, color="tab:red", linestyle=":", linewidth=1,
               label=f"lost_patience={LOST_PATIENCE} (real rover's hard cutoff)")
    if cross is not None:
        ax.axvline(cross, color="tab:blue", linestyle=":", linewidth=1,
                   label=f"belief confidence crosses threshold at tick {cross}")
    ax.set_xlabel("ticks since target lost")
    ax.set_ylabel("belief confidence")
    ax.set_title(f"Belief confidence decay while occluded (decay_factor={DECAY_FACTOR}/tick)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "confidence_decay.png"), dpi=130)
    plt.close(fig)
    return cross


def write_csv(results):
    path = os.path.join(OUTDIR, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion", "v", "w", "occlusion_ticks", "mean_err_frozen_m", "std_err_frozen_m",
                    "mean_err_belief_m", "std_err_belief_m", "frozen_over_belief_ratio"])
        for mname, (v, wv) in MOTIONS.items():
            for dur in DURATIONS:
                r = results[(mname, dur)]
                ratio = r["mean_frozen"] / r["mean_belief"] if r["mean_belief"] > 1e-9 else float("inf")
                w.writerow([mname, v, wv, dur, f"{r['mean_frozen']:.4f}", f"{r['std_frozen']:.4f}",
                            f"{r['mean_belief']:.4f}", f"{r['std_belief']:.4f}", f"{ratio:.2f}"])
    return path


if __name__ == "__main__":
    print(f"[INFO] output dir: {OUTDIR}")
    print("[INFO] running example episode plot...")
    plot_example_episode()
    print("[INFO] running duration/motion grid (this is the main result)...")
    results = run_grid()
    plot_duration_grid(results)
    print("[INFO] running confidence decay...")
    cross = plot_confidence_decay()
    print("[INFO] writing results.csv...")
    csv_path = write_csv(results)

    print("\n=== SUMMARY (end-of-occlusion error, mean over {} trials) ===".format(N_TRIALS))
    for mname in MOTIONS:
        r5 = results[(mname, LOST_PATIENCE)]
        print(f"  {mname:10s} @ lost_patience={LOST_PATIENCE} ticks: "
              f"frozen={r5['mean_frozen']:.3f}m  belief={r5['mean_belief']:.3f}m")
    print(f"\nbelief confidence crosses {BELIEF_CONF_MIN} threshold at tick {cross} "
          f"(real rover's hard cutoff is tick {LOST_PATIENCE})")
    print(f"\nSaved: {OUTDIR}/error_vs_time_example.png")
    print(f"       {OUTDIR}/error_vs_occlusion_duration.png")
    print(f"       {OUTDIR}/confidence_decay.png")
    print(f"       {csv_path}")
