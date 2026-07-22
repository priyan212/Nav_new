"""Unit test: footprint-swept clearance accounts for rover size + heading."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from nav_pipeline.obstacle_guard import _CIRCLE_R, GuardConfig, swept_clearance

cfg = GuardConfig()
straight = np.stack([np.linspace(0, 2, 24), np.zeros(24), np.zeros(24)], axis=1)[None]

# 1. side obstacle 0.30 m off the centerline: hull (r~0.23) leaves ~0.07 gap -> veto
gap = swept_clearance(straight, np.array([[1.0, 0.30]]))[0]
print(f"1. side pass 0.30m: gap={gap:.3f} (veto if < {cfg.margin})")
assert gap < cfg.margin

# 2. same obstacle 0.45 m off: gap ~0.22 -> safe
gap2 = swept_clearance(straight, np.array([[1.0, 0.45]]))[0]
print(f"2. side pass 0.45m: gap={gap2:.3f} (safe)")
assert gap2 > cfg.margin

# 3. rotation-in-place safety: obstacle 0.28 m ahead overlaps hull at START pose
gap3 = swept_clearance(straight, np.array([[0.28, 0.0]]))[0]
print(f"3. obstacle 0.28m ahead: gap={gap3:.3f} (negative = hull overlap)")
assert gap3 < 0

# 4. heading matters: pose turned 90 deg sweeps the long axis laterally
turned = straight.copy()
turned[0, :, 2] = np.pi / 2
gs = swept_clearance(straight, np.array([[1.0, 0.33]]))[0]
gt = swept_clearance(turned, np.array([[1.0, 0.33]]))[0]
print(f"4. same point: straight gap={gs:.3f} vs turned gap={gt:.3f} (turned reaches wider)")
assert gt < gs

print(f"(circle cover radius = {_CIRCLE_R:.3f} m)")
print("FOOTPRINT GUARD TEST PASSED")
