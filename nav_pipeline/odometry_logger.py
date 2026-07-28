"""Dead-reckoned odometry from the ESP32's signed wheel-encoder RPM feed.

The rover has no LiDAR and no dedicated odometry node, but the firmware
(esp32/rover_6wd_complete.ino) already publishes real encoder data on
/rover/rpm (std_msgs/Float32MultiArray [left_rpm, right_rpm], signed, +ve =
wheel drives the robot forward, 10 Hz). This integrates that into a
differential-drive pose (x, y, theta) and appends one CSV row per sample.

One file per GOAL, not per run: start_new_goal() closes whatever file is
open and starts a fresh one (fresh x/y/theta origin too), named after the
target text, so each goal's odometry is self-contained in odometry_log/.
"""

import csv
import math
import os
import re
import time
from collections import deque
from typing import Optional

# Must match esp32/rover_6wd_complete.ino WHEEL_RADIUS_M / TRACK_WIDTH_M.
WHEEL_RADIUS_M = 0.056
TRACK_WIDTH_M = 0.345


class OdometryLogger:
    # retention for the spin_delta() rolling window -- must be >= the widest
    # window_s a caller will query
    HISTORY_WINDOW_S = 30.0

    def __init__(self, log_dir: str = "odometry_log"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path: Optional[str] = None
        self._file = None
        self._writer = None
        self.x = self.y = self.theta = 0.0
        self._last_t: Optional[float] = None
        self._history: deque = deque()  # (t, x, y, theta), newest last

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
        return s[:40] or "target"

    def start_new_goal(self, target: str):
        """Close the current file (if any) and start a fresh one + fresh pose for this goal."""
        if self._file is not None:
            self._file.close()
        fname = f"odom_{self._slug(target)}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = os.path.join(self.log_dir, fname)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", "dt", "left_rpm", "right_rpm", "v", "w", "x", "y", "theta"])
        self.x = self.y = self.theta = 0.0
        self._last_t = None
        self._history.clear()
        print(f"[odometry] goal '{target}' -> logging to {self.path}")

    def update(self, left_rpm: float, right_rpm: float, t: Optional[float] = None):
        """Feed one /rover/rpm sample; integrates and appends a CSV row.

        No-op (returns None) until start_new_goal() has opened a file.
        """
        if self._writer is None:
            return None
        t = time.time() if t is None else t
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        v_left = left_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v_right = right_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v = 0.5 * (v_left + v_right)
        w = (v_right - v_left) / TRACK_WIDTH_M

        if dt > 0.0:
            self.theta += w * dt
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt

        self._writer.writerow([f"{t:.6f}", f"{dt:.4f}", left_rpm, right_rpm,
                               f"{v:.4f}", f"{w:.4f}", f"{self.x:.4f}", f"{self.y:.4f}", f"{self.theta:.4f}"])
        self._file.flush()

        self._history.append((t, self.x, self.y, self.theta))
        cutoff = t - self.HISTORY_WINDOW_S
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return self.x, self.y, self.theta

    def spin_delta(self, window_s: float):
        """(|dtheta|, dist) over the trailing window_s, or None if not enough history yet.

        Used to catch a rover that's spinning in place (e.g. bouncing between
        several same-class detections around it) without net translation --
        see _select_detection's "ping-pong" note in pipeline.py.
        """
        if not self._history:
            return None
        t_now = self._history[-1][0]
        oldest = None
        for sample in self._history:
            if t_now - sample[0] <= window_s:
                oldest = sample
                break
        if oldest is None or oldest[0] == t_now:
            return None
        t0, x0, y0, th0 = oldest
        _, x1, y1, th1 = self._history[-1]
        return abs(th1 - th0), math.hypot(x1 - x0, y1 - y0)

    def close(self):
        if self._file is not None:
            self._file.close()
