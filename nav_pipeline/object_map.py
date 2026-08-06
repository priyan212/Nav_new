"""Persistent per-object world-location memory.

REMIND (REMIND/remind-reid-tracker) gives each object a persistent ID +
BLIP caption within a live tracking session, but that's purely
camera-relative -- it says nothing about WHERE the object is in the world.
This module fills that gap: every tick a REMIND object is actually visible
with a valid mask+depth, its local-frame goal point (nav_pipeline/
goal_utils.py's pixel_depth_to_point, the same math pipeline.py uses to
drive toward a live detection) is transformed into the rover's continuous
world frame and folded into a running per-ID estimate.

Requires pose to be a genuine continuous world frame, not reset per goal --
see odometry_logger.py's module docstring; this module's world coordinates
are meaningless against a pose that keeps re-zeroing at wherever the rover
happens to be standing.

Persisted to disk (JSON) so it survives GUI restarts within a room/building,
not just goal switches. NOT safe to trust across a power cycle or a physical
pick-up-and-move of the rover -- there's no way to detect that the odometry
origin is no longer valid, so a stale map from a previous session should be
discarded (delete the file) if the rover was moved by hand since it was
written.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple


def local_to_world(local_xy: Tuple[float, float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """local_xy: (x fwd, y left) in the rover's CURRENT frame. pose: (x, y,
    theta) of that frame in the world (see odometry_logger.py). -> world (x, y)."""
    lx, ly = local_xy
    px, py, pth = pose
    wx = px + lx * math.cos(pth) - ly * math.sin(pth)
    wy = py + lx * math.sin(pth) + ly * math.cos(pth)
    return wx, wy


def world_to_local(world_xy: Tuple[float, float], pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """Inverse of local_to_world -- world (x, y) as seen from the rover's
    current pose, i.e. the (x fwd, y left) convention pipeline.py's goal
    expects (see goal_utils.pixel_depth_to_point)."""
    wx, wy = world_xy
    px, py, pth = pose
    dx, dy = wx - px, wy - py
    lx = dx * math.cos(pth) + dy * math.sin(pth)
    ly = -dx * math.sin(pth) + dy * math.cos(pth)
    return lx, ly


class ObjectMap:
    """object_id -> {"caption", "world_x", "world_y", "last_seen", "n_obs",
    "embedding"}.

    caption is kept purely for internal bookkeeping/debugging -- the GUI
    displays IDs only (see remind_gui.py), not this text. embedding is an
    optional CLIP image-embedding (list[float], L2-normalized) of the crop
    the object was first seen in -- see object_query.py, which matches free
    text against THIS vector, not the (often noisy/verbose) BLIP caption
    text. Cached once per object_id and never overwritten, since the point
    is a stable fingerprint of "what this object looks like" that free-text
    queries can be scored against consistently across the object's lifetime.
    """

    def __init__(self, path: str, ema_alpha: float = 0.3, save_period_s: float = 2.0):
        self.path = path
        self.ema_alpha = float(ema_alpha)
        self.save_period_s = float(save_period_s)
        self._entries: Dict[int, dict] = {}
        self._dirty = False
        self._last_save_t = 0.0
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            self._entries = {int(k): v for k, v in raw.items()}
            print(f"[object-map] loaded {len(self._entries)} remembered object(s) from {self.path}")
        except (OSError, ValueError) as e:
            print(f"[object-map] failed to load {self.path}: {e} -- starting empty")
            self._entries = {}

    def save(self, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        now = time.time()
        if not force and (now - self._last_save_t) < self.save_period_s:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._entries, f, indent=2)
        os.replace(tmp_path, self.path)
        self._dirty = False
        self._last_save_t = now

    def update(
        self,
        object_id: int,
        caption: Optional[str],
        world_xy: Tuple[float, float],
        timestamp: float,
    ) -> None:
        oid = int(object_id)
        wx, wy = float(world_xy[0]), float(world_xy[1])
        entry = self._entries.get(oid)
        if entry is None:
            self._entries[oid] = {
                "caption": caption,
                "world_x": wx,
                "world_y": wy,
                "last_seen": float(timestamp),
                "n_obs": 1,
            }
        else:
            a = self.ema_alpha
            entry["world_x"] = (1.0 - a) * float(entry["world_x"]) + a * wx
            entry["world_y"] = (1.0 - a) * float(entry["world_y"]) + a * wy
            entry["last_seen"] = float(timestamp)
            entry["n_obs"] = int(entry.get("n_obs", 0)) + 1
            if caption:
                entry["caption"] = caption
        self._dirty = True
        self.save()

    def get(self, object_id: int) -> Optional[dict]:
        return self._entries.get(int(object_id))

    def has_embedding(self, object_id: int) -> bool:
        entry = self._entries.get(int(object_id))
        return entry is not None and entry.get("embedding") is not None

    def set_embedding(self, object_id: int, embedding: List[float]) -> None:
        """Cache the CLIP image embedding the first time this object_id is
        seen -- see the class docstring. No-op if object_id isn't already
        in the map (must go through update() first) or already has one
        (deliberately never overwritten)."""
        entry = self._entries.get(int(object_id))
        if entry is None or entry.get("embedding") is not None:
            return
        entry["embedding"] = [float(v) for v in embedding]
        self._dirty = True
        self.save()

    def get_embedding(self, object_id: int) -> Optional[List[float]]:
        entry = self._entries.get(int(object_id))
        return entry.get("embedding") if entry is not None else None

    def all_ids(self):
        return sorted(self._entries.keys())

    def items(self):
        return sorted(self._entries.items())

    def forget(self, object_id: int) -> None:
        if int(object_id) in self._entries:
            del self._entries[int(object_id)]
            self._dirty = True
            self.save(force=True)

    def clear(self) -> None:
        self._entries = {}
        self._dirty = True
        self.save(force=True)
