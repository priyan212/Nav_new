"""HTTP client for the REMIND live re-identification server.

REMIND (REMIND/remind-reid-tracker) runs in its own conda env -- its
torch/transformers/ultralytics pins (requirements.txt: torch 2.11+cu128,
transformers 5.14) are incompatible with this project's internnav env
(torch 2.5.1+cu118, transformers 5.9), so it can't be imported in-process.
It runs instead as a separate local service
(REMIND/remind-reid-tracker/scripts/live_server.py) and this client talks
to it over loopback HTTP -- see launch_rover_remind.sh for how both
processes are brought up together.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class RemindObject:
    det_id: int
    object_id: Optional[int]   # REMIND's persistent identity; None until confirmed
    kind: str                  # "match" | "new" | "ambiguous" | "provisional" | "detection"
    class_name: Optional[str]
    confidence: float
    bbox: np.ndarray           # [x0, y0, x1, y1] pixels
    mask: Optional[np.ndarray] = None  # full-frame bool HxW, or None if REMIND had no mask

    @property
    def label(self) -> str:
        """Matches the "<CLASS> ID <n>" format the GUI's text entry expects
        (see remind_target.parse_object_target) -- what's drawn on screen is
        exactly what the operator can type back."""
        if self.object_id is not None:
            return f"{(self.class_name or '?').upper()} ID {self.object_id}"
        return f"{(self.class_name or '?').upper()} ({self.kind})"


class RemindClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8765", timeout: float = 3.0,
                 jpeg_quality: int = 85):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=self.timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def reset(self) -> None:
        """Clears REMIND's object catalogue (fresh IDs from 0) -- call when
        starting a new run/room so stale IDs from a previous session don't
        linger in the GUI's known-objects list."""
        req = urllib.request.Request(f"{self.server_url}/reset", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=max(self.timeout, 10.0)):
            pass

    def infer(self, rgb: np.ndarray) -> List[RemindObject]:
        """rgb: HxWx3 uint8 RGB (this project's convention -- see
        zenoh_node.py's parse_image/parse_compressed_image). REMIND expects
        BGR (its native video pipeline reads frames via cv2.VideoCapture) --
        flip channels before JPEG-encoding so colors round-trip correctly."""
        h, w = rgb.shape[:2]
        bgr = rgb[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        req = urllib.request.Request(
            f"{self.server_url}/infer", data=buf.tobytes(), method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
        objects = []
        for o in payload.get("objects", []):
            bbox = o.get("bbox")
            if bbox is None:
                continue
            objects.append(RemindObject(
                det_id=int(o["det_id"]),
                object_id=(int(o["object_id"]) if o.get("object_id") is not None else None),
                kind=str(o.get("kind", "detection")),
                class_name=o.get("class_name"),
                confidence=float(o.get("confidence", 0.0)),
                bbox=np.array(bbox, dtype=np.float32),
                mask=self._decode_mask(o.get("mask_png_b64"), o.get("mask_bbox"), (h, w)),
            ))
        return objects

    @staticmethod
    def _decode_mask(mask_png_b64: Optional[str], mask_bbox, frame_hw) -> Optional[np.ndarray]:
        """Reconstruct a full-frame bool mask from the bbox-cropped PNG the
        server sent (see live_server.py's _encode_mask_crop) -- pipeline.py's
        mask_median_depth/mask_centroid index it against the full depth/rgb
        frame, so it must match that frame's exact shape."""
        if not mask_png_b64 or not mask_bbox:
            return None
        raw = base64.b64decode(mask_png_b64)
        crop = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if crop is None:
            return None
        h, w = frame_hw
        x0, y0, x1, y1 = [int(v) for v in mask_bbox]
        full = np.zeros((h, w), dtype=bool)
        full[y0:y1, x0:x1] = crop > 127
        return full
