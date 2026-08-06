# features/blip_captioner.py
"""BLIP image captioning -- fills the human-readable label SAM's
class-agnostic segmentation can't provide (no classifier).

Runs ONCE per newly created tracked object (see update/memory_manager.py's
create_new_objects / materialize_committed_new_objects_for_ambiguous_items),
not every frame: BLIP captions aren't perfectly stable frame-to-frame on the
same physical object, and the operator-facing "<CAPTION> ID <n>" label (see
Nav_new's remind_gui.py) needs to be a string the operator can read once and
type back, not one that drifts every tick.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class BlipCaptioner:
    def __init__(self, config: dict, device: str):
        self.config = config or {}
        self.device = device

        self.model_id = str(self.config.get("model_id", "Salesforce/blip-image-captioning-base"))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 12))
        self.min_crop_px = int(self.config.get("min_crop_px", 8))
        self.mask_background_fill = bool(self.config.get("mask_background_fill", True))

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def caption(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """rgb: HxWx3 uint8 RGB frame. mask/bbox: object region to crop to
        (bbox alone if mask is None) -- captioning the whole frame would
        describe the room, not the object. Returns None on a degenerate
        crop or empty caption (caller should fall back to a generic label)."""
        if self.model is None:
            raise RuntimeError("BLIP is not loaded. Call load_model() before caption().")

        crop = self._crop(rgb, mask, bbox)
        if crop is None:
            return None

        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
        return text or None

    def _crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        if bbox is not None:
            x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
        elif mask is not None:
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                return None
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        else:
            return None

        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 - x0 < self.min_crop_px or y1 - y0 < self.min_crop_px:
            return None

        crop = rgb[y0:y1, x0:x1]
        if mask is not None and self.mask_background_fill:
            local_mask = mask[y0:y1, x0:x1]
            if local_mask.shape[:2] == crop.shape[:2]:
                crop = crop.copy()
                crop[~local_mask] = 255
        return crop
