# features/internvl_classifier.py
"""InternVL open-vocabulary classification -- alternative to blip_captioner.py
for filling the human-readable label SAM's class-agnostic segmentation can't
provide on its own.

BLIP is a pure captioning model with no way to constrain its output, which is
exactly why it captioned flat floor/wall/ceiling patches as things like
"black tiles" instead of giving a short, stable answer (or declining) --
there's no way to ask a captioner a targeted question. InternVL is an
instruction-following VQA model: prompted as a CLASSIFIER ("what object is
this? one or two words only") rather than a captioner, it gives a much more
constrained, stable answer for the same crop. Natively supported by
transformers as of this env's version (InternVLForConditionalGeneration /
AutoProcessor, no trust_remote_code, no extra deps beyond what's already in
requirements.txt) -- verified against OpenGVLab/InternVL3_5-1B-HF: ~2.2GB
VRAM, ~12s load, ~0.5s/call.

Same call contract as BlipCaptioner (caption(rgb, mask, bbox) -> Optional[str])
on purpose -- update/memory_manager.py calls self.captioner.caption(...)
generically and doesn't care which backend built it (see
pipeline/initialization.py's build_captioner). Runs ONCE per newly created
tracked object, not every frame -- same reasoning as BLIP: a stable label the
operator reads once, not one that drifts every tick.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, InternVLForConditionalGeneration


class InternVLClassifier:
    DEFAULT_PROMPT = "What object is this? Answer with one or two words only, no punctuation."
    # Nav_new's VLM arrival-confirmation gate (nav_pipeline/remind_gui_vlm.py)
    # -- see confirm_arrival() below. Full-scene question, not the crop-only
    # classifier prompt above: arrival is a judgment about the robot's
    # relationship to the whole frame, not what the target object looks like.
    ARRIVAL_PROMPT_TEMPLATE = (
        "You are looking through a mobile robot's front camera. The robot is "
        "trying to reach: '{target}'. Has the robot arrived at / gotten close "
        "enough to this target that it should stop moving? Answer with exactly "
        "one word: yes or no."
    )

    def __init__(self, config: dict, device: str):
        self.config = config or {}
        self.device = device

        self.model_id = str(self.config.get("model_id", "OpenGVLab/InternVL3_5-1B-HF"))
        self.prompt = str(self.config.get("prompt", self.DEFAULT_PROMPT))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 8))
        self.min_crop_px = int(self.config.get("min_crop_px", 8))
        self.mask_background_fill = bool(self.config.get("mask_background_fill", True))

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = InternVLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16
        ).to(self.device).eval()

    @torch.no_grad()
    def caption(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Same contract as BlipCaptioner.caption -- see that class's
        docstring for the mask/bbox crop convention. Returns None on a
        degenerate crop or empty answer (caller falls back to a generic
        label)."""
        if self.model is None:
            raise RuntimeError("InternVL is not loaded. Call load_model() before caption().")

        crop = self._crop(rgb, mask, bbox)
        if crop is None:
            return None

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=crop, text=prompt, return_tensors="pt").to(
            self.device, torch.bfloat16
        )
        # Greedy, not sampled -- this is a constrained classification answer,
        # not creative captioning, and it's the one-shot label an object
        # keeps for its whole tracked lifetime (see this class's docstring),
        # so it should be deterministic rather than rolling different labels
        # for visually-identical crops across restarts.
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        # inputs.input_ids includes the whole prompt (chat template + image
        # placeholder tokens) -- slice it off so decode() returns only the
        # newly generated answer, not an echo of the question.
        gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen_only[0], skip_special_tokens=True).strip()
        text = text.rstrip(".").strip()
        return text or None

    @torch.no_grad()
    def confirm_arrival(self, rgb: np.ndarray, target_desc: str) -> Optional[bool]:
        """VQA arrival check on the FULL frame -- unlike caption() above,
        which classifies a cropped detection, arrival is a judgment about
        the robot's relationship to the whole scene (how close it looks,
        whether the object fills the frame), not just what the target crop
        looks like. Called by live_server.py's /confirm_arrival endpoint,
        itself only hit once Nav_new's metric depth-threshold stop has
        already fired (see nav_pipeline/remind_gui_vlm.py's
        VLMArrivalGate) -- this is a slower semantic confirmation layered
        ON TOP of that, never a replacement for it.

        Returns True/False, or None if the generated answer didn't parse
        as a clear yes/no (caller treats None as "not yet")."""
        if self.model is None:
            raise RuntimeError("InternVL is not loaded. Call load_model() before confirm_arrival().")

        prompt = self.ARRIVAL_PROMPT_TEMPLATE.format(target=target_desc)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        chat_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=rgb, text=chat_prompt, return_tensors="pt").to(
            self.device, torch.bfloat16
        )
        out_ids = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen_only[0], skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
        return None

    def _crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        bbox: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Identical crop convention to BlipCaptioner._crop -- duplicated
        rather than imported so the two captioner backends stay independent
        (either can be enabled/disabled/removed without touching the
        other)."""
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
