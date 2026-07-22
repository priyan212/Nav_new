"""Grounding DINO open-vocabulary detector wrapper.

Loads IDEA-Research/grounding-dino-base from the local HF cache
(HF_HOME=/mnt/bigdisk/hf_cache) and returns the best box for a free-text
target phrase, e.g. "a red chair." -> (bbox, score).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class Detection:
    box: np.ndarray  # [x0, y0, x1, y1] pixels in the input image
    score: float
    label: str

    @property
    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        # Grounding DINO expects lowercase phrases terminated by periods.
        text = text.strip().lower()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def detect(self, image: np.ndarray, text: str) -> List[Detection]:
        """image: HxWx3 uint8 RGB. Returns detections sorted by score desc."""
        pil = Image.fromarray(image)
        prompt = self._normalize_prompt(text)
        inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        dets = [
            Detection(box=np.array(b, dtype=np.float32), score=float(s), label=str(lbl))
            for b, s, lbl in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results.get("text_labels", results.get("labels", [])),
            )
        ]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_best(self, image: np.ndarray, text: str) -> Optional[Detection]:
        dets = self.detect(image, text)
        return dets[0] if dets else None
