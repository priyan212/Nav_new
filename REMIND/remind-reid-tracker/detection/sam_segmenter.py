# detection/sam_segmenter.py
"""SAM 2.1 automatic mask generation -- class-agnostic segmentation-only
detector backend, replacing YOLO for live REMIND operation.

Unlike YoloSegmenter/DavisSegmenter, SAM has no classifier: every mask gets
the same constant class_id (0), so the rest of the pipeline (association's
per-class Hungarian partitioning in association/engine/assignment.py,
MemoryStore's class-based index) collapses onto one universal bucket
instead of needing a rewrite -- see REMIND_METHOD.md's per-class-partition
design, which was only ever driven by YOLO's output. A real per-object
label is filled in later by BLIP (features/blip_captioner.py) when an
object is first created, not here.

transformers' built-in "mask-generation" pipeline only targets SAM1
checkpoints, not SAM2, so this runs its own grid-point automatic mask
generation on top of the same Sam2Model/Sam2Processor pattern used
elsewhere in this project (see nav_pipeline/sam_segmenter.py's box-prompted
usage): batch every grid point as its own single-point prompt in one
forward pass, keep the best of the 3 multimask outputs per point by
predicted IoU, drop low-confidence/low-stability/degenerate-area masks,
then de-duplicate near-identical masks across points (many grid points
land on the same object) via greedy mask-IoU NMS -- the standard SAM
automatic-mask-generator recipe (Meta's original AMG).

Two more RGB-only filters reject flat-surface false positives (measured
live: SAM happily segments floor tiles, wall patches, and ceiling patches
as if they were discrete objects, and BLIP then dutifully captions them
as "black tiles"/"floor"/"ceiling" -- these are genuinely not objects, no
captioner fixes that after the fact):
  - min_mask_std: grayscale intensity std-dev inside the mask. A tile-sized
    patch of floor/wall/ceiling material is much flatter/more uniform than
    a real object of similar size (which has shading, edges, printed
    detail); reject masks below the threshold.
  - edge_touch_reject: floor/wall/ceiling segments characteristically hug a
    large fraction of one whole image border (floor along the bottom, wall
    along a side, ceiling along the top), unlike a discrete object that
    might clip a border but doesn't run its length. Reject large masks that
    do.
A real depth-based check (reject near-planar masks via a plane fit) isn't
available here: REMIND's own process only ever receives RGB frames over
the wire (see live_server.py's /infer contract) -- there is no depth
channel to fit a plane to without a bigger client/server contract change.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from detection.detection import Detection


class SamSegmenter:
    """
    Wrapper for class-agnostic SAM automatic mask generation.

    Public contract (matches YoloSegmenter/DavisSegmenter):
      - load_model()
      - segment(frame, frame_id, timestamp) -> list[Detection]
      - class_id_to_name
    """

    def __init__(self, config: dict, device: str):
        self.config = config or {}
        self.device = device

        self.sam_cfg = (self.config.get("sam", {}) or {})
        self.model_id = str(self.sam_cfg.get("model_id", "facebook/sam2.1-hiera-small"))
        self.points_per_side = int(self.sam_cfg.get("points_per_side", 16))
        self.points_batch_size = int(self.sam_cfg.get("points_batch_size", 64))
        self.pred_iou_thresh = float(self.sam_cfg.get("pred_iou_thresh", 0.86))
        self.stability_score_offset = float(self.sam_cfg.get("stability_score_offset", 1.0))
        self.stability_score_thresh = float(self.sam_cfg.get("stability_score_thresh", 0.90))
        self.min_mask_area_frac = float(self.sam_cfg.get("min_mask_area_frac", 0.001))
        self.max_mask_area_frac = float(self.sam_cfg.get("max_mask_area_frac", 0.85))
        self.mask_nms_iou_thresh = float(self.sam_cfg.get("mask_nms_iou_thresh", 0.7))
        self.border_margin_px = int(self.sam_cfg.get("border_margin_px", 4))

        # Flat-surface rejection (floor/wall/ceiling/tile false positives --
        # see module docstring). 0 disables a given filter.
        self.min_mask_std = float(self.sam_cfg.get("min_mask_std", 10.0))
        self.edge_touch_reject = bool(self.sam_cfg.get("edge_touch_reject", True))
        self.edge_touch_frac = float(self.sam_cfg.get("edge_touch_frac", 0.5))
        self.edge_touch_min_area_frac = float(self.sam_cfg.get("edge_touch_min_area_frac", 0.05))
        self.edge_touch_margin_px = int(self.sam_cfg.get("edge_touch_margin_px", 3))

        self.processor = None
        self.model = None
        # Single universal class -- see module docstring. Kept as a dict
        # (rather than None) so PerceptionEngine's ignored-classes-by-name
        # resolution (which reads yolo.class_id_to_name) keeps working.
        # NOT named "object": default_config.yaml's detector.ignored_classes
        # (meant for YOLO/DAVIS) already contains the literal string
        # "object" -- naming this class that silently filtered out every
        # single SAM detection via that name collision (caught in testing).
        self.class_id_to_name = {0: "unlabeled"}
        self.last_timings_seconds: dict = {}

    def load_model(self) -> None:
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()

    def _grid_points(self, h: int, w: int) -> np.ndarray:
        n = max(1, self.points_per_side)
        margin = int(self.border_margin_px)
        xs = np.linspace(margin, max(margin, w - 1 - margin), n)
        ys = np.linspace(margin, max(margin, h - 1 - margin), n)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        return grid.astype(np.float32)

    @staticmethod
    def _mask_stability_score(logits: np.ndarray, offset: float) -> float:
        """Fraction of pixels stable under a +/-offset logit threshold shift
        (Meta's SAM AMG stability-score heuristic) -- filters masks whose
        boundary is a coin-flip rather than a confident edge."""
        hi = float((logits > offset).sum())
        lo = float((logits > -offset).sum())
        if lo <= 0.0:
            return 0.0
        return hi / lo

    @staticmethod
    def _mask_std(gray: np.ndarray, mask: np.ndarray) -> float:
        """Grayscale intensity std-dev inside the mask -- low values mean a
        flat, low-detail region (paint, tile, plain wall material); real
        objects of similar size typically show shading/edges/printed
        detail and score noticeably higher."""
        vals = gray[mask]
        if vals.size == 0:
            return 0.0
        return float(vals.std())

    def _touches_border_heavily(self, mask: np.ndarray, h: int, w: int) -> bool:
        """True if the mask hugs a large fraction of the length of any
        single full image border -- the geometric signature of a flat
        surface (floor along the bottom, wall along a side, ceiling along
        the top) filling the frame from that edge inward, as opposed to a
        discrete object that might merely clip a border."""
        margin = max(1, int(self.edge_touch_margin_px))
        top = mask[:margin, :].any(axis=0)
        bottom = mask[-margin:, :].any(axis=0)
        left = mask[:, :margin].any(axis=1)
        right = mask[:, -margin:].any(axis=1)
        frac_top = float(top.sum()) / float(w)
        frac_bottom = float(bottom.sum()) / float(w)
        frac_left = float(left.sum()) / float(h)
        frac_right = float(right.sum()) / float(h)
        return max(frac_top, frac_bottom, frac_left, frac_right) >= self.edge_touch_frac

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = float(np.logical_and(a, b).sum())
        if inter <= 0.0:
            return 0.0
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0.0 else 0.0

    def _greedy_mask_nms(self, masks: List[np.ndarray], scores: List[float]) -> List[int]:
        order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
        keep: List[int] = []
        for i in order:
            if all(self._mask_iou(masks[i], masks[j]) < self.mask_nms_iou_thresh for j in keep):
                keep.append(i)
        return keep

    @torch.no_grad()
    def _run_grid_batch(self, image_embeddings, original_sizes, batch_pts: np.ndarray):
        """Prompt the (already-encoded) image with one point-batch.

        Passing image_embeddings (from get_image_embeddings(), computed ONCE
        per frame in segment()) instead of pixel_values skips re-running the
        Hiera vision encoder on every batch -- measured ~2x end-to-end
        speedup at points_per_side=16 (920ms -> ~440ms on an RTX 3090 Ti),
        since the encoder dominates cost and the mask decoder is cheap
        per-point."""
        input_points = [[[[float(px), float(py)]] for px, py in batch_pts]]
        input_labels = [[[1] for _ in batch_pts]]

        prompt_inputs = self.processor(
            input_points=input_points,
            input_labels=input_labels,
            original_sizes=original_sizes,
            return_tensors="pt",
        )
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}

        outputs = self.model(image_embeddings=image_embeddings, multimask_output=True, **prompt_inputs)

        iou_scores = outputs.iou_scores[0].float().cpu().numpy()  # (n_pts, n_masks)
        best_idx = iou_scores.argmax(axis=1)
        best_iou = iou_scores[np.arange(len(best_idx)), best_idx]

        low_res_masks = outputs.pred_masks[0].detach().cpu().numpy()  # (n_pts, n_masks, 256, 256)
        processed = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), prompt_inputs["original_sizes"].cpu()
        )[0]  # (n_pts, n_masks, H, W)

        return best_idx, best_iou, low_res_masks, processed

    def segment(self, frame, frame_id: int, timestamp: float) -> list:
        if self.model is None:
            raise RuntimeError("SAM is not loaded. Call load_model() before segment().")

        h, w = frame.shape[:2]
        # frame arrives BGR (cv2 convention, same as YoloSegmenter's input --
        # see perception_engine.py's frame_aligned); SAM/transformers expect RGB.
        frame_rgb = np.ascontiguousarray(frame[:, :, ::-1]) if frame.ndim == 3 and frame.shape[2] == 3 else frame
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        points = self._grid_points(h, w)

        with torch.no_grad():
            img_inputs = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)
            image_embeddings = self.model.get_image_embeddings(img_inputs["pixel_values"])
        original_sizes = img_inputs["original_sizes"].cpu()

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        batch_size = max(1, self.points_batch_size)
        for start in range(0, len(points), batch_size):
            batch_pts = points[start:start + batch_size]
            best_idx, best_iou, low_res_masks, processed = self._run_grid_batch(
                image_embeddings, original_sizes, batch_pts
            )

            for i in range(len(batch_pts)):
                idx = int(best_idx[i])
                if float(best_iou[i]) < self.pred_iou_thresh:
                    continue
                stability = self._mask_stability_score(low_res_masks[i, idx], self.stability_score_offset)
                if stability < self.stability_score_thresh:
                    continue
                mask = processed[i, idx].numpy().astype(bool)
                area_frac = float(mask.sum()) / float(h * w)
                if area_frac < self.min_mask_area_frac or area_frac > self.max_mask_area_frac:
                    continue
                if self.min_mask_std > 0.0 and self._mask_std(gray, mask) < self.min_mask_std:
                    continue
                if (
                    self.edge_touch_reject
                    and area_frac >= self.edge_touch_min_area_frac
                    and self._touches_border_heavily(mask, h, w)
                ):
                    continue
                all_masks.append(mask)
                all_scores.append(float(best_iou[i]))

        keep_idx = self._greedy_mask_nms(all_masks, all_scores)

        detections = []
        for det_id, i in enumerate(keep_idx):
            mask = all_masks[i]
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            geom = {"center": (float(xs.mean()), float(ys.mean())), "area": float(xs.size)}
            det = Detection(
                detection_id=int(det_id),
                class_id=0,
                frame_id=frame_id,
                timestamp=timestamp,
                bbox=bbox,
                mask=mask,
                confidence=float(all_scores[i]),
                geom=geom,
            )
            det.class_name = None
            det.original_class_name = None
            detections.append(det)

        self.last_timings_seconds = {}
        return detections
