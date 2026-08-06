"""Free-text targeting ("go to the black chair") -> a specific object_id.

BLIP's caption text (object_map.py's "caption" field) is internal
bookkeeping only and is often long/unstable ("black chair with black
board" one tick, something else the next) -- matching against it directly
would inherit that instability. Instead this module scores a query against
each remembered object's CACHED CLIP IMAGE embedding (object_map.py's
"embedding" field, set once when the object is first seen and never
overwritten -- see ClipObjectMatcher/its caller in remind_gui.py), using
CLIP's text encoder purely as the query side. Once resolved to an
object_id, the caller just sets that as the normal ID-based target --
REMIND's per-tick matching is already `object_id == target_id` (see
remind_gui.remind_inference_loop), so "multiple instances of the same
class in frame" is a non-issue: resolution happens once, then everything
downstream locks onto that one ID exactly as if it had been typed
directly.

Two additional patterns, resolved the same way relational_target.py
already resolves them against pixel-space DINO detections, just against
object_map's world positions instead:

- RELATIONAL ("chair near the window"): rank objects by CLIP similarity to
  "window" to find the anchor's world position, then rank CLIP-plausible
  "chair" candidates by world-frame distance to that anchor.
- POSITIONAL ("leftmost chair"): rank CLIP-plausible "chair" candidates by
  their position in the rover's CURRENT local frame (world_to_local against
  the live pose), not a stored frame -- "leftmost" only means something
  relative to where the rover is standing right now.

Falls back to plain top-CLIP-score matching if neither pattern is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .object_map import ObjectMap, world_to_local
from .relational_target import parse_positional_target, parse_relational_target

# Typed into the GUI's free-text field, an operator naturally leads with an
# imperative ("go to the chair near door") that relational_target.py's
# parsers were never written to expect -- they were built for a bare
# category phrase (pipeline.py's target_text). Left unstripped, "go to the
# chair near door" splits into target="go to the chair" / anchor="door"
# (the article-stripper only strips from the very START of a string, and
# "go to the chair" starts with "go", not "the"), which then gets embedded
# as the CLIP text query "a photo of a go to the chair" -- a garbled prompt
# that won't score well against any real chair crop. Strip ONE leading
# command phrase (longest first, so "go to " doesn't leave a dangling "to")
# before any parsing happens.
_COMMAND_PREFIXES = (
    "go to the ", "go to ", "navigate to the ", "navigate to ",
    "take me to the ", "take me to ", "drive to the ", "drive to ",
    "head to the ", "head to ", "go find the ", "go find ",
    "find the ", "find ",
)


def _strip_command_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for prefix in _COMMAND_PREFIXES:
        if low.startswith(prefix):
            return t[len(prefix):].strip()
    return t


def _as_tensor(out):
    """CLIPModel.get_image_features/get_text_features's return shape is
    transformers-version-dependent: some versions return the raw pooled
    embedding tensor directly, others (e.g. this env's 5.8.1, decorated with
    @can_return_tuple) return a BaseModelOutputWithPooling wrapper with the
    actual (already visual/text-projected) embedding at .pooler_output.
    Handle both so this doesn't silently break on the next transformers
    bump either."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


class ClipObjectMatcher:
    """Embedding-based CLIP wrapper, distinct from clip_verifier.ClipVerifier
    (which only ever does a live crop-vs-text-list softmax verify). Here the
    image side is embedded ONCE per object and cached (object_map.py), and
    the text side is embedded fresh per query -- so it needs independent
    embed_crop/embed_text methods rather than verify()'s single combined
    call."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def embed_crop(self, rgb: np.ndarray, box: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[List[float]]:
        """box: [x0,y0,x1,y1] px. mask: optional full-frame bool HxW -- if
        given, background outside the mask is white-filled within the crop
        (same convention as blip_captioner.py's _crop) so the embedding
        describes the object, not its surrounding clutter."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgb[y0:y1, x0:x1]
        if mask is not None and mask.shape[:2] == (h, w):
            local_mask = mask[y0:y1, x0:x1]
            if local_mask.shape[:2] == crop.shape[:2]:
                crop = crop.copy()
                crop[~local_mask] = 255
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        feat = _as_tensor(self.model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=[f"a photo of a {text}"], return_tensors="pt", padding=True).to(self.device)
        feat = _as_tensor(self.model.get_text_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().tolist()


def cosine_sim(a: List[float], b: List[float]) -> float:
    av, bv = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(av, bv))


@dataclass
class QueryResult:
    object_id: Optional[int]
    ambiguous: bool
    candidates: List[Tuple[int, float]] = field(default_factory=list)  # (object_id, score), best first
    message: str = ""


# Candidates within this fraction of the top score are treated as "also
# plausible" for AMBIGUOUS reporting, and (for relational/positional
# queries) as the pool ranked by geometry instead of trusting CLIP's exact
# ranking among near-ties.
_SCORE_MARGIN = 0.03


def _rank_by_clip(query_text: str, object_map: ObjectMap, matcher: ClipObjectMatcher) -> List[Tuple[int, float]]:
    text_emb = matcher.embed_text(query_text)
    scored = []
    for oid, entry in object_map.items():
        emb = entry.get("embedding")
        if emb is None:
            continue
        scored.append((oid, cosine_sim(text_emb, emb)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def resolve_object_query(
    query: str,
    object_map: ObjectMap,
    matcher: ClipObjectMatcher,
    pose: Optional[Tuple[float, float, float]] = None,
) -> QueryResult:
    query = _strip_command_prefix(query)
    if not query:
        return QueryResult(None, False, [], "empty query")

    relational = parse_relational_target(query)
    positional = None if relational is not None else parse_positional_target(query)
    target_phrase = relational[0] if relational is not None else (positional[0] if positional is not None else query)

    candidates = _rank_by_clip(target_phrase, object_map, matcher)
    if not candidates:
        return QueryResult(None, False, [], f"no remembered object matches '{target_phrase}'")

    best_score = candidates[0][1]
    plausible = [c for c in candidates if (best_score - c[1]) <= _SCORE_MARGIN]

    if relational is not None:
        _, anchor_phrase = relational
        anchor_candidates = _rank_by_clip(anchor_phrase, object_map, matcher)
        if not anchor_candidates:
            # can't resolve "near X" without a remembered X -- fall back to
            # plain top CLIP score on the target phrase alone.
            pass
        else:
            anchor_id, _ = anchor_candidates[0]
            anchor_entry = object_map.get(anchor_id)
            anchor_xy = (anchor_entry["world_x"], anchor_entry["world_y"])
            ranked = sorted(
                plausible,
                key=lambda c: np.hypot(
                    object_map.get(c[0])["world_x"] - anchor_xy[0],
                    object_map.get(c[0])["world_y"] - anchor_xy[1],
                ),
            )
            return QueryResult(ranked[0][0], len(plausible) > 1, candidates,
                                f"'{target_phrase}' near '{anchor_phrase}' (ID {anchor_id})")

    if positional is not None and pose is not None:
        _, position = positional
        ranked = sorted(plausible, key=lambda c: world_to_local(
            (object_map.get(c[0])["world_x"], object_map.get(c[0])["world_y"]), pose)[1])
        if position == "leftmost":
            chosen = ranked[-1]  # max y (left, ROS convention: +y is left)
        elif position == "rightmost":
            chosen = ranked[0]   # min y (right)
        else:  # "middle"
            chosen = ranked[len(ranked) // 2]
        return QueryResult(chosen[0], False, candidates, f"'{target_phrase}' ({position})")

    ambiguous = len(plausible) > 1
    msg = f"best match for '{target_phrase}'" + (" (ambiguous)" if ambiguous else "")
    return QueryResult(candidates[0][0], ambiguous, candidates, msg)
