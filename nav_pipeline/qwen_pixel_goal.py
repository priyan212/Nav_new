"""Qwen2.5-VL instruction-grounded pixel goal for NavDP trajectory following.

Free-text navigation instructions ("walk through the doorway and stop near
the desk") aren't Grounding DINO's job -- DINO finds a named OBJECT, not a
described maneuver. This lets a frozen Qwen2.5-VL-7B-Instruct ground the
instruction directly to a 2D pixel (u, v) in the current frame -- the next
waypoint to head toward. pipeline.py turns that pixel into a 3D goal point
via depth (goal_utils.pixel_depth_to_point) and hands it to NavDP exactly
like a DINO detection would -- NavDP samples trajectories against it and the
pipeline's existing trajectory-selection/obstacle-guard machinery follows
it, completely unchanged.

Sibling of MARS/mars-habitatsim/navdp/navdp/extensions/system2_pixel_goal.py
(the Habitat-sim version, which renders the pixel into a goal-MASK channel
for a different NavDP variant's own point-conditioning input). Kept as a
separate module rather than shared code: different conda env/subproject,
and this one feeds the real-rover pipeline's depth-derived 3D goal path
instead of a mask channel.

Distinct from qwen_search_guide.py: that module only ever steers the
SEARCH state while chasing a DINO *text target* (a bearing, not a point,
and only a fallback when DINO can't see the target). This module is the
PRIMARY goal source for a free-text *instruction* -- there is no DINO
target in this mode at all.

Between Qwen calls (throttled -- see QwenInstructionGate), pipeline.py does
NOT hold a stale pixel: it lets GoalBelief propagate the last 3D goal by
ego-motion, the same belief-coasting machinery a lost DINO detection already
falls back to. That is more physically correct than re-using a fixed pixel
column across ticks where the rover has since turned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_PROMPT = (
    "You are guiding a mobile ground robot. Instruction: \"{instruction}\". "
    "Look at this image and point to the single best next waypoint pixel to "
    "move toward in order to follow the instruction. Reply with only one "
    "pixel coordinate as (x, y)."
)

# Distractor-robustness fix: asking for ONE point forces a guess the instant
# there's real ambiguity (e.g. two similar doors) -- there's no way to tell
# "confidently the only door" apart from "arbitrarily picked one of two"
# from a single (x, y) alone. Asking for several ranked, confidence-scored
# candidates instead lets pipeline.py score each against BOTH goal
# continuity (does this look like what we're already heading toward) and
# obstacle cost (see PipelineConfig.qwen_max_candidates and _step_inner's
# instruction branch) -- closer to the "semantic score minus collision
# cost" combination arxiv 2605.19420 ("Beyond Waypoints: Dual-Heatmap
# Grounding") uses a purpose-trained heatmap network for, adapted here to a
# frozen general VLM that can only be prompted for a handful of discrete
# points, not a dense field.
MULTI_CANDIDATE_PROMPT = (
    "You are guiding a mobile ground robot. Instruction: \"{instruction}\". "
    "Look at this image and identify up to {max_candidates} candidate "
    "waypoint pixels that could satisfy the instruction. Point to a spot ON "
    "THE FLOOR/GROUND at or just in front of the target, NOT on the "
    "target's own surface (a door frame, wall, or object body) and not "
    "through a doorway/opening into whatever is beyond it -- a robot-height "
    "camera cannot get a real distance reading through a gap or off a flat "
    "surface far away, only off the floor in front of it. If the "
    "instruction is to go through a doorway/opening, point at the floor "
    "centered in the opening, not the top of the frame or the room beyond "
    "it. If there is more than one plausible match (e.g. two similar doors "
    "or openings), list each one separately instead of picking just one. "
    "List your best candidate first. Reply with ONE candidate per line, "
    "each formatted exactly as: x, y, confidence -- where confidence is "
    "your certainty from 0 (unsure) to 1 (certain) that this candidate is "
    "correct."
)


def parse_pixel_coordinate(text: str, image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Pull the first (x, y) pixel pair out of a VLM answer.

    Handles bare ``(x, y)``, ``x, y``, JSON-ish ``[x, y]`` and Qwen box tags.
    If the numbers look normalized (<=1) they are scaled by the image size; if
    they look like Qwen's 0-1000 grounding scale they are rescaled too.
    """
    w, h = image_size
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:  # normalized
        return x * (w - 1), y * (h - 1)
    if x > w or y > h:  # likely Qwen 0-1000 grounding scale
        return x / 1000.0 * (w - 1), y / 1000.0 * (h - 1)
    return x, y


def parse_pixel_candidates(text: str, image_size: Tuple[int, int],
                           max_candidates: int = 3) -> List["PixelGoal"]:
    """Pull up to max_candidates (x, y[, confidence]) groups out of a VLM
    answer, one per line (see MULTI_CANDIDATE_PROMPT). Confidence defaults
    to a rank-based decay (1.0, 0.7, 0.5, ...) when a line omits it or the
    model didn't follow the format -- still usable, just less informative
    than a real per-candidate confidence. Same normalization rules as
    parse_pixel_coordinate, applied per line independently.
    """
    w, h = image_size
    default_confs = [1.0, 0.7, 0.5, 0.35, 0.25]
    results: List[PixelGoal] = []
    # A numbered list ("1. 300, 200, 0.95") would otherwise have its
    # leading "1." misread as x itself, shifting every value over by one
    # (caught by testing before this shipped) -- strip a SHORT (1-2 digit,
    # so a real 3-digit pixel coordinate can never match) leading list
    # marker before grabbing numbers, rather than requiring strict
    # comma-separation (which broke on label-prefixed replies like
    # "x=300, y=200, confidence=0.8" -- also caught by testing).
    leading_marker = re.compile(r"^\s*[\(\[]?\d{1,2}[.\):\]]\s+")
    lines = [ln for ln in re.split(r"[\n;]", text) if ln.strip()] or [text]
    for line in lines:
        line = leading_marker.sub("", line, count=1)
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(nums) < 2:
            continue
        x, y = float(nums[0]), float(nums[1])
        raw_conf = float(nums[2]) if len(nums) >= 3 else None
        conf = raw_conf if raw_conf is not None and 0.0 <= raw_conf <= 1.0 else None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x, y = x * (w - 1), y * (h - 1)
        elif x > w or y > h:
            x, y = x / 1000.0 * (w - 1), y / 1000.0 * (h - 1)
        if conf is None:
            conf = default_confs[len(results)] if len(results) < len(default_confs) else 0.2
        results.append(PixelGoal(float(x), float(y), float(conf), in_view=(0 <= x < w and 0 <= y < h)))
        if len(results) >= max_candidates:
            break
    return results


# --- Direct motion instructions -------------------------------------- #
# Simple imperative commands ("go straight", "turn left", "turn right") name
# a MANEUVER, not a landmark -- there's no salient pixel in the frame for a
# VLM to point at ("where in this image is 'turn right'?"), so routing them
# through QwenVLPixelGoal's grounding produced unreliable, inconsistent
# waypoints (real complaint 2026-09-07). Recognized here as an EXACT-phrase
# match only, never a substring -- "go to the door on the left" must still
# ground normally through Qwen, only a bare "left" on its own is a direct
# command. Matches turn into a fixed local-frame offset that pipeline.py
# feeds through the SAME TRACK/obstacle-guard/NavDP path a grounded goal
# uses (see its instruction branch), so full obstacle-avoidance safety is
# unchanged. Recomputed fresh every tick -- no Qwen throttle needed, it's a
# dict lookup, not a model call -- so it acts as a persistent "keep doing
# this" command until a different instruction (a landmark phrase, or a new
# directional one) is sent.
_FORWARD_PHRASES = frozenset({
    "go straight", "straight", "straight ahead", "go forward",
    "move forward", "forward", "keep going", "keep going straight",
    "continue straight", "drive straight", "go ahead",
})
_LEFT_PHRASES = frozenset({
    "turn left", "go left", "left", "veer left", "turn to the left",
})
_RIGHT_PHRASES = frozenset({
    "turn right", "go right", "right", "veer right", "turn to the right",
})
_STOP_PHRASES = frozenset({"stop", "halt", "stop now"})


def parse_direct_motion(instruction: str) -> Optional[str]:
    """Exact-phrase match for a bare directional/stop instruction -> its
    kind ("forward" | "left" | "right" | "stop"). Returns None if
    `instruction` isn't one of the recognized short phrases -- i.e. it's a
    landmark instruction and should ground through Qwen as usual.

    Classification only -- this module has no pose/heading tracking to
    turn a kind into an actual command. pipeline.py does that: "forward"
    becomes a look-ahead NavDP goal point (obstacle-aware, like a grounded
    goal), "left"/"right" become a heading-bounded in-place turn (exactly
    PipelineConfig.direct_turn_deg degrees, tracked via odometry pose --
    real user spec 2026-09-07: turns must be an exact 90 degrees, not an
    open-ended curve-toward-a-point, which is what an earlier version of
    this function produced), "stop" becomes zero velocity.
    """
    key = " ".join(instruction.strip().lower().split())
    if not key:
        return None
    if key in _STOP_PHRASES:
        return "stop"
    if key in _FORWARD_PHRASES:
        return "forward"
    if key in _LEFT_PHRASES:
        return "left"
    if key in _RIGHT_PHRASES:
        return "right"
    return None


# --- Compound directional instructions --------------------------------- #
# "go straight up to 2.9 meters and turn right" is NOT one bare direct-motion
# phrase (parse_direct_motion above returns None for it) and it isn't a
# landmark either -- it's a SEQUENCE of directional steps with a distance
# trigger between them. A single Qwen grounding call can't represent "do A,
# then after N meters do B" as one pixel: sending the whole sentence to
# Qwen just gets some single point pointed at (real failure observed
# 2026-09-07 -- 30+ ticks of far-mode/rejected-candidate, the "turn right"
# clause never once influenced the robot). Parsed here into an ordered list
# of MotionStep and stepped through by pipeline.py using real odometry
# distance -- never sent to Qwen at all. Only engages when EVERY clause is
# a recognized directional/stop phrase (reusing the same clause vocabulary
# as parse_direct_motion, matched as a keyword within each clause rather
# than an exact full-string match since a clause is already an isolated
# fragment); a compound instruction naming a real landmark anywhere ("go
# straight to the kitchen then turn right") returns None and falls through
# to Qwen grounding on the whole sentence, unchanged from before this
# existed.
_CLAUSE_SPLIT_RE = re.compile(r"\s*,\s*|\b(?:and\s+then|then|and)\b", re.IGNORECASE)
# "1.6 more meters"/"1.6 further metres" -- the optional filler between
# number and unit is common in natural phrasing ("after 1.6 more meters
# stop at...") and was silently defeating the plain number+unit match
# (caught by testing before this shipped), wrongly treating a perfectly
# distanced clause as undistanced and aborting the whole compound parse.
_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:more|further|additional)?\s*(?:m|meter|meters|metre|metres)\b",
    re.IGNORECASE)
# Whitelist, not a keyword search: a clause classifies as directional only
# if EVERY word in it (after stripping the distance) is drawn from this set
# -- one unrecognized word means a real content word (a landmark) is
# present and the whole instruction must be abandoned to Qwen. A keyword
# search alone was tried first and wrongly matched "go straight to the
# kitchen" as a bare forward clause just because it contains "straight"
# (caught by testing before this shipped) -- this whitelist is what fixes
# that: "kitchen" isn't in it, so the clause (and the whole instruction)
# correctly falls through instead.
_ALLOWED_CLAUSE_WORDS = frozenset({
    "go", "turn", "veer", "move", "keep", "going", "continue", "drive",
    "to", "the", "towards", "toward", "now",
    "straight", "forward", "ahead", "left", "right", "stop", "halt",
    "up", "upto", "for", "about", "approx", "approximately",
})


@dataclass
class MotionStep:
    kind: str                       # "forward" | "left" | "right" | "stop" | "landmark"
    distance_m: Optional[float]     # None = open-ended -- holds until the instruction changes
    text: Optional[str] = None      # "landmark" only: the clause text to Qwen-ground for BEARING each tick


def _classify_clause(clause: str) -> Optional[str]:
    """One clause of a compound instruction -> a MotionStep.kind, or None
    if it isn't a recognized directional/stop clause (i.e. it names a real
    landmark and the whole instruction should be abandoned to Qwen)."""
    words = re.findall(r"[a-z]+", _DISTANCE_RE.sub(" ", clause).lower())
    if not words or any(w not in _ALLOWED_CLAUSE_WORDS for w in words):
        return None
    has_left, has_right = "left" in words, "right" in words
    if "stop" in words or "halt" in words:
        return "stop"
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if "straight" in words or "forward" in words or "ahead" in words:
        return "forward"
    return None


def parse_direct_motion_sequence(instruction: str) -> Optional[List[MotionStep]]:
    """Compound instruction (clauses split on "and"/"then"/",") -> ordered
    [MotionStep, ...], or None if it isn't one (a single bare phrase like
    "turn left" still goes through parse_direct_motion above, not this --
    this only fires for multi-clause instructions parse_direct_motion
    already rejected).

    Each clause becomes either a pure directional/stop step (see
    _classify_clause) or, if it carries a distance but ISN'T purely
    directional ("go straight upto 1.5 meters passing a container"), a
    "landmark" step: Qwen grounds the clause's own text for BEARING every
    tick this step is active (pipeline.py's _ground_landmark_bearing) --
    but never for ARRIVAL; the pipeline advances past it once `distance_m`
    metres have been travelled, exactly like a plain "forward" step, using
    real odometry the same way a distance-bounded direct-motion step
    already does. A clause that's neither purely directional NOR carries a
    distance (a landmark named with no distance at all, e.g. bare "go to
    the kitchen") still aborts the WHOLE parse -- there's no odometry-
    independent way to decide when an undistanced landmark leg is "done"
    without touching the shared belief-coast/SEARCH arrival logic every
    other pipeline mode (DINO included) also relies on, which is
    deliberately out of scope here; that clause's instruction falls
    through to Qwen grounding on the whole sentence instead, unchanged
    from before this "landmark" step kind existed.

    distance_m on a directional (non-landmark) step is POSITION-based
    (straight-line displacement from where that step began, via world-
    frame odometry pose), for every kind including turns: "turn right for
    1 meter" means "keep curving right until you've moved 1m from where
    the turn started", not a heading/angle bound -- there is no heading-
    delta tracking for that phrasing, only position (contrast a bare "turn
    right" with no distance, which IS heading-bounded -- see
    PipelineConfig.direct_turn_deg).
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(instruction) if c.strip()]
    if len(clauses) < 2:
        return None  # not a compound instruction
    steps: List[MotionStep] = []
    for i, clause in enumerate(clauses):
        kind = _classify_clause(clause)
        dist_match = _DISTANCE_RE.search(clause)
        dist = float(dist_match.group(1)) if dist_match else None
        if kind is not None:
            steps.append(MotionStep(kind, None if kind == "stop" else dist))
        elif dist is not None:
            steps.append(MotionStep("landmark", dist, text=clause))
        elif i == len(clauses) - 1:
            # Undistanced landmark clause, but it's the LAST leg -- allowed
            # as a special case (unlike a mid-sequence one, just below):
            # arrival is decided by the normal proximity-based
            # stop_distance check, same as a standalone landmark
            # instruction already uses, reused ONLY for this terminal step
            # (see pipeline.py's landmark dispatch) -- never the shared
            # belief-coast/SEARCH fallback every other pipeline mode (DINO
            # included) also relies on, so this stays fully scoped to the
            # sequence path.
            steps.append(MotionStep("landmark", None, text=clause))
        else:
            return None  # a MID-sequence undistanced landmark clause -- there's no signal
            # to know when to advance past it to a LATER step; abandon, let Qwen ground
            # the whole sentence instead.
    return steps


@dataclass
class PixelGoal:
    u: float          # column (x) in image pixels
    v: float          # row (y) in image pixels
    confidence: float
    in_view: bool = True


class QwenVLPixelGoal:
    """Frozen Qwen2.5-VL-7B-Instruct grounder: (rgb, instruction) -> PixelGoal.

    Inference only -- no gradients, no finetuning. load_in_4bit=True
    (default, needs bitsandbytes) measured ~6.2GB VRAM / ~0.7-1.0s per call
    on a 3090 Ti alongside the rest of nav_pipeline's stack; fp16 needs
    ~16.6GB / ~0.4-0.9s. See qwen_search_guide.QwenVLSearchGuide for the
    same numbers measured on that sibling module -- this one loads an
    equivalent model the same way.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_new_tokens: int = 48,
        max_new_tokens_multi: int = 160,
        prompt_template: Optional[str] = None,
        multi_prompt_template: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.max_new_tokens_multi = int(max_new_tokens_multi)
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self.multi_prompt_template = multi_prompt_template or MULTI_CANDIDATE_PROMPT
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        # Model-class-agnostic load so a Qwen3-VL id (needs
        # Qwen3VLForConditionalGeneration) works alongside the original
        # Qwen2.5-VL default: AutoModelForImageTextToText reads whichever
        # class the checkpoint's own config names. Fall back to the explicit
        # 2.5-VL class on a transformers too old to expose the Auto mapping.
        try:
            from transformers import AutoModelForImageTextToText as _VLMClass
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLMClass

        kwargs = {"torch_dtype": torch.float16}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        self._model = _VLMClass.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def ground(self, rgb: np.ndarray, instruction: str) -> Optional[PixelGoal]:
        """Single best point -- thin convenience wrapper, kept for callers
        that don't need candidate scoring (e.g. qwen_search_guide's sibling
        use case has no obstacle/continuity scoring to apply). Prefer
        ground_candidates() wherever a distractor could plausibly be in
        view -- see this module's docstring on why."""
        candidates = self.ground_candidates(rgb, instruction, max_candidates=1)
        return candidates[0] if candidates else None

    def ground_candidates(self, rgb: np.ndarray, instruction: str,
                          max_candidates: int = 3) -> List[PixelGoal]:
        """Up to max_candidates ranked, confidence-scored waypoint pixels
        for `instruction` -- see MULTI_CANDIDATE_PROMPT and this module's
        docstring. Caller (pipeline.py) scores each against goal continuity
        and obstacle cost and picks the winner; this method does no
        scoring of its own, just grounding."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        prompt = self.multi_prompt_template.format(instruction=instruction, max_candidates=max_candidates)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens_multi, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(gen, skip_special_tokens=True)
        return parse_pixel_candidates(answer, image_size=(w, h), max_candidates=max_candidates)


class QwenInstructionGate:
    """Wall-clock throttle: is it time to call Qwen again? Same pattern as
    sam_period_s/scene_tag_period_s/qwen_search_period_s elsewhere in this
    pipeline -- 7B inference is seconds, not one pipeline tick. Between due
    ticks, pipeline.py lets GoalBelief coast the goal by ego-motion instead
    of holding a stale pixel here (see this module's docstring)."""

    def __init__(self, period_s: float = 1.5):
        self.period_s = float(period_s)
        self._last_t = 0.0

    def due(self, now: float) -> bool:
        if now - self._last_t < self.period_s:
            return False
        self._last_t = now
        return True

    def reset(self) -> None:
        self._last_t = 0.0
