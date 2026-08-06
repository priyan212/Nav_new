#!/usr/bin/env python3
"""Live per-frame REMIND re-identification server.

Nav_new's rover pipeline (nav_pipeline/, a separate conda env -- see
../SETUP.md) wants REMIND's persistent per-object identities on every
camera frame, but REMIND's own torch/transformers/ultralytics pins
(requirements.txt: torch 2.11+cu128, transformers 5.14) are incompatible
with that env (torch 2.5.1+cu118, transformers 5.9), so it can't be
imported in-process. Instead this runs as a standalone service inside
REMIND's own .venv and answers over local HTTP -- see
nav_pipeline/remind_client.py for the caller side and
launch_rover_remind.sh for how the two processes are brought up together.

Detection backend: SAM 2.1 automatic mask generation (class-agnostic --
detection/sam_segmenter.py), not YOLO. There's no classifier, so every
newly-tracked object's class_name comes from a one-shot InternVL
classification call (features/internvl_classifier.py, run once per object
at creation, not per frame; prompted as a classifier -- "what object is
this, one/two words" -- not a free-form captioner, which is what actually
fixed BLIP's instability on flat floor/wall/ceiling patches, e.g. captioning
them as "black tiles"). Pass --no-internvl to fall back to the generic
"object" label and skip that model entirely, or --use-blip to go back to
the old BLIP captioner (features/blip_captioner.py) instead.

Endpoints
---------
  GET  /health   -> {"status": "ok", "frame_count": int}
  POST /infer      body: one JPEG-encoded frame (BGR/cv2 convention)
                    -> {"frame_id": int, "objects": [
                         {"det_id", "object_id" (persistent, null if not
                          yet confirmed), "kind" (match|new|ambiguous|
                          provisional|detection), "class_name" (an InternVL
                          classification once the object has been confirmed/
                          created, e.g. "chair"; null for detections
                          that haven't been assigned an object_id yet),
                          "confidence", "bbox": [x0,y0,x1,y1],
                          "mask_bbox": [x0,y0,x1,y1] (int, rounded -- the
                          crop region "mask_png_b64" was encoded from) or
                          null, "mask_png_b64": base64 PNG of the mask
                          CROPPED to mask_bbox (not full-frame -- REMIND's
                          own SAM backend already produced this mask as
                          part of its own per-object descriptors, see
                          detection/detection.py's Detection.mask; sending
                          only the bbox-sized crop keeps the payload small)
                          or null if this detection had no mask}, ...]}
  POST /reset      clears the object memory catalogue and restarts frame
                    numbering from 0, WITHOUT reloading SAM/DINOv3/BLIP
                    (fast) -> {"status": "reset"}
  POST /confirm_arrival?target=<url-encoded text>
                    body: one JPEG-encoded frame (BGR/cv2 convention), same
                    as /infer -- asks the already-loaded InternVL model
                    (features/internvl_classifier.py's confirm_arrival) a
                    full-frame yes/no VQA question: has the robot reached
                    `target`? -> {"arrived": true|false|null} (null = the
                    model's answer didn't parse as yes/no). 501 if InternVL
                    wasn't loaded (server started with --no-internvl or
                    --use-blip). Used by nav_pipeline/remind_gui_vlm.py as
                    a confirmation layer on top of (never instead of) the
                    metric depth-threshold STOP -- see that module's
                    docstring.

Run (inside this repo's .venv -- see run_live_server.sh):
    python scripts/live_server.py --port 8765
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_loader import Config  # noqa: E402
from memory.memory_store import MemoryStore  # noqa: E402
from pipeline.initialization import initialize_system  # noqa: E402
from pipeline.reid_pipeline import ReIDPipeline  # noqa: E402


def _build_config(args: argparse.Namespace) -> dict:
    cfg = Config(args.config).to_dict()
    cfg.setdefault("paths", {})["output_dir"] = str(args.output_dir)
    det_cfg = cfg.setdefault("detector", {})
    det_cfg["backend"] = "sam"
    # Name-based ignored_classes (default_config.yaml's list, meant for
    # YOLO/DAVIS's real class names) doesn't apply to a class-agnostic
    # backend -- SAM's own area-fraction filtering (sam.min/max_mask_area_frac)
    # does that job instead. Leaving the default list in place caused every
    # detection to be silently dropped (see detection/sam_segmenter.py).
    det_cfg["ignored_classes"] = []

    sam_cfg = cfg.setdefault("sam", {})
    sam_cfg["points_per_side"] = int(args.sam_points_per_side)

    cfg.setdefault("system", {})["input_width_size"] = int(args.input_width)
    cfg.setdefault("runtime", {})["device"] = str(args.device)

    blip_cfg = cfg.setdefault("blip", {})
    blip_cfg["enabled"] = bool(args.use_blip)

    internvl_cfg = cfg.setdefault("internvl", {})
    internvl_cfg["enabled"] = bool(not args.no_internvl and not args.use_blip)

    # This runs unattended as a background service -- silence REMIND's
    # per-frame console timing table and association/update debug dumps
    # (both gated by these top-level flags; see association/engine/
    # data_association.py's want_debug = debug.enabled AND debug.<stage>.enabled).
    cfg.setdefault("timing", {})["enabled"] = False
    cfg.setdefault("debug", {})["enabled"] = False
    return cfg


def _bbox_to_original_space(bbox, transforms: dict, frame_hw: Tuple[int, int]) -> list:
    """REMIND detects on a resized+patch-aligned COPY of the submitted frame
    (perception/perception_engine.py: resize_keep_aspect_by_width, then
    align_frame_to_patches_crop) -- bbox/mask on p_out.detections are in
    THAT copy's pixel space, not the original frame's. Map back using the
    same transforms PerceptionEngine.process_frame recorded (p_out.transforms),
    so the API always returns coordinates in the frame the caller sent."""
    scale_in = float(transforms.get("scale_in", 1.0)) or 1.0
    align_meta = transforms.get("align_meta", {}) or {}
    crop = align_meta.get("crop", None)
    ox, oy = (float(crop[0]), float(crop[1])) if crop is not None else (0.0, 0.0)
    h, w = frame_hw
    x0, y0, x1, y1 = [float(v) for v in bbox]
    out = [(x0 + ox) / scale_in, (y0 + oy) / scale_in, (x1 + ox) / scale_in, (y1 + oy) / scale_in]
    return [max(0.0, min(w, out[0])), max(0.0, min(h, out[1])),
            max(0.0, min(w, out[2])), max(0.0, min(h, out[3]))]


def _mask_to_original_space(mask: np.ndarray, transforms: dict, frame_hw: Tuple[int, int]) -> np.ndarray:
    """Same mapping as _bbox_to_original_space, for the pixel mask: pad the
    aligned-space mask back into the pre-crop (resized) canvas at its exact
    crop offset, then scale that canvas up to the original frame size."""
    align_meta = transforms.get("align_meta", {}) or {}
    crop = align_meta.get("crop", None)
    proc_size = align_meta.get("orig_size", None)  # (h, w) of the resized frame, BEFORE the align-crop

    m = mask.astype(np.uint8, copy=False)
    if crop is not None and proc_size is not None:
        x0, y0, x1, y1 = [int(v) for v in crop]
        h_proc, w_proc = int(proc_size[0]), int(proc_size[1])
        canvas = np.zeros((h_proc, w_proc), dtype=np.uint8)
        canvas[y0:y1, x0:x1] = m
    else:
        canvas = m

    h, w = frame_hw
    if canvas.shape[:2] != (h, w):
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)
    return canvas.astype(bool, copy=False)


def _encode_mask_crop(mask: np.ndarray, bbox, frame_hw: Tuple[int, int]) -> Tuple[Optional[str], Optional[list]]:
    """Crop the (already original-frame-resolution) mask to bbox and
    PNG+base64-encode just that crop -- masks are mostly background, so
    sending the full frame-sized array every tick would be needless
    payload for no benefit."""
    h, w = frame_hw
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None, None
    crop = mask[y0:y1, x0:x1]
    if not np.any(crop):
        return None, None
    ok, buf = cv2.imencode(".png", (crop.astype(np.uint8) * 255))
    if not ok:
        return None, None
    return base64.b64encode(buf.tobytes()).decode("ascii"), [x0, y0, x1, y1]


def _pack_detections(p_out, u_out, frame_hw: Tuple[int, int], memory=None) -> list:
    transforms = getattr(p_out, "transforms", {}) or {}
    entries: dict = {}
    for item in getattr(u_out, "matches", []) or []:
        entries[int(item["det_id"])] = {"kind": "match", "object_id": int(item["object_id"])}
    for item in getattr(u_out, "created", []) or []:
        entries[int(item["det_id"])] = {"kind": "new", "object_id": int(item["object_id"])}
    for item in getattr(u_out, "ambiguous", []) or []:
        entries[int(item["det_id"])] = {"kind": "ambiguous", "object_id": None}
    for item in getattr(u_out, "provisional", []) or []:
        entries[int(item["det_id"])] = {"kind": "provisional", "object_id": None}

    objects = []
    for det in getattr(p_out, "detections", None) or []:
        det_id = int(getattr(det, "detection_id", -1))
        entry = entries.get(det_id, {"kind": "detection", "object_id": None})
        raw_bbox = getattr(det, "bbox", None)
        if raw_bbox is None:
            continue
        bbox = _bbox_to_original_space(raw_bbox, transforms, frame_hw)

        raw_mask = getattr(det, "mask", None)
        mask_png_b64, mask_bbox = (None, None)
        if raw_mask is not None:
            full_mask = _mask_to_original_space(np.asarray(raw_mask), transforms, frame_hw)
            mask_png_b64, mask_bbox = _encode_mask_crop(full_mask, bbox, frame_hw)

        # SAM's own Detection.class_name is always None (class-agnostic --
        # see detection/sam_segmenter.py); the real label is the BLIP
        # caption cached on the tracked object once it's created/confirmed
        # (update/memory_manager.py's resolve_creation_class_name). Detections
        # not yet assigned an object_id (kind="detection") have no caption
        # yet and stay unlabeled.
        class_name = getattr(det, "class_name", None)
        object_id = entry["object_id"]
        if object_id is not None and memory is not None:
            tracked = memory.get(int(object_id))
            if tracked is not None and getattr(tracked, "class_name", None):
                class_name = str(tracked.class_name)

        objects.append({
            "det_id": det_id,
            "object_id": object_id,
            "kind": entry["kind"],
            "class_name": class_name,
            "confidence": float(getattr(det, "confidence", 0.0) or 0.0),
            "bbox": bbox,
            "mask_bbox": mask_bbox,
            "mask_png_b64": mask_png_b64,
        })
    return objects


class Tracker:
    """Owns the ReIDPipeline. reset() rebuilds only the memory/association/
    update state (cheap) -- it reuses the already-loaded YOLO + DINOv3
    models on ctx rather than reinitialize_system()-ing from scratch, since
    a memory reset (start a fresh room/run) is a routine GUI action and
    reloading both models each time would make it a multi-second stall."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.frame_id = 0
        self.config = _build_config(args)
        label_backend = (" + InternVL" if self.config.get("internvl", {}).get("enabled")
                         else " + BLIP" if self.config.get("blip", {}).get("enabled") else "")
        print(f"[remind-live] loading SAM + DINOv3{label_backend} ...")
        self.ctx = initialize_system(self.config)
        self.pipeline = ReIDPipeline(self.ctx)

    def reset(self) -> None:
        with self.lock:
            mem_cfg = self.config.get("memory", {}) or {}
            self.ctx.memory = MemoryStore(
                config=self.config, start_object_id=int(mem_cfg.get("start_object_id", 0)),
            )
            self.pipeline = ReIDPipeline(self.ctx)
            self.frame_id = 0

    def infer(self, frame_bgr: np.ndarray) -> dict:
        with self.lock:
            fid = self.frame_id
            self.frame_id += 1
            p_out, _a_out, u_out = self.pipeline.process_frame(
                frame=frame_bgr, frame_id=fid, timestamp=time.time(),
            )
            frame_hw = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))
            return {"frame_id": fid, "objects": _pack_detections(p_out, u_out, frame_hw, memory=self.ctx.memory)}

    def confirm_arrival(self, frame_bgr: np.ndarray, target_desc: str) -> dict:
        """Full-frame VLM arrival VQA -- see InternVLClassifier.confirm_arrival
        and this module's /confirm_arrival docstring. Reuses ctx.captioner,
        the SAME already-loaded InternVL model detection labeling uses (no
        second model load) -- guarded with self.lock since generate() isn't
        safe to call concurrently with infer()'s pipeline.process_frame."""
        captioner = getattr(self.ctx, "captioner", None)
        if captioner is None or not hasattr(captioner, "confirm_arrival"):
            raise RuntimeError(
                "InternVL classifier not loaded on this server (started with "
                "--no-internvl or --use-blip?) -- /confirm_arrival unavailable"
            )
        rgb = frame_bgr[:, :, ::-1]
        with self.lock:
            arrived = captioner.confirm_arrival(rgb, target_desc)
        return {"arrived": arrived}


class Handler(BaseHTTPRequestHandler):
    tracker: "Tracker" = None  # set by main() before serve_forever

    def log_message(self, fmt, *a):  # quiet the default per-request access log
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "frame_count": self.tracker.frame_id})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/infer":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b""
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self._send_json(400, {"error": "could not decode JPEG"})
                return
            try:
                result = self.tracker.infer(frame)
            except Exception as e:  # keep the server alive on a single bad frame
                print(f"[remind-live] inference error: {e}")
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, result)
        elif self.path == "/reset":
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                self.rfile.read(length)
            self.tracker.reset()
            self._send_json(200, {"status": "reset"})
        elif self.path.startswith("/confirm_arrival"):
            qs = parse_qs(urlparse(self.path).query)
            target_desc = qs.get("target", ["the target object"])[0]
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b""
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self._send_json(400, {"error": "could not decode JPEG"})
                return
            try:
                result = self.tracker.confirm_arrival(frame, target_desc)
            except RuntimeError as e:
                self._send_json(501, {"error": str(e)})
                return
            except Exception as e:  # keep the server alive on a single bad call
                print(f"[remind-live] confirm_arrival error: {e}")
                self._send_json(500, {"error": str(e)})
                return
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "not found"})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="REMIND live per-frame inference server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "default_config.yaml"))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "live"))
    ap.add_argument("--sam-points-per-side", type=int, default=8,
                     help="SAM automatic-mask-generation grid density -- the main latency knob; "
                          "see config/default_config.yaml's sam: block for measured timings")
    ap.add_argument("--input-width", type=int, default=1280)
    ap.add_argument("--no-internvl", action="store_true",
                     help="skip loading InternVL; newly tracked objects are labeled 'object' instead "
                          "of getting a real class name (unless --use-blip is also given)")
    ap.add_argument("--use-blip", action="store_true",
                     help="use the old BLIP free-form captioner instead of InternVL's constrained "
                          "classification -- BLIP tends to caption flat floor/wall/ceiling patches "
                          "with unstable text (e.g. 'black tiles'); prefer InternVL (the default) "
                          "unless specifically debugging/comparing the two")
    ap.add_argument("--device", default="cuda:0")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    tracker = Tracker(args)
    Handler.tracker = tracker
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[remind-live] ready -- listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
