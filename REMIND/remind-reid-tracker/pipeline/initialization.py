# pipeline/initialization.py

from __future__ import annotations

from pathlib import Path
import random
import numpy as np

import torch

from memory.memory_store import MemoryStore
from features.dino_extractor import DinoExtractor


class RuntimeContext:
    """
    Contexto global del sistema.
    """

    def __init__(
        self,
        config: dict,
        device: str,
        memory: MemoryStore,
        yolo,
        dino: DinoExtractor,
        output_dir: Path,
        captioner=None,
    ):
        self.config = config
        self.device = device
        self.memory = memory
        self.detector = yolo
        self.yolo = yolo
        self.dino = dino
        self.output_dir = output_dir
        # BLIP captioner, only built when config['blip']['enabled'] -- see
        # initialize_system(). None for the yolo/davis backends, which
        # already carry real class names and don't need it.
        self.captioner = captioner


def set_seeds(seed: int | None) -> None:
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def select_device(config: dict) -> str:
    runtime_cfg = config.get("runtime", {}) or {}
    requested = runtime_cfg.get("device", "auto")

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def prepare_output_dir(config: dict) -> Path:
    paths_cfg = config.get("paths", {})
    out_cfg = paths_cfg.get("output_dir", "outputs")

    project_root = Path(__file__).resolve().parents[2]
    out = (project_root / out_cfg).resolve()

    out.mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    return out


def build_segmenter(config: dict, device: str):
    det_cfg = config.get("detector", {}) or {}
    backend = str(det_cfg.get("backend", "yolo")).strip().lower()

    if backend == "yolo":
        from detection.yolo_segmenter import YoloSegmenter

        seg = YoloSegmenter(config=config, device=device)
        seg.load_model()
        return seg

    if backend == "davis":
        from detection.davis_segmenter import DavisSegmenter

        seg = DavisSegmenter(config=config, device=device)
        seg.load_model()
        return seg

    if backend == "sam":
        from detection.sam_segmenter import SamSegmenter

        seg = SamSegmenter(config=config, device=device)
        seg.load_model()
        return seg

    raise ValueError(f"Backend de detector no soportado: {backend}")


def build_captioner(config: dict, device: str):
    """Builds whichever labeling backend is enabled for class-agnostic
    detections (SAM has no real class name of its own -- see
    detection/sam_segmenter.py). internvl takes priority if both are
    somehow enabled at once (mutually exclusive in practice -- see
    default_config.yaml's comments) since it's the better-constrained
    classifier (see features/internvl_classifier.py's docstring for why
    BLIP's free-form captioning was replaced)."""
    internvl_cfg = config.get("internvl", {}) or {}
    blip_cfg = config.get("blip", {}) or {}
    internvl_on = bool(internvl_cfg.get("enabled", False))
    blip_on = bool(blip_cfg.get("enabled", False))

    if internvl_on and blip_on:
        print("[WARN] both internvl.enabled and blip.enabled are true -- using internvl")

    if internvl_on:
        from features.internvl_classifier import InternVLClassifier

        captioner = InternVLClassifier(config=internvl_cfg, device=device)
        captioner.load_model()
        return captioner

    if blip_on:
        from features.blip_captioner import BlipCaptioner

        captioner = BlipCaptioner(config=blip_cfg, device=device)
        captioner.load_model()
        return captioner

    return None


def initialize_system(config: dict) -> RuntimeContext:
    device = select_device(config)

    seed = (config.get("runtime", {}) or {}).get("seed", None)
    set_seeds(seed)

    output_dir = prepare_output_dir(config)

    # Detector/segmentador
    yolo = build_segmenter(config=config, device=device)

    # Memory
    mem_cfg = config.get("memory", {}) or {}
    memory = MemoryStore(
        config=config,
        start_object_id=int(mem_cfg.get("start_object_id", 0)),
    )

    # DINO
    dino_cfg = config.get("dino", {}) or {}
    dino = DinoExtractor(config=dino_cfg, device=device)
    dino.load_model()

    # BLIP (optional -- only for backends without real class names, e.g. sam)
    captioner = build_captioner(config=config, device=device)

    return RuntimeContext(
        config=config,
        device=device,
        memory=memory,
        yolo=yolo,
        dino=dino,
        output_dir=output_dir,
        captioner=captioner,
    )
