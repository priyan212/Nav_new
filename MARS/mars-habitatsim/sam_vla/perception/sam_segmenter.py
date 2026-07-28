"""Single-frame wrapper around the custom SimpleSAM2Seg 4-class head.

SimpleSAM2Seg (sam/train_sam2_simple_fast.py) currently only has a
process_video entry point. This module is the missing single-frame path:
one forward pass on one RGB frame, then per-pixel logits turned into
per-instance bounding boxes for the two classes this pipeline surfaces.
"""

import pprint
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam_vla.perception import sam_weights_loader

IMAGE_SIZE = 1024
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model output vocabulary is index -> class; only bedrock/big_rock surface
# downstream (soil/sand are filtered here, not passed further).
SURFACED_CLASSES = {1: "bedrock", 3: "bigrock"}

# Minimum connected-component area (px, at model resolution) to keep as a real instance instead of segmentation noise.
_MIN_INSTANCE_AREA = 4

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = sam_weights_loader.load_sam_model()
    return _model_cache


def _preprocess(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def segment_frame(rgb: np.ndarray, model=None) -> list[dict]:
    if model is None:
        model = _get_model()

    device = next(model.parameters()).device
    h0, w0 = rgb.shape[:2]
    scale_x = w0 / IMAGE_SIZE
    scale_y = h0 / IMAGE_SIZE

    with torch.no_grad():
        tensor = _preprocess(rgb, device)
        logits = model(tensor)  # (1, 4, IMAGE_SIZE, IMAGE_SIZE)
        probs = F.softmax(logits, dim=1)[0]  # (4, IMAGE_SIZE, IMAGE_SIZE)
        class_map = probs.argmax(dim=0).cpu().numpy()
        probs_np = probs.cpu().numpy()

    detections = []
    for class_idx, class_name in SURFACED_CLASSES.items():
        mask = (class_map == class_idx).astype(np.uint8)
        if not mask.any():
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label_id in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[label_id]
            if area < _MIN_INSTANCE_AREA:
                continue

            score = float(probs_np[class_idx][labels == label_id].mean())
            detections.append({
                "class_name": class_name,
                "x": float(x * scale_x),
                "y": float(y * scale_y),
                "width": float(w * scale_x),
                "height": float(h * scale_y),
                "score": score,
            })

    return detections


if __name__ == "__main__":
    image_path = sys.argv[1]
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pprint.pprint(segment_frame(rgb_image))
