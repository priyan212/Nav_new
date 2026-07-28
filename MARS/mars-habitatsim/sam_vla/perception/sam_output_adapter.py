"""
Converts sam_segmenter.segment_frame's raw pixel-space dicts into Detections.

"""

from sam_vla.core.types import Detection

# bigrock is the only discrete-object class the goal/obstacle VLM selection
# should ever see. bedrock is terrain/background segmentation (its contours
# can span most of the frame), not a candidate rock - surfacing it here lets
# the VLM pick a bedrock region as the "goal" or "obstacle", which is wrong.
# Any class not in this map (bedrock included) is dropped in to_detections.
_CLASS_MAP = {
    "bigrock": "obstacle",
}


def to_detections(
    raw_detections: list[dict], image_width: int, image_height: int
) -> list[Detection]:
    detections = []
    for raw in raw_detections:
        if raw["class_name"] not in _CLASS_MAP:
            continue
        class_name = _CLASS_MAP[raw["class_name"]]
        x0 = raw["x"] / image_width
        y0 = raw["y"] / image_height
        x1 = (raw["x"] + raw["width"]) / image_width
        y1 = (raw["y"] + raw["height"]) / image_height

        det = Detection(
            class_name=class_name,
            bbox_norm=(x0, y0, x1, y1),
            confidence=raw["score"],
        )
        try:
            det.validate()
        except ValueError as e:
            print(f"Warning: skipping invalid detection {raw}: {e}")
            continue
        detections.append(det)

    return detections


if __name__ == "__main__":
    raw_examples = [
        {"class_name": "bedrock", "x": 100.0, "y": 50.0, "width": 200.0, "height": 150.0, "score": 0.92},
        {"class_name": "bigrock", "x": 400.0, "y": 300.0, "width": 80.0, "height": 80.0, "score": 0.77},
        # Invalid: zero width -> x0 == x1, fails validate()
        {"class_name": "bedrock", "x": 600.0, "y": 200.0, "width": 0.0, "height": 40.0, "score": 0.5},
    ]

    result = to_detections(raw_examples, image_width=1280, image_height=720)
    print(f"\n{len(result)} valid detections:")
    for d in result:
        print(d)
