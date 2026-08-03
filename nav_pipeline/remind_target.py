"""Parse the REMIND GUI's "<CLASS> ID <n>" target text, e.g. "CHAIR ID 1",
into (class_name, object_id) -- matching the labels REMIND overlays on
tracked objects (see remind_client.RemindObject.label) so the operator can
read an ID straight off the video feed and type it back.
"""
import re
from typing import Optional, Tuple

_PATTERN = re.compile(r"^\s*(.+?)\s+id\s*#?\s*(\d+)\s*$", re.IGNORECASE)


def parse_object_target(text: str) -> Optional[Tuple[str, int]]:
    """"CHAIR ID 1" -> ("chair", 1). None if text doesn't match the format
    (caller should reject/prompt rather than guess -- there's no bare-phrase
    fallback here, unlike relational_target.py's DINO-phrase parsers)."""
    m = _PATTERN.match(text or "")
    if not m:
        return None
    class_name = m.group(1).strip().lower()
    if not class_name:
        return None
    return class_name, int(m.group(2))
