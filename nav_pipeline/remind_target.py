"""Parse the REMIND GUI's target text into a bare object_id.

Targeting is ID-only now (see remind_gui.py): the operator picks an ID off
the video overlay or the known-objects list -- both display "ID <n>" only,
never REMIND's BLIP caption (that stays internal bookkeeping, see
object_map.py) -- and types/clicks that same number back. No class name
involved, unlike the earlier "<CLASS> ID <n>" format this replaces.
"""
import re
from typing import Optional

_PATTERN = re.compile(r"^\s*(?:id\s*#?\s*)?(\d+)", re.IGNORECASE)


def parse_object_target(text: str) -> Optional[int]:
    """"ID 3", "id3", bare "3", or "ID 3 (visible)" (the known-objects
    list's visibility suffix, see remind_gui.py's refresh()) -> 3. No
    end-anchor on purpose -- trailing text after the number is ignored
    rather than rejected. None if text doesn't match at all (caller should
    reject/prompt rather than guess)."""
    m = _PATTERN.match(text or "")
    if not m:
        return None
    return int(m.group(1))
