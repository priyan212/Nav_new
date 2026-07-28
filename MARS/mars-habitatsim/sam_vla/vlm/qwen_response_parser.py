import json
import re

from sam_vla.core.types import Action

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw_text: str) -> str:
    return _FENCE_RE.sub("", raw_text.strip()).strip()


def _load_json_object(raw_text: str) -> dict:
    stripped = _strip_fences(raw_text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON from model response: {e}\nraw_text={raw_text!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(parsed).__name__}\nraw_text={raw_text!r}"
        )
    return parsed


def parse_select_goal_response(raw_text: str) -> dict:
    parsed = _load_json_object(raw_text)
    goal_index = parsed.get("goal_index")
    if not isinstance(goal_index, int) or isinstance(goal_index, bool):
        raise ValueError(
            f"Missing or non-integer 'goal_index' in response\nraw_text={raw_text!r}"
        )
    return parsed


DIRECTIONS = ("forward", "turn_left", "turn_right")


def parse_direction_response(raw_text: str) -> dict:
    """Parse a discrete-direction response (build_direction_prompt), validating
    that 'direction' is one of DIRECTIONS rather than an arbitrary continuous
    action."""
    parsed = _load_json_object(raw_text)

    direction = parsed.get("direction")
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError(
            f"Missing or invalid 'direction' in response (expected one of "
            f"{DIRECTIONS})\nraw_text={raw_text!r}"
        )
    return parsed


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_drive_action_response(raw_text: str) -> Action:
    parsed = _load_json_object(raw_text)

    for key in ("v_fwd", "v_lat", "yaw_rate"):
        value = parsed.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Missing or non-numeric '{key}' in response\nraw_text={raw_text!r}"
            )

    v_fwd = _clamp(float(parsed["v_fwd"]), 0.0, 1.0)
    v_lat = _clamp(float(parsed["v_lat"]), -1.0, 1.0)
    yaw_rate = _clamp(float(parsed["yaw_rate"]), -1.0, 1.0)

    return Action(v_fwd=v_fwd, v_lat=v_lat, yaw_rate=yaw_rate)


if __name__ == "__main__":
    well_formed = '{"v_fwd": 0.5, "v_lat": -0.2, "yaw_rate": 0.1}'
    fenced = """```json
{"goal_index": 2}
```"""
    malformed = '{"v_fwd": 0.5, "v_lat": "left", yaw_rate: 0.1}'

    print("parsed drive action:")
    print(parse_drive_action_response(well_formed))

    print("\n fenced goal selection:")
    print(parse_select_goal_response(fenced))

    print("\n primitive input:")
    try:
        parse_drive_action_response(malformed)
    except ValueError as e:
        print(f"caught ValueError: {e}")

    print("\n parsed direction:")
    print(parse_direction_response('{"direction": "turn_left", "reasoning": "goal is left"}'))

    print("\n invalid direction:")
    try:
        parse_direction_response('{"direction": "spin", "reasoning": "nope"}')
    except ValueError as e:
        print(f"caught ValueError: {e}")
