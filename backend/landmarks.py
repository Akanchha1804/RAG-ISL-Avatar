"""
Landmark dataset loading and validation (backend side).

The same structural rules the Unity client enforces are enforced here before
the backend points any client at a landmark file:
  - top level must be an object with a "frames" list
  - every hand used for animation must carry exactly 21 landmarks

Malformed data raises LandmarkValidationError with a specific reason instead
of being silently swallowed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

LANDMARK_KEYS = ("x", "y", "z")
EXPECTED_LANDMARKS_PER_HAND = 21


class LandmarkValidationError(Exception):
    """Raised when a landmark file is missing or structurally invalid."""


def validate_hand(hand) -> Tuple[bool, str]:
    """Validate one hand entry. Returns (ok, reason)."""
    if not isinstance(hand, dict):
        return False, "hand entry is not an object"

    landmarks = hand.get("landmarks")
    if not isinstance(landmarks, list):
        return False, "hand has no landmark list"

    if len(landmarks) != EXPECTED_LANDMARKS_PER_HAND:
        return False, (
            f"expected {EXPECTED_LANDMARKS_PER_HAND} landmarks, "
            f"got {len(landmarks)}"
        )

    for i, point in enumerate(landmarks):
        if not isinstance(point, dict):
            return False, f"landmark {i} is not an object"
        for key in LANDMARK_KEYS:
            if key not in point:
                return False, f"landmark {i} missing '{key}'"

    return True, "ok"


def validate_dataset(data) -> dict:
    """Validate dataset structure and return summary statistics.

    Raises LandmarkValidationError on structural problems.
    Invalid hands are counted (not fatal) so callers can report them.
    """
    if not isinstance(data, dict):
        raise LandmarkValidationError("landmark dataset must be a JSON object")

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise LandmarkValidationError("landmark dataset has no 'frames' list")

    if not frames:
        raise LandmarkValidationError("landmark dataset contains no frames")

    frames_with_hands = 0
    valid_hands = 0
    invalid_hands = 0

    for frame in frames:
        if not isinstance(frame, dict):
            raise LandmarkValidationError("frame entry is not an object")

        hands = frame.get("hands")
        if not hands:
            continue

        if not isinstance(hands, list):
            raise LandmarkValidationError("frame 'hands' is not a list")

        frames_with_hands += 1
        for hand in hands:
            ok, _ = validate_hand(hand)
            if ok:
                valid_hands += 1
            else:
                invalid_hands += 1

    return {
        "frame_count": len(frames),
        "frames_with_hands": frames_with_hands,
        "valid_hands": valid_hands,
        "invalid_hands": invalid_hands,
    }


def load_landmark_dataset(path: Union[str, Path]) -> Tuple[dict, dict]:
    """Load and validate a landmark JSON file.

    Returns (dataset, stats). Raises LandmarkValidationError with a clear
    message when the file is missing, unreadable, or malformed.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise LandmarkValidationError(f"Landmark file not found: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise LandmarkValidationError(
            f"Cannot read landmark file '{file_path}': {e}"
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LandmarkValidationError(
            f"Invalid JSON in landmark file '{file_path}': {e}"
        ) from e

    stats = validate_dataset(data)
    return data, stats
