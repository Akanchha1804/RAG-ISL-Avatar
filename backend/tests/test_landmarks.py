import json
from pathlib import Path

import pytest

from landmarks import (
    EXPECTED_LANDMARKS_PER_HAND,
    LandmarkValidationError,
    load_landmark_dataset,
    validate_dataset,
    validate_hand,
)

FIXTURE_LM = Path(__file__).resolve().parent / "fixtures" / "data" / "ISL_MediaPipe" / "output_landmarks"


def test_valid_hand_accepted():
    hand = {
        "handedness": "Right",
        "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0} for _ in range(EXPECTED_LANDMARKS_PER_HAND)],
    }
    ok, reason = validate_hand(hand)
    assert ok, reason


def test_short_hand_rejected():
    hand = {
        "handedness": "Right",
        "landmarks": [{"x": 0, "y": 0, "z": 0} for _ in range(20)],
    }
    ok, reason = validate_hand(hand)
    assert not ok
    assert "21" in reason


def test_missing_xyz_rejected():
    landmarks = [{"x": 0, "y": 0, "z": 0} for _ in range(21)]
    landmarks[3] = {"x": 0, "y": 0}
    ok, reason = validate_hand({"handedness": "Left", "landmarks": landmarks})
    assert not ok
    assert "z" in reason


def test_load_valid_fixture():
    data, stats = load_landmark_dataset(FIXTURE_LM / "hello_landmarks.json")
    assert stats["frame_count"] == 3
    assert stats["invalid_hands"] == 0
    assert stats["valid_hands"] == 6


def test_malformed_short_hand_counts_invalid():
    data, stats = load_landmark_dataset(FIXTURE_LM / "malformed_short_hand.json")
    assert stats["invalid_hands"] == 1
    assert stats["valid_hands"] == 0


def test_malformed_not_json_raises():
    with pytest.raises(LandmarkValidationError, match="Invalid JSON"):
        load_landmark_dataset(FIXTURE_LM / "malformed_not_json.json")


def test_missing_file_raises():
    with pytest.raises(LandmarkValidationError, match="not found"):
        load_landmark_dataset(FIXTURE_LM / "does_not_exist.json")


def test_no_frames_raises():
    with pytest.raises(LandmarkValidationError, match="frames"):
        load_landmark_dataset(FIXTURE_LM / "malformed_no_frames.json")


def test_validate_dataset_rejects_non_object():
    with pytest.raises(LandmarkValidationError):
        validate_dataset([1, 2, 3])


def test_validate_dataset_rejects_empty_frames():
    with pytest.raises(LandmarkValidationError, match="no frames"):
        validate_dataset({"frames": []})
