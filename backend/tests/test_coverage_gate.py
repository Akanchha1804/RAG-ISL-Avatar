"""Coverage gate: the signable share may go up, never down silently.

`eval/run_animation_eval.py` measures what the avatar can actually sign, from
the real per-gloss clips and the real reference set. These floors are the
numbers measured when the gate was added; a change that drops one of them is
either a regression or a deliberate decision, and in both cases this test
should fail until the number in the test is updated on purpose.

The fixtures point DATA_DIR at a tiny corpus, so the landmark-clip lookup is
redirected at the real repo file: this is a check on shipped data, not on the
test fixtures.
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "eval"))

import animation_resolver  # noqa: E402
import gloss_lookup  # noqa: E402
import run_animation_eval as coverage  # noqa: E402

GLOSS_CLIPS = REPO / "ISL_MediaPipe" / "gloss_landmarks.json"

FLOORS = {
    "n_tokens": 328,
    "clip_hit_rate": 0.89,
    "per_token_landmark_hit_rate": 0.87,
    "refs.fully_animated": 68,
    "vocabulary.animated_glosses": 184,
    "vocabulary.size": 4765,
}


@pytest.fixture()
def payload(monkeypatch):
    if not GLOSS_CLIPS.exists():
        pytest.skip("gloss_landmarks.json is not in this checkout")
    monkeypatch.setattr(animation_resolver, "_gloss_landmarks_path", lambda: GLOSS_CLIPS)
    return coverage.compute()


def _get(payload, dotted):
    value = payload
    for part in dotted.split("."):
        value = value[part]
    return value


def test_coverage_does_not_regress(payload):
    for key, floor in FLOORS.items():
        measured = _get(payload, key)
        assert measured >= floor, (
            f"{key} dropped to {measured} (floor {floor}). If this is "
            f"intentional, update FLOORS; otherwise a clip or reference was lost."
        )


def test_the_backlog_accounts_for_every_missing_token(payload):
    """A coverage failure must say which glosses are missing, not just a rate."""
    top = payload["missing_landmark_top"]
    assert top, "no missing tokens recorded, but coverage is below 100%"
    assert payload["n_missing_landmark"] == payload["n_tokens"] - payload["n_landmark_hits"]
    # The payload reports the 20 most frequent misses, so this is a lower bound.
    assert 0 < sum(count for _, count in top) <= payload["n_missing_landmark"]


def test_coverage_measurement_restores_the_vocabulary_cache():
    """compute() must not leak the real vocabulary into fixture-based tests."""
    before = gloss_lookup._gloss_vocab
    coverage.compute()
    assert gloss_lookup._gloss_vocab is before
