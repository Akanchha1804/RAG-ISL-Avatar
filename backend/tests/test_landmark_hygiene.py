"""The landmark corpus must stay free of the artefacts that broke the avatar.

The retargeters cluster detections and ignore the handedness label precisely
because the shipped clips could not be trusted, but the data itself was left
dirty: 79 of 184 per-gloss clips flipped a hand's label mid-clip and 113
reported one physical hand twice. ``ISL_MediaPipe/clean_landmarks.py`` repairs
those in place; this test is the gate that stops a re-generated corpus from
quietly reintroducing them.

It runs against the checked-in JSON, so it needs a full checkout (CI has one).
"""

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LANDMARKS = REPO / "ISL_MediaPipe" / "output_landmarks"

sys.path.insert(0, str(REPO / "ISL_MediaPipe"))
import clean_landmarks as cl  # noqa: E402  (path has to be set up first)


def _files() -> list[Path]:
    if not LANDMARKS.is_dir():
        return []
    return [
        p
        for p in sorted(LANDMARKS.glob("*.json"))
        if not p.name.endswith("_mapping.json")
    ]


_CACHE: dict = {}


def _audit_all() -> dict:
    """path -> list of hygiene problems, computed once per session."""
    if "audit" not in _CACHE:
        results = {}
        for path in _files():
            dataset = cl.json.loads(path.read_text(encoding="utf-8"))
            problems = cl.audit(dataset)
            if problems:
                results[path.name] = problems
        _CACHE["audit"] = results
    return _CACHE["audit"]


def _hand(label: str, x: float, y: float, score: float = 0.9) -> dict:
    point = {"x": x, "y": y, "z": 0.0}
    return {
        "handedness": label,
        "handedness_score": score,
        "landmarks": [dict(point) for _ in range(21)],
    }


def _dirty_clip() -> dict:
    """A clip with every artefact the cleaner exists for."""
    frames = []
    for i in range(10):
        # One flip at the end: the arm used to hop mid-sign.
        label = "Right" if i < 9 else "Left"
        frames.append({"hands": [_hand(label, 0.30 + i * 0.01, 0.50)]})
    # One hand reported twice, same label, 1 cm apart.
    frames.append({"hands": [_hand("Left", 0.40, 0.50), _hand("Left", 0.41, 0.50, 0.4)]})
    # A hallucinated second hand, far enough apart not to merge by distance.
    frames.append({"hands": [_hand("Right", 0.42, 0.50), _hand("Left", 0.46, 0.56, 0.3)]})
    return {"fps": 30.0, "frame_count": len(frames), "frames": frames}


def test_the_corpus_is_present():
    assert LANDMARKS.is_dir(), f"missing landmark directory: {LANDMARKS}"
    assert len(_files()) > 100, "landmark corpus looks truncated"


def test_every_clip_passes_the_hygiene_audit():
    dirty = _audit_all()
    assert not dirty, (
        f"{len(dirty)} clip(s) still carry tracker artefacts; run "
        f"`python ISL_MediaPipe/clean_landmarks.py --apply`:\n"
        + "\n".join(f"  {name}: {problems[:3]}" for name, problems in list(dirty.items())[:20])
    )


def test_the_cleaner_is_idempotent_on_the_shipped_clips():
    changed = {}
    for path in _files():
        dataset = cl.json.loads(path.read_text(encoding="utf-8"))
        stats = cl.clean_dataset(dataset)
        if any(stats.values()):
            changed[path.name] = stats
    assert not changed, f"cleaner would still change {len(changed)} clip(s): {list(changed)[:10]}"


def test_every_hand_has_21_landmarks():
    bad = []
    for path in _files():
        dataset = cl.json.loads(path.read_text(encoding="utf-8"))
        for i, frame in enumerate(dataset.get("frames") or []):
            for hand in frame.get("hands") or []:
                if len(hand.get("landmarks") or []) != 21:
                    bad.append(f"{path.name} frame {i}")
    assert not bad, f"malformed hands in {len(bad)} place(s): {bad[:10]}"


def test_the_audit_catches_the_artefacts():
    clip = _dirty_clip()
    problems = cl.audit(clip)
    assert any("flips" in p for p in problems), problems
    assert any("share a label" in p or "duplicate" in p for p in problems), problems


def test_the_cleaner_repairs_a_dirty_clip():
    clip = _dirty_clip()
    before = copy.deepcopy(clip)
    stats = cl.clean_dataset(clip)

    assert stats["merged"] >= 1, "duplicate detection was not merged"
    assert stats["dropped"] >= 1, "hallucinated second hand was not dropped"
    assert stats["relabelled"] >= 1, "flipping label was not stabilised"
    assert cl.audit(clip) == [], "cleaner left the clip dirty"
    assert before != clip, "cleaner reported changes but wrote none"

    # One hand per frame in a strongly single-handed clip, one label throughout.
    for frame in clip["frames"]:
        hands = frame["hands"]
        assert len(hands) == 1
        assert hands[0]["handedness"] == "Right"

    # Running it again must not move anything.
    again = cl.clean_dataset(clip)
    assert all(value == 0 for value in again.values()), again


def test_no_clip_has_a_dead_stretch_longer_than_the_hold():
    """A hole longer than HAND_GAP_HOLD is missing footage and must be spliced."""
    bad = []
    for path in _files():
        dataset = cl.json.loads(path.read_text(encoding="utf-8"))
        fps = float(dataset.get("fps") or 25.0)
        run = 0
        for i, frame in enumerate(dataset.get("frames") or []):
            has_hand = any(
                len(h.get("landmarks") or []) == 21 for h in (frame.get("hands") or [])
            )
            if has_hand:
                if run and i - run > 0 and run / fps >= cl.MAX_GAP_SECONDS:
                    bad.append(f"{path.name}: {run} dead frames ({run / fps:.2f}s) at {i - run}")
                run = 0
            else:
                run += 1
    assert not bad, (
        f"{len(bad)} clip(s) still freeze mid-sign; run "
        f"`python ISL_MediaPipe/clean_landmarks.py --apply`:\n" + "\n".join(bad[:10])
    )


def test_the_cleaner_splices_long_holes_but_keeps_short_dropouts():
    frames = [{"hands": [_hand("Right", 0.30, 0.50)]} for _ in range(10)]
    frames += [{"hands": []} for _ in range(10)]  # 0.33 s: bridged at runtime
    frames += [{"hands": [_hand("Right", 0.32, 0.50)]} for _ in range(10)]
    frames += [{"hands": []} for _ in range(30)]  # 1.0 s: missing footage
    frames += [{"hands": [_hand("Right", 0.34, 0.50)]} for _ in range(10)]
    clip = {"fps": 30.0, "frame_count": len(frames), "frames": frames}

    removed = cl.splice_long_gaps(clip)

    assert removed == 30
    assert len(clip["frames"]) == 40
    assert clip["cleaned_frame_count"] == 40
    # The short dropout stays: the retargeters hold the pose through it.
    assert sum(1 for f in clip["frames"] if not f["hands"]) == 10


def test_the_cleaner_keeps_genuine_two_handed_clips():
    """A clip with many two-hand frames must keep both hands and both labels."""
    frames = []
    for i in range(10):
        frames.append({
            "hands": [
                _hand("Right", 0.45, 0.52),
                _hand("Left", 0.55, 0.52),
            ]
        })
    clip = {"fps": 30.0, "frame_count": len(frames), "frames": frames}
    cl.clean_dataset(clip)
    for frame in clip["frames"]:
        labels = sorted(h["handedness"] for h in frame["hands"])
        assert labels == ["Left", "Right"]
    assert cl.audit(clip) == []
