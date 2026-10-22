"""Remove tracker artefacts from the landmark clips, in place.

The MediaPipe pass that produced ``output_landmarks`` trusts the model's
per-frame handedness label and keeps every detection it emits. Measured on
the clips shipped in this repo:

* 79 of the 184 per-gloss clips have a hand label that flips mid-clip
  (283 flips), and 38 of the 104 sentence clips (144 flips);
* 113 gloss clips and 86 sentence clips contain frames where ONE physical
  hand is reported twice (520 and 486 frames).

Both artefacts reached the avatars: the web and Unity retargeters bucket
hands by that label, so a flip moved the sign to the other arm - across the
body, into the space the idle hand occupied - and a duplicate drove two
arms to the same place. That was the overlapping-hands symptom.

This pass repairs the data rather than the two consumers, so both clients
are fixed at once:

R1  Within a frame, drop duplicate detections of one hand: same label twice
    (left/right are unique per frame) or wrists closer than MERGE_DIST.
    The higher ``handedness_score`` wins.
R2  In a clip whose frames are mostly single-handed, rewrite the label of
    single detections to the clip's majority label, so it cannot flip.
R3  In a STRONGLY single-handed clip (two-hand frames <= 10%), keep at most
    one detection per frame, chosen by continuity with the previous kept
    wrist. Frames that briefly hallucinate a second hand are the ones that
    made both arms lunge at the same spot.

Mixed clips (many genuine two-hand frames) keep their two-hand frames; only
R1 and R2 apply there. The pass is idempotent, writes only files that
changed, and prints a before/after summary.

Usage (from the repo root):

    python ISL_MediaPipe/clean_landmarks.py            # report only
    python ISL_MediaPipe/clean_landmarks.py --apply    # rewrite in place
    python ISL_MediaPipe/clean_landmarks.py --dir PATH # another directory
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

MERGE_DIST = 0.04  # normalised image distance treated as the same hand
SINGLE_MAJORITY = 0.5  # >= this share of single-hand frames -> stabilise labels
STRONG_SINGLE = 0.10  # <= this share of two-hand frames -> keep one hand/frame


def wrist(hand: dict) -> dict:
    return hand["landmarks"][0]


def wrist_dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def valid_hands(frame: dict) -> list[dict]:
    hands = []
    for hand in frame.get("hands") or []:
        lms = hand.get("landmarks")
        if isinstance(lms, list) and len(lms) == 21:
            hands.append(hand)
    return hands


def dedupe(hands: list[dict]) -> list[dict]:
    """R1: keep one detection per physical hand, in place order."""
    kept: list[dict] = []
    for hand in hands:
        dup = None
        for i, other in enumerate(kept):
            same_label = hand.get("handedness") == other.get("handedness")
            close = wrist_dist(wrist(hand), wrist(other)) < MERGE_DIST
            if same_label or close:
                dup = i
                break
        if dup is None:
            kept.append(hand)
        elif (hand.get("handedness_score") or 0.0) > (kept[dup].get("handedness_score") or 0.0):
            kept[dup] = hand
    return kept


def clean_dataset(dataset: dict) -> dict:
    """Return {'merged', 'relabelled', 'dropped'} for one clip."""
    stats = {"merged": 0, "relabelled": 0, "dropped": 0}
    frames = dataset.get("frames") or []
    with_hands = 0
    single = 0
    two = 0

    # Pass 1: R1 dedupe, and count the shape of the clip.
    cleaned: list[list[dict]] = []
    for frame in frames:
        before = valid_hands(frame)
        after = dedupe(before)
        stats["merged"] += max(0, len(before) - len(after))
        cleaned.append(after)
        if after:
            with_hands += 1
            if len(after) == 1:
                single += 1
            else:
                two += 1

    if with_hands == 0:
        return stats

    single_share = single / with_hands
    majority = None
    if single_share >= SINGLE_MAJORITY:
        counts = collections.Counter(
            hands[0].get("handedness") for hands in cleaned if len(hands) == 1
        )
        # Ties keep the earliest-seen label: deterministic and stable.
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        majority = ordered[0][0]

    strong_single = two / with_hands <= STRONG_SINGLE

    # Pass 2: R2 relabel, R3 one hand per frame for strongly single-handed
    # clips, and a consistent pair of labels on two-hand frames (left/right
    # are unique per frame, so a shared label is always a tracker error).
    prev = None
    for frame, hands in zip(frames, cleaned):
        if not hands:
            continue
        if strong_single and len(hands) > 1:
            # Continuity: the hand nearest the previous kept wrist is the
            # signer's real hand; the rest are hallucinations.
            hands.sort(
                key=lambda h: (
                    wrist_dist(wrist(h), prev) if prev is not None else 0.0,
                    -(h.get("handedness_score") or 0.0),
                )
            )
            stats["dropped"] += len(hands) - 1
            hands = hands[:1]
        if len(hands) >= 2:
            if len(hands) > 2:
                hands = sorted(
                    hands, key=lambda h: -(h.get("handedness_score") or 0.0)
                )[:2]
            # A front-facing signer's right hand sits on the viewer's left,
            # which is also how the two retargeters map image x to an arm.
            hands.sort(key=lambda h: wrist(h)["x"])
            for hand, want in zip(hands, ("Right", "Left")):
                if hand.get("handedness") != want:
                    hand["handedness"] = want
                    stats["relabelled"] += 1
        elif majority is not None and hands[0].get("handedness") != majority:
            hands[0]["handedness"] = majority
            stats["relabelled"] += 1
        prev = wrist(hands[0])
        frame["hands"] = hands

    return stats


def audit(dataset: dict) -> list[str]:
    """The invariants clean_dataset enforces; empty means clean."""
    problems = []
    frames = dataset.get("frames") or []
    with_hands = single = two = 0
    labels = []
    for i, frame in enumerate(frames):
        hands = valid_hands(frame)
        if not hands:
            continue
        with_hands += 1
        if len(hands) == 1:
            single += 1
            labels.append(hands[0].get("handedness"))
        else:
            two += 1
            for a in range(len(hands)):
                for b in range(a + 1, len(hands)):
                    if hands[a].get("handedness") == hands[b].get("handedness"):
                        problems.append(f"frame {i}: two hands share a label")
                    elif wrist_dist(wrist(hands[a]), wrist(hands[b])) < MERGE_DIST:
                        problems.append(f"frame {i}: duplicate detection")
    if with_hands:
        if single / with_hands >= SINGLE_MAJORITY and len(set(labels)) > 1:
            problems.append(f"hand label flips mid-clip ({len(set(labels))} labels)")
        if two / with_hands <= STRONG_SINGLE and two:
            problems.append(f"{two} two-hand frame(s) in a single-handed clip")
    return problems


def iter_files(root: Path):
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_mapping.json"):
            continue
        yield path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None, help="landmark directory (default: output_landmarks)")
    ap.add_argument("--apply", action="store_true", help="rewrite changed files in place")
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else Path(__file__).resolve().parent / "output_landmarks"
    totals = collections.Counter()
    changed = 0
    dirty_before = 0
    dirty_after = 0
    for path in iter_files(root):
        dataset = json.loads(path.read_text(encoding="utf-8"))
        before = audit(dataset)
        if before:
            dirty_before += 1
        stats = clean_dataset(dataset)
        after = audit(dataset)
        if after:
            dirty_after += 1
            print(f"STILL DIRTY {path.name}: {after[:3]}")
        for key, value in stats.items():
            totals[key] += value
        if any(stats.values()):
            changed += 1
            if args.apply:
                path.write_text(
                    json.dumps(dataset, ensure_ascii=False),
                    encoding="utf-8",
                )
    print(f"files        : {changed} of {sum(1 for _ in iter_files(root))} needed changes")
    print(f"dirty before : {dirty_before}")
    print(f"dirty after  : {dirty_after}")
    print(f"merged dups  : {totals['merged']}")
    print(f"relabelled   : {totals['relabelled']}")
    print(f"dropped      : {totals['dropped']}")
    if not args.apply and changed:
        print("(report only - pass --apply to rewrite)")
    return 0 if dirty_after == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
