"""Animation asset harness (M6/M7 input, Phase 4): landmark-hit rates.

For every ref's corrected tokens, resolves via animation_resolver (offline
JSON registry, production files) and reports clip vs per-token landmark
coverage plus the missing-token backlog (M5 coverage queue). Sentence-level
playback uses the sentence file, so per-token landmark gaps only bite the
composed path - both numbers are reported separately.

Since Tier 2 the payload also rolls the same data up by sentence (how many
references are signable end to end with the per-gloss clips) and against the
whole CISLR vocabulary (how much of the corpus has motion at all). The
regression gate in `tests/test_coverage_gate.py` treats these as floors: the
numbers may go up, never down without a deliberate change.

    .venv\\Scripts\\python.exe eval/run_animation_eval.py
"""

import sys
from collections import Counter

import metrics_eval
from metrics_eval import BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

import gloss_lookup  # noqa: E402
from animation_resolver import (  # noqa: E402
    load_gloss_landmarks,
    reset_registry,
    resolve_gloss_token,
)


def compute() -> dict:
    """Measure coverage without leaving the resolver cache mutated."""
    import csv

    vocab = {}
    with open(BACKEND_DIR / "dataset.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            g = (row.get("gloss") or "").strip().lower()
            if g and g not in vocab:
                vocab[g] = {"uid": row.get("uid"), "category": row.get("category"),
                            "duration": row.get("duration")}

    saved_vocab = gloss_lookup._gloss_vocab
    gloss_lookup._gloss_vocab = vocab
    reset_registry()
    try:
        refs = metrics_eval.load_refs()
        total = clip_hits = landmark_hits = 0
        missing_clip: Counter = Counter()
        missing_landmark: Counter = Counter()
        fully = partial = none = 0

        for r in refs:
            # THANK YOU is one sign: score it as a unit, not as THANK + YOU.
            raw = r["reference"].replace("THANK YOU", "THANK\x00YOU").split()
            tokens = [t.replace("THANK\x00YOU", "THANK YOU") for t in raw]
            ref_ok = 0
            for token in tokens:
                total += 1
                entry = resolve_gloss_token(token)
                if entry is None:
                    missing_clip[token] += 1
                    missing_landmark[token] += 1
                    continue
                if entry.get("clip_url"):
                    clip_hits += 1
                else:
                    missing_clip[token] += 1
                if entry.get("landmark_clip_url"):
                    landmark_hits += 1
                    ref_ok += 1
                else:
                    missing_landmark[token] += 1
            if tokens:
                if ref_ok == len(tokens):
                    fully += 1
                elif ref_ok:
                    partial += 1
                else:
                    none += 1

        # Per-gloss motion actually built by ISL_MediaPipe/batch_gloss_landmarks.py.
        animated = sum(
            1 for e in load_gloss_landmarks().values()
            if isinstance(e, dict) and e.get("status") == "ready" and e.get("file")
        )
    finally:
        gloss_lookup._gloss_vocab = saved_vocab
        reset_registry()

    return {
        "locked_refs": False,
        "n_tokens": total,
        "n_landmark_hits": landmark_hits,
        "n_missing_landmark": total - landmark_hits,
        "clip_hit_rate": round(clip_hits / total, 4) if total else 0.0,
        "per_token_landmark_hit_rate": round(landmark_hits / total, 4) if total else 0.0,
        "refs": {
            "total": len(refs),
            "fully_animated": fully,
            "partially_animated": partial,
            "not_animated": none,
        },
        "vocabulary": {
            "size": len(vocab),
            "animated_glosses": animated,
            "animated_share": round(animated / len(vocab), 4) if vocab else 0.0,
        },
        "missing_clip_top": missing_clip.most_common(20),
        "missing_landmark_top": missing_landmark.most_common(20),
    }


def main() -> int:
    payload = compute()
    path = metrics_eval.save_result("animation_eval", payload)
    refs = payload["refs"]
    vocab = payload["vocabulary"]
    print(
        f"tokens={payload['n_tokens']} "
        f"clip_hit={payload['clip_hit_rate']} "
        f"landmark_hit={payload['per_token_landmark_hit_rate']} "
        f"| refs fully={refs['fully_animated']}/{refs['total']} "
        f"partial={refs['partially_animated']} none={refs['not_animated']} "
        f"| glosses with motion={vocab['animated_glosses']}/{vocab['size']} "
        f"({vocab['animated_share'] * 100:.1f}%) "
        f"-> {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
