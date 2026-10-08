"""Animation asset harness (M6/M7 input, Phase 4): landmark-hit rates.

For every ref's corrected tokens, resolves via animation_resolver (offline
JSON registry, production files) and reports clip vs per-token landmark
coverage plus the missing-token backlog (M5 coverage queue). Sentence-level
playback uses the sentence file, so per-token landmark gaps only bite the
composed path — both numbers are reported separately.

    .venv\\Scripts\\python.exe eval/run_animation_eval.py
"""

import sys
from collections import Counter

import metrics_eval
from metrics_eval import BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

import gloss_lookup  # noqa: E402
from animation_resolver import resolve_gloss_token  # noqa: E402


def main() -> int:
    import csv

    vocab = {}
    with open(BACKEND_DIR / "dataset.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            g = (row.get("gloss") or "").strip().lower()
            if g and g not in vocab:
                vocab[g] = {"uid": row.get("uid"), "category": row.get("category"),
                            "duration": row.get("duration")}
    gloss_lookup._gloss_vocab = vocab

    refs = metrics_eval.load_refs()
    total = clip_hits = landmark_hits = 0
    missing_clip: Counter = Counter()
    missing_landmark: Counter = Counter()
    for r in refs:
        # THANK YOU is one sign: score it as a unit, not as THANK + YOU.
        raw = r["reference"].replace("THANK YOU", "THANK\x00YOU").split()
        tokens = [t.replace("THANK\x00YOU", "THANK YOU") for t in raw]
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
            else:
                missing_landmark[token] += 1

    payload = {
        "n_tokens": total,
        "clip_hit_rate": round(clip_hits / total, 4) if total else 0.0,
        "per_token_landmark_hit_rate": round(landmark_hits / total, 4) if total else 0.0,
        "missing_clip_top": missing_clip.most_common(20),
        "missing_landmark_top": missing_landmark.most_common(20),
    }
    path = metrics_eval.save_result("animation_eval", payload)
    print(f"tokens={total} clip_hit={payload['clip_hit_rate']} "
          f"landmark_hit={payload['per_token_landmark_hit_rate']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
