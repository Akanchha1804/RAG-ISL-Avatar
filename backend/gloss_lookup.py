"""
Gloss Vocabulary Lookup.
Maps English words to CISLR gloss entries.

Best-clip selection (M5): 1363 glosses have multiple CISLR videos. The
winner is picked by (ready landmark clip, shortest duration, uid) so the
served reference clip is the one the avatar can actually mirror, and the
choice is deterministic. iter_best_cislr_rows() is the single source used
by both this loader and the PG seed, so both stores always agree.
"""

import csv

from paths import get_paths

# =====================================================
# PATHS
# =====================================================

CISLR_CSV = get_paths().cislr_csv

# =====================================================
# GLOSS VOCABULARY
# =====================================================

_gloss_vocab = None
_landmark_uids = None


def _parse_duration(raw) -> float | None:
    """CISLR ships 2 rows with duration '#N/A'; never let them break loading."""
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _is_junk_gloss(gloss: str) -> bool:
    """Dataset artifacts (e.g. '#N/A') are never valid signs."""
    return not gloss or gloss.startswith("#")


def gloss_landmark_uids() -> set:
    """UIDs with a ready per-gloss landmark clip (avatar-mirrorable)."""
    global _landmark_uids
    if _landmark_uids is not None:
        return _landmark_uids
    _landmark_uids = set()
    glm_path = get_paths().mapping_file.parent / "gloss_landmarks.json"
    try:
        import json

        data = json.loads(glm_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _landmark_uids = {
                v.get("uid") for v in data.values()
                if isinstance(v, dict) and v.get("status") == "ready" and v.get("uid")
            }
    except (OSError, ValueError) as e:
        print(f"[GlossLookup] gloss landmarks unreadable ({e}); duration-only ranking.")
    return _landmark_uids


def iter_best_cislr_rows(csv_path=None):
    """Yield the winning (gloss, uid, category, duration) per gloss.

    Rank: ready landmark clip first, then shortest duration (None sorts
    last), then uid for determinism. Junk rows skipped.
    """
    from collections import defaultdict

    path = csv_path or CISLR_CSV
    buckets: dict[str, list] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                gloss = (row.get("gloss") or "").strip().lower()
                if _is_junk_gloss(gloss):
                    continue
                uid = (row.get("uid") or "").strip()
                category = (row.get("category") or "").strip()
                duration = _parse_duration(row.get("duration"))
            except Exception:
                continue
            buckets[gloss].append((uid, category, duration))

    ready = gloss_landmark_uids()
    for gloss in sorted(buckets):
        rows = buckets[gloss]
        if len(rows) == 1:
            uid, category, duration = rows[0]
        else:
            uid, category, duration = min(
                rows,
                key=lambda r: (
                    0 if r[0] in ready else 1,
                    r[2] if r[2] is not None else float("inf"),
                    r[0],
                ),
            )
        yield gloss, uid, category, duration


def reset_vocabulary_state():
    """Test hook: forget cached vocab and landmark uids."""
    global _gloss_vocab, _landmark_uids
    _gloss_vocab = None
    _landmark_uids = None


def load_vocabulary():
    global _gloss_vocab
    if _gloss_vocab is not None:
        return _gloss_vocab

    _gloss_vocab = {}

    if not CISLR_CSV.exists():
        print(f"WARNING: CISLR CSV not found: {CISLR_CSV}")
        return _gloss_vocab

    for gloss, uid, category, duration in iter_best_cislr_rows():
        _gloss_vocab[gloss] = {
            "uid": uid,
            "category": category,
            "duration": duration,
        }

    print(f"Loaded {len(_gloss_vocab)} unique glosses from CISLR.")
    return _gloss_vocab


def lookup_word(word: str) -> dict | None:
    vocab = load_vocabulary()
    return vocab.get(word.lower().strip())


def sentence_to_glosses(sentence: str) -> list:
    words = sentence.lower().strip().split()
    results = []

    for word in words:
        cleaned = word.strip(".,!?;:'\"")
        if not cleaned:
            continue

        match = lookup_word(cleaned)
        results.append({
            "word": cleaned,
            "found": match is not None,
            "gloss": cleaned.upper() if match else None,
            "uid": match["uid"] if match else None,
            "category": match["category"] if match else None
        })

    return results


def get_available_glosses() -> list:
    vocab = load_vocabulary()
    return sorted(vocab.keys())
