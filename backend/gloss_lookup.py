"""
Gloss Vocabulary Lookup.
Maps English words to CISLR gloss entries.
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


def load_vocabulary():
    global _gloss_vocab
    if _gloss_vocab is not None:
        return _gloss_vocab

    _gloss_vocab = {}

    if not CISLR_CSV.exists():
        print(f"WARNING: CISLR CSV not found: {CISLR_CSV}")
        return _gloss_vocab

    with open(CISLR_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row["gloss"].strip().lower()
            uid = row["uid"].strip()
            category = row["category"].strip()
            duration = float(row["duration"])

            if gloss not in _gloss_vocab:
                _gloss_vocab[gloss] = {
                    "uid": uid,
                    "category": category,
                    "duration": duration
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
