"""Build backend/eval/references.csv: canonical ISL refs for BLEU eval (M4 Phase 2).

Source: backend/sentence_mapping.json (101 corpus sentences).
Canonical column: isl_grammar.canonicalize applied to the corpus gloss tokens
with the PRODUCTION vocabulary (backend/dataset.csv), so refs match exactly
what the live pipeline emits for these sentences.

Regenerate only after a deliberate grammar rule change:
    .venv\\Scripts\\python.exe eval/build_references.py
then review the git diff of references.csv before locking.

Manual corrections (user-verified) live in eval/corrections.csv
(sentence,corrected_gloss) and override the generated column on every run,
so fixes survive regeneration.
"""

import csv
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = BACKEND_DIR / "eval"
MAPPING_FILE = BACKEND_DIR / "sentence_mapping.json"
CISLR_CSV = BACKEND_DIR / "dataset.csv"
REFERENCES_FILE = EVAL_DIR / "references.csv"
CORRECTIONS_FILE = EVAL_DIR / "corrections.csv"


def load_production_vocab() -> dict:
    vocab = {}
    with open(CISLR_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            gloss = (row.get("gloss") or "").strip().lower()
            if gloss and gloss not in vocab:
                vocab[gloss] = {
                    "uid": row.get("uid"),
                    "category": row.get("category"),
                    "duration": row.get("duration"),
                }
    return vocab


def load_corrections() -> dict:
    corrections = {}
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sentence = (row.get("sentence") or "").strip().lower()
                gloss = " ".join((row.get("corrected_gloss") or "").split())
                if sentence and gloss:
                    corrections[sentence] = gloss
    return corrections


def main() -> int:
    import gloss_lookup
    from isl_grammar import canonicalize

    gloss_lookup._gloss_vocab = load_production_vocab()
    print(f"production vocab: {len(gloss_lookup._gloss_vocab)} glosses")

    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    corrections = load_corrections()
    print(f"manual corrections: {len(corrections)}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sentence in sorted(mapping):
        entry = mapping[sentence]
        corpus_gloss = " ".join(str(entry.get("glosses", "")).split())
        tokens = [t for t in corpus_gloss.split(" ") if t]
        canon, dropped = canonicalize(tokens)
        canonical = " ".join(canon)
        key = sentence.lower().strip()
        final = corrections.get(key, canonical)
        rows.append({
            "sentence": sentence,
            "corpus_gloss": corpus_gloss,
            "canonical_gloss": canonical,
            "corrected_gloss": final,
            "dropped": " ".join(dropped),
            "corrected": "yes" if key in corrections else "no",
        })

    with open(REFERENCES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentence", "corpus_gloss", "canonical_gloss",
            "corrected_gloss", "dropped", "corrected",
        ])
        writer.writeheader()
        writer.writerows(rows)

    n_corr = sum(1 for r in rows if r["corrected"] == "yes")
    print(f"wrote {len(rows)} refs -> {REFERENCES_FILE} ({n_corr} manually corrected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
