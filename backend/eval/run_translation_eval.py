"""Translation fidelity harness (M4 Phase 4): BLEU/ROUGE-L/METEOR vs refs.

Runs every corpus sentence through stage_generate (exact path: deterministic,
no models) and scores the canonical glosses against corrected_gloss refs.
Also reports the rule-vs-human gap: corrected rows should score below the
uncorrected (exact-match) rows until the rules converge.

    .venv\\Scripts\\python.exe eval/run_translation_eval.py
"""

import sys

import metrics_eval
from metrics_eval import BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

# Production files (same set the snapshot pin uses), independent of DATA_DIR.
import gloss_lookup  # noqa: E402
from main import gloss_to_sequence, stage_generate  # noqa: E402


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
    hyps, golds, per_sentence = [], [], []
    for r in refs:
        out = stage_generate(r["sentence"], [], True)
        hyp = " ".join(out["gloss_sequence"])
        hyps.append(hyp)
        golds.append(r["reference"])
        per_sentence.append({
            "sentence": r["sentence"],
            "method": out["method"],
            "hypothesis": hyp,
            "reference": r["reference"],
            "corrected": r["corrected"],
            "exact": hyp == r["reference"],
        })

    exact = sum(1 for p in per_sentence if p["exact"])
    payload = {
        "n": len(refs),
        "exact_match": exact,
        "exact_match_rate": round(exact / len(refs), 4),
        "bleu": metrics_eval.bleu_score(hyps, golds),
        "rougeL": metrics_eval.rouge_l_mean(hyps, golds),
        "meteor": metrics_eval.meteor_mean(hyps, golds),
        "per_sentence": per_sentence,
    }
    path = metrics_eval.save_result("translation_eval", payload)
    print(f"n={len(refs)} exact={exact} "
          f"BLEU={payload['bleu']['bleu']} "
          f"ROUGE-L={payload['rougeL']['rougeL_f1_mean']} "
          f"METEOR={payload['meteor']['meteor_mean']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
