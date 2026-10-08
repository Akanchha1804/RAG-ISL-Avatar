"""Shared eval helpers: refs loading + BLEU/ROUGE-L/METEOR (M4 Phase 4).

Each harness (run_*_eval.py) runs independently so a retrieval miss can
never lower a translation score and vice versa. Translation results are
provisional until the BLEU reference set locks (review_hold.txt empty);
the harness stamps every result file with locked: true/false.
"""

import csv
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = BACKEND_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"


def refs_locked() -> bool:
    """True when no sentence is waiting for separate review."""
    hold_file = EVAL_DIR / "review_hold.txt"
    try:
        holds = [l.strip() for l in hold_file.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    except OSError:
        holds = []
    return not holds


def load_refs() -> list:
    """All 101 refs: {sentence, corpus_gloss, reference, corrected}."""
    rows = []
    with open(EVAL_DIR / "references.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "sentence": row["sentence"],
                "corpus_gloss": row["corpus_gloss"],
                "reference": " ".join((row["corrected_gloss"] or "").split()),
                "corrected": row.get("corrected") == "yes",
            })
    return rows


def bleu_score(hypotheses: list, references: list) -> dict:
    """Corpus BLEU on pre-tokenized gloss strings (sacrebleu, tokenize=none)."""
    import sacrebleu

    # NOTE: ref_streams is one stream per reference SET, i.e. [[ref...]].
    bleu = sacrebleu.corpus_bleu(hypotheses, [[r for r in references]], tokenize="none")
    return {
        "bleu": round(bleu.score, 2),
        "precisions": [round(p, 2) for p in bleu.precisions],
        "brevity_penalty": round(bleu.bp, 3),
        "hyp_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
    }


def rouge_l_mean(hypotheses: list, references: list) -> dict:
    """Mean ROUGE-L F1 (rouge-score, space-tokenized glosses)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
              for hyp, ref in zip(hypotheses, references)]
    return {
        "rougeL_f1_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "n": len(scores),
    }


def meteor_mean(hypotheses: list, references: list) -> dict:
    """Mean sentence METEOR (nltk, needs wordnet; exact+stem+synonym)."""
    from nltk.translate.meteor_score import meteor_score

    scores = [meteor_score([ref.split()], hyp.split())
              for hyp, ref in zip(hypotheses, references)]
    return {
        "meteor_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "n": len(scores),
    }


def save_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"locked_refs": refs_locked(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
