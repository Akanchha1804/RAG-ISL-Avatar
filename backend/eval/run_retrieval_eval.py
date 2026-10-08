"""Retrieval quality harness (M5/Phase 4): Recall@K + admission accuracy.

Fixed paraphrase probes with known targets, scored against the LIVE FAISS
index (real MiniLM, CPU). Admission = top-1 similarity >= the extractive
threshold (similarity.EXTRACTIVE_MATCH_THRESHOLD). Independent of
translation scores by design.

    .venv\\Scripts\\python.exe eval/run_retrieval_eval.py
"""

import sys

import metrics_eval
from metrics_eval import BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

# (query, expected_sentence or None, expect_extractive)
PROBES = [
    ("hi how are you", "hi how are you", True),
    ("Hi, How Are You?", "hi how are you", True),
    ("hi", "hi how are you", False),
    ("please help me", "help me", True),
    ("can you please repeat that", "can you repeat that please", True),
    ("i need some water", "i need water", True),
    ("how are you", "hi how are you", True),
    ("good morning", None, False),
    ("namaste", None, False),
    ("quantum physics is difficult", None, False),
    ("the weather is nice today", None, False),
    ("what is your name", None, False),
]


def main() -> int:
    from similarity import EXTRACTIVE_MATCH_THRESHOLD

    import rag

    per_probe = []
    for query, expected, expect_extract in PROBES:
        results = rag.search(query, top_k=3)
        top1 = results[0] if results else None
        admitted = bool(top1 and top1["similarity"] >= EXTRACTIVE_MATCH_THRESHOLD)
        per_probe.append({
            "query": query,
            "expected": expected,
            "expect_extractive": expect_extract,
            "top1": top1["sentence"] if top1 else None,
            "top1_similarity": top1["similarity"] if top1 else None,
            "top3": [r["sentence"] for r in results],
            "recall_at_1": bool(expected and top1 and top1["sentence"] == expected),
            "recall_at_3": bool(expected and any(r["sentence"] == expected for r in results)),
            "admission_correct": admitted == expect_extract,
        })

    expected_probes = [p for p in per_probe if p["expected"]]
    payload = {
        "threshold": EXTRACTIVE_MATCH_THRESHOLD,
        "n_probes": len(per_probe),
        "recall_at_1": round(sum(p["recall_at_1"] for p in expected_probes) / len(expected_probes), 4),
        "recall_at_3": round(sum(p["recall_at_3"] for p in expected_probes) / len(expected_probes), 4),
        "admission_accuracy": round(sum(p["admission_correct"] for p in per_probe) / len(per_probe), 4),
        "per_probe": per_probe,
    }
    path = metrics_eval.save_result("retrieval_eval", payload)
    print(f"R@1={payload['recall_at_1']} R@3={payload['recall_at_3']} "
          f"admission={payload['admission_accuracy']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
