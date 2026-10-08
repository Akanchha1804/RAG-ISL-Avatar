"""Phase 4 eval smoke: metric math + harness wiring. Fast, offline models-free.

Full harnesses run via backend/eval/run_all.py (translation needs T5
weights present for import only; retrieval needs the FAISS index).
"""

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sacrebleu") is None,
    reason="sacrebleu not installed",
)


def test_bleu_perfect_and_zero():
    from eval.metrics_eval import bleu_score

    # BLEU-4 needs 4+ tokens for non-degenerate precisions.
    perfect = bleu_score(["I YOU YOUR CAREER PLAN WHAT"],
                         ["I YOU YOUR CAREER PLAN WHAT"])
    assert perfect["bleu"] == pytest.approx(100.0)
    bad = bleu_score(["A B C D E F"], ["W X Y Z P Q"])
    assert bad["bleu"] == pytest.approx(0.0)


def test_rouge_l_perfect_and_partial():
    from eval.metrics_eval import rouge_l_mean

    assert rouge_l_mean(["I YOU HELP"], ["I YOU HELP"])["rougeL_f1_mean"] == pytest.approx(1.0)
    partial = rouge_l_mean(["I YOU"], ["I YOU HELP"])["rougeL_f1_mean"]
    assert 0.0 < partial < 1.0


def test_meteor_perfect_and_partial():
    from eval.metrics_eval import meteor_mean

    assert meteor_mean(["I YOU HELP"], ["I YOU HELP"])["meteor_mean"] > 0.9
    partial = meteor_mean(["I YOU"], ["I YOU HELP"])["meteor_mean"]
    assert 0.0 < partial < 1.0


def test_retrieval_probes_have_known_shape():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import run_retrieval_eval

    assert len(run_retrieval_eval.PROBES) >= 10
    for query, expected, expect_extract in run_retrieval_eval.PROBES:
        assert isinstance(query, str) and query.strip()
        assert isinstance(expect_extract, bool)
        assert expected is None or isinstance(expected, str)


def test_results_stamped_with_lock_state():
    import eval.metrics_eval as metrics_eval

    assert isinstance(metrics_eval.refs_locked(), bool)
