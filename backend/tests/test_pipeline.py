"""Full pipeline stage composition (text input, no STT)."""

import asyncio

from main import run_pipeline


def test_run_pipeline_retrieve_then_generate_then_animate(rag_module, monkeypatch):
    import gloss_generator

    seen = {}

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        seen["retrieved"] = retrieved_examples
        return "HOW ARE YOU"

    # Below-threshold result so the pipeline reaches generation
    # (the default fake scores this query ~0.83 -> extractive path).
    rag_module.set_results(
        [
            {
                "sentence": "hello how are you",
                "glosses": "HI HOW ARE_YOU",
                "landmark_file": "hello_landmarks.json",
                "similarity": 0.6,
                "distance": 0.625,
            }
        ]
    )

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    result = asyncio.run(
        run_pipeline(text="hello how are you friend", use_rag=True, top_k=4)
    )

    # stage order evidence: retrieval recorded before generation used results
    assert rag_module.calls == [{"query": "hello how are you friend", "top_k": 4}]
    # generation must not receive retrieved examples (stitched-gloss fix)
    assert seen["retrieved"] is None

    assert result["use_rag"] is True
    assert result["top_k"] == 4
    assert result["method"] in ("t5", "rag", "word_by_word")
    # M4: fake T5 "HOW ARE YOU" canonicalized (ARE dropped, HOW final).
    assert result["gloss_sequence"] == ["YOU", "HOW"]
    assert result["animation"]["clip_playlist"]
    assert result["animation"]["unresolved_tokens"] == []


def test_run_pipeline_rag_disabled(rag_module):
    result = asyncio.run(
        run_pipeline(text="hello how are you", use_rag=False)
    )
    assert rag_module.calls == []
    assert result["use_rag"] is False
    assert result["retrieved_examples"] == []
    assert result["method"] == "direct_lookup"


def test_run_pipeline_top_k_clamped(rag_module):
    asyncio.run(run_pipeline(text="hello", use_rag=True, top_k=999))
    assert rag_module.calls[0]["top_k"] == 20

    rag_module.clear()
    asyncio.run(run_pipeline(text="hello", use_rag=True, top_k=-5))
    assert rag_module.calls[0]["top_k"] == 1
