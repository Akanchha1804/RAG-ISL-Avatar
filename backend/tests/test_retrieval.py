"""RAG retrieval: top_k, similarity math, query recording, order relative to generation."""

import asyncio

import pytest

import gloss_generator
from main import run_pipeline, stage_retrieve
from similarity import distance_to_similarity, exact_match_similarity


def test_stage_retrieve_records_query_and_top_k(rag_module):
    results = stage_retrieve("hello how are you", use_rag=True, top_k=3)
    assert rag_module.calls == [{"query": "hello how are you", "top_k": 3}]
    assert len(results) <= 3
    for r in results:
        assert {"sentence", "glosses", "landmark_file", "similarity", "distance"} <= set(r)
        assert r["similarity"] == pytest.approx(
            distance_to_similarity(r["distance"]), abs=1e-3
        )


def test_stage_retrieve_skipped_when_rag_disabled(rag_module):
    assert stage_retrieve("hello", use_rag=False, top_k=5) == []
    assert rag_module.calls == []


def test_stage_retrieve_clamps_top_k(rag_module):
    stage_retrieve("hello", use_rag=True, top_k=999)
    assert rag_module.calls[0]["top_k"] == 20

    rag_module.clear()
    stage_retrieve("hello", use_rag=True, top_k=0)
    assert rag_module.calls[0]["top_k"] == 1


def test_exact_match_similarity_is_one():
    assert exact_match_similarity() == 1.0
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(1.0) == 0.5


def test_retrieval_runs_before_generation(rag_module, monkeypatch):
    order = []
    original_search = rag_module.search

    def tracking_search(query, top_k=5):
        order.append("retrieve")
        return original_search(query, top_k)

    def tracking_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        order.append("generate")
        assert use_rag is True
        assert retrieved_examples is not None, "generator must receive retrieved context"
        assert len(retrieved_examples) >= 1
        return "HI HOW ARE_YOU"

    monkeypatch.setattr(rag_module, "search", tracking_search)
    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", tracking_generate)

    result = asyncio.run(
        run_pipeline(text="hello how are you friend", use_rag=True, top_k=5)
    )

    assert order.index("retrieve") < order.index("generate")
    assert result["method"] == "t5_rag"


def test_disabled_rag_never_calls_search(rag_module):
    result = asyncio.run(run_pipeline(text="hello how are you", use_rag=False))
    assert rag_module.calls == []
    assert result["retrieved_examples"] == []
    assert result["method"] == "direct_lookup"
