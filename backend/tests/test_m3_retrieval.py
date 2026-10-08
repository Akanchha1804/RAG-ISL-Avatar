"""M3: threshold single-source + index-drift logic. No real models.

NOTE: conftest replaces `rag` in sys.modules with FakeRag, so the real
module is loaded from its file path here to test its pure logic with a
stubbed search (no MiniLM/FAISS touched).
"""

import importlib.util
import json
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent


def load_real_rag():
    import sys

    for mod in list(sys.modules):
        if mod == "rag" or mod.startswith("rag."):
            saved = sys.modules.pop(mod)
            break
    else:
        saved = None
    try:
        spec = importlib.util.spec_from_file_location(
            "real_rag_under_test", str(BACKEND_DIR / "rag.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is not None:
            sys.modules["rag"] = saved


@pytest.fixture(scope="module")
def real_rag():
    return load_real_rag()


def test_threshold_single_source(real_rag):
    """Pipeline and helper must agree on the extractive threshold."""
    import main
    from similarity import EXTRACTIVE_MATCH_THRESHOLD

    assert main.SIMILARITY_MATCH_THRESHOLD == EXTRACTIVE_MATCH_THRESHOLD == 0.75
    import inspect

    default = inspect.signature(real_rag.get_closest_match).parameters["threshold"].default
    assert default == EXTRACTIVE_MATCH_THRESHOLD


def test_get_closest_match_threshold(real_rag, monkeypatch):
    hit = {"sentence": "help me", "glosses": "HELP ME", "landmark_file": "h.json",
           "similarity": 0.80, "distance": 0.25}
    monkeypatch.setattr(real_rag, "search", lambda q, top_k=1: [dict(hit)])
    assert real_rag.get_closest_match("please help me") == hit
    assert real_rag.get_closest_match("please help me", threshold=0.9) is None
    monkeypatch.setattr(real_rag, "search", lambda q, top_k=1: [])
    assert real_rag.get_closest_match("anything") is None


def test_index_matches_mapping(real_rag, tmp_path):
    meta = tmp_path / "metadata.json"
    monkeypatch_meta = real_rag.METADATA_FILE
    real_rag.METADATA_FILE = meta
    try:
        sents = ["a", "b"]
        assert real_rag._index_matches_mapping(sents) is False  # missing file
        meta.write_text(json.dumps({"sentences": sents}), encoding="utf-8")
        assert real_rag._index_matches_mapping(sents) is True
        assert real_rag._index_matches_mapping(["a", "c"]) is False  # drifted
        meta.write_text("{not json", encoding="utf-8")
        assert real_rag._index_matches_mapping(sents) is False  # corrupt
    finally:
        real_rag.METADATA_FILE = monkeypatch_meta
