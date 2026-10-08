"""Shared test fixtures: DATA_DIR, fake rag, DB isolation."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

# Paths must be configured BEFORE importing backend modules (paths.get_paths is lru_cached).
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
FIXTURE_DATA_DIR = TESTS_DIR / "fixtures" / "data"

os.environ["DATA_DIR"] = str(FIXTURE_DATA_DIR)
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@127.0.0.1:1/isl_test"
os.environ["ISL_WARMUP_MODELS"] = "0"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import paths
paths.reset_paths_cache()

# Fixture mapping used by the default fake search (retrieval must return
# results unless a test calls set_results for an exact scenario).
_MAPPING_FILE = FIXTURE_DATA_DIR / "ISL_MediaPipe" / "sentence_mapping.json"
_FIXTURE_MAPPING = json.loads(_MAPPING_FILE.read_text(encoding="utf-8"))


class FakeRag(types.ModuleType):
    """Pre-seeded stand-in for the rag module: records calls, returns results.

    Default scoring uses the documented contract similarity = 1 / (1 + distance):
      - token overlap >= 1  -> distance = 1/(1+overlap)  (similarity >= 0.5)
      - no overlap          -> distance = 2.0           (similarity = 1/3 < 0.4)
    Tests may override with set_results() for exact scenarios.
    """

    def __init__(self):
        super().__init__("rag")
        self.calls = []
        self._results = None

    def set_results(self, results):
        self._results = list(results) if results is not None else None

    def clear(self):
        self.calls.clear()
        self._results = None

    def search(self, query, top_k=5):
        self.calls.append({"query": query, "top_k": top_k})
        if self._results is not None:
            return list(self._results)[:top_k]

        q_tokens = set(str(query).lower().split())
        results = []
        for sentence, meta in _FIXTURE_MAPPING.items():
            s_tokens = set(sentence.lower().split())
            overlap = len(q_tokens & s_tokens)
            distance = (1.0 / (1.0 + overlap)) if overlap else 2.0
            results.append(
                {
                    "sentence": sentence,
                    "glosses": meta.get("glosses", ""),
                    "landmark_file": meta.get("landmark_file", ""),
                    "similarity": round(1.0 / (1.0 + distance), 4),
                    "distance": round(distance, 4),
                }
            )
        results.sort(key=lambda r: r["distance"])
        return results[:top_k]


fake_rag = FakeRag()
sys.modules["rag"] = fake_rag


@pytest.fixture(autouse=True)
def _reset_fake_rag():
    fake_rag.clear()
    yield
    fake_rag.clear()


@pytest.fixture
def rag_module():
    return fake_rag


@pytest.fixture
def fixture_mapping():
    path = FIXTURE_DATA_DIR / "ISL_MediaPipe" / "sentence_mapping.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as c:
        yield c
