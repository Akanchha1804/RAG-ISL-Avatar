"""
RAG Retrieval Module.
Uses sentence-transformers + FAISS to find similar ISL training examples.

Similarity values follow similarity.distance_to_similarity():
    similarity = 1 / (1 + L2_distance), range (0, 1].
Exact string match has distance 0 and therefore similarity 1.0.
"""

import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from paths import get_paths
from similarity import distance_to_similarity

# =====================================================
# PATHS
# =====================================================

_paths = get_paths()
MAPPING_FILE = _paths.mapping_file
INDEX_DIR = _paths.index_dir
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# GLOBALS
# =====================================================

_model = None
_index = None
_metadata = None
_sentences = None


def get_model():
    global _model
    if _model is None:
        print("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")
    return _model


def load_mapping():
    global _sentences, _metadata
    if _sentences is None:
        if not MAPPING_FILE.exists():
            print(f"WARNING: Mapping file not found: {MAPPING_FILE}")
            _sentences = []
            _metadata = {}
            return

        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

        _sentences = list(_metadata.keys())
        print(f"Loaded {len(_sentences)} sentences from mapping.")
    return _sentences


def build_index():
    global _index, _sentences

    sentences = load_mapping()
    if not sentences:
        return

    model = get_model()
    print("Encoding sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dimension)
    _index.add(embeddings)

    faiss.write_index(_index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "sentences": sentences,
            "dimension": dimension,
            "count": len(sentences)
        }, f, indent=2)

    print(f"FAISS index built: {len(sentences)} vectors, dim={dimension}")


def load_index():
    global _index

    if _index is not None:
        return

    load_mapping()

    if INDEX_FILE.exists():
        print("Loading existing FAISS index...")
        _index = faiss.read_index(str(INDEX_FILE))
        print(f"Index loaded: {_index.ntotal} vectors.")
    else:
        print("No index found, building...")
        build_index()


def search(query: str, top_k: int = 5) -> list:
    """Retrieve the top_k most similar corpus sentences for a query.

    Returns a list of dicts ordered by descending similarity:
        {sentence, glosses, landmark_file, similarity, distance}
    similarity = 1 / (1 + L2_distance)  (see similarity.py).
    """
    load_index()

    if _index is None or _index.ntotal == 0:
        return []

    model = get_model()
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = _index.search(query_embedding, top_k)

    results = []
    sentences = load_mapping()

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(sentences):
            continue

        sentence = sentences[idx]
        entry = _metadata.get(sentence, {})

        results.append({
            "sentence": sentence,
            "glosses": entry.get("glosses", ""),
            "landmark_file": entry.get("landmark_file", ""),
            "similarity": round(distance_to_similarity(float(dist)), 4),
            "distance": round(float(dist), 4)
        })

    return results


def get_closest_match(query: str, threshold: float = 0.6) -> dict | None:
    results = search(query, top_k=1)
    if results and results[0]["similarity"] >= threshold:
        return results[0]
    return None
