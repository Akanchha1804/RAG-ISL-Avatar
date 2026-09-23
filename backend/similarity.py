"""
Documented similarity conversion.

Single source of truth used by the RAG retrieval module and by exact-match
handling so that every similarity value reported by the system follows the
same, stated rule.

Conversion:
    similarity = 1 / (1 + L2_distance)

Properties (documented in Docs/Description/05_API_Specification-1.md):
  - similarity is in (0, 1]
  - distance 0 (identical embeddings / exact string match) => similarity 1.0
  - the conversion is strictly monotonic decreasing in distance
"""

from __future__ import annotations


def distance_to_similarity(distance: float) -> float:
    """Map a non-negative L2 distance to a similarity in (0, 1]."""
    d = float(distance)
    if d < 0:
        raise ValueError(f"L2 distance must be non-negative, got {distance}")
    return 1.0 / (1.0 + d)


def exact_match_similarity() -> float:
    """Similarity reported for an exact corpus hit (distance = 0)."""
    return distance_to_similarity(0.0)
