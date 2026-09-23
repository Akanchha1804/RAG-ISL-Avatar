"""
Animation resolution layer.

Canonical mapping structure (built from two existing data sources):

    gloss token
      -> sign/animation ID   (CISLR uid when the token exists in the gloss
                              vocabulary; otherwise a corpus-namespaced ID)
      -> clip / landmark resource
           clip_id           (CISLR uid, or "corpus::<token>" reference ID)
           landmark_file     (sentence-level reference sequence from the
                              ISL-CSLTR corpus that contains this token;
                              the corpus has no per-gloss landmark clips)
      -> duration / metadata (CISLR duration in ms + category when known)

Resolution never crashes on unknown glosses: tokens found in neither source
are reported in unresolved_tokens.

This layer is intentionally separate from sentence retrieval and gloss
generation so animation resolution can be tested (and extended) independently.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from gloss_lookup import load_vocabulary
from paths import get_paths

SOURCE_CISLR = "cislr_vocabulary"
SOURCE_CORPUS = "sentence_corpus"

_registry = None


def _token_key(token: str) -> str:
    return token.strip().strip(".,!?;:'\"").lower()


def _load_corpus_token_index() -> dict:
    """token_key -> first corpus sentence landmark file containing it."""
    mapping_file = get_paths().mapping_file
    index: dict = {}
    if not mapping_file.exists():
        return index

    import json

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for entry in mapping.values():
        if not isinstance(entry, dict):
            continue
        glosses = entry.get("glosses") or ""
        landmark_file = entry.get("landmark_file") or ""
        for token in glosses.split():
            key = _token_key(token)
            if key and key not in index:
                index[key] = landmark_file
    return index


def get_registry() -> dict:
    """Build (once) and return the canonical gloss registry."""
    global _registry
    if _registry is not None:
        return _registry

    vocabulary = load_vocabulary()
    cislr = {}
    for gloss_key, info in vocabulary.items():
        try:
            duration_s = float(info.get("duration") or 0)
        except (TypeError, ValueError):
            duration_s = 0.0
        cislr[gloss_key] = {
            "sign_id": info.get("uid"),
            "clip_id": info.get("uid"),
            "duration_ms": int(duration_s * 1000) if duration_s > 0 else None,
            "category": info.get("category"),
            "source": SOURCE_CISLR,
        }

    _registry = {
        "cislr": cislr,
        "corpus_token_landmarks": _load_corpus_token_index(),
    }
    return _registry


def reset_registry() -> None:
    """Test hook: force a registry rebuild on next use."""
    global _registry
    _registry = None


def resolve_gloss_token(token: str, registry: Optional[dict] = None) -> Optional[dict]:
    """Resolve one gloss token to its canonical sign resource dict, or None."""
    reg = registry if registry is not None else get_registry()
    key = _token_key(token)
    if not key:
        return None

    corpus_landmark = reg["corpus_token_landmarks"].get(key, "")

    if key in reg["cislr"]:
        entry = dict(reg["cislr"][key])
        entry["gloss"] = token
        entry["landmark_file"] = corpus_landmark or None
        return entry

    if corpus_landmark:
        return {
            "gloss": token,
            "sign_id": None,
            "clip_id": f"corpus::{key}",
            "duration_ms": None,
            "category": None,
            "source": SOURCE_CORPUS,
            "landmark_file": corpus_landmark,
        }

    return None


def resolve_gloss_sequence(gloss_sequence: Iterable) -> dict:
    """Resolve a full gloss sequence to a clip playlist.

    Returns:
        {
          "clip_playlist":   [ {gloss, sign_id, clip_id, duration_ms,
                                category, source, landmark_file}, ... ],
          "resolved_tokens": [token, ...],   # same order as input
          "unresolved_tokens": [token, ...]  # reported, never raised
        }

    Never raises for unknown glosses or empty input.
    """
    registry = get_registry()
    clip_playlist: List[dict] = []
    resolved_tokens: List[str] = []
    unresolved_tokens: List[str] = []

    if not gloss_sequence:
        return {
            "clip_playlist": clip_playlist,
            "resolved_tokens": resolved_tokens,
            "unresolved_tokens": unresolved_tokens,
        }

    for token in gloss_sequence:
        if not isinstance(token, str) or not token.strip():
            continue

        entry = resolve_gloss_token(token, registry=registry)
        if entry is None:
            unresolved_tokens.append(token)
            continue

        resolved_tokens.append(token)
        clip_playlist.append(entry)

    return {
        "clip_playlist": clip_playlist,
        "resolved_tokens": resolved_tokens,
        "unresolved_tokens": unresolved_tokens,
    }
