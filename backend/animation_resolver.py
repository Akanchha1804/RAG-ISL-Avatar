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

# English words that map onto a canonical CISLR gloss token. Used to
# widen clip coverage for novel sentences: the fine-tuned model and the
# word-by-word fallback emit the source English word, but a word only
# animates if it is (or maps to) a gloss in the CISLR vocabulary.
NORMALIZATION_MAP = {
    # greetings - CISLR signs the Indian greeting
    "hello": "namaste",
    "hi": "namaste",
    "hey": "namaste",
    # thanks - CISLR has the two-word gloss "thank you"
    "thank": "thank you",
    "thanks": "thank you",
    # go family - CISLR signs the base form
    "going": "go",
    "gone": "go",
    "went": "go",
    "living": "live",
}

# English "to be" forms. ISL does not sign copular be-verbs, so they
# are dropped from the word-by-word gloss rather than left as an
# unresolvable token.
BE_VERBS = {
    "am", "are", "is", "was", "were", "be", "being", "been", "art",
}

# Punctuation stripped when matching a token against the vocabulary.
_TOKEN_PUNCT = ".,!?;:''\""

_registry = None

# Per-gloss landmark clips built by ISL_MediaPipe/batch_gloss_landmarks.py:
# gloss key -> {"status": "ready", "file": "gloss_<key>.json", ...}.
# Cached by file mtime so clips appear as soon as the batch writes them.
_gloss_landmarks_cache = {"mtime": None, "data": {}}


def _gloss_landmarks_path():
    return get_paths().mapping_file.parent / "gloss_landmarks.json"


def load_gloss_landmarks() -> dict:
    """Per-gloss landmark clip mapping (refreshed when the file changes)."""
    import json

    path = _gloss_landmarks_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _gloss_landmarks_cache["mtime"] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _gloss_landmarks_cache["data"] = data if isinstance(data, dict) else {}
        except Exception:
            _gloss_landmarks_cache["data"] = {}
        _gloss_landmarks_cache["mtime"] = mtime
    return _gloss_landmarks_cache["data"]


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
        uid = info.get("uid")
        cislr[gloss_key] = {
            "sign_id": uid,
            "clip_id": uid,
            "clip_url": f"/clips/{uid}.mp4" if uid else None,
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
        # Per-gloss landmark clip for the avatar playlist (the sentence
        # landmark_file above only applies to exact corpus sentences).
        gloss_entry = load_gloss_landmarks().get(key)
        if (
            isinstance(gloss_entry, dict)
            and gloss_entry.get("status") == "ready"
            and gloss_entry.get("file")
        ):
            entry["landmark_clip_url"] = f"/landmarks/{gloss_entry['file']}"
        else:
            entry["landmark_clip_url"] = None
        return entry

    if corpus_landmark:
        return {
            "gloss": token,
            "sign_id": None,
            "clip_id": f"corpus::{key}",
            "clip_url": None,
            "duration_ms": None,
            "category": None,
            "source": SOURCE_CORPUS,
            "landmark_file": corpus_landmark,
            "landmark_clip_url": None,
        }

    return None


def resolve_gloss_sequence(gloss_sequence: Iterable) -> dict:
    """Resolve a full gloss sequence to a clip playlist.

    Returns:
        {
          "clip_playlist":   [ {gloss, sign_id, clip_id, clip_url,
                                 duration_ms, category, source,
                                 landmark_file, landmark_clip_url}, ... ],
          "resolved_tokens": [token, ...],   # same order as input
          "unresolved_tokens": [token, ...]  # reported, never raised
        }

    landmark_clip_url is the per-gloss landmark clip for the avatar
    playlist (/landmarks/gloss_<key>.json) when one has been built,
    else None. clip_url is the CISLR reference video (API-level).

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


def _stem_to_gloss(key: str, cislr: dict) -> Optional[str]:
    """Map an inflected English word onto a base-form gloss, or None.

    Only applies when the base form is a real CISLR gloss, so it
    never invents a sign ("going" -> "go", "living" -> "live",
    "walks" -> "walk"). Returns the canonical gloss key (lowercase).
    """
    if not key:
        return None
    candidates: List[str] = []
    if key.endswith("ing"):
        base = key[:-3]
        candidates.append(base)            # going -> go
        candidates.append(base + "e")      # living -> live
        if len(base) > 1 and base[-1] == base[-2]:
            candidates.append(base[:-1])   # running -> run
    if key.endswith("s") and not key.endswith("ss"):
        candidates.append(key[:-1])        # walks -> walk
        if key.endswith("es"):
            candidates.append(key[:-2])    # goes -> go
    for candidate in candidates:
        if candidate and candidate in cislr:
            return candidate
    return None


def normalize_gloss_sequence(gloss_sequence: Iterable) -> tuple:
    """Normalize a gloss sequence to canonical CISLR gloss tokens.

    Widens clip coverage for novel sentences, where the source is the
    raw English text (T5 output or the word-by-word fallback). It:

      * keeps tokens that are already CISLR glosses,
      * maps English words that are not glosses onto their canonical
        ISL gloss (hello/hi -> namaste, thank -> thank you,
        going -> go),
      * drops copular be-verbs (ISL does not sign "am/are/is"),
      * applies safe suffix stemming only when the base form is a
        real gloss.

    Curated corpus glosses (direct_lookup / rag) must NOT pass
    through here - their tokens are already correct.

    Returns (normalized_tokens, dropped_tokens). Multi-word glosses
    such as "thank you" stay a single token.
    """
    registry = get_registry()
    cislr = registry["cislr"]

    normalized: List[str] = []
    dropped: List[str] = []
    tokens = [t for t in gloss_sequence if isinstance(t, str) and t.strip()]

    i = 0
    while i < len(tokens):
        key = _token_key(tokens[i])

        # "thank you" is one two-word gloss in CISLR.
        if (
            key == "thank"
            and i + 1 < len(tokens)
            and _token_key(tokens[i + 1]) == "you"
        ):
            normalized.append("THANK YOU")
            i += 2
            continue

        if key in cislr:
            normalized.append(tokens[i])
        elif key in NORMALIZATION_MAP:
            normalized.append(NORMALIZATION_MAP[key].upper())
        elif key in BE_VERBS:
            dropped.append(tokens[i])
        else:
            stemmed = _stem_to_gloss(key, cislr)
            if stemmed:
                normalized.append(stemmed.upper())
            else:
                normalized.append(tokens[i])
        i += 1

    return normalized, dropped
