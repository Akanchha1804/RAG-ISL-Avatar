"""
Authoritative clip inventory with JSON fallback (M5/Phase 3).

PostgreSQL (sentence_clips / sign_clips tables) is authoritative when
reachable; otherwise the exact files the seed was built from
(sentence_mapping.json, CISLR dataset.csv, gloss_landmarks.json) serve the
same records via animation_resolver, so both stores always agree in shape.

Record shapes intentionally mirror animation_resolver.resolve_gloss_token
so switching stores never changes an API response shape.
"""

from __future__ import annotations

import json
import time
from typing import Optional

SOURCE_CISLR = "cislr_vocabulary"
SOURCE_CORPUS = "sentence_corpus"

_pg_ok: bool | None = None
_pg_checked_at: float = 0.0
PG_RETRY_SECONDS = 60

_corpus_index_cache: list | None = None
_corpus_index_mtime: float | None = None


def reset_store_state():
    """Test hook: forget PG reachability and cached indexes."""
    global _pg_ok, _pg_checked_at, _corpus_index_cache, _corpus_index_mtime
    _pg_ok = None
    _pg_checked_at = 0.0
    _corpus_index_cache = None
    _corpus_index_mtime = None


async def pg_available() -> bool:
    """True when the inventory tables are reachable (cached, retried)."""
    global _pg_ok, _pg_checked_at
    now = time.monotonic()
    if _pg_ok is True:
        return True
    if _pg_ok is False and (now - _pg_checked_at) < PG_RETRY_SECONDS:
        return False
    try:
        from sqlalchemy import text as sql_text

        from database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            await session.execute(sql_text("SELECT 1 FROM sign_clips LIMIT 1"))
        _pg_ok = True
    except Exception:
        _pg_ok = False
    _pg_checked_at = now
    return _pg_ok


async def store_name() -> str:
    return "postgres" if await pg_available() else "json_fallback"


def _token_key(token: str) -> str:
    return token.strip().strip(".,!?;:'\"").lower()


def _pg_sign_to_record(display_token: str, row) -> dict:
    key = _token_key(display_token)
    uid = row.uid
    if uid:
        return {
            "gloss": display_token,
            "sign_id": uid,
            "clip_id": uid,
            "clip_url": f"/clips/{uid}.mp4",
            "duration_ms": int(float(row.duration) * 1000) if row.duration else None,
            "category": row.category,
            "source": SOURCE_CISLR,
            "landmark_file": row.landmark_file or None,
            "landmark_clip_url": f"/landmarks/{row.landmark_clip}" if row.landmark_clip else None,
        }
    return {
        "gloss": display_token,
        "sign_id": None,
        "clip_id": f"corpus::{key}",
        "clip_url": None,
        "duration_ms": None,
        "category": None,
        "source": SOURCE_CORPUS,
        "landmark_file": row.landmark_file or None,
        "landmark_clip_url": None,
    }


async def get_sign(display_token: str) -> dict | None:
    """One gloss token -> animation-shaped record, or None when unknown."""
    key = _token_key(display_token)
    if not key:
        return None
    if await pg_available():
        try:
            from sqlalchemy import select

            from database import AsyncSessionLocal
            from models import SignClip

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(SignClip).where(SignClip.gloss == key).limit(1)
                )
                row = result.scalars().first()
            if row is None:
                return None
            return _pg_sign_to_record(display_token, row)
        except Exception:
            pass
    from animation_resolver import resolve_gloss_token

    return resolve_gloss_token(display_token)


async def get_sentence(text_norm: str) -> dict | None:
    """Exact sentence lookup by normalized text (level-1 hierarchy)."""
    norm = (text_norm or "").lower().strip()
    if not norm:
        return None
    if await pg_available():
        try:
            from sqlalchemy import select

            from database import AsyncSessionLocal
            from models import SentenceClip

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(SentenceClip).where(SentenceClip.text_norm == norm).limit(1)
                )
                row = result.scalars().first()
            if row is None:
                return None
            return {
                "sentence": row.text_display,
                "glosses": row.glosses or "",
                "landmark_file": row.landmark_file or "",
                "status": row.status,
            }
        except Exception:
            pass
    from paths import get_paths

    mapping_file = get_paths().mapping_file
    try:
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for sentence, entry in mapping.items():
        if isinstance(entry, dict) and sentence.lower().strip() == norm:
            return {
                "sentence": sentence,
                "glosses": entry.get("glosses", ""),
                "landmark_file": entry.get("landmark_file", ""),
                "status": entry.get("status", "ready"),
            }
    return None


async def corpus_canonical_index() -> list:
    """All corpus sentences with canonical gloss tokens (phrase scoring).

    Uses locked corrected refs when present (eval/references.csv), else the
    live grammar. Cached per process; reset via reset_store_state().
    """
    global _corpus_index_cache
    if _corpus_index_cache is not None:
        return _corpus_index_cache
    from isl_grammar import canonicalize
    from paths import get_paths

    refs = _load_locked_refs()
    mapping_file = get_paths().mapping_file
    try:
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    index = []
    for sentence, entry in mapping.items():
        if not isinstance(entry, dict):
            continue
        key = sentence.lower().strip()
        if key in refs:
            tokens = refs[key].split()
        else:
            tokens, _ = canonicalize(
                [t for t in str(entry.get("glosses", "")).split(" ") if t]
            )
        index.append({
            "sentence": sentence,
            "canonical_tokens": tokens,
            "landmark_file": entry.get("landmark_file", ""),
        })
    _corpus_index_cache = index
    return index


def _load_locked_refs() -> dict:
    """corrected_gloss refs, only when review holds are empty (else live)."""
    from paths import get_paths

    backend_dir = get_paths().model_dir.parent.parent
    refs_file = backend_dir / "eval" / "references.csv"
    hold_file = backend_dir / "eval" / "review_hold.txt"
    try:
        holds = {l.strip().lower() for l in hold_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    except OSError:
        holds = set()
    if holds or not refs_file.exists():
        return {}
    import csv

    refs = {}
    with open(refs_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("corrected") == "yes":
                refs[(row.get("sentence") or "").lower().strip()] = " ".join(
                    (row.get("corrected_gloss") or "").split()
                )
    return refs
