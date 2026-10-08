"""Seed the authoritative clip inventory from JSON/CSV sources (M5/Phase 3).

Idempotent: re-running updates changed rows, never duplicates.

    $env:DATABASE_URL = "postgresql+asyncpg://isl_user:isl_password@localhost:5432/isl_avatar"
    .venv\\Scripts\\python.exe seed_clip_db.py

Sources (same files the JSON fallback reads, so both stores agree):
  sentences <- sentence_mapping.json (101 rows)
  signs     <- CISLR dataset.csv (4765 unique) + corpus-only canonical tokens
  phrases   <- none yet (schema-ready for M6)
"""

import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from database import AsyncSessionLocal, engine, Base
from models import SentenceClip, SignClip
from paths import get_paths


def safe_key(key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in key.lower()).strip("_")


async def main() -> int:
    paths = get_paths()

    with open(paths.mapping_file, encoding="utf-8") as f:
        mapping = json.load(f)

    gloss_landmarks = {}
    glm_path = paths.mapping_file.parent / "gloss_landmarks.json"
    if glm_path.exists():
        try:
            data = json.loads(glm_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                gloss_landmarks = data
        except Exception as e:
            print(f"[seed] gloss_landmarks.json unreadable: {e}")

    # Production vocab for canonicalization (same CSV the app reads).
    # Best-clip winners come from the shared helper so PG and JSON agree.
    import gloss_lookup

    vocab = gloss_lookup.load_vocabulary()
    print(f"[seed] vocab: {len(vocab)} glosses, mapping: {len(mapping)} sentences")

    from isl_grammar import canonicalize

    sentence_rows = []
    corpus_tokens: dict[str, str] = {}
    for sentence, entry in mapping.items():
        if not isinstance(entry, dict):
            continue
        landmark_file = entry.get("landmark_file") or ""
        sentence_rows.append({
            "text_norm": sentence.lower().strip(),
            "text_display": sentence,
            "glosses": entry.get("glosses", ""),
            "canonical_gloss": None,  # filled when the BLEU set locks
            "landmark_file": landmark_file,
            "status": entry.get("status", "ready"),
            "video_count": entry.get("video_count"),
            "detection_rate": entry.get("detection_rate"),
        })
        tokens, _ = canonicalize(
            [t for t in str(entry.get("glosses", "")).split(" ") if t]
        )
        for token in tokens:
            if token not in corpus_tokens:
                corpus_tokens[token] = landmark_file

    cislr_rows = []
    for gloss, uid, category, duration in gloss_lookup.iter_best_cislr_rows(paths.cislr_csv):
        glm = gloss_landmarks.get(gloss, {})
        landmark_clip = None
        if (isinstance(glm, dict) and glm.get("status") == "ready"
                and glm.get("file")):
            landmark_clip = glm["file"]
        cislr_rows.append({
            "gloss": gloss,
            "uid": uid,
            "category": category,
            "duration": duration,
            "landmark_clip": landmark_clip,
            "landmark_file": corpus_tokens.get(gloss.upper(), ""),
            "source": "cislr",
        })

    cislr_keys = {r["gloss"] for r in cislr_rows}
    corpus_rows = [
        {
            "gloss": token.lower(),
            "uid": None,
            "category": None,
            "duration": None,
            "landmark_clip": None,
            "landmark_file": landmark_file,
            "source": "corpus",
        }
        for token, landmark_file in sorted(corpus_tokens.items())
        if token.lower() not in cislr_keys
    ]
    print(f"[seed] cislr signs: {len(cislr_rows)}, "
          f"corpus-only signs: {len(corpus_rows)}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        for row in sentence_rows:
            stmt = pg_insert(SentenceClip).values(**row)
            await session.execute(stmt.on_conflict_do_update(
                index_elements=["text_norm"],
                set_={k: v for k, v in row.items() if k != "text_norm"},
            ))
        for row in cislr_rows + corpus_rows:
            stmt = pg_insert(SignClip).values(**row)
            await session.execute(stmt.on_conflict_do_update(
                index_elements=["gloss"],
                set_={k: v for k, v in row.items() if k != "gloss"},
            ))
        # Upserts never remove: drop dataset artifacts and rows for glosses
        # that no longer exist (e.g. after junk filtering was added).
        from sqlalchemy import delete
        from models import SignClip as _SignClip

        valid = {r["gloss"] for r in cislr_rows + corpus_rows}
        await session.execute(delete(_SignClip).where(_SignClip.gloss.like("#%")))
        await session.execute(delete(_SignClip).where(_SignClip.gloss == ""))
        await session.execute(delete(_SignClip).where(_SignClip.gloss.notin_(valid)))
        await session.commit()

    print(f"[seed] upserted {len(sentence_rows)} sentences, "
          f"{len(cislr_rows) + len(corpus_rows)} signs")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
