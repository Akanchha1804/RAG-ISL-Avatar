"""
ISL Avatar Backend - FastAPI Application

Pipeline (stages are separate functions so REST and WebSocket share them):

    Speech/Text
      -> 1. STT                (stage_transcribe)
      -> 1b. Translation       (stage_translate, Hindi->English; English passthrough)
      -> 2. Sentence retrieval (stage_retrieve)      [only when use_rag=True]
      -> 3. Gloss generation   (stage_generate)      [receives retrieved context]
      -> 4. Animation resolve  (stage_resolve_animation)  (gloss -> clip playlist)
      -> 5. Landmark/clip delivery (REST response, WS broadcast, Unity HTTP fetch)

RAG architecture: retrieval runs BEFORE generation, but the
retrieved examples are deliberately NOT injected into the
T5 prompt (the model was fine-tuned only on the single-line
format; see stage_generate). Retrieval instead feeds the
extractive "rag" method when a near-identical corpus
sentence is found. When use_rag=False no retrieval runs and
the generator works from the original sentence alone.

Similarity convention (similarity.py): similarity = 1 / (1 + L2 distance),
range (0, 1]; exact corpus match reports 1.0 (distance 0).
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import engine, Base, AsyncSessionLocal
from crud import log_translation_background
from paths import get_paths
from schemas import (
    TextInput,
    TranslationResponse,
    AnimateRequest,
    AnimateResponse,
    GlossEntry,
)
from similarity import exact_match_similarity, EXTRACTIVE_MATCH_THRESHOLD

# =====================================================
# PATHS (DATA_DIR mechanism, resolved explicitly in paths.py)
# =====================================================

_paths = get_paths()
BACKEND_DIR = _paths.model_dir.parent.parent
DATA_DIR = _paths.data_dir
LANDMARKS_DIR = _paths.landmarks_dir
MAPPING_FILE = _paths.mapping_file
CISLR_VIDEOS_DIR = _paths.cislr_videos_dir
# The VRM avatar model shipped with the project (single source of truth;
# served read-only so the web viewport can load the actual avatar).
VRM_MODEL_PATH = DATA_DIR / "ISL-3D Avatar" / "Assets" / "model_isl.vrm"

# Minimum retrieval similarity before a retrieved sentence's glosses or
# landmark file are trusted for the extractive "rag" method. Single source
# of truth: similarity.EXTRACTIVE_MATCH_THRESHOLD (M3); this alias keeps
# the pipeline call sites readable.
SIMILARITY_MATCH_THRESHOLD = EXTRACTIVE_MATCH_THRESHOLD

# =====================================================
# LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[Startup] Database tables created/verified.")
    except Exception as e:
        print(f"[Startup] Database not available: {e}")
        print("[Startup] Running without database - translations won't be persisted.")
    if os.environ.get("ISL_WARMUP_MODELS", "1") != "0":
        try:
            async def prefetch():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: __import__("rag", fromlist=["search"]))
                print("[Startup] RAG module prefetched.")
                # M1: warm the STT model too, so the first voice request
                # does not pay the CPU load cost inside the request (which
                # previously tripped the frontend timeout). Failure here
                # must not block startup; the request path loads lazily.
                try:
                    from stt import get_model as _get_stt_model
                    await loop.run_in_executor(None, _get_stt_model)
                    print("[Startup] STT model prefetched.")
                except Exception as e:
                    print(f"[Startup] STT prefetch skipped: {e}")
                # M2: warm the gloss + translation models so the first text
                # request does not pay the CPU load cost inside the request
                # (T5 ~17s, OPUS-MT ~20s cold). Failures never block startup.
                try:
                    from gloss_generator import load_model as _load_t5
                    await loop.run_in_executor(None, _load_t5)
                    print("[Startup] T5 gloss model prefetched.")
                except Exception as e:
                    print(f"[Startup] T5 prefetch skipped: {e}")
                try:
                    from mt import load_model as _load_mt
                    await loop.run_in_executor(None, _load_mt)
                    print("[Startup] MT model prefetched.")
                except Exception as e:
                    print(f"[Startup] MT prefetch skipped: {e}")
            await prefetch()
        except Exception as e:
            print(f"[Startup] RAG prefetch skipped: {e}")
    yield

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(
    title="ISL Avatar Backend",
    description="RAG-Augmented Text-to-ISL Avatar Generator API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # No cookies or HTTP auth are used, so credentials are never sent. The
    # wildcard origin would be rejected by browsers if combined with
    # allow_credentials=True, so it is explicitly disabled here.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if LANDMARKS_DIR.exists():
    app.mount("/landmarks", StaticFiles(directory=str(LANDMARKS_DIR)), name="landmarks")
else:
    print(
        "[Startup][ERROR] Landmarks directory not found - /landmarks disabled. "
        f"Tried: {LANDMARKS_DIR} (source={_paths.landmarks_dir_source}); "
        f"DATA_DIR={DATA_DIR}"
    )

# Per-gloss CISLR sign videos (<uid>.mp4), served read-only so the
# avatar can play the actual sign for any gloss in the vocabulary.
if CISLR_VIDEOS_DIR.exists():
    app.mount("/clips", StaticFiles(directory=str(CISLR_VIDEOS_DIR)), name="clips")
else:
    print(
        "[Startup][ERROR] CISLR videos directory not found - /clips disabled. "
        f"Tried: {CISLR_VIDEOS_DIR} (source={_paths.cislr_videos_dir_source}); "
        f"DATA_DIR={DATA_DIR}"
    )

if not VRM_MODEL_PATH.exists():
    print(
        "[Startup][ERROR] VRM avatar model not found - /avatar/model.vrm will 404. "
        f"Tried: {VRM_MODEL_PATH}"
    )

# =====================================================
# DATA MODELS (re-exported; schemas.py is the single source)
# =====================================================

# =====================================================
# LAZY LOADERS
# =====================================================

_mapping = None
_mapping_warned = False


def load_mapping():
    global _mapping, _mapping_warned
    if _mapping is None:
        if MAPPING_FILE.exists():
            with open(MAPPING_FILE, "r", encoding="utf-8") as f:
                _mapping = json.load(f)
        else:
            if not _mapping_warned:
                print(f"[Config][ERROR] Sentence mapping not found: {MAPPING_FILE}")
                _mapping_warned = True
            _mapping = {}
    return _mapping


def gloss_to_sequence(glosses: str) -> list:
    """Tokenize a gloss string into a gloss sequence (empty tokens dropped)."""
    if not glosses:
        return []
    return [
        token
        for token in (t.strip(".,!?;:'\"") for t in glosses.split())
        if token
    ]


def gloss_tokens_in_input(gloss_sequence: list, text_lower: str) -> bool:
    """True when every gloss token appears in the source text.

    The fine-tuned model is small and sometimes invents a word that
    was never in the input ("i am going to school" -> "I WANT TO
    SCHOOL", "hello" -> "HAPPY"). Accepting that would show a
    hallucinated gloss, so a generation that introduces foreign
    tokens is treated as a failure and the faithful word-by-word
    translation is used instead. Dropping or reordering input words
    (e.g. "nice to meet you" -> "NICE MEET YOU") still passes.
    """
    input_tokens = {
        token.strip(".,!?;:'\"").lower() for token in text_lower.split()
    }
    return all(token.lower() in input_tokens for token in gloss_sequence)


def landmark_url_for(landmark_file: str) -> str:
    return f"/landmarks/{landmark_file}" if landmark_file else ""


def clamp_top_k(top_k) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 5
    return max(1, min(value, 20))


async def _run_sync(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def log_translation_in_background(background_tasks, result, input_text):
    """Schedule the non-fatal translation history write (shared by REST routes).

    Every failure is swallowed inside crud.log_translation_background, so a
    missing database never fails the request.
    """
    if not background_tasks:
        return
    stt = result.get("stt") or {}
    background_tasks.add_task(
        log_translation_background,
        input_text=input_text,
        output_glosses=result.get("glosses"),
        matched_sentence=result.get("matched_sentence"),
        similarity=result.get("similarity"),
        landmark_file=result.get("landmark_file"),
        method=result.get("method"),
        stt_transcript=stt.get("transcript"),
        stt_language=stt.get("language"),
        stt_duration=stt.get("duration"),
    )


# =====================================================
# PIPELINE STAGES (shared by REST and WebSocket)
# =====================================================

async def stage_transcribe(
    text: Optional[str],
    audio_bytes: Optional[bytes],
    filename: str = "audio.wav",
    language: Optional[str] = None,
):
    """Stage 1: speech -> text. Returns (transcript, stt_result|None)."""
    if audio_bytes:
        from stt import transcribe_bytes
        stt_result = await _run_sync(transcribe_bytes, audio_bytes, filename, language)
        return stt_result.get("transcript", ""), stt_result
    return text or "", None


def stage_translate(transcript: str, source_language: Optional[str] = None) -> tuple:
    """M2 translation stage: Hindi -> English before retrieval/generation.

    Returns (english_text, translation|None) where translation is
    {detected_language, source_text, translated_text} when a translation
    decision was made, else None (plain English path).

    Rules:
      - Devanagari present (or explicit "hi" from STT) -> OPUS-MT hi->en.
        Translated English flows into retrieval (M3) and gloss generation,
        so no Hindi-specific branches exist downstream.
      - Explicit "hi" but Latin-script (romanized Hinglish) -> left as-is
        with a warning: the hi->en model mistranslates romanized input,
        which is worse than the English best-effort path.
      - MT unavailable -> original text flows through, flagged with
        "translation unavailable" (degraded, never silent).
      - Non-Latin, non-Devanagari scripts -> ("", unsupported marker);
        the caller short-circuits to method "unsupported_language".
    """
    text = transcript or ""
    if not text.strip():
        return text, None
    from mt import is_hindi_text, is_unsupported_script, translate_hi_to_en

    if is_unsupported_script(text):
        return "", {
            "detected_language": "unsupported",
            "source_text": transcript,
            "translated_text": "",
        }

    lang = (source_language or "").strip().lower()
    if lang.startswith("hi") or (not lang and is_hindi_text(text)):
        if not is_hindi_text(text):
            return text, {
                "detected_language": "hi-latn",
                "source_text": transcript,
                "translated_text": None,
                "warning": "romanized Hindi is not translated; English path used",
            }
        english = translate_hi_to_en(text)
        if english:
            return english, {
                "detected_language": "hi",
                "source_text": transcript,
                "translated_text": english,
            }
        return text, {
            "detected_language": "hi",
            "source_text": transcript,
            "translated_text": None,
            "warning": "translation unavailable; original text used",
        }
    return text, None


def unsupported_language_result(translation: dict) -> dict:
    """Explicit result for scripts no pipeline stage can handle."""
    return {
        "glosses": "",
        "gloss_sequence": [],
        "method": "unsupported_language",
        "matched_sentence": None,
        "similarity": None,
        "landmark_file": "",
        "landmark_url": "",
        "translation": translation,
    }


def stage_retrieve(text: str, use_rag: bool, top_k: int = 5) -> list:
    """Stage 2: sentence retrieval.

    Returns [] when use_rag is False (the generator then operates without
    retrieved context). Otherwise returns up to top_k corpus examples ordered
    by descending similarity.
    """
    if not use_rag or not text or not text.strip():
        return []
    from rag import search
    return search(text, top_k=clamp_top_k(top_k))


def stage_generate(text: str, retrieved: list, use_rag: bool) -> dict:
    """Stage 3: gloss generation.

    Priority:
      1. exact corpus lookup (direct_lookup)
      2. high-confidence retrieval match -> curated glosses (rag)
      3. T5 generation on the bare training-format prompt (t5)
      4. word-by-word fallback (word_by_word)

    Every path's tokens pass through isl_grammar.canonicalize last, so ALL
    methods emit canonical ISL order (M4). Retrieval metadata (matched
    sentence, landmark file) is unaffected: the sentence clip still plays
    for exact/rag hits while the displayed gloss is canonical.

    Retrieved examples are deliberately NOT injected into the T5
    prompt. The model was fine-tuned only on the single-line format
    "translate English to ISL: <sentence>" -> <glosses>; the previous
    few-shot wrapper ("Examples: EN: .. -> ISL: ..") was a format it
    never saw, so it stitched the retrieved glosses together instead
    of translating the query (e.g. "hi" produced
    "WHY YOU -> ISL: HI HOW YOU -> ISL: NICE MEET YOU").
    """
    from isl_grammar import canonicalize

    text_lower = text.lower().strip()
    mapping = load_mapping()

    # 1. Exact corpus hit: deterministic dictionary lookup (similarity
    #    1.0 by the documented convention: exact match <=> distance 0
    #    <=> 1/(1+0) = 1.0). Keys are matched case-insensitively: 3 of the
    #    101 corpus keys are capitalized ("No need to worry dont worry")
    #    and a literal dict lookup would miss them for lowercased input.
    entry = mapping.get(text_lower)
    if entry is None:
        for key, candidate in mapping.items():
            if isinstance(candidate, dict) and key.lower().strip() == text_lower:
                entry = candidate
                break
    if entry is not None and isinstance(entry, dict):
        landmark_file = entry.get("landmark_file", "")
        sequence, _dropped = canonicalize(
            gloss_to_sequence(entry.get("glosses", ""))
        )
        return {
            "glosses": " ".join(sequence),
            "gloss_sequence": sequence,
            "method": "direct_lookup",
            "matched_sentence": text_lower,
            "similarity": exact_match_similarity(),
            "landmark_file": landmark_file,
            "landmark_url": landmark_url_for(landmark_file),
        }

    # 2. High-confidence retrieval match: reuse the curated glosses of
    #    a near-identical corpus sentence. The threshold is high on
    #    purpose - all-MiniLM-L6-v2 similarities for "same sentence"
    #    and "different sentence" overlap in the 0.5-0.78 band, so a
    #    low threshold returns the wrong sentence's glosses (the old
    #    0.4 made "hi" match "hi how are you" at 0.62).
    if use_rag and retrieved:
        top = retrieved[0]
        if top["similarity"] >= SIMILARITY_MATCH_THRESHOLD:
            landmark_file = top.get("landmark_file", "")
            sequence, _dropped = canonicalize(
                gloss_to_sequence(top.get("glosses", ""))
            )
            return {
                "glosses": " ".join(sequence),
                "gloss_sequence": sequence,
                "matched_sentence": top["sentence"],
                "similarity": top["similarity"],
                "landmark_file": landmark_file,
                "landmark_url": landmark_url_for(landmark_file),
                "method": "rag",
            }

    # 3. Generate with T5 on the bare training-format prompt (no
    #    retrieved examples - see the docstring).
    from gloss_generator import is_model_available, generate_gloss

    if is_model_available():
        t5_gloss = generate_gloss(
            text,
            retrieved_examples=None,
            use_rag=False,
        )
        if t5_gloss:
            # Gloss convention is UPPERCASE (word_by_word uppercases too);
            # without this a T5 echo leaks lowercase tokens downstream.
            sequence = [t.upper() for t in gloss_to_sequence(t5_gloss)]
            # Reject a generation that invents words absent from the
            # input and fall through to the faithful word-by-word
            # translation (see gloss_tokens_in_input).
            if sequence and gloss_tokens_in_input(sequence, text_lower):
                # Canonical ISL order (drops + SOV/WH/NEG). Empty (e.g.
                # only be-verbs) falls through to the word-by-word fallback.
                canonical, _dropped = canonicalize(sequence)
                if canonical:
                    return {
                        "glosses": " ".join(canonical),
                        "gloss_sequence": canonical,
                        "method": "t5",
                        "matched_sentence": None,
                        "similarity": None,
                        "landmark_file": "",
                        "landmark_url": "",
                    }

    # 4. Word-by-word fallback (no model, or generation failed).
    words = text_lower.split()
    gloss_results = [w.upper().strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"")]
    # Canonical ISL order so the fallback signs ordered glosses,
    # not raw English sequence.
    canonical, _dropped = canonicalize(gloss_results)
    glosses = " ".join(canonical)
    return {
        "glosses": glosses,
        "gloss_sequence": canonical,
        "landmark_file": "",
        "landmark_url": "",
        "matched_sentence": None,
        "similarity": None,
        "method": "word_by_word",
    }


def sentence_hit_from_generated(generated: dict) -> dict | None:
    """Level-1 retrieval evidence from the generation result.

    Only exact/rag hits with a landmark file qualify: the sentence clip
    covers the whole sequence. Anything else composes sign clips.
    """
    if generated.get("method") in ("direct_lookup", "rag") and generated.get("landmark_file"):
        return {
            "sentence": generated.get("matched_sentence"),
            "landmark_file": generated.get("landmark_file"),
            "similarity": generated.get("similarity"),
        }
    return None


async def stage_resolve_animation(
    gloss_sequence: list,
    sentence_hit: dict | None = None,
) -> dict:
    """Stage 4: canonical gloss sequence -> hierarchical clip playlist."""
    from retrieval import resolve_playlist
    return await resolve_playlist(gloss_sequence, sentence_hit=sentence_hit)


async def run_pipeline(
    text: str = None,
    audio_bytes: bytes = None,
    filename: str = "audio.wav",
    use_rag: bool = True,
    top_k: int = 5,
    language: Optional[str] = None,
):
    """Run the full pipeline. See module docstring for the architecture."""
    top_k = clamp_top_k(top_k)
    result = {}

    # Stage 1: transcription (or text passthrough)
    transcript, stt_result = await stage_transcribe(text, audio_bytes, filename, language)
    result["transcript"] = transcript
    result["input_text"] = transcript
    if stt_result is not None:
        result["stt"] = stt_result

    # Stage 1b (M2): Hindi -> English. Retrieval and generation below always
    # see English; the original transcript is preserved above and in
    # result["translation"].
    stt_language = (stt_result or {}).get("language") or language
    english_text, translation = await _run_sync(stage_translate, transcript, stt_language)
    if translation is not None:
        result["translation"] = translation
    if translation and translation.get("detected_language") == "unsupported":
        result["retrieved_examples"] = []
        result["use_rag"] = bool(use_rag)
        result["top_k"] = top_k
        result.update(unsupported_language_result(translation))
        result["animation"] = await stage_resolve_animation([])
        return result

    # Stage 2: retrieval (skipped entirely when use_rag=False)
    retrieved = await _run_sync(stage_retrieve, english_text, use_rag, top_k)
    result["retrieved_examples"] = retrieved
    result["use_rag"] = bool(use_rag)
    result["top_k"] = top_k

    # Stage 3: gloss generation with retrieved context
    generated = await _run_sync(stage_generate, english_text, retrieved, use_rag)
    result.update(generated)
    if translation is not None and "translation" not in generated:
        result["translation"] = translation

    # Stage 4: animation resolution (hierarchical retrieval)
    animation = await stage_resolve_animation(
        result.get("gloss_sequence", []),
        sentence_hit_from_generated(generated),
    )
    result["animation"] = animation

    return result


# =====================================================
# WEBSOCKET CONNECTION MANAGER
# =====================================================

class ConnectionManager:
    def __init__(self):
        self.frontend_connections: list[WebSocket] = []
        self.avatar_connections: list[WebSocket] = []

    async def connect_frontend(self, websocket: WebSocket):
        await websocket.accept()
        self.frontend_connections.append(websocket)

    async def connect_avatar(self, websocket: WebSocket):
        await websocket.accept()
        self.avatar_connections.append(websocket)

    def disconnect_frontend(self, websocket: WebSocket):
        if websocket in self.frontend_connections:
            self.frontend_connections.remove(websocket)

    def disconnect_avatar(self, websocket: WebSocket):
        if websocket in self.avatar_connections:
            self.avatar_connections.remove(websocket)

    async def broadcast_to_avatars(self, data: dict):
        for connection in self.avatar_connections[:]:
            try:
                await connection.send_json(data)
            except Exception:
                self.avatar_connections.remove(connection)

ws_manager = ConnectionManager()

# =====================================================
# REST ROUTES
# =====================================================

@app.get("/")
def root():
    return {"message": "ISL Avatar Backend", "version": "2.0.0"}


@app.get("/api/health")
async def health():
    mapping = load_mapping()

    db_ok = False
    db_error = None
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        db_error = str(e)

    from gloss_generator import is_model_available
    from gloss_lookup import load_vocabulary

    landmark_files = (
        len(list(LANDMARKS_DIR.glob("*.json"))) if LANDMARKS_DIR.exists() else 0
    )

    try:
        from clip_store import store_name as _clip_store_name

        clip_store = await _clip_store_name()
    except Exception:
        clip_store = "json_fallback"

    index_info = {"exists": _paths.index_dir.exists(), "vectors": None}
    metadata_file = _paths.index_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                index_info["vectors"] = json.load(f).get("count")
        except (OSError, json.JSONDecodeError) as e:
            index_info["error"] = str(e)

    return {
        "status": "healthy",
        "version": app.version,
        "landmark_files": landmark_files,
        "sentence_mappings": len(mapping),
        "database": "connected" if db_ok else "disconnected",
        "database_error": db_error,
        "t5_model": "loaded" if is_model_available() else "not loaded",
        "gloss_vocabulary_size": len(load_vocabulary()),
        "faiss_index": index_info,
        "clip_store": clip_store,
        "paths": _paths.describe(),
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = Query(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    from stt import transcribe_bytes
    try:
        result = await _run_sync(transcribe_bytes, content, file.filename or "audio.wav", language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/api/translate")
async def translate(
    input_data: TextInput,
    background_tasks: BackgroundTasks = None,
):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = await run_pipeline(
        text=input_data.text,
        use_rag=input_data.use_rag,
        top_k=input_data.top_k,
    )

    log_translation_in_background(background_tasks, result, input_data.text)

    return TranslationResponse(
        input_text=input_data.text,
        transcript=result.get("transcript"),
        translation=result.get("translation"),
        matched_sentence=result.get("matched_sentence"),
        glosses=result.get("glosses", ""),
        gloss_sequence=result.get("gloss_sequence", []),
        retrieved_examples=result.get("retrieved_examples", []),
        use_rag=result.get("use_rag", input_data.use_rag),
        top_k=result.get("top_k", input_data.top_k),
        landmark_file=result.get("landmark_file", ""),
        similarity=result.get("similarity"),
        landmark_url=result.get("landmark_url", ""),
        method=result.get("method", "unknown"),
        animation=result.get("animation"),
    )


@app.post("/api/animate", response_model=AnimateResponse)
async def animate(request: AnimateRequest):
    """Resolve a gloss sequence to a hierarchical clip playlist.

    Unknown gloss tokens are reported in unresolved_tokens, tokens with no
    playable clip or landmark in unsupported_tokens; this endpoint never
    crashes on them and never substitutes other signs.
    """
    result = await stage_resolve_animation(request.gloss_sequence)
    return AnimateResponse(**result)


@app.get("/avatar/model.vrm")
def get_avatar_model():
    """Serve the project's VRM avatar model (single file, read-only).

    The web viewport loads this exact file (ISL-3D Avatar/Assets),
    so the avatar on screen is the project's own model.
    """
    if not VRM_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="VRM avatar model not found on server")
    return FileResponse(
        path=str(VRM_MODEL_PATH),
        media_type="application/octet-stream",
        filename="model_isl.vrm",
    )


@app.post("/api/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    use_rag: bool = Query(True),
    top_k: int = Query(5, ge=1, le=20),
    language: Optional[str] = Query(None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    try:
        result = await run_pipeline(
            audio_bytes=content,
            filename=file.filename or "audio.wav",
            use_rag=use_rag,
            top_k=top_k,
            language=language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log_translation_in_background(background_tasks, result, result.get("transcript", ""))

    return result


@app.get("/api/history")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        from crud import get_translation_history, get_translation_count
        async with AsyncSessionLocal() as db:
            history = await get_translation_history(db, limit=limit, offset=offset)
            total = await get_translation_count(db)
        return {
            "total": total,
            "showing": len(history),
            "history": [
                {
                    "id": str(h.id),
                    "input_text": h.input_text,
                    "output_glosses": h.output_glosses,
                    "matched_sentence": h.matched_sentence,
                    "similarity": h.similarity,
                    "method": h.method,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                }
                for h in history
            ],
        }
    except Exception as e:
        return {
            "total": 0,
            "showing": 0,
            "history": [],
            "note": "Database not available",
            "error": str(e),
        }


@app.get("/api/gloss-vocabulary")
def get_gloss_vocabulary(limit: int = Query(100, ge=1, le=10000)):
    from gloss_lookup import load_vocabulary
    vocab = load_vocabulary()
    entries = []
    for i, (gloss, info) in enumerate(sorted(vocab.items())):
        if i >= limit:
            break
        entries.append({
            "gloss": gloss,
            "uid": info["uid"],
            "category": info["category"],
            "duration": info["duration"]
        })
    return {"total": len(vocab), "showing": len(entries), "entries": entries}


@app.get("/api/sentences")
def get_sentences():
    mapping = load_mapping()
    sentences = []
    for sentence, info in mapping.items():
        sentences.append({
            "sentence": sentence,
            "glosses": info.get("glosses", ""),
            "landmark_file": info.get("landmark_file", "")
        })
    return {"total": len(sentences), "sentences": sentences}


@app.get("/api/search")
def search_sentences(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    from rag import search
    results = search(q, top_k=top_k)
    return {"query": q, "results": results}


# =====================================================
# WEBSOCKET: Frontend Pipeline (stage streaming)
# =====================================================

@app.websocket("/api/pipeline/ws")
async def pipeline_websocket(websocket: WebSocket):
    await ws_manager.connect_frontend(websocket)
    try:
        while True:
            raw = await websocket.receive_json()

            input_mode = raw.get("input_mode", "text")
            text = raw.get("text", "")
            audio_base64 = raw.get("audio_base64")
            use_rag = bool(raw.get("use_rag", True))
            top_k = clamp_top_k(raw.get("top_k", 5))
            language = raw.get("language") or None
            current_stage = "transcribing"

            try:
                # Stage 1: transcription
                await websocket.send_json({"stage": "transcribing", "status": "in_progress"})

                audio_bytes = None
                if input_mode == "speech" and audio_base64:
                    import base64
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                    except Exception:
                        raise ValueError("Invalid audio_base64 payload")

                transcript, stt_result = await stage_transcribe(
                    text if input_mode == "text" else None,
                    audio_bytes,
                    "audio.webm" if audio_bytes else "audio.wav",
                    language,
                )
                if not transcript.strip():
                    raise ValueError("Empty input: nothing to translate")

                result = {
                    "transcript": transcript,
                    "input_text": transcript,
                    "use_rag": use_rag,
                    "top_k": top_k,
                }
                if stt_result is not None:
                    result["stt"] = stt_result

                await websocket.send_json({
                    "stage": "transcribing",
                    "status": "done",
                    "transcript": transcript,
                })

                # Stage 1b (M2): Hindi -> English; retrieval/generation see English.
                stt_language = (stt_result or {}).get("language") or language
                english_text, translation = await _run_sync(
                    stage_translate, transcript, stt_language
                )
                if translation is not None:
                    result["translation"] = translation

                if translation and translation.get("detected_language") == "unsupported":
                    generated = unsupported_language_result(translation)
                    result.update(generated)
                    result["retrieved_examples"] = []
                    animation = await stage_resolve_animation([])
                    result["animation"] = animation
                    await websocket.send_json({
                        "stage": "retrieving",
                        "status": "done",
                        "retrieved_count": 0,
                        "retrieved_examples": [],
                        "similarity": None,
                    })
                    await websocket.send_json({
                        "stage": "generating",
                        "status": "done",
                        "glosses": "",
                        "gloss_sequence": [],
                        "method": "unsupported_language",
                    })
                    await websocket.send_json({
                        "stage": "animating",
                        "status": "done",
                        "landmark_url": "",
                        "landmark_file": "",
                        "clip_playlist": [],
                        "resolved_count": 0,
                        "unresolved_tokens": [],
                    })
                else:
                    # Stage 2: retrieval
                    current_stage = "retrieving"
                    retrieved = await _run_sync(stage_retrieve, english_text, use_rag, top_k)
                    result["retrieved_examples"] = retrieved
                    await websocket.send_json({
                        "stage": "retrieving",
                        "status": "done",
                        "retrieved_count": len(retrieved),
                        "retrieved_examples": retrieved,
                        "similarity": retrieved[0]["similarity"] if retrieved else None,
                    })

                    # Stage 3: gloss generation (receives retrieved context)
                    current_stage = "generating"
                    generated = await _run_sync(stage_generate, english_text, retrieved, use_rag)
                    result.update(generated)
                    if translation is not None and "translation" not in generated:
                        result["translation"] = translation
                    await websocket.send_json({
                        "stage": "generating",
                        "status": "done",
                        "glosses": result.get("glosses", ""),
                        "gloss_sequence": result.get("gloss_sequence", []),
                        "method": result.get("method"),
                    })

                    # Stage 4: animation resolution (hierarchical retrieval)
                    current_stage = "animating"
                    animation = await stage_resolve_animation(
                        result.get("gloss_sequence", []),
                        sentence_hit_from_generated(generated),
                    )
                    result["animation"] = animation
                    await websocket.send_json({
                        "stage": "animating",
                        "status": "done",
                        "landmark_url": result.get("landmark_url", ""),
                        "landmark_file": result.get("landmark_file", ""),
                        "clip_playlist": animation["clip_playlist"],
                        "resolved_count": len(animation["resolved_tokens"]),
                        "unresolved_tokens": animation["unresolved_tokens"],
                        "unsupported_tokens": animation.get("unsupported_tokens", []),
                    })

                # Complete
                await websocket.send_json({
                    "stage": "complete",
                    "status": "done",
                    "result": {
                        "input_text": result.get("input_text"),
                        "transcript": result.get("transcript"),
                        "translation": result.get("translation"),
                        "matched_sentence": result.get("matched_sentence"),
                        "glosses": result.get("glosses"),
                        "gloss_sequence": result.get("gloss_sequence"),
                        "retrieved_examples": result.get("retrieved_examples"),
                        "use_rag": result.get("use_rag", use_rag),
                        "top_k": result.get("top_k", top_k),
                        "similarity": result.get("similarity"),
                        "landmark_file": result.get("landmark_file"),
                        "landmark_url": result.get("landmark_url"),
                        "method": result.get("method"),
                        "animation": result.get("animation"),
                    },
                })

                # Notify Unity avatars: file name + URL only (data over HTTP)
                await ws_manager.broadcast_to_avatars({
                    "type": "landmark_update",
                    "landmark_file": result.get("landmark_file", ""),
                    "landmark_url": result.get("landmark_url", ""),
                    "glosses": result.get("glosses", ""),
                })

            except WebSocketDisconnect:
                raise
            except Exception as e:
                await websocket.send_json({
                    "stage": current_stage,
                    "status": "error",
                    "message": str(e),
                })
                # Always finish so connected frontends stop showing a spinner.
                await websocket.send_json({
                    "stage": "complete",
                    "status": "error",
                    "result": {"error": str(e)},
                })

    except WebSocketDisconnect:
        ws_manager.disconnect_frontend(websocket)
    except Exception:
        # A malformed frame must not leak the connection: without this the
        # dead socket stays in frontend_connections forever.
        ws_manager.disconnect_frontend(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# =====================================================
# WEBSOCKET: Unity Avatar (file notifications; data via HTTP)
# =====================================================

@app.websocket("/api/avatar/ws")
async def avatar_websocket(websocket: WebSocket):
    await ws_manager.connect_avatar(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "request_landmark":
                landmark_file = data.get("landmark_file", "")

                # Reject path traversal and empty names explicitly. A plain
                # prefix check is not enough: "../landmarks_backup/x.json"
                # still starts with the landmarks directory as a string, so
                # containment must be checked against the resolved path.
                candidate = (LANDMARKS_DIR / landmark_file).resolve() if landmark_file else None
                inside = candidate is not None and candidate.is_relative_to(
                    LANDMARKS_DIR.resolve()
                )

                if not inside:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid landmark file name: {landmark_file!r}",
                    })
                    continue

                try:
                    from landmarks import load_landmark_dataset, LandmarkValidationError
                    load_landmark_dataset(candidate)
                except LandmarkValidationError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                # WebSocket carries the file name/URL only; the landmark data
                # itself is fetched over HTTP to avoid duplicate transfer.
                await websocket.send_json({
                    "type": "landmark_file",
                    "landmark_file": landmark_file,
                    "landmark_url": f"/landmarks/{landmark_file}",
                })

    except WebSocketDisconnect:
        ws_manager.disconnect_avatar(websocket)
    except Exception:
        ws_manager.disconnect_avatar(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
