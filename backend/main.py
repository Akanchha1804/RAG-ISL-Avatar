"""
ISL Avatar Backend - FastAPI Application

Pipeline (stages are separate functions so REST and WebSocket share them):

    Speech/Text
      -> 1. STT                (stage_transcribe)
      -> 2. Sentence retrieval (stage_retrieve)      [only when use_rag=True]
      -> 3. Gloss generation   (stage_generate)      [receives retrieved context]
      -> 4. Animation resolve  (stage_resolve_animation)  (gloss -> clip playlist)
      -> 5. Landmark/clip delivery (REST response, WS broadcast, Unity HTTP fetch)

RAG architecture: retrieval happens BEFORE generation and the retrieved
examples are passed into the gloss generator as few-shot context
(see gloss_generator.build_prompt). When use_rag=False no retrieval runs and
the generator works from the original sentence alone.

Similarity convention (similarity.py): similarity = 1 / (1 + L2 distance),
range (0, 1]; exact corpus match reports 1.0 (distance 0).
"""

import json
import csv
import os
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, engine, Base, AsyncSessionLocal
from crud import log_translation_background
from paths import get_paths
from schemas import TextInput, TranslationResponse, AnimateRequest, AnimateResponse
from similarity import exact_match_similarity

# =====================================================
# PATHS (DATA_DIR mechanism, resolved explicitly in paths.py)
# =====================================================

_paths = get_paths()
BACKEND_DIR = _paths.model_dir.parent.parent
DATA_DIR = _paths.data_dir
LANDMARKS_DIR = _paths.landmarks_dir
MAPPING_FILE = _paths.mapping_file
ISL_CSV = _paths.isl_corpus_csv

# Minimum retrieval similarity before a retrieved sentence's glosses or
# landmark file are trusted for the extractive "rag" method.
SIMILARITY_MATCH_THRESHOLD = 0.4

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
            loop = asyncio.get_event_loop()
            async def prefetch():
                await loop.run_in_executor(None, lambda: __import__("rag", fromlist=["search"]))
                print("[Startup] RAG module prefetched.")
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
    allow_origins=["*"],
    allow_credentials=True,
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

# =====================================================
# DATA MODELS (re-exported; schemas.py is the single source)
# =====================================================

class GlossEntry(BaseModel):
    word: str
    found: bool
    gloss: Optional[str]
    uid: Optional[str]
    category: Optional[str]

# =====================================================
# LAZY LOADERS
# =====================================================

_mapping = None
_mapping_warned = False
_gloss_csv = None


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


def load_gloss_csv():
    global _gloss_csv
    if _gloss_csv is None:
        _gloss_csv = {}
        if ISL_CSV.exists():
            with open(ISL_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sentence = row["Sentence"].strip().lower()
                    glosses = row["SIGN GLOSSES"].strip()
                    _gloss_csv[sentence] = glosses
        else:
            print(f"[Config][WARN] ISL corpus CSV not found: {ISL_CSV}")
    return _gloss_csv


def gloss_to_sequence(glosses: str) -> list:
    """Tokenize a gloss string into a gloss sequence (empty tokens dropped)."""
    if not glosses:
        return []
    return [
        token
        for token in (t.strip(".,!?;:'\"") for t in glosses.split())
        if token
    ]


def landmark_url_for(landmark_file: str) -> str:
    return f"/landmarks/{landmark_file}" if landmark_file else ""


def clamp_top_k(top_k) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 5
    return max(1, min(value, 20))


async def _run_sync(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)


# =====================================================
# PIPELINE STAGES (shared by REST and WebSocket)
# =====================================================

async def stage_transcribe(
    text: Optional[str],
    audio_bytes: Optional[bytes],
    filename: str = "audio.wav",
):
    """Stage 1: speech -> text. Returns (transcript, stt_result|None)."""
    if audio_bytes:
        from stt import transcribe_bytes
        stt_result = await _run_sync(transcribe_bytes, audio_bytes, filename)
        return stt_result.get("transcript", ""), stt_result
    return text or "", None


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

    Priority: exact corpus lookup -> RAG-conditioned T5 generation ->
    zero-shot T5 generation (only when use_rag=False) -> extractive reuse of
    the top retrieved example -> word-by-word fallback.

    The retrieved examples are passed into the generator so they actually
    influence generation (gloss_generator.build_prompt).
    """
    text_lower = text.lower().strip()
    mapping = load_mapping()

    # Exact corpus hit: deterministic dictionary lookup (similarity 1.0 by the
    # documented convention: exact match <=> distance 0 <=> 1/(1+0) = 1.0).
    if text_lower in mapping:
        entry = mapping[text_lower]
        glosses = entry.get("glosses", "")
        landmark_file = entry.get("landmark_file", "")
        return {
            "glosses": glosses,
            "gloss_sequence": gloss_to_sequence(glosses),
            "method": "direct_lookup",
            "matched_sentence": text_lower,
            "similarity": exact_match_similarity(),
            "landmark_file": landmark_file,
            "landmark_url": landmark_url_for(landmark_file),
        }

    from gloss_generator import is_model_available, generate_gloss

    if is_model_available():
        t5_gloss = generate_gloss(
            text,
            retrieved_examples=retrieved,
            use_rag=use_rag,
        )
        if t5_gloss:
            top = retrieved[0] if retrieved else None
            similarity = top["similarity"] if top else None
            landmark_file = ""
            if top and similarity is not None and similarity >= SIMILARITY_MATCH_THRESHOLD:
                landmark_file = top.get("landmark_file", "")
            return {
                "glosses": t5_gloss,
                "gloss_sequence": gloss_to_sequence(t5_gloss),
                "method": "t5_rag" if (use_rag and retrieved) else "t5",
                "matched_sentence": top["sentence"] if top else None,
                "similarity": similarity,
                "landmark_file": landmark_file,
                "landmark_url": landmark_url_for(landmark_file),
            }

    # Extractive fallback: only meaningful when retrieval ran.
    if use_rag and retrieved and retrieved[0]["similarity"] >= SIMILARITY_MATCH_THRESHOLD:
        best = retrieved[0]
        glosses = best.get("glosses", "")
        landmark_file = best.get("landmark_file", "")
        return {
            "glosses": glosses,
            "gloss_sequence": gloss_to_sequence(glosses),
            "matched_sentence": best["sentence"],
            "similarity": best["similarity"],
            "landmark_file": landmark_file,
            "landmark_url": landmark_url_for(landmark_file),
            "method": "rag",
        }

    words = text_lower.split()
    gloss_results = [w.upper().strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"")]
    glosses = " ".join(gloss_results)
    return {
        "glosses": glosses,
        "gloss_sequence": gloss_to_sequence(glosses),
        "landmark_file": "",
        "landmark_url": "",
        "matched_sentence": None,
        "similarity": None,
        "method": "word_by_word",
    }


def stage_resolve_animation(gloss_sequence: list) -> dict:
    """Stage 4: gloss sequence -> clip playlist (canonical sign resources)."""
    from animation_resolver import resolve_gloss_sequence
    return resolve_gloss_sequence(gloss_sequence)


async def run_pipeline(
    text: str = None,
    audio_bytes: bytes = None,
    filename: str = "audio.wav",
    use_rag: bool = True,
    top_k: int = 5,
):
    """Run the full pipeline. See module docstring for the architecture."""
    top_k = clamp_top_k(top_k)
    result = {}

    # Stage 1: transcription (or text passthrough)
    transcript, stt_result = await stage_transcribe(text, audio_bytes, filename)
    result["transcript"] = transcript
    result["input_text"] = transcript
    if stt_result is not None:
        result["stt"] = stt_result

    # Stage 2: retrieval (skipped entirely when use_rag=False)
    retrieved = await _run_sync(stage_retrieve, transcript, use_rag, top_k)
    result["retrieved_examples"] = retrieved
    result["use_rag"] = bool(use_rag)
    result["top_k"] = top_k

    # Stage 3: gloss generation with retrieved context
    generated = await _run_sync(stage_generate, transcript, retrieved, use_rag)
    result.update(generated)

    # Stage 4: animation resolution
    animation = await _run_sync(
        stage_resolve_animation, result.get("gloss_sequence", [])
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
        "paths": _paths.describe(),
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    from stt import transcribe_bytes
    result = await _run_sync(transcribe_bytes, content, file.filename or "audio.wav")
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

    if background_tasks:
        background_tasks.add_task(
            log_translation_background,
            input_text=input_data.text,
            output_glosses=result.get("glosses"),
            matched_sentence=result.get("matched_sentence"),
            similarity=result.get("similarity"),
            landmark_file=result.get("landmark_file"),
            method=result.get("method"),
            stt_transcript=result.get("stt", {}).get("transcript") if result.get("stt") else None,
            stt_language=result.get("stt", {}).get("language") if result.get("stt") else None,
            stt_duration=result.get("stt", {}).get("duration") if result.get("stt") else None,
        )

    return TranslationResponse(
        input_text=input_data.text,
        transcript=result.get("transcript"),
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
    """Resolve a gloss sequence to a clip playlist.

    Unknown gloss tokens are reported in unresolved_tokens; this endpoint
    never crashes on them.
    """
    result = stage_resolve_animation(request.gloss_sequence)
    return AnimateResponse(**result)


@app.post("/api/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    use_rag: bool = Query(True),
    top_k: int = Query(5, ge=1, le=20),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    result = await run_pipeline(
        audio_bytes=content,
        filename=file.filename or "audio.wav",
        use_rag=use_rag,
        top_k=top_k,
    )

    if background_tasks:
        background_tasks.add_task(
            log_translation_background,
            input_text=result.get("transcript", ""),
            output_glosses=result.get("glosses"),
            matched_sentence=result.get("matched_sentence"),
            similarity=result.get("similarity"),
            landmark_file=result.get("landmark_file"),
            method=result.get("method"),
            stt_transcript=result.get("stt", {}).get("transcript") if result.get("stt") else None,
            stt_language=result.get("stt", {}).get("language") if result.get("stt") else None,
            stt_duration=result.get("stt", {}).get("duration") if result.get("stt") else None,
        )

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
            current_stage = "transcribing"

            try:
                # Stage 1: transcription
                await websocket.send_json({"stage": "transcribing", "status": "in_progress"})

                audio_bytes = None
                if input_mode == "speech" and audio_base64:
                    import base64
                    audio_bytes = base64.b64decode(audio_base64)

                transcript, stt_result = await stage_transcribe(
                    text if input_mode == "text" else None,
                    audio_bytes,
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

                # Stage 2: retrieval
                current_stage = "retrieving"
                retrieved = await _run_sync(stage_retrieve, transcript, use_rag, top_k)
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
                generated = await _run_sync(stage_generate, transcript, retrieved, use_rag)
                result.update(generated)
                await websocket.send_json({
                    "stage": "generating",
                    "status": "done",
                    "glosses": result.get("glosses", ""),
                    "gloss_sequence": result.get("gloss_sequence", []),
                    "method": result.get("method"),
                })

                # Stage 4: animation resolution
                current_stage = "animating"
                animation = await _run_sync(
                    stage_resolve_animation, result.get("gloss_sequence", [])
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
                })

                # Complete
                await websocket.send_json({
                    "stage": "complete",
                    "status": "done",
                    "result": {
                        "input_text": result.get("input_text"),
                        "transcript": result.get("transcript"),
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

                # Reject path traversal and empty names explicitly.
                candidate = (LANDMARKS_DIR / landmark_file).resolve() if landmark_file else None
                inside = (
                    candidate is not None
                    and str(candidate).startswith(str(LANDMARKS_DIR.resolve()))
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
