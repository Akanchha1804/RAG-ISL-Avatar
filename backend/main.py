"""
ISL Avatar Backend - FastAPI Application
Pipeline: Speech/Text -> STT -> RAG -> Gloss -> Landmarks -> Avatar
"""

import json
import csv
import os
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, engine, Base
from crud import log_translation

# =====================================================
# PATHS (support both local dev and deployment)
# =====================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_DIR)))
LANDMARKS_DIR = DATA_DIR / "ISL_MediaPipe" / "output_landmarks"
MAPPING_FILE = DATA_DIR / "ISL_MediaPipe" / "sentence_mapping.json"
ISL_CSV = DATA_DIR / "Dataset" / "data" / "isl_csltr" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "corpus_csv_files" / "ISL Corpus sign glosses.csv"

if not LANDMARKS_DIR.exists():
    LANDMARKS_DIR = BACKEND_DIR / "landmarks"
if not MAPPING_FILE.exists():
    MAPPING_FILE = BACKEND_DIR / "sentence_mapping.json"
if not ISL_CSV.exists():
    ISL_CSV = BACKEND_DIR / "ISL Corpus sign glosses.csv"

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

app.mount("/landmarks", StaticFiles(directory=str(LANDMARKS_DIR)), name="landmarks")

# =====================================================
# DATA MODELS
# =====================================================

class TextInput(BaseModel):
    text: str
    use_rag: bool = True
    top_k: int = 5

class TranslationResponse(BaseModel):
    input_text: str
    transcript: Optional[str] = None
    matched_sentence: Optional[str] = None
    glosses: str
    landmark_file: str
    similarity: Optional[float] = None
    landmark_url: str
    method: str

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
_gloss_csv = None


def load_mapping():
    global _mapping
    if _mapping is None:
        if MAPPING_FILE.exists():
            with open(MAPPING_FILE, "r", encoding="utf-8") as f:
                _mapping = json.load(f)
        else:
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
    return _gloss_csv


# =====================================================
# PIPELINE LOGIC (shared by REST and WebSocket)
# =====================================================

async def run_pipeline(text: str = None, audio_bytes: bytes = None, filename: str = "audio.wav"):
    result = {}

    if audio_bytes:
        from stt import transcribe_bytes
        stt_result = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_bytes, audio_bytes, filename
        )
        result["stt"] = stt_result
        text = stt_result["transcript"]
        result["transcript"] = text
    else:
        result["transcript"] = text

    mapping = load_mapping()
    text_lower = text.lower().strip()

    if text_lower in mapping:
        entry = mapping[text_lower]
        result.update({
            "matched_sentence": text_lower,
            "glosses": entry.get("glosses", ""),
            "landmark_file": entry.get("landmark_file", ""),
            "landmark_url": f"/landmarks/{entry.get('landmark_file', '')}",
            "method": "direct_lookup",
            "similarity": 1.0,
        })
        return result

    from gloss_generator import is_model_available, generate_gloss
    if is_model_available():
        t5_gloss = await asyncio.get_event_loop().run_in_executor(
            None, generate_gloss, text
        )
        if t5_gloss:
            from rag import search
            results = await asyncio.get_event_loop().run_in_executor(None, search, t5_gloss, 1)
            landmark_file = ""
            landmark_url = ""
            if results and results[0]["similarity"] >= 0.3:
                landmark_file = results[0].get("landmark_file", "")
                landmark_url = f"/landmarks/{landmark_file}" if landmark_file else ""
            result.update({
                "glosses": t5_gloss,
                "landmark_file": landmark_file,
                "landmark_url": landmark_url,
                "method": "t5_gloss",
                "similarity": results[0]["similarity"] if results else None,
            })
            return result

    from rag import search
    results = await asyncio.get_event_loop().run_in_executor(None, search, text, 3)

    if results and results[0]["similarity"] >= 0.4:
        best = results[0]
        result.update({
            "matched_sentence": best["sentence"],
            "glosses": best["glosses"],
            "landmark_file": best["landmark_file"],
            "landmark_url": f"/landmarks/{best['landmark_file']}",
            "similarity": best["similarity"],
            "method": "rag",
        })
        return result

    words = text_lower.split()
    gloss_results = [w.upper().strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"")]
    result.update({
        "glosses": " ".join(gloss_results),
        "landmark_file": "",
        "landmark_url": "",
        "method": "word_by_word",
    })
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
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    from gloss_generator import is_model_available
    return {
        "status": "healthy",
        "landmark_files": len(list(LANDMARKS_DIR.glob("*.json"))),
        "sentence_mappings": len(mapping),
        "database": "connected" if db_ok else "disconnected",
        "t5_model": "loaded" if is_model_available() else "not loaded",
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content = await file.read()
    from stt import transcribe_bytes
    result = await asyncio.get_event_loop().run_in_executor(
        None, transcribe_bytes, content, file.filename or "audio.wav"
    )
    return result


@app.post("/api/translate")
async def translate(
    input_data: TextInput,
    background_tasks: BackgroundTasks = None,
):
    result = await run_pipeline(text=input_data.text)

    if background_tasks:
        try:
            from database import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                background_tasks.add_task(
                    log_translation,
                    db,
                    input_text=input_data.text,
                    output_glosses=result.get("glosses"),
                    matched_sentence=result.get("matched_sentence"),
                    similarity=result.get("similarity"),
                    landmark_file=result.get("landmark_file"),
                    method=result.get("method"),
                )
        except Exception:
            pass

    return TranslationResponse(
        input_text=input_data.text,
        transcript=result.get("transcript"),
        matched_sentence=result.get("matched_sentence"),
        glosses=result.get("glosses", ""),
        landmark_file=result.get("landmark_file", ""),
        similarity=result.get("similarity"),
        landmark_url=result.get("landmark_url", ""),
        method=result.get("method", "unknown"),
    )


@app.post("/api/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    result = await run_pipeline(audio_bytes=content, filename=file.filename or "audio.wav")

    if background_tasks and db:
        background_tasks.add_task(
            log_translation,
            db,
            input_text=result.get("transcript", ""),
            output_glosses=result.get("glosses"),
            matched_sentence=result.get("matched_sentence"),
            similarity=result.get("similarity"),
            landmark_file=result.get("landmark_file"),
            method=result.get("method"),
            stt_transcript=result.get("stt", {}).get("transcript"),
            stt_language=result.get("stt", {}).get("language"),
            stt_duration=result.get("stt", {}).get("duration"),
        )

    return result


@app.get("/api/history")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        from database import AsyncSessionLocal
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
    except Exception:
        return {"total": 0, "showing": 0, "history": [], "note": "Database not available"}


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

            await websocket.send_json({"stage": "transcribing", "status": "in_progress"})

            audio_bytes = None
            if input_mode == "speech" and audio_base64:
                import base64
                audio_bytes = base64.b64decode(audio_base64)

            result = await run_pipeline(
                text=text if input_mode == "text" else None,
                audio_bytes=audio_bytes,
            )

            await websocket.send_json({
                "stage": "transcribing",
                "status": "done",
                "transcript": result.get("transcript", text),
            })

            await websocket.send_json({
                "stage": "retrieving",
                "status": "done",
                "matched_sentence": result.get("matched_sentence"),
                "similarity": result.get("similarity"),
            })

            await websocket.send_json({
                "stage": "generating",
                "status": "done",
                "glosses": result.get("glosses", ""),
                "gloss_sequence": result.get("glosses", "").split(),
            })

            await websocket.send_json({
                "stage": "animating",
                "status": "done",
                "landmark_url": result.get("landmark_url", ""),
                "landmark_file": result.get("landmark_file", ""),
            })

            await websocket.send_json({
                "stage": "complete",
                "status": "done",
                "result": {
                    "transcript": result.get("transcript"),
                    "matched_sentence": result.get("matched_sentence"),
                    "glosses": result.get("glosses"),
                    "landmark_url": result.get("landmark_url"),
                    "similarity": result.get("similarity"),
                    "method": result.get("method"),
                },
            })

            await ws_manager.broadcast_to_avatars({
                "type": "landmark_update",
                "landmark_file": result.get("landmark_file", ""),
                "landmark_url": result.get("landmark_url", ""),
                "glosses": result.get("glosses", ""),
            })

    except WebSocketDisconnect:
        ws_manager.disconnect_frontend(websocket)


# =====================================================
# WEBSOCKET: Unity Avatar (landmark streaming)
# =====================================================

@app.websocket("/api/avatar/ws")
async def avatar_websocket(websocket: WebSocket):
    await ws_manager.connect_avatar(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "request_landmark":
                landmark_file = data.get("landmark_file", "")
                landmark_path = LANDMARKS_DIR / landmark_file
                if landmark_path.exists():
                    with open(landmark_path, "r", encoding="utf-8") as f:
                        landmark_data = json.load(f)
                    await websocket.send_json({
                        "type": "landmark_data",
                        "landmark_file": landmark_file,
                        "data": landmark_data,
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Landmark file not found: {landmark_file}",
                    })

    except WebSocketDisconnect:
        ws_manager.disconnect_avatar(websocket)
