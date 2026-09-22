"""
ISL Avatar Backend - FastAPI Application
Pipeline: Speech/Text -> STT -> RAG -> Gloss -> Landmarks -> Avatar
"""

import json
import csv
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
# APP SETUP
# =====================================================

app = FastAPI(
    title="ISL Avatar Backend",
    description="RAG-Augmented Text-to-ISL Avatar Generator API",
    version="1.0.0"
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
# ROUTES
# =====================================================

@app.get("/")
def root():
    return {"message": "ISL Avatar Backend", "version": "1.0.0"}


@app.get("/api/health")
def health():
    mapping = load_mapping()
    return {
        "status": "healthy",
        "landmark_files": len(list(LANDMARKS_DIR.glob("*.json"))),
        "sentence_mappings": len(mapping)
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    content = await file.read()
    from stt import transcribe_bytes
    result = transcribe_bytes(content, file.filename or "audio.wav")
    return result


@app.post("/api/translate")
def translate(input_data: TextInput):
    mapping = load_mapping()
    text = input_data.text.strip()
    text_lower = text.lower()

    if input_data.use_rag:
        from rag import search
        results = search(text, top_k=input_data.top_k)

        if results:
            best = results[0]
            if best["similarity"] >= 0.6 and best["landmark_file"]:
                return TranslationResponse(
                    input_text=text,
                    matched_sentence=best["sentence"],
                    glosses=best["glosses"],
                    landmark_file=best["landmark_file"],
                    similarity=best["similarity"],
                    landmark_url=f"/landmarks/{best['landmark_file']}",
                    method="rag_exact"
                )

    if text_lower in mapping:
        entry = mapping[text_lower]
        return TranslationResponse(
            input_text=text,
            matched_sentence=text_lower,
            glosses=entry.get("glosses", ""),
            landmark_file=entry.get("landmark_file", ""),
            similarity=1.0,
            landmark_url=f"/landmarks/{entry.get('landmark_file', '')}",
            method="direct_lookup"
        )

    from rag import search
    results = search(text, top_k=3)
    if results and results[0]["similarity"] >= 0.4:
        best = results[0]
        return TranslationResponse(
            input_text=text,
            matched_sentence=best["sentence"],
            glosses=best["glosses"],
            landmark_file=best["landmark_file"],
            similarity=best["similarity"],
            landmark_url=f"/landmarks/{best['landmark_file']}",
            method="rag_fuzzy"
        )

    words = text_lower.split()
    gloss_results = []
    for word in words:
        cleaned = word.strip(".,!?;:'\"")
        if not cleaned:
            continue
        gloss_results.append(cleaned.upper())

    return TranslationResponse(
        input_text=text,
        glosses=" ".join(gloss_results),
        landmark_file="",
        landmark_url="",
        method="word_by_word"
    )


@app.post("/api/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    content = await file.read()
    from stt import transcribe_bytes
    stt_result = transcribe_bytes(content, file.filename or "audio.wav")
    transcript = stt_result["transcript"]

    mapping = load_mapping()
    transcript_lower = transcript.lower().strip()

    if transcript_lower in mapping:
        entry = mapping[transcript_lower]
        return {
            "transcript": transcript,
            "stt": stt_result,
            "matched_sentence": transcript_lower,
            "glosses": entry.get("glosses", ""),
            "landmark_file": entry.get("landmark_file", ""),
            "landmark_url": f"/landmarks/{entry.get('landmark_file', '')}",
            "method": "direct_lookup"
        }

    from rag import search
    results = search(transcript, top_k=3)

    if results and results[0]["similarity"] >= 0.4:
        best = results[0]
        return {
            "transcript": transcript,
            "stt": stt_result,
            "matched_sentence": best["sentence"],
            "glosses": best["glosses"],
            "landmark_file": best["landmark_file"],
            "landmark_url": f"/landmarks/{best['landmark_file']}",
            "similarity": best["similarity"],
            "method": "rag"
        }

    words = transcript_lower.split()
    gloss_results = [w.upper().strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"")]

    return {
        "transcript": transcript,
        "stt": stt_result,
        "glosses": " ".join(gloss_results),
        "landmark_file": "",
        "landmark_url": "",
        "method": "word_by_word"
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
