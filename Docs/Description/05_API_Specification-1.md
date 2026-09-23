# API Specification
## RAG-Augmented Speech-to-ISL Avatar Generator

**Base URL (dev):** `http://localhost:8000`
**Framework:** FastAPI
**Format:** JSON (REST) / JSON messages over WebSocket

All REST paths below are absolute from the base URL (e.g. `POST /api/translate`).

---

## 1. REST Endpoints

### 1.1 `POST /api/transcribe`
Convert speech audio to text.

**Request:**
```json
{
  "audio_base64": "<base64-encoded audio blob>",
  "language_hint": "en"
}
```

**Response (200):**
```json
{
  "transcript": "I am going to school today.",
  "confidence": 0.94,
  "language": "en"
}
```

**Errors:**
- `400` — malformed/empty audio
- `422` — unsupported audio format

---

### 1.2 `POST /api/translate`
Generate an ISL gloss sequence from input text. Retrieve-then-generate: top-k similar examples are retrieved (FAISS + MiniLM, `similarity = 1/(1+L2)`, exact match = `1.0`) and injected as few-shot context when `use_rag` is true.

**Request:**
```json
{
  "text": "I am going to school today.",
  "use_rag": true,
  "top_k": 5
}
```

`top_k` is clamped to `1..20` (default `5`). Empty `text` returns `400`.

**Response (200):**
```json
{
  "gloss_sequence": ["TODAY", "I", "GO", "SCHOOL"],
  "retrieved_examples": [
    {
      "sentence": "I am going to market today.",
      "glosses": "TODAY I GO MARKET",
      "landmark_file": "uid-hi.json",
      "similarity": 0.89,
      "distance": 0.12
    }
  ],
  "animation": {
    "clip_playlist": [{"gloss": "GO", "clip_id": "clip_go_01"}],
    "resolved_tokens": ["TODAY", "I", "GO", "SCHOOL"],
    "unresolved_tokens": []
  },
  "landmark_url": "/landmarks/uid-hi.json",
  "use_rag": true,
  "top_k": 5,
  "pipeline_method": "t5_rag",
  "stage_ms": {"transcribe": 0, "retrieve": 12, "generate": 180, "resolve_animation": 1}
}
```

`pipeline_method` is one of: `direct_lookup` (similarity ≥ `0.4` shortcut), `t5_rag`, `t5`, `rag`, `word_by_word`. Exact-match retrieval returns `similarity: 1.0`. DB logging runs in the background and never fails the request.

**Errors:**
- `400` — empty text input
- `503` — gloss model not loaded/ready (only when no fallback path applies)

---

### 1.3 `POST /api/animate`
Map a gloss sequence to an animation clip playlist (same resolver used inside `/api/translate`). Never raises on unknown tokens; unresolved tokens are reported.

**Request:**
```json
{
  "gloss_sequence": ["TODAY", "I", "GO", "SCHOOL"]
}
```

**Response (200):**
```json
{
  "clip_playlist": [
    {"gloss": "TODAY", "clip_id": "clip_today_01", "duration_ms": 800},
    {"gloss": "I", "clip_id": "clip_i_01", "duration_ms": 500},
    {"gloss": "GO", "clip_id": "clip_go_01", "duration_ms": 600},
    {"gloss": "SCHOOL", "clip_id": "clip_school_01", "duration_ms": 900}
  ],
  "resolved_tokens": ["TODAY", "I", "GO", "SCHOOL"],
  "unresolved_tokens": []
}
```

**Errors:**
- `206` — returned with `unresolved_tokens` populated if one or more gloss tokens have no clip mapping (also may be `200` with a partial list, depending on resolver mode; clients must read `unresolved_tokens`)

---

### 1.4 `GET /api/health`
Liveness + dependency status. Rich diagnostics for paths, DB, and FAISS.

**Response (200):**
```json
{
  "status": "ok",
  "models_loaded": {
    "stt": true,
    "embedding": true,
    "gloss_generator": false
  },
  "database": {"connected": true, "database_error": null},
  "faiss_index": {"exists": true, "size": 477},
  "paths": {
    "source": "env",
    "data_dir": "/app/data",
    "cislr_csv": "/app/data/cislr/dataset.csv",
    "landmarks_dir": "/app/data/landmarks"
  }
}
```

`models_loaded.gloss_generator` is `false` until Flan-T5 weights exist in `backend/models/isl_gloss_t5/` (fine-tune via Colab; directory is gitignored).

---

### 1.5 `GET /landmarks/{filename}`
Static landmark JSON files (mounted only when the landmarks directory exists). Path traversal is rejected (`400`).

---

## 2. WebSocket Endpoints

### 2.1 `WS /api/pipeline/ws`
Full end-to-end pipeline with streamed status stages (`transcribing` → `retrieving` → `generating` → `animating` → `complete`). Used by the frontend live demo.

**Client → Server (initial message):**
```json
{
  "input_mode": "speech",
  "audio_base64": "<base64 audio>"
}
```
or
```json
{
  "input_mode": "text",
  "text": "I am going to school today.",
  "use_rag": true,
  "top_k": 5
}
```

**Server → Client (streamed status messages):**
```json
{"stage": "transcribing", "status": "in_progress"}
{"stage": "transcribing", "status": "done", "transcript": "I am going to school today."}
{"stage": "retrieving", "status": "done", "retrieved_count": 5}
{"stage": "generating", "status": "done", "gloss_sequence": ["TODAY","I","GO","SCHOOL"], "use_rag": true, "top_k": 5}
{"stage": "animating", "status": "done", "clip_playlist": [ ... ]}
{"stage": "complete", "status": "done", "result": { ... same shape as /api/translate ... }}
```

**Error message format:**
```json
{"stage": "<stage_name>", "status": "error", "message": "<description>"}
```

---

### 2.2 `WS /api/avatar/ws`
Avatar-facing protocol. The server sends **metadata only** (landmark file name / update URL); binary/text landmark payloads are fetched over HTTP from `/landmarks/...` to avoid WS frame fragmentation issues.

**Server → Client examples:**
```json
{"type": "landmark_file", "landmark_file": "uid-hi.json", "url": "/landmarks/uid-hi.json", "force": true}
{"type": "landmark_update", "landmark_file": "seq-003.json", "url": "/landmarks/seq-003.json"}
{"type": "error", "message": "path traversal rejected"}
```

`force: true` tells the Unity client to always reload the file. Path traversal in `landmark_file` is rejected before any file access.

**Client → Server:** control messages (e.g. request current landmark) forwarded as JSON; large payloads are not echoed back by the server.

---

## 3. Authentication & Rate Limiting
For the academic-demo scope, no authentication is required (single-user local/dev deployment). If deployed publicly, minimum recommended additions:
- API key or session-token auth on all endpoints
- Rate limiting on `/api/transcribe` and `/api/translate` (compute-expensive)
- CORS restricted to the deployed frontend origin

---

## 4. Removed / non-existent endpoints
The following appeared in earlier draft docs and **do not exist** in this codebase — do not call them:
- `POST /api/eval/ablation` (evaluation is offline via held-out split + `use_rag` A/B, see Test Plan §4)
- MCP tool exposure section (no MCP wrapper ships in this repo)
- `rag_enabled` request field (renamed to `use_rag`)
