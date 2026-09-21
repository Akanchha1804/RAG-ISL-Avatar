# API Specification
## RAG-Augmented Speech-to-ISL Avatar Generator

**Base URL (dev):** `http://localhost:8000/api`
**Framework:** FastAPI
**Format:** JSON (REST) / JSON messages over WebSocket

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
Generate an ISL gloss sequence from input text, using RAG retrieval.

**Request:**
```json
{
  "text": "I am going to school today.",
  "rag_enabled": true,
  "top_k": 5
}
```

**Response (200):**
```json
{
  "gloss_sequence": ["TODAY", "I", "GO", "SCHOOL"],
  "retrieved_examples": [
    {"sentence": "I am going to market today.", "gloss": "TODAY I GO MARKET", "score": 0.89},
    {"sentence": "She is going to school.", "gloss": "SHE GO SCHOOL", "score": 0.81}
  ],
  "rag_enabled": true
}
```

**Errors:**
- `400` — empty text input
- `503` — gloss model not loaded/ready

---

### 1.3 `POST /api/animate`
Map a gloss sequence to an animation clip playlist.

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
  "unresolved_tokens": []
}
```

**Errors:**
- `206` (partial content) — returned with `unresolved_tokens` populated if one or more gloss tokens have no clip mapping

---

### 1.4 `POST /api/eval/ablation`
Run a RAG-enabled vs. RAG-disabled comparison for evaluation reporting (dev/eval use only).

**Request:**
```json
{
  "test_sentence_ids": [12, 45, 78]
}
```

**Response (200):**
```json
{
  "results": [
    {
      "sentence_id": 12,
      "reference_gloss": "TODAY I GO SCHOOL",
      "rag_on_gloss": "TODAY I GO SCHOOL",
      "rag_off_gloss": "I GO TO SCHOOL TODAY",
      "rag_on_bleu": 1.0,
      "rag_off_bleu": 0.62
    }
  ]
}
```

---

## 2. WebSocket Endpoint

### 2.1 `WS /api/pipeline`
Full end-to-end pipeline with streamed status updates, used by the frontend for the live demo experience.

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
  "text": "I am going to school today."
}
```

**Server → Client (streamed status messages):**
```json
{"stage": "transcribing", "status": "in_progress"}
{"stage": "transcribing", "status": "done", "transcript": "I am going to school today."}
{"stage": "retrieving", "status": "done", "retrieved_count": 5}
{"stage": "generating", "status": "done", "gloss_sequence": ["TODAY","I","GO","SCHOOL"]}
{"stage": "animating", "status": "done", "clip_playlist": [ ... ]}
{"stage": "complete", "status": "done"}
```

**Error message format:**
```json
{"stage": "<stage_name>", "status": "error", "message": "<description>"}
```

---

## 3. Health & Status

### 3.1 `GET /api/health`
```json
{
  "status": "ok",
  "models_loaded": {
    "stt": true,
    "embedding": true,
    "gloss_generator": true
  }
}
```

---

## 4. MCP Tool Exposure (Optional)

When the MCP wrapper is enabled, the pipeline is exposed as a single callable tool:

```json
{
  "tool": "translate_to_isl",
  "input": {"text": "I am going to school today."},
  "output": {"gloss_sequence": ["TODAY", "I", "GO", "SCHOOL"], "clip_playlist": [ ... ]}
}
```

This allows external MCP-compatible applications (chat clients, classroom tools) to call the pipeline as a single tool without needing to know the internal REST/WebSocket structure.

---

## 5. Authentication & Rate Limiting
For the academic-demo scope, no authentication is required (single-user local/dev deployment). If deployed publicly, minimum recommended additions:
- API key or session-token auth on all endpoints
- Rate limiting on `/api/transcribe` and `/api/translate` (compute-expensive)
- CORS restricted to the deployed frontend origin
