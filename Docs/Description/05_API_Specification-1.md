# API Specification
## RAG-Augmented Speech-to-ISL Avatar Generator

**Base URL (dev):** `http://localhost:8000`
**Framework:** FastAPI
**Format:** JSON (REST) / JSON messages over WebSocket

All REST paths below are absolute from the base URL (e.g. `POST /api/translate`).

> **Conformance note.** This document was reconciled against the implementation.
> Where an earlier draft disagreed with the code, the code is authoritative and
> the draft text has been corrected. See §4 for endpoints that do not exist.

---

## 1. REST Endpoints

### 1.0 `GET /`
Service banner.

```json
{"message": "ISL Avatar Backend", "version": "2.0.0"}
```

---

### 1.1 `POST /api/transcribe`
Convert speech audio to text (faster-whisper `distil-small.en`).

**Request:** `multipart/form-data` with a single file part named `file`.
The filename is sanitised to its base name server-side before it is written to
the temp directory.

**Response (200)** (shape produced by `stt.transcribe_bytes`):
```json
{
  "transcript": "I am going to school today.",
  "language": "en",
  "language_probability": 0.98,
  "duration": 3.42
}
```

**Errors:**
- `400` — empty audio upload

> An earlier draft specified a JSON body (`audio_base64` + `language_hint`).
> That shape is **not** implemented; use multipart upload.

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

**Response (200)** — `TranslationResponse`:
```json
{
  "input_text": "please bring water for me",
  "transcript": "please bring water for me",
  "matched_sentence": "bring water for me",
  "glosses": "PLEASE BRING WATER ME",
  "gloss_sequence": ["PLEASE", "BRING", "WATER", "ME"],
  "retrieved_examples": [
    {
      "sentence": "bring water for me",
      "glosses": "BRING WATER ME",
      "landmark_file": "bring_water_for_me_landmarks.json",
      "similarity": 0.8721,
      "distance": 0.1466
    }
  ],
  "use_rag": true,
  "top_k": 5,
  "landmark_file": "bring_water_for_me_landmarks.json",
  "similarity": 0.8721,
  "landmark_url": "/landmarks/bring_water_for_me_landmarks.json",
  "method": "rag",
  "animation": {
    "clip_playlist": [
      {"gloss": "BRING", "sign_id": "JsdNgRcC4wU_1", "clip_id": "JsdNgRcC4wU_1",
       "duration_ms": 6000, "category": "action",
       "source": "cislr_vocabulary",
       "landmark_file": "bring_water_for_me_landmarks.json"}
    ],
    "resolved_tokens": ["BRING", "WATER", "ME"],
    "unresolved_tokens": []
  }
}
```

`method` is one of:

- `direct_lookup` — exact corpus hit (similarity `1.0`); curated glosses + landmark file.
- `rag` — top retrieval similarity is at or above the `0.75` extractive
  threshold; curated glosses + landmark file of the matched sentence.
- `t5` — fine-tuned model generated glosses on the bare training-format
  prompt (no retrieved examples injected). Accepted only when every gloss
  token appears in the source text; the accepted tokens are then
  normalized onto canonical CISLR glosses (see below). No landmark file.
- `word_by_word` — fallback (no model, or the generation invented a word
  absent from the input); the input mapped word-for-word onto canonical
  CISLR glosses. No landmark file.

**Gloss normalization** (applies to `t5` and `word_by_word` only;
curated `direct_lookup` / `rag` glosses are kept verbatim): English
words that are not themselves CISLR glosses are mapped onto their
canonical ISL sign (`hello`/`hi` → `namaste`, `thank`/`thanks` →
`thank you`, `going` → `go`, safe `-ing`/`-s` stemming only when the
base form is a real gloss), and copular be-verbs (`am`, `are`, `is`,
…) are dropped because ISL does not sign them. Multi-word glosses
such as `thank you` stay a single token. E.g. "i am going to school"
returns `I GO TO SCHOOL`, and "hi" returns `NAMASTE` — every token
resolving to a sign clip below.

**Animation:** the avatar viewport shows only the project's VRM avatar
— never video clips. Every entry of the `animation.clip_playlist`
whose `source` is `cislr_vocabulary` carries a `clip_url`
(`GET /clips/<uid>.mp4`, the CISLR reference video, API-level) and,
when a per-gloss landmark clip has been built, a `landmark_clip_url`
(`GET /landmarks/gloss_<gloss>.json`). The avatar signs the sentence
landmark file when one exists, and otherwise signs the per-gloss
landmark clips in sequence. `unresolved_tokens` lists tokens with no
clip (reported, never an error).

`similarity` and `landmark_file` are `null` / `""` for `t5` and
`word_by_word` (no curated match). For `rag`, a generation whose glosses
contain a word that was never in the input is rejected and the faithful
`word_by_word` gloss is returned instead (e.g. "i am going to school"
would otherwise become "I WANT TO SCHOOL"). DB logging runs in the
background and never fails the request.

> Earlier drafts called this field `pipeline_method`, advertised a
> `stage_ms` breakdown, and listed a `t5_rag` method that injected
> retrieved few-shot examples into the T5 prompt. The field is `method`;
> `stage_ms` is **not** returned; `t5_rag` no longer exists (the
> few-shot wrapper made the model stitch retrieved glosses together
> instead of translating the query, e.g. "hi" produced "HI HOW YOU").

**Errors:**
- `400` — empty text input

> The `503` "gloss model not loaded" error from an earlier draft is **never**
> raised: the pipeline always degrades to a fallback method instead.

---

### 1.3 `POST /api/animate`
Map a gloss sequence to an animation clip playlist (same resolver used inside `/api/translate`). Never raises on unknown tokens; unresolved tokens are reported.

**Request:**
```json
{"gloss_sequence": ["BRING", "WATER", "ME", "XYZZY"]}
```

**Response (200):**
```json
{
  "clip_playlist": [
    {"gloss": "BRING", "sign_id": "JsdNgRcC4wU_1", "clip_id": "JsdNgRcC4wU_1",
     "clip_url": "/clips/JsdNgRcC4wU_1.mp4",
     "duration_ms": 6000, "category": "action",
     "source": "cislr_vocabulary",
     "landmark_file": "bring_water_for_me_landmarks.json",
     "landmark_clip_url": "/landmarks/gloss_bring.json"}
  ],
  "resolved_tokens": ["BRING", "WATER", "ME"],
  "unresolved_tokens": ["XYZZY"]
}
```

**Notes:** always `200`; clients must read `unresolved_tokens` rather than rely on a partial-status code. Entries with a `clip_url` are CISLR reference videos (`GET /clips/<uid>.mp4`, API-level only — the viewport never shows video); `landmark_clip_url` is the per-gloss landmark clip the avatar actually performs (`GET /landmarks/gloss_<gloss>.json`, built by `ISL_MediaPipe/batch_gloss_landmarks.py`, `null` until built). Corpus-only entries carry a `landmark_file` reference instead and have both URLs `null`.

---

### 1.4 `POST /api/pipeline`
Full pipeline from an uploaded audio file. REST counterpart of `WS /api/pipeline/ws`, used by the frontend when the socket is unavailable.

**Request:** `multipart/form-data`, field `file`; query params `use_rag` (default `true`) and `top_k` (`1..20`, default `5`).

**Response (200):** the raw pipeline result dict — same content as the `complete` frame of `WS /api/pipeline/ws`.

**Errors:** `400` on an empty upload.

---

### 1.5 `GET /api/search`
Raw FAISS retrieval, no generation.

**Query:** `q` (required), `top_k` (`1..20`, default `5`).

```json
{
  "query": "bring water",
  "results": [
    {"sentence": "bring water for me", "glosses": "BRING WATER ME",
     "landmark_file": "bring_water_for_me_landmarks.json",
     "similarity": 0.8331, "distance": 0.2003}
  ]
}
```

---

### 1.6 `GET /api/sentences`
Every sentence in the mapping, with its glosses and landmark file.

```json
{"total": 101, "sentences": [{"sentence": "...", "glosses": "...", "landmark_file": "..."}]}
```

---

### 1.7 `GET /api/gloss-vocabulary`
CISLR gloss vocabulary. **Query:** `limit` (`1..10000`, default `100`).

```json
{"total": 4765, "showing": 100, "entries": [{"gloss": "...", "uid": "...", "category": "...", "duration": 6.0}]}
```

`duration` is `null` for the two CISLR rows whose `duration` is `#N/A`; such rows are skipped rather than raising.

---

### 1.8 `GET /api/history`
Paginated translation history. **Query:** `limit` (`1..200`, default `50`), `offset` (default `0`).

When PostgreSQL is unreachable the endpoint returns a degraded payload with
`note: "Database not available"` and HTTP `200` — it never 5xxs.

```json
{"total": 0, "showing": 0, "history": [], "note": "Database not available", "error": "..."}
```

---

### 1.9 `GET /api/health`
Liveness + dependency status. Rich diagnostics for paths, DB, and FAISS.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "landmark_files": 104,
  "sentence_mappings": 101,
  "database": "connected",
  "database_error": null,
  "t5_model": "loaded",
  "gloss_vocabulary_size": 4765,
  "faiss_index": {"exists": true, "vectors": 101},
  "paths": {
    "data_dir": "/app",
    "landmarks_dir": {"path": "...", "exists": true, "source": "data_dir"},
    "mapping_file":   {"path": "...", "exists": true, "source": "data_dir"},
    "isl_corpus_csv": {"path": "...", "exists": true, "source": "data_dir"},
    "cislr_csv":      {"path": "...", "exists": true, "source": "data_dir"},
    "index_dir":      {"path": "...", "exists": true, "source": "backend"},
    "model_dir":      {"path": "...", "exists": true, "source": "backend"}
  }
}
```

`database` is the string `"connected"` / `"disconnected"` (not an object).
`t5_model` is `"not loaded"` until Flan-T5 weights exist in
`backend/models/isl_gloss_t5/` (fine-tune via Colab; directory is gitignored).
`source` is `"data_dir"` or `"backend_fallback"` and records which candidate path
was selected.

---

### 1.10 `GET /landmarks/{filename}`
Static landmark JSON files (mounted only when the landmarks directory exists).
Names are validated for containment in the landmarks directory before any file
access; a traversal attempt is rejected with `404` (StaticFiles) — for the
WebSocket variant see §2.2, which returns an explicit `{"type":"error"}` frame.

---

### 1.11 `GET /clips/{uid}.mp4`
Static per-gloss CISLR sign videos (`Dataset/data/cislr/CISLR_v1.5-a_videos`,
mounted only when the directory exists; reported as `cislr_videos_dir` in
`/api/health`). The filename is the clip's `uid` from the gloss vocabulary
(`clip_id` in the playlist) plus `.mp4`; supports HTTP range requests, so
`<video>` elements stream with `206 Partial Content`. The dev server and
nginx proxy `/clips/` to the backend like `/landmarks/`.

---

### 1.12 `GET /avatar/model.vrm`
The project's VRM avatar model (`ISL-3D Avatar/Assets/model_isl.vrm`,
served read-only, `404` when the file is absent). The web viewport loads
this exact file and drives its humanoid hand bones from the landmark
sequence (`HandPoseMapper` palm-frame retargeting, ported to the browser),
so the avatar on screen is the project's own model performing the
translated signs.

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
{"stage": "retrieving", "status": "done", "retrieved_count": 5, "retrieved_examples": [ ... ], "similarity": 0.87}
{"stage": "generating", "status": "done", "glosses": "TODAY I GO SCHOOL", "gloss_sequence": ["TODAY","I","GO","SCHOOL"], "method": "t5"}
{"stage": "animating", "status": "done", "landmark_url": "...", "landmark_file": "...", "clip_playlist": [ ... ], "resolved_count": 4, "unresolved_tokens": []}
{"stage": "complete", "status": "done", "result": { ... same content as /api/translate ... }}
```

**Error message format:**
```json
{"stage": "<stage_name>", "status": "error", "message": "<description>"}
{"stage": "complete", "status": "error", "result": {"error": "<description>"}}
```
A terminal `complete` frame is **always** sent, including on error, so clients
can always stop their spinner.

---

### 2.2 `WS /api/avatar/ws`
Avatar-facing protocol. The server sends **metadata only** (landmark file name / update URL); binary/text landmark payloads are fetched over HTTP from `/landmarks/...` to avoid WS frame fragmentation issues.

**Server → Client examples:**
```json
{"type": "landmark_file", "landmark_file": "hi_how_are_you_landmarks.json", "landmark_url": "/landmarks/hi_how_are_you_landmarks.json"}
{"type": "landmark_update", "landmark_file": "bring_water_for_me_landmarks.json", "landmark_url": "/landmarks/...", "glosses": "BRING WATER ME"}
{"type": "error", "message": "Invalid landmark file name: '../../main.py'"}
```

**Client → Server:**
```json
{"action": "request_landmark", "landmark_file": "hi_how_are_you_landmarks.json"}
```

The file name is resolved and checked with `Path.is_relative_to()` against the
landmarks directory, so `../../main.py`, `../landmarks_backup/evil.json` and the
empty string are all rejected before any file access. The validated dataset is
then parsed and its structure (21 landmarks per hand) is verified before the
server replies.

`landmark_update` is broadcast to every connected avatar whenever a
`/api/pipeline/ws` run completes.

---

## 3. Authentication & Rate Limiting
For the academic-demo scope, no authentication is required (single-user local/dev deployment). If deployed publicly, minimum recommended additions:
- API key or session-token auth on all endpoints
- Rate limiting on `/api/transcribe` and `/api/translate` (compute-expensive)
- CORS restricted to the deployed frontend origin (the demo ships
  `allow_origins=["*"]` with `allow_credentials=False`; do not enable
  credentials without restricting the origin)

---

## 4. Removed / non-existent endpoints
The following appeared in earlier draft docs and **do not exist** in this codebase — do not call them:
- `POST /api/eval/ablation` (evaluation is offline via held-out split + `use_rag` A/B, see Test Plan §4)
- MCP tool exposure section (no MCP wrapper ships in this repo)
- `rag_enabled` request field (renamed to `use_rag`)

## 5. Conformance deltas from the original draft

| Item | Draft | Implemented |
|---|---|---|
| `/api/transcribe` body | JSON `audio_base64` | multipart `file` |
| `/api/transcribe` errors | 400 + 422 | 400 only |
| translate response field | `pipeline_method` | `method` |
| translate `stage_ms` | present | not returned |
| translate `503` | documented | never raised (fallback always applies) |
| `/api/health` `status` | `"ok"` | `"healthy"` |
| `/api/health` `models_loaded` | object | `t5_model` string + `gloss_vocabulary_size` |
| `/api/health` `database` | object | string `"connected"`/`"disconnected"` |
| `/api/health` `faiss_index.size` | `477` | `vectors` = `101` |
| avatar WS URL field | `url` | `landmark_url` |
| `/api/pipeline`, `/api/search`, `/api/sentences`, `/api/gloss-vocabulary`, `/api/history`, `/` | not documented | implemented (§1.0, §1.4–§1.8) |
