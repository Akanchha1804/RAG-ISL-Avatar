# RAG-ISL-Avatar

An AI-powered RAG-based system that translates text and speech into Indian Sign Language (ISL) using intelligent gloss generation and 3D avatar animation, enabling accessible and inclusive communication.

## Architecture

**Retrieve-then-generate:** sentences are embedded with `all-MiniLM-L6-v2`, searched in FAISS (distance converted with `similarity = 1/(1+L2)`). A top hit at or above the `0.75` similarity threshold reuses that sentence's curated glosses + landmark file (`rag`); otherwise the fine-tuned T5 generates glosses on the bare training-format prompt (retrieved examples are deliberately *not* injected — the few-shot wrapper made the model stitch retrieved glosses together), accepted only when every token appears in the input, else a word-by-word fallback. Both non-curated paths are normalized onto canonical CISLR gloss tokens (`hi` → `namaste`, `going` → `go`, be-verbs dropped) so every token resolves to a sign clip. The viewport shows only the project's VRM avatar: it signs the sentence landmark file when one exists, and otherwise signs per-gloss landmark clips (`/landmarks/gloss_<gloss>.json`, built by `ISL_MediaPipe/batch_gloss_landmarks.py`) in sequence. Full pipeline: STT → retrieval → generation → animation resolution → WebSocket/REST response.

## Quick start

```bash
# Backend (Python 3.11+)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api, /landmarks and /clips to :8000

# Tests (no GPU, no real models required)
cd backend
pytest -q          # 69 tests
```

One-click launcher (uses the project venv, not the system Python):

```bat
start-dev.bat
```

### Database

The app expects PostgreSQL at `isl_user` / `isl_password` on the
host port the backend can reach, database `isl_avatar`.

**With Docker:** `docker compose up -d --build` (PostgreSQL exposed on
host port `5433`, which is what the app's default `DATABASE_URL` points
at).

**Without Docker (local install):** the default port 5433 is just the
docker-compose host mapping — a bare local PostgreSQL listens on 5432.
Install PostgreSQL, then either:

```bash
# Option A (recommended): override the URL, no repo edits
$env:DATABASE_URL = "postgresql+asyncpg://isl_user:isl_password@localhost:5432/isl_avatar"
uvicorn main:app --reload --port 8000

# Option B: keep the default port by reconfiguring the server to 5433
```

Create the role/database once:

```sql
CREATE ROLE isl_user LOGIN PASSWORD 'isl_password';
CREATE DATABASE isl_avatar OWNER isl_user;
```

Then apply the schema:

```bash
cd backend
$env:DATABASE_URL = "postgresql+asyncpg://isl_user:isl_password@localhost:5432/isl_avatar"
python -m alembic upgrade head
```

`alembic/env.py` reads `DATABASE_URL` (same as the app) and only falls
back to the hardcoded `alembic.ini` URL, so both always target the same
server.

A missing database never fails a request — `/api/history` returns a
degraded payload and translation logging is skipped (logged as
`[DB] Translation log skipped (non-fatal)`).

Frontend API base defaults to **same-origin** (proxied by Vite in dev and
nginx in Docker). Set `VITE_API_URL` only when the backend is on another
host — see `frontend/.env.example`.

## API summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service name/version |
| `/api/transcribe` | POST | Audio (multipart `file`) → text |
| `/api/translate` | POST | Text → gloss sequence (`use_rag`, `top_k`) |
| `/api/animate` | POST | Gloss sequence → clip playlist |
| `/api/pipeline` | POST | Full pipeline from an audio file (REST fallback for the WS path) |
| `/api/search` | GET | Raw FAISS retrieval (`?q=&top_k=`) |
| `/api/sentences` | GET | All mapped corpus sentences |
| `/api/gloss-vocabulary` | GET | CISLR gloss vocabulary (uid/category/duration) |
| `/api/history` | GET | Paginated translation history (degrades if the DB is down) |
| `/api/health` | GET | Model/DB/path status |
| `/api/pipeline/ws` | WS | Full pipeline with stage streaming |
| `/api/avatar/ws` | WS | Avatar protocol (file URL / landmark update metadata; data via HTTP `/landmarks/...`) |
| `/landmarks/{file}` | GET | Sentence landmark JSON (corpus sentences) |
| `/clips/{uid}.mp4` | GET | Per-gloss CISLR sign video (local dataset) |
| `/avatar/model.vrm` | GET | Project VRM avatar model (driven by landmark sequences in the viewport) |

Similarity exact match = `1.0`; threshold for the extractive shortcut = `0.75`.
Full spec: `Docs/Description/05_API_Specification-1.md`.

## Project layout

- `backend/` — FastAPI app, RAG, gloss generator, animation resolver, tests (`backend/tests/`, 69 automated tests)
- `frontend/` — React + Vite UI
- `ISL_MediaPipe/` — offline landmark extraction: `batch_process.py` (video → landmark JSON + `sentence_mapping.json`), `generate_mapping.py` (mapping repair), `scripts/`
- `ISL-3D Avatar/` — Unity 6 + UniVRM scripts (`LandmarkModels.cs`, `HandPoseMapper.cs`, `LandmarkLoader.cs`, `LandmarkFetcher.cs`, `AvatarWebSocketClient.cs`)
- `Dataset/` — ISL-CSLTR and CISLR corpora (10 GB, gitignored)
- `Docs/Description/` — SRS, system design, API spec, test plan
- `notebooks/` — data augmentation and Flan-T5 fine-tuning scripts (run on Colab)

## Regenerating landmarks

```bash
# Optional: ISL_CSLTR_DIR=<path to ISL_CSLRT_Corpus> to override auto-detection
cd ISL_MediaPipe
python batch_process.py      # writes output_landmarks/*.json + sentence_mapping.json
python generate_mapping.py   # optional repair; validates content, not just existence
```

`batch_process.py` keeps MediaPipe timestamps increasing **across** videos and
marks a sentence `ready` only when the written file actually contains hand
landmarks. The current corpus yields 101/101 sentences with usable landmarks
(min detection rate 0.72, mean 0.96).

## Training note

`backend/models/isl_gloss_t5/` holds the fine-tuned weights but is **not**
checked in. Fine-tune via `notebooks/finetune_t5.py` (Colab) and drop the
weights there. Without them, `/api/translate` still works by falling back
(RAG extractive / word-by-word) and `GET /api/health` reports
`"t5_model": "not loaded"`.

## License / scope

Academic final-year project. No auth in demo scope; do not deploy publicly without auth and rate limiting (see API spec §5).
