# RAG-ISL-Avatar
An AI-powered RAG-based system that translates text and speech into Indian Sign Language (ISL) using intelligent gloss generation and 3D avatar animation, enabling accessible and inclusive communication.

## Architecture

**Retrieve-then-generate:** sentences are embedded with `all-MiniLM-L6-v2`, searched in FAISS (distance converted with `similarity = 1/(1+L2)`), and top-k examples are injected as few-shot context into the gloss generator (`use_rag`). Animation resolution is a separate stage that maps resolved gloss tokens to clip playlists. Full pipeline: STT → retrieval → generation → animation resolution → WebSocket/REST response.

## Quick start

```bash
# Backend (Python 3.11+)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev

# Tests (no GPU, no real models required)
cd backend
pytest -q
```

Docker: `docker compose up -d --build` (PostgreSQL on host port `5433`, db `isl_avatar`).

## API summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/transcribe` | POST | Audio → text |
| `/api/translate` | POST | Text → gloss sequence (`use_rag`, `top_k`) |
| `/api/animate` | POST | Gloss sequence → clip playlist |
| `/api/health` | GET | Model/DB/path status |
| `/api/pipeline/ws` | WS | Full pipeline with stage streaming |
| `/api/avatar/ws` | WS | Avatar protocol (file URL / landmark update metadata; data via HTTP `/landmarks/...`) |

Similarity exact match = `1.0`; threshold for exact-match shortcut = `0.4`. Full spec: `Docs/Description/05_API_Specification-1.md`.

## Project layout

- `backend/` — FastAPI app, RAG, gloss generator, animation resolver, tests (`backend/tests/`, 58 automated tests)
- `frontend/` — React + Vite UI
- `ISL-3D Avatar/` — Unity 6 + UniVRM scripts (`LandmarkModels.cs`, `HandPoseMapper.cs`, `LandmarkLoader.cs`, `LandmarkFetcher.cs`, `AvatarWebSocketClient.cs`)
- `Docs/Description/` — SRS, system design, API spec, test plan
- `notebooks/` — data augmentation and Flan-T5 fine-tuning (Colab)

## Training note

`backend/models/isl_gloss_t5/` is the expected weight directory but is **not** checked in. Fine-tune via `notebooks/finetune_t5.py` (Colab) and drop weights there. Without weights, `/api/translate` falls back gracefully (RAG few-shot / word-by-word); `GET /api/health` reports `gloss_generator: false`.

## License / scope

Academic final-year project. No auth in demo scope; do not deploy publicly without auth and rate limiting (see API spec §5).
