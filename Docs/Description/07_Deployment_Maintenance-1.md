# Deployment & Maintenance Guide
## RAG-Augmented Speech-to-ISL Avatar Generator

---

## 1. Deployment Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                  │
│                                                             │
│  ┌───────────────┐   ┌────────────────┐   ┌─────────────┐ │
│  │ frontend       │   │ backend         │   │ postgres     │ │
│  │ (React build,  │   │ (FastAPI +      │   │ (PostgreSQL) │ │
│  │  served via    │   │  ML models +    │   │              │ │
│  │  Nginx)        │──▶│  FAISS index)   │──▶│              │ │
│  └───────────────┘   └────────────────┘   └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────┘
        Volumes: model weights, FAISS index file, animation clip assets
```

## 2. Prerequisites
- Docker & Docker Compose installed
- (For fine-tuning/retraining only, not required for running the demo) Python 3.11+, PyTorch, CUDA-capable GPU recommended
- ISL-CSLTR and CISLR datasets placed in `/data` volume before first FAISS index build
- Rigged avatar + animation clip assets placed in `/assets` volume

## 3. Environment Configuration
Recommended `.env` variables:
```
DATABASE_URL=postgresql://user:password@postgres:5432/isl_db
FAISS_INDEX_PATH=/data/isl_corpus.index
MODEL_STT=distil-whisper
MODEL_GLOSS=t5-small-isl-finetuned
TOP_K_RETRIEVAL=5
RAG_ENABLED_DEFAULT=true
```

## 4. Build & Run

```bash
# Build all services
docker compose build

# Start the stack
docker compose up -d

# View logs
docker compose logs -f backend

# Stop
docker compose down
```

On first run, the backend should:
1. Connect to PostgreSQL and run schema migrations
2. Load the FAISS index (build it from `corpus_entries` if not already present)
3. Load STT and gloss-generation model weights into memory
4. Expose `/api/health` returning `models_loaded: {stt: true, embedding: true, gloss_generator: true}` once ready

## 5. Model & Data Updates

| Task | Procedure |
|---|---|
| Update ISL knowledge base (new sentence-gloss pairs) | Insert into `corpus_entries`, re-embed new rows, rebuild/append to FAISS index — no model retraining required (this is the key advantage of the RAG design) |
| Retrain/fine-tune gloss generator | Run training script offline, replace model weights volume, restart backend |
| Add new animation clips | Add clip asset file, insert row into `gloss_vocabulary` mapping gloss token → clip path |
| Expand vocabulary | Add corpus entries + corresponding animation clips together; verify via `/api/animate` that new tokens resolve |

## 6. Monitoring & Logging
- `translation_logs` table captures every request (input, retrieved examples, generated gloss, latency) for ongoing quality monitoring
- Recommend periodic review of `unresolved_tokens` occurrences to prioritize which missing signs to add next
- Basic latency dashboard (even a simple query against `translation_logs.latency_ms`) helps track NFR-1 compliance over time

## 7. Backup & Recovery
- PostgreSQL: standard `pg_dump` scheduled backup of corpus/logs
- FAISS index: back up the index file alongside the database (they are logically paired via `embedding_id`) — if they fall out of sync, rebuild the index from `corpus_entries` rather than attempting a partial patch

## 8. Known Limitations (for documentation transparency)
- No authentication layer in the current scope — not intended for public multi-user deployment as-is
- No facial expression/non-manual marker rendering — current output covers manual signs only
- Retrieval corpus and animation vocabulary are limited to what's been sourced; sentences well outside this domain will produce lower-quality or partial output
- Real-time performance is "near real-time," not sub-second live-caption speed

## 9. Maintenance Roadmap (maps to SRS Future Enhancements)
1. Hindi input support
2. Non-manual marker generation (facial expressions)
3. Finger-spelling fallback for out-of-vocabulary words
4. Browser extension / video-call integration
5. Native mobile application
6. Expanded vocabulary and animation clip library
