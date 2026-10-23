# RAG-ISL-Avatar

An AI-powered RAG-based system that translates text and speech into Indian Sign Language (ISL) using intelligent gloss generation and 3D avatar animation, enabling accessible and inclusive communication.

## Architecture

**Retrieve-then-generate:** sentences are embedded with `all-MiniLM-L6-v2`, searched in FAISS (distance converted with `similarity = 1/(1+L2)`). A top hit at or above the `0.75` similarity threshold reuses that sentence's curated glosses + landmark file (`rag`); otherwise the fine-tuned T5 generates glosses on the bare training-format prompt (retrieved examples are deliberately *not* injected — the few-shot wrapper made the model stitch retrieved glosses together), accepted only when every token appears in the input, else a word-by-word fallback. Both non-curated paths are normalized onto canonical CISLR gloss tokens (`hi` → `namaste`, `going` → `go`, be-verbs dropped) so a token resolves to a canonical gloss wherever a recorded sign exists. The viewport shows only the project's VRM avatar: it signs the sentence landmark file when one exists, and otherwise signs per-gloss landmark clips (`/landmarks/gloss_<gloss>.json`, built by `ISL_MediaPipe/batch_gloss_landmarks.py`) in sequence. Full pipeline: STT → retrieval → generation → animation resolution → WebSocket/REST response.

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
pytest -q          # full suite: API, pipeline, retrieval, eval metrics,
                   # path resolution, landmark hygiene and web/Unity
                   # retargeting conformance
```

The metrics tests and the eval harnesses additionally need the WordNet
corpus, a one-time download: `python -m nltk.downloader wordnet omw-1.4`
(without it that one metric skips).

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

- `backend/` - FastAPI app, RAG, gloss generator, animation resolver, tests (`backend/tests/`, run in CI)
- `frontend/` — React + Vite UI
- `ISL_MediaPipe/` — offline landmark extraction: `batch_process.py` (video → landmark JSON + `sentence_mapping.json`), `generate_mapping.py` (mapping repair), `scripts/`
- `ISL-3D Avatar/` — Unity 6 + UniVRM scripts (`LandmarkModels.cs`, `HandPoseMapper.cs`, `LandmarkLoader.cs`, `LandmarkFetcher.cs`, `AvatarWebSocketClient.cs`)
- `Dataset/` — ISL-CSLTR and CISLR corpora (10 GB, gitignored)
- `Docs/Description/` - SRS, system design, API spec, test plan, avatar retargeting spec
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

The shipped weights predate the `CONGRATULATIIONS` → `CONGRATULATIONS`
correction in `augmented_glosses.csv` (the misspelling was flagged by
`backend/eval/flag_references.py` and reached the corpus CSV, both
`sentence_mapping.json` copies, the eval references and the training data), so
the model path can still emit the old spelling. Re-run the fine-tune to bring
the weights in line with the data; the curated/RAG path is already correct.

## Coverage

How much of the language the avatar can actually sign, as opposed to how much
the pipeline can gloss. Numbers come from `backend/eval/results/` and
`ISL_MediaPipe/output_landmarks/`; re-run `backend/eval/run_all.py` to refresh.

| | |
|---|---|
| Curated corpus sentences (what RAG can retrieve) | 101 |
| Sentences signable end to end with per-gloss clips | **68** (32 partial, 1 none) |
| Per-gloss landmark clips | 184 |
| Landmark files total (incl. sentence-level) | 288 |
| Tokens in the 101-sentence corpus with per-token motion | **87.8%** |
| Tokens with CISLR reference footage | 89.3% |
| CISLR gloss vocabulary | 4,765 |

So the corpus is covered well, but the *vocabulary* is not: 184 of 4,765
glosses (3.9%) carry motion. Input that resolves onto a corpus sentence plays
in full; a generated gloss outside those tokens is reported as unsupported
rather than silently skipped (`_playable` in `retrieval.py`), and the result
panel now says `Signed N of M glosses` and names the words with no motion.
The missing-token backlog is `missing_landmark_top` in
`animation_eval.json`. `backend/tests/test_coverage_gate.py` fails if any of
these numbers drops.

The CISLR videos behind `/clips` (10 GB) and the Flan-T5 weights are not
checked in, so a fresh clone runs without reference footage and with
`"t5_model": "not loaded"`.

### Landmark hygiene

The MediaPipe pass trusted the tracker's per-frame handedness label and kept
every detection it emitted, which produced two artefacts that reached the
avatar: a label that flipped mid-clip (79 of the 184 per-gloss clips, 283
flips) and one physical hand reported twice (113 clips, 520 frames). The web
and Unity retargeters bucket hands by that label, so a flip moved the sign to
the other arm - across the body, into the space the resting hand occupied -
and a duplicate drove two arms to the same place. That was the overlapping
hands in composed playback.

`ISL_MediaPipe/clean_landmarks.py` repairs the data in place: merge duplicate
detections, stabilise labels, keep one hand per frame in strongly
single-handed clips, and splice out stretches of 0.5 s or more with no
detection at all (7 clips, 242 frames) so a sign cannot freeze halfway
through. It is idempotent and prints before/after counts:
`python ISL_MediaPipe/clean_landmarks.py --apply`.
`backend/tests/test_landmark_hygiene.py` re-audits all 288 clips in CI, so a
regenerated dataset cannot quietly reintroduce the artefacts or the dead
stretches.

The retargeters themselves no longer trust the label either: they cluster
detections and assign each to the arm whose previous wrist is nearest. A hand
that disappears for a frame or two is a tracker dropout, not the sign ending,
so its arm holds the last pose for `HAND_GAP_HOLD` (0.5 s) before easing to
the arms-down idle - without that, every dropout dropped the arm to the hip
and snapped it back, which read as the arm pumping mid-sign. A side with no
landmarks for longer keeps the pose it started the clip in until the hold
expires, instead of snapping to the model's T-pose (see
`Docs/Description/10_Avatar_Retargeting_Spec.md`).
In a dev build, `__vrmAudit([urls])` in the browser console replays clips
through the real retargeter and reports, per clip, which arm was driven, hand
separation, arm-to-idle drops, held frames and wrist travel - the batch check
that replaces watching each animation.

## License / scope

MIT - see `LICENSE`.

Academic final-year project. No auth in demo scope; do not deploy publicly without auth and rate limiting (see API spec §5).
