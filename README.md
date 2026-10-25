# RAG-ISL-Avatar

**English or Hindi in. ISL glosses and 3D avatar playback out.**

[![CI](https://github.com/Akanchha1804/RAG-ISL-Avatar/actions/workflows/ci.yml/badge.svg)](https://github.com/Akanchha1804/RAG-ISL-Avatar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![The app translating सुप्रभात: Hindi input, translated English, ISL glosses and the avatar mid-sign](Docs/assets/demo.png)

A final-year accessibility project that runs on a laptop: English or Hindi
text (or English speech) goes in, the pipeline translates Hindi to English,
converts the sentence into Indian Sign Language glosses — retrieval first, a
fine-tuned Flan-T5 second — and retargets recorded sign motion onto a VRM
avatar. It reports what it could *not* sign instead of dropping words, and the
whole stack runs on CPU.

The defensible capability is **recorded sentence retrieval and replay**. Novel
sentences fall back to **experimental gloss-by-gloss playback**: isolated
recorded signs in sequence, not a fluent ISL translation. This is a demo, not
a substitute for a human interpreter, and it has not been validated with Deaf
signers.

## 60-second demo

1. **Windows:** double-click `start-dev.bat`. It picks a free port, waits for
   the API, warms the models, and opens <http://localhost:5173>.
2. Click an example under the text box — each one is a curated sentence whose
   glosses all have motion:
   `hi how are you` · `help me` · `do not worry` · `are you free today` ·
   `can you repeat that please` · `congratulations` · `सुप्रभात`
3. Follow the panel: input → `TRANSLATED (HI → EN)` (Hindi only) → ISL glosses
   in ISL order → avatar. The result panel says whether it is replaying a
   `Recorded sentence clip` or the `Experimental gloss playback` path. The
   Hindi chip uses the composed path; a curated sentence plays its
   sentence-level capture instead.
4. Optional: switch to **Voice Input** and say "good morning". The first voice
   request loads Whisper on CPU (~20 s); later ones are quick.

What the demo shows:

- **Hindi → English → ISL** in one request: `सुप्रभात` → *Good morning* →
  `MORNING GOOD`.
- **ISL word order from the gloss model**, not English order: *are you free
  today* → `TODAY YOU FREE`.
- **Non-manual markers:** WH questions raise the brows, negation furrows them,
  timed to the sign they accompany rather than the whole clip.
- **Honest coverage:** when a gloss has no recorded motion the panel says
  `Playing 2 of 3 gloss clips` and names the missing word.
- **Honest sourcing:** composed playback is labelled experimental, and
  one-handed source clips say so instead of hiding the resting hand.
- **Coarticulation:** back-to-back clips crossfade at the wrist when the jump is
  small, with a neutral pass at sentence edges.

## How it works

```text
text / speech
   │  (Hindi) OPUS-MT hi→en
   ▼
glossing ── FAISS retrieval of 101 curated sentences (≥0.75 similarity)
   │      └─ else fine-tuned Flan-T5 → token-validated → word-by-word fallback
   ▼
animation resolver ── sentence landmark file, else per-gloss clip playlist
   ▼
WebSocket / REST → React + three.js VRM avatar (or the Unity scene)
```

1. **Retrieve** with `all-MiniLM-L6-v2` and FAISS (`similarity = 1/(1+L2)`); a
   hit at or above `0.75` reuses that sentence's curated glosses and landmark
   file. Retrieved examples are deliberately *not* injected into the T5 prompt —
   the few-shot wrapper made the model stitch retrieved glosses together.
2. **Generate** with the fine-tuned T5 on the bare training-format prompt,
   accepting a sequence only when every token appears in the input; otherwise
   fall back word by word.
3. **Normalize** every non-curated token onto a canonical CISLR gloss (`hi` →
   `namaste`, `going` → `go`, be-verbs dropped) so it resolves wherever a
   recorded sign exists.
4. **Resolve + retarget** onto the avatar: retargeters cluster detections and
   assign hands by nearest wrist instead of trusting MediaPipe's handedness
   label, hold brief dropouts for 0.5 s, and crossfade between clips.

The long version (and the data-repair story behind the landmarks) is in
[Architecture](#architecture) and [Landmark hygiene](#landmark-hygiene).

## What works, and what does not

| Works | Not yet |
|---|---|
| English + Hindi text; English speech | Languages other than Hindi/English |
| 101 curated sentences; **68/101 fully covered by recorded motion** | Only **184 of 4,765** glosses (3.9%) have motion |
| **Recorded sentence playback** (one source clip) | Composed playback is experimental — isolated clips, not fluent ISL |
| Experimental per-gloss playback, with named misses | Fingerspelling unknown words — needs signer-verified handshapes |
| NMM (brow raise/furrow) timed per gloss | Full ISL grammar/word-order rules — needs signer validation |
| One-handed sources replay faithfully and are labelled | Two-handed output for a one-handed source — that would invent signs |
| Clip crossfades and dropout holds | Reference footage is 10 GB and is not checked in |
| Web avatar + Unity scene | Unity scene base URL is configurable but unset; no PlayMode CI |
| History via Postgres, degrades gracefully without it | No auth or rate limiting in demo scope; no Deaf comprehension study |

The coverage table below has the exact numbers and how to reproduce them.

### Scope and limits

- **Not fluent ISL.** Composed playback concatenates isolated gloss clips. It
  does not add ISL grammar, spatial agreement, or non-manual grammar beyond the
  timed brow raise/furrow.
- **One-handed source clips.** Many recorded signs use one hand; the signer's
  other hand rests at the hip. The avatar copies the source and the panel
  labels it. Making the resting hand sign would invent language.
- **Not validated with Deaf signers.** There has been no comprehension study;
  treat the output as a visualisation of retrieved sign footage, not an
  accessibility substitute.
- **Coverage is partial.** 68 of 101 curated sentences are fully covered; a
  novel sentence may play only some of its gloss clips.

## Quick start

### Windows: one click

```bat
start-dev.bat
```

Uses `backend\.venv`, does not reinstall packages, picks port `8000` or `8010`
if 8000 is taken, waits for `/api/health`, pre-warms the models with one
translation, starts Vite with the matching proxy, and opens the browser. Node
is required for the frontend; the launcher prints the install link if it is
missing.

### Manual

Python **3.12+** (numpy's floor) and Node **22+** (CI uses 24).

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
.venv\Scripts\python -m uvicorn main:app --port 8000

# Frontend (second terminal)
cd frontend
npm ci
npm run dev        # http://localhost:5173
```

If port 8000 is occupied, start uvicorn on 8010 and point Vite at it:
`$env:ISL_BACKEND_URL = "http://127.0.0.1:8010"` before `npm run dev` (the
one-click launcher does this automatically).

```bash
# Tests — no GPU and no real models required
cd backend
.venv\Scripts\python -m pytest -q    # API, pipeline, retrieval, eval metrics,
                                     # path resolution, landmark hygiene,
                                     # web/Unity retargeting conformance
cd ../frontend
npm test                             # vitest + jsdom: playback, coverage
                                     # panel, websocket reconnect, demo chips
```

The metrics tests and eval harnesses also need the WordNet corpus, a one-time
download: `python -m nltk.downloader wordnet omw-1.4` (without it that one
metric skips).

### Docker

```bash
docker compose up -d --build
# frontend http://localhost:5173, API http://localhost:8000
```

The compose stack includes PostgreSQL on host port `5433`, healthchecks, and
read-only mounts for landmarks and mapping data. CI builds both images and
smoke-starts them on every push, so a broken Dockerfile cannot reach `main`
unnoticed.

### Database

The app expects PostgreSQL at `isl_user` / `isl_password`, database
`isl_avatar`, on the host port the backend can reach (compose maps `5433`).

**Without Docker:** a bare local PostgreSQL listens on 5432, so override the
URL rather than editing the repo:

```bash
$env:DATABASE_URL = "postgresql+asyncpg://isl_user:isl_password@localhost:5432/isl_avatar"
```

Create the role/database once:

```sql
CREATE ROLE isl_user LOGIN PASSWORD 'isl_password';
CREATE DATABASE isl_avatar OWNER isl_user;
```

Then apply the schema (same env var, so Alembic and the app target the same
server):

```bash
cd backend
python -m alembic upgrade head
```

A missing database never fails a request — `/api/history` returns a degraded
payload and translation logging is skipped (logged as
`[DB] Translation log skipped (non-fatal)`).

Frontend API base defaults to **same-origin** (proxied by Vite in dev and
nginx in Docker). Set `VITE_API_URL` only when the backend is on another host —
see `frontend/.env.example`. Landmark and avatar URLs follow it too, so a
cross-origin build can still load clips.

## Architecture

**Retrieve-then-generate:** sentences are embedded with `all-MiniLM-L6-v2`, searched in FAISS (distance converted with `similarity = 1/(1+L2)`). A top hit at or above the `0.75` similarity threshold reuses that sentence's curated glosses + landmark file (`rag`); otherwise the fine-tuned T5 generates glosses on the bare training-format prompt (retrieved examples are deliberately *not* injected — the few-shot wrapper made the model stitch retrieved glosses together), accepted only when every token appears in the input, else a word-by-word fallback. Both non-curated paths are normalized onto canonical CISLR gloss tokens (`hi` → `namaste`, `going` → `go`, be-verbs dropped) so a token resolves to a canonical gloss wherever a recorded sign exists. The viewport shows only the project's VRM avatar: it signs the sentence landmark file when one exists, and otherwise signs per-gloss landmark clips (`/landmarks/gloss_<gloss>.json`, built by `ISL_MediaPipe/batch_gloss_landmarks.py`) in sequence. Full pipeline: STT → retrieval → generation → animation resolution → WebSocket/REST response.

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
panel says `Signed N of M glosses` and names the words with no motion.
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
the other arm - across the body, into the space the resting hand occupied - and
a duplicate drove two arms to the same place. That was the overlapping
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

## Deploy notes

`render.yaml` describes a CPU-only Render blueprint for the backend
(Python 3.12, CPU torch, `/api/health` check, models warmed at boot). The
frontend is a static build (`npm run build` in `frontend/`); serve it with a
same-origin proxy as nginx does in `frontend/Dockerfile`, or set
`VITE_API_URL` at build time and allow that origin (CORS is currently `*`).

## License / scope

MIT - see `LICENSE`.

Academic final-year project. No auth or rate limiting in demo scope; CORS is
`*`, so do not expose it publicly without adding both (see API spec §5).
