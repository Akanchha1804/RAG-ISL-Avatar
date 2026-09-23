# Test Plan
## RAG-Augmented Speech-to-ISL Avatar Generator

---

## 1. Objectives
Verify that each pipeline module functions correctly in isolation and that the
integrated system meets the functional and non-functional requirements defined
in the SRS, with particular focus on validating the RAG contribution (the
project's core evaluation claim).

## 2. Scope
Covers unit testing of individual modules, integration testing of the full
pipeline, WebSocket protocol tests, and a dedicated evaluation/ablation study
comparing RAG-enabled vs. RAG-disabled gloss generation.

**Automated suite (implemented):** `backend/tests/` — 58 tests, run with:

```bash
cd backend
pytest -q
```

Tests never import the real `rag` module (sentence-transformers is slow); `conftest.py` injects a fake `rag`/`stt` and uses fixtures under `backend/tests/fixtures/data/`. Env set in tests: `DATA_DIR`, `DATABASE_URL` (non-routable), `ISL_WARMUP_MODELS=0`.

| Test file | Covers |
|---|---|
| `test_similarity.py` | `distance_to_similarity`, exact match = 1.0, threshold 0.4 behavior |
| `test_landmarks.py` | 21-landmark validation, malformed frames, dataset load |
| `test_animation_resolver.py` | clip playlist, unresolved tokens, no-raise guarantee |
| `test_retrieval.py` | top-k, similarity scores, empty index fallback |
| `test_gloss_generation.py` | RAG few-shot prompt vs zero-shot, method selection |
| `test_pipeline.py` | staged `run_pipeline` methods, graceful degradation |
| `test_endpoints.py` | REST schemas, 400 on empty text, health shape, animate |
| `test_websocket.py` | pipeline stage order, avatar file-only protocol, traversal guard |
| `test_db_background.py` | background translation log never fails the request |

## 3. Test Levels

### 3.1 Unit Testing

| Module | Test Cases |
|---|---|
| STT | Clear audio → correct transcript; noisy audio → degraded but non-empty transcript; empty audio → graceful error |
| Embedding/Retrieval | Known sentence → retrieves itself (or near-duplicate) as top match; out-of-domain sentence → low similarity scores returned, no crash; `similarity = 1/(1+L2)`, exact = 1.0 |
| Gloss Generator | In-corpus sentence → gloss matches reference reasonably closely; novel sentence → produces plausible gloss ordering (topic-comment); `use_rag` on/off changes prompt |
| Animation Mapping | Known gloss sequence → correct clip playlist; unknown gloss token → flagged in `unresolved_tokens`, not a crash |
| Backend API | Each endpoint returns correct schema for valid input; returns appropriate error codes for invalid input |
| Landmark validation | Exactly 21 landmarks per hand; malformed JSON rejected without crashing pipeline |

### 3.2 Integration Testing

| Test ID | Scenario | Expected Result |
|---|---|---|
| IT-01 | Text input → full pipeline → animation playlist | Playlist returned within latency target, gloss sequence logically ordered |
| IT-02 | Speech input → full pipeline → animation playlist | STT transcript flows correctly into retrieval/generation stages |
| IT-03 | WebSocket status streaming (`/api/pipeline/ws`) | All expected stage messages (`transcribing`, `retrieving`, `generating`, `animating`, `complete`) received in order; `complete` includes `use_rag`/`top_k` |
| IT-04 | RAG retrieval failure (empty index) | System falls back to zero-shot generation without crashing |
| IT-05 | Gloss token with no animation clip | System returns partial playlist with `unresolved_tokens`, frontend displays fallback indicator |
| IT-06 | Concurrent session isolation | Two simultaneous sessions do not cross-contaminate logs or retrieval context |
| IT-07 | Avatar WS (`/api/avatar/ws`) | Server sends `landmark_file`/`landmark_update` metadata + URL only; path traversal rejected (`400`/`error` message); client fetches data via HTTP `/landmarks/...` |
| IT-08 | Background DB log failure | Translation still returns 200; log error surfaced only in health/logs |

### 3.3 System Testing
- End-to-end demo walkthrough covering at least 20 representative sentences across the supported vocabulary
- Cross-browser check (Chrome, Firefox, Edge) for WebGL/avatar rendering compatibility
- Docker container build-and-run verification on a clean machine (reproducibility check)
- Unity: open project in Unity 6, enter Play mode, confirm `LandmarkModels.cs` compiles (no local CLI compile available)

### 3.4 Non-Functional Testing

| NFR | Test Method | Target |
|---|---|---|
| Latency (NFR-1) | Measure end-to-end pipeline time across 20+ runs (`stage_ms` in response) | < 3–5 seconds average on dev hardware |
| STT accuracy (NFR-2) | WER on a held-out audio test set | Comparable to published Distil-Whisper benchmarks |
| Gloss quality (NFR-3) | BLEU/ROUGE against held-out gloss references | Report score; primary comparison is RAG-on vs. RAG-off (see Section 4) |
| Usability (NFR-4) | Informal walkthrough with 2–3 test users unfamiliar with the system | Users complete a translation without guidance |
| Portability (NFR-5) | Fresh `docker compose up` on a separate machine | System starts and serves requests without manual fixes |

---

## 4. Evaluation Study (Core Research Validation)

This is the project's primary evidence for its RAG contribution and should be
treated as a first-class deliverable, not just a QA step.

**Method:**
1. Hold out a test split of ISL-CSLTR and CISLR sentence-gloss pairs (not used in fine-tuning or retrieval index).
2. For each test sentence, generate gloss output twice: once with RAG retrieval enabled, once disabled (`use_rag: false` in `/api/translate`).
3. Score both outputs against the reference gloss using BLEU (and/or exact-match rate for short sequences).
4. Additionally collect qualitative judgments (e.g., 1–5 fluency/correctness rating) from a small number of reviewers, since automatic metrics like BLEU are known to be weak proxies for sign-language grammaticality.
5. Report aggregate scores for both conditions and discuss cases where RAG helped vs. did not (e.g., sentences with no close corpus match).

**Note:** There is no in-app `/api/eval/ablation` endpoint; run the A/B offline against the held-out split (or a script calling `/api/translate` with `use_rag` toggled).

**Deliverable:** A results table/chart (RAG-on vs. RAG-off, BLEU + qualitative scores) included in the evaluation section of the final report — this is the honest, self-measured percentage-improvement claim discussed earlier in project planning, since no external RAG-for-SLP baseline exists to compare against.

---

## 5. Test Environment
- Local development machine (primary): Python 3.11+, Node.js (frontend), Docker Desktop
- GPU (optional): for faster fine-tuning/inference during testing iterations
- Test data: ISL-CSLTR and CISLR held-out split; fixtures in `backend/tests/fixtures/data/`; synthetic edge-case sentences (empty input, very long input, mixed-language input)
- Automated tests require no GPU and no downloaded models (fakes injected)

## 6. Defect Tracking
Maintain a simple issue log (spreadsheet or GitHub Issues) with: ID, module, description, severity (blocker/major/minor), status. Recommended severity triage: pipeline crashes and incorrect gloss ordering = blocker/major; cosmetic animation blend imperfections = minor.

## 7. Exit Criteria
- All unit and integration test cases pass or have documented known limitations (`pytest -q` green)
- Evaluation study (Section 4) completed with results included in the final report
- No blocker/major defects open at submission
- Docker-based reproducible setup verified on a second machine
