# Project Development Flow
## RAG-Augmented Speech-to-ISL Avatar Generator

This document walks through the end-to-end development flow, from raw data to
final demo, tying together the SRS, System Design, Database Design, API
Specification, Test Plan, and Work Allocation docs into a single sequential
view.

---

## Flow Overview

```text
1. Data Preparation
        │
        ▼
2. Avatar Creation (Rig)
        │
        ▼
3. Sign Movement Acquisition (Animation Content)
        │
        ▼
4. RAG Retrieval Pipeline
        │
        ▼
5. Gloss Generation Model
        │
        ▼
6. Animation Mapping Layer
        │
        ▼
7. Backend Integration
        │
        ▼
8. Frontend Integration
        │
        ▼
9. End-to-End Pipeline Testing
        │
        ▼
10. Evaluation (RAG-on vs. RAG-off)
        │
        ▼
11. Deployment (Docker)
        │
        ▼
12. Documentation & Demo
```

---

## Stage 1 — Data Preparation
- Source ISL-CSLTR and CISLR sentence-gloss corpora
- Clean and deduplicate entries; tag each with a `source` field (ISL-CSLTR / CISLR / curated)
- Assemble the additional curated retrieval knowledge base referenced in the project abstract
- Split into train / validation / test sets
- **Output:** `corpus_entries` table populated (see Database Design doc)

## Stage 2 — Avatar Creation (Rig)
- Build the avatar in VRoid Studio (primary) or via Reallusion AccuRIG + ActorCore (backup)
- Confirm the exported rig includes individual finger/thumb bones (not a single fused hand bone) and facial blend shapes
- Export as VRM; convert to FBX/GLB for Unity import
- **Output:** Rigged avatar file, movement-capable, imported into Unity/Three.js project

## Stage 3 — Sign Movement Acquisition (Animation Content)
- Primary method: run pose estimation (MediaPipe Hands/Pose) on ISL-CSLTR/CISLR sign videos to extract per-frame hand, finger, arm, and head keypoints
- Retarget extracted keypoints onto the avatar's skeleton (map estimated joint positions to bone rotations)
- Secondary method: manually keyframe-animate any signs where extraction quality is poor (occlusion, fast motion) or entirely missing from source video
- Populate `gloss_vocabulary` table (gloss token → clip file mapping)
- **Output:** Animation clip library covering the target vocabulary (50–100+ signs)

## Stage 4 — RAG Retrieval Pipeline
- Generate sentence embeddings for all `corpus_entries` using sentence-transformers (MiniLM)
- Build and persist a FAISS index (`IndexFlatL2`)
- Implement query-time retrieval: embed incoming sentence → return top-k similar sentence-gloss pairs
- **Output:** Working retrieval module, testable independently via `/api/translate` with `rag_enabled: true`

## Stage 5 — Gloss Generation Model
- Fine-tune Flan-T5/mT5 on the prepared corpus
- Implement the RAG-prompted generation flow: user sentence + retrieved examples → gloss sequence
- Implement the ablation flag (`rag_enabled: false`) for later evaluation
- **Output:** Gloss generator producing ISL-ordered token sequences (e.g., topic-comment structure)

## Stage 6 — Animation Mapping Layer
- Implement gloss-sequence → clip-playlist lookup against `gloss_vocabulary`
- Implement blend-tree/transition logic between consecutive clips (co-articulation handling)
- Handle unresolved tokens gracefully (flagged, not fatal)
- **Output:** `/api/animate` functional, returns a clip playlist for a given gloss sequence

## Stage 7 — Backend Integration
- Wire together STT (Distil-Whisper) → RAG retrieval → gloss generation → animation mapping behind the FastAPI backend
- Implement REST endpoints (`/api/transcribe`, `/api/translate`, `/api/animate`, `/api/health`) and the WebSocket `/api/pipeline` endpoint
- Log each request to `translation_logs` for later evaluation and debugging
- **Output:** Fully functional backend, testable via API calls without the frontend

## Stage 8 — Frontend Integration
- Build the React + Tailwind interface: text/mic input, live pipeline status indicator, avatar viewport
- Connect to the backend via WebSocket for streamed status updates
- Render the avatar animation playlist in the viewport (Unity WebGL build or Three.js scene)
- **Output:** End-to-end usable demo — speak or type, watch the avatar sign

## Stage 9 — End-to-End Pipeline Testing
- Run integration test cases (per Test Plan: IT-01 through IT-06)
- Validate latency against NFR-1 target (< 3–5 seconds)
- Cross-browser check for avatar rendering compatibility
- **Output:** Stable, demo-ready pipeline with logged/triaged defects

## Stage 10 — Evaluation (RAG-on vs. RAG-off)
- Run the held-out test split through `/api/translate` twice per sentence (RAG enabled and disabled)
- Score against reference gloss (BLEU / exact-match) and collect qualitative reviewer ratings
- Compile results table — this is the project's primary research evidence
- **Output:** Evaluation results ready for the final report

## Stage 11 — Deployment (Docker)
- Containerize frontend, backend, and PostgreSQL via `docker compose`
- Verify reproducible setup on a clean machine
- Confirm model weights, FAISS index, and animation assets load correctly via mounted volumes
- **Output:** Reproducible, submittable deployment package

## Stage 12 — Documentation & Demo
- Finalize all SDLC documents with results from Stages 9–10
- Prepare live demo walkthrough and viva Q&A
- Final proofread and submission
- **Output:** Complete project submission package

---

## Cross-Reference to Work Allocation (10-Week Plan)

| Development Stage | Primarily Maps to Weeks | Primary Owner(s) |
|---|---|---|
| 1. Data Preparation | 1–2 | Harshnandni |
| 2. Avatar Creation | 2 | Divyanshi |
| 3. Sign Movement Acquisition | 3–6 | Divyanshi |
| 4. RAG Retrieval Pipeline | 3–4 | Akanchha |
| 5. Gloss Generation Model | 5 | Akanchha |
| 6. Animation Mapping Layer | 6 | Divyanshi + Anshika |
| 7. Backend Integration | 3–7 (incremental) | Anshika |
| 8. Frontend Integration | 3–8 (incremental) | Divyansh |
| 9. End-to-End Pipeline Testing | 7 | Harshnandni + All |
| 10. Evaluation | 8 | Akanchha + Harshnandni |
| 11. Deployment | 9 | All |
| 12. Documentation & Demo | 9–10 | Harshnandni + Divyansh |

## Dependency Notes
- Stage 3 (animation content) is the highest-risk, longest-running stage — it starts early (Week 3) and runs in parallel with the ML modules rather than waiting for them, since it has no dependency on retrieval or gloss generation being finished.
- Stage 6 (animation mapping) depends on both Stage 3 (clips must exist) and a stable gloss token vocabulary from Stage 5 — coordinate before starting.
- Stage 10 (evaluation) depends on Stage 5 being feature-complete (ablation flag implemented) and Stage 1's test split being held out from the start — don't let the test split leak into fine-tuning data.
