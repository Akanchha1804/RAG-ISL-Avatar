# Work Allocation Plan
## RAG-Augmented Speech-to-ISL Avatar Generator — 10-Week Timeline

**Group:** Akanchha Singh (2300040100003) · Anshika Pachauri (2300040100012) ·
Divyansh Sharma (2300040100029) · Divyanshi Goyal (2300040100030) ·
Harshnandni (2300040100035)
**Batch:** B.Tech (B1 Batch) — CSE, Raja Balwant Singh Engineering Technical Campus

---

## 1. Role Summary

| Member | Responsibility | Load |
|---|---|---|
| Akanchha Singh | RAG retrieval module (embeddings + FAISS) + Gloss generation (Flan-T5/mT5 fine-tuning) | Heavy |
| Anshika Pachauri | Backend API (FastAPI, all endpoints) + STT integration (Distil-Whisper) | Heavy |
| Divyanshi Goyal | Animation mapping module + Unity avatar rigging/clip integration | Heavy |
| Divyansh Sharma | Frontend (React + Tailwind UI, avatar viewport, input controls) | Light |
| Harshnandni | Dataset curation (ISL-CSLTR/CISLR prep) + Documentation + QA/testing | Light |

---

## 2. Week-by-Week Plan (10 weeks)

### Week 1 — Setup & Requirements
- **All:** Finalize SRS, confirm scope, set up shared repo and dev environments
- **Harshnandni:** Begin sourcing ISL-CSLTR and CISLR datasets; inventory format/fields
- **Divyansh:** Set up React + Tailwind project skeleton, wireframe the input/avatar viewport screens

### Week 2 — Design
- **Akanchha:** Design RAG retrieval architecture (embedding model choice, FAISS index type)
- **Anshika:** Design backend API contract (endpoints, request/response schemas)
- **Divyanshi:** Source/evaluate rigged avatar options (Unity asset or custom rig); confirm animation clip format
- **Divyansh:** Build static UI shell against the finalized API spec (no live data yet)
- **Harshnandni:** Clean and deduplicate corpus data; produce a corpus summary report (size, vocabulary coverage)

### Week 3 — Core Module Build (Part 1)
- **Akanchha:** Implement embedding pipeline (MiniLM) + build initial FAISS index from cleaned corpus
- **Anshika:** Implement `/api/transcribe` (Distil-Whisper integration) and `/api/health`
- **Divyanshi:** Begin sourcing/recording core-vocabulary animation clips (target: first 25 signs)
- **Divyansh:** Wire up text input form to call `/api/transcribe` (stubbed backend acceptable initially)
- **Harshnandni:** Draft Database Design doc updates as schema stabilizes; begin test-case list for STT

### Week 4 — Core Module Build (Part 2)
- **Akanchha:** Implement retrieval query logic (`/api/translate` retrieval half); validate retrieval quality manually
- **Anshika:** Implement `/api/translate` endpoint scaffold, integrate with Akanchha's retrieval module
- **Divyanshi:** Continue clip sourcing (target: 50 signs); begin gloss-to-clip mapping table
- **Divyansh:** Build avatar viewport placeholder (static model load, no animation yet)
- **Harshnandni:** Expand test-case list to cover retrieval edge cases; log first round of QA findings

### Week 5 — Gloss Generation
- **Akanchha:** Fine-tune Flan-T5/mT5 on ISL-CSLTR + CISLR corpus; integrate RAG-prompted generation
- **Anshika:** Complete `/api/translate` end-to-end (retrieval + generation); add ablation flag (`rag_enabled`)
- **Divyanshi:** Reach 75-sign clip library; start blend-transition prototyping
- **Divyansh:** Connect frontend to live `/api/translate`, display gloss output as text (pre-animation)
- **Harshnandni:** Run first fine-tuning sanity checks against held-out sentences; log gloss quality observations

### Week 6 — Animation Pipeline
- **Divyanshi:** Implement `/api/animate` mapping logic + blend-tree transitions in Unity
- **Akanchha:** Tune retrieval top-k and prompt format based on Week 5 quality observations
- **Anshika:** Implement `/api/animate` backend endpoint, connect to Divyanshi's Unity mapping logic
- **Divyansh:** Build avatar animation playback in frontend viewport (consume clip playlist)
- **Harshnandni:** Update Test Plan doc with animation test cases; test unresolved-token handling

### Week 7 — Integration
- **All (paired):** Full pipeline integration — WebSocket `/api/pipeline` wiring speech → transcript → retrieval → gloss → animation
- **Akanchha + Anshika:** Debug end-to-end latency and retrieval/generation handoff
- **Divyanshi + Divyansh:** Debug animation playback and blend smoothness in the live pipeline
- **Harshnandni:** Run integration test pass (IT-01 through IT-06 from Test Plan), log defects

### Week 8 — Evaluation & Refinement
- **Akanchha:** Run RAG-on vs. RAG-off ablation study (core evaluation deliverable); compute BLEU scores
- **Anshika:** Fix defects from Week 7 integration testing; optimize API latency
- **Divyanshi:** Expand clip library toward final target (100+ signs) based on demo sentence set
- **Divyansh:** UI polish, status indicators (transcribing/retrieving/generating/animating), responsive layout
- **Harshnandni:** Collect qualitative reviewer ratings for ablation study; compile results table

### Week 9 — Hardening & Documentation
- **All:** Bug fixes from full-system testing; finalize Docker deployment
- **Harshnandni:** Finalize all SDLC docs and final project report using Week 8 evaluation results
- **Divyansh:** Cross-browser testing, final UI fixes
- **Akanchha + Anshika:** Verify reproducible Docker setup on a clean machine
- **Divyanshi:** Final animation QA pass across full demo sentence set

### Week 10 — Final Demo & Submission
- **All:** Rehearse live demo, prepare viva Q&A, final report review and submission
- **Harshnandni:** Compile and proofread final submission package
- **Divyansh:** Prepare demo-day UI walkthrough script

---

## 3. Milestones

| Milestone | Week | Owner(s) |
|---|---|---|
| M1: Design sign-off | 2 | All |
| M2: Retrieval + STT functional | 4 | Akanchha, Anshika |
| M3: Gloss generation functional | 5 | Akanchha |
| M4: Animation pipeline functional | 6 | Divyanshi, Anshika |
| M5: End-to-end integration complete | 7 | All |
| M6: Evaluation results finalized | 8 | Akanchha, Harshnandni |
| M7: Submission-ready | 10 | All |

## 4. Notes on Load Balancing
- Divyansh's and Harshnandni's tracks are intentionally decoupled from model-training risk — a delayed fine-tuning run or retrieval tuning issue (Akanchha's track) does not block frontend or documentation progress, since both can work against the API spec and Test Plan already defined.
- If the 10-week timeline slips, the first cut should come from Divyanshi's clip-library target (reduce from 100+ signs to the 50–75 range) rather than compressing the evaluation study in Week 8, since the ablation results are the project's core research claim.
