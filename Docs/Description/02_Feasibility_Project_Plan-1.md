# Feasibility Study & Project Plan
## RAG-Augmented Speech-to-ISL Avatar Generator

---

## 1. Feasibility Study

### 1.1 Technical Feasibility
| Component | Feasibility | Notes |
|---|---|---|
| Speech-to-text | High | Distil-Whisper is open-source, pre-trained, fine-tunable, and runs on modest hardware |
| RAG retrieval | High | Sentence-transformer embeddings + FAISS is lightweight, well-documented, no server infra needed |
| Gloss generation | Medium | Requires fine-tuning on ISL-CSLTR and CISLR; data volume is limited, so quality will depend on retrieval augmentation working as intended |
| Animation/avatar | Medium | Sourcing or building a rigged sign clip library is the most time-intensive component |
| Real-time pipeline | Medium | Achievable for "near real-time" (few seconds), not broadcast-grade live latency |

**Verdict:** Technically feasible for a solo B.Tech final-year timeline, provided the animation clip library is scoped early (50–100 core signs is sufficient for a compelling demo) and full diffusion-based motion synthesis is explicitly excluded from v1 scope.

### 1.2 Economic Feasibility
All core components (Distil-Whisper, sentence-transformers, FAISS, Flan-T5/mT5, PyTorch, FastAPI, React, Unity Personal/Three.js, Docker, PostgreSQL) are free/open-source. Only cost is compute time for fine-tuning (feasible on a single consumer GPU or free-tier cloud GPU credits) and optional cloud hosting for the demo deployment.

### 1.3 Operational Feasibility
System is intended as a demo/research prototype, not a production accessibility tool at this stage. Operationally realistic for coursework submission, viva demonstration, and portfolio use.

### 1.4 Schedule Feasibility
Achievable within a single semester (~14–16 weeks) if scoped per the timeline in Section 2, with the animation library and RAG evaluation identified as the highest-risk, earliest-start items.

### 1.5 Key Risks
| Risk | Impact | Mitigation |
|---|---|---|
| ISL-CSLTR/CISLR too small for generalizable fine-tuning | High | RAG retrieval is the primary mitigation; scope demo sentences within corpus domain coverage |
| No existing rigged ISL sign clip library available | High | Start sourcing/building early (Week 1–2); fall back to a smaller vocabulary if needed |
| Real-time latency exceeds target | Medium | Cache embeddings, use smaller/distilled models (Distil-Whisper, Flan-T5-small), pre-warm model loading |
| Co-articulation (jerky transitions) | Medium | Budget explicit time for blend-tree/interpolation tuning, not just clip triggering |
| No published RAG-for-SLP baseline to compare against | Low (expected) | Report own ablation (with RAG vs. without RAG) as primary evaluation evidence |

---

## 2. Project Plan

### 2.1 Development Methodology
Iterative/incremental (mini-waterfall per module), given solo development and fixed academic milestones. Each pipeline stage (STT → RAG → Gloss → Animation) is built and validated independently before integration.

### 2.2 Work Breakdown Structure & Timeline (16-week reference plan)

| Phase | Weeks | Deliverables |
|---|---|---|
| 1. Requirements & Literature Review | 1–2 | SRS finalized, related-work summary, dataset acquisition (ISL-CSLTR, CISLR) |
| 2. Design | 3–4 | System design (HLD/LLD), database schema, API spec |
| 3. STT Module | 5 | Distil-Whisper integration, basic transcription tested |
| 4. RAG Retrieval Module | 6–7 | Embedding pipeline, FAISS index built from corpus, retrieval tested |
| 5. Gloss Generation Module | 8–9 | Flan-T5/mT5 fine-tuned, RAG-prompted generation tested |
| 6. Animation Clip Library | 6–10 (parallel) | Sign clips sourced/recorded, rigged avatar set up in Unity/Three.js |
| 7. Animation Mapping & Blending | 10–11 | Gloss-to-clip lookup, transition blending implemented |
| 8. Pipeline Integration | 12 | End-to-end pipeline: speech → avatar, via FastAPI + WebSocket |
| 9. Frontend | 12–13 | React + Tailwind UI, avatar viewport, input controls |
| 10. Evaluation | 13–14 | RAG vs. no-RAG ablation, BLEU/qualitative results, latency measurement |
| 11. Documentation & Report | 14–15 | Final report, SDLC docs, user manual |
| 12. Buffer / Viva Prep | 15–16 | Bug fixes, demo rehearsal, MCP wrapper (optional) |

### 2.3 Milestones
- M1 (Week 2): Requirements sign-off
- M2 (Week 4): Design sign-off
- M3 (Week 9): Core ML pipeline (STT + RAG + Gloss) functional standalone
- M4 (Week 11): Animation pipeline functional standalone
- M5 (Week 12): End-to-end integration complete
- M6 (Week 14): Evaluation results finalized
- M7 (Week 16): Final submission ready

### 2.4 Resource Plan
- **Personnel:** 1 developer (solo), 1 project guide/mentor for review checkpoints
- **Hardware:** Development laptop/desktop; GPU access (personal or free-tier cloud) for fine-tuning
- **Software/Data:** ISL-CSLTR and CISLR datasets, open-source libraries listed in Technology Stack (see System Design document)

### 2.5 Success Criteria
- End-to-end pipeline produces a signed animation for at least 50–100 core-vocabulary sentences
- Measurable quality improvement (BLEU/qualitative) with RAG enabled vs. disabled
- Latency within near-real-time target (few seconds)
- Documented, reproducible setup via Docker
