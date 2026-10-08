# System Design Document (HLD + LLD)
## RAG-Augmented Speech-to-ISL Avatar Generator

---

## Part A: High-Level Design (HLD)

### A.1 System Architecture Overview

```text
                         ┌─────────────────────────┐
                         │   Frontend (React +     │
                         │   Tailwind CSS)          │
                         │  - Text/mic input        │
                         │  - Avatar viewport        │
                         └───────────┬──────────────┘
                                     │ REST / WebSocket
                                     ▼
                         ┌─────────────────────────┐
                         │ Backend API             │
                         │ (FastAPI)               │
                         └───────────┬──────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                             ▼
┌───────────────┐          ┌───────────────────┐         ┌──────────────────┐
│ STT Module     │          │ RAG Retrieval      │         │ Gloss Generation  │
│ faster-whisper │  text →  │ MiniLM embeddings   │ examples│ Flan-T5-small      │
│ (distil-small) │          │ + FAISS index       │────────▶│ (use_rag few-shot)│
└───────────────┘          └───────────────────┘         └────────┬─────────┘
                                                                     │ gloss sequence
                                                                     ▼
                                                          ┌──────────────────────┐
                                                          │ Animation Mapping     │
                                                          │ Gloss → clip lookup   │
                                                          │ + transition blending │
                                                          └──────────┬───────────┘
                                                                     │
                                                                     ▼
                                                          ┌──────────────────────┐
                                                          │ 3D Avatar Renderer    │
                                                          │ (Unity / Three.js)     │
                                                          └──────────────────────┘

        Supporting stores: PostgreSQL (metadata, logs, users)
                            FAISS index (persisted vector store)
                            Animation clip library (file storage)
```

### A.2 Module Overview
| Module | Responsibility |
|---|---|
| Frontend | Capture input, display status, render avatar animation |
| Backend API | Orchestrate pipeline stages, expose REST/WebSocket endpoints |
| STT Module | Convert speech to text |
| RAG Retrieval Module | Embed input, retrieve similar sentence-gloss pairs |
| Gloss Generation Module | Generate ISL gloss sequence using retrieved context |
| Animation Mapping Module | Map gloss tokens to clips, blend transitions |
| Avatar Renderer | Play animation sequence on rigged 3D model |

### A.3 Technology Stack
| Layer | Technology |
|---|---|
| Frontend | React + Tailwind CSS |
| Backend | FastAPI (Python) |
| Speech Recognition | faster-whisper (`distil-small.en`) |
| Embeddings | sentence-transformers (MiniLM) |
| Vector Database | FAISS |
| Gloss Generator | Fine-tuned Flan-T5-small (weights via Colab; graceful fallback without weights) |
| Deep Learning Framework | PyTorch + HuggingFace `transformers` |
| Avatar Engine | Unity (WebGL build) or Three.js |
| Relational Database | PostgreSQL |
| Deployment | Docker |

### A.4 Design Principles
- **Modularity:** Each pipeline stage is independently testable and replaceable (e.g., swap in a different fine-tuned seq2seq model without touching retrieval or animation code).
- **Graceful degradation:** If RAG retrieval returns no close matches, the generator falls back to zero-shot generation rather than failing.
- **Separation of deterministic and learned components:** Animation clip playback is deterministic; only STT and gloss generation are model-driven, keeping the system debuggable.

---

## Part B: Low-Level Design (LLD)

### B.1 STT Module
- **Input:** Audio stream (WAV/PCM from browser mic) or uploaded audio file
- **Process:** faster-whisper (`distil-small.en`) inference → transcript string
- **Output:** `{ "transcript": str, "confidence": float, "language": str }`
- **Error handling:** On low-confidence transcription, prompt user to confirm/edit text before proceeding

### B.2 RAG Retrieval Module
- **Index build (offline, one-time):**
  1. Load ISL-CSLTR and CISLR sentence-gloss pairs
  2. Generate embeddings via `sentence-transformers/all-MiniLM-L6-v2`
  3. Build FAISS index (IndexFlatL2 or IndexIVFFlat for larger corpora), persist to disk
- **Query (runtime):**
  1. Embed incoming sentence
  2. `index.search(query_embedding, k=5)` → top-k nearest sentence-gloss pairs
  3. Return retrieved pairs with similarity scores
- **Output:** List of `{ "sentence": str, "glosses": str, "landmark_file": str, "similarity": float, "distance": float }` (`similarity = 1/(1+L2)`, exact match = `1.0`)

### B.3 Gloss Generation Module
- **Input:** User sentence + retrieved examples (few-shot context)
- **Prompt structure (conceptual):**
  ```
  translate English to ISL:
  EN: <retrieved_sentence_1> -> ISL: <retrieved_gloss_1>
  EN: <retrieved_sentence_2> -> ISL: <retrieved_gloss_2>
  ...
  EN: <user_sentence> ->
  ```
  (few-shot lines only when `use_rag`; trained prefix appended when Flan-T5 weights are present)
- **Model:** Fine-tuned Flan-T5-small on ISL-CSLTR and CISLR sentence-gloss pairs (prefix `translate English to ISL: `), prompted with retrieved context at inference time
- **Output:** Ordered gloss token sequence, e.g., `["TODAY", "I", "GO", "SCHOOL"]`
- **Ablation mode:** Flag `use_rag: false` bypasses retrieved context (RAG off) for evaluation comparison; there is no in-app ablation endpoint

### B.4 Animation Mapping Module
- **Input:** Gloss token sequence
- **Process:**
  1. Look up each gloss token in the clip registry (gloss → clip file/ID mapping table)
  2. For unknown tokens: log warning, skip (v1) or trigger finger-spelling fallback (future)
  3. Build ordered clip playlist with blend metadata (transition duration, blend curve)
- **Output:** Playback instruction set sent to frontend/avatar engine

### B.5 Avatar Renderer
- **Input:** Ordered clip playlist
- **Process:** Load rigged avatar, play clips sequentially with blend-tree interpolation between consecutive signs
- **Output:** Rendered animation in browser viewport (Unity WebGL canvas or Three.js scene)

### B.6 Backend API Endpoints (summary — full spec in API document)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/transcribe` | POST | Audio → text |
| `/api/translate` | POST | Text → gloss sequence (RAG-augmented, `use_rag`/`top_k`) |
| `/api/animate` | POST | Gloss sequence → clip playlist |
| `/api/health` | GET | Model/DB/path/FAISS status |
| `/api/pipeline/ws` | WS | Full pipeline, streamed stage updates |
| `/api/avatar/ws` | WS | Avatar protocol (landmark file URL metadata; data via HTTP `/landmarks/...`) |

Evaluation (RAG-on vs. RAG-off) is run offline against a held-out split — see Test Plan §4; no `/api/eval/ablation` endpoint exists.

### B.7 Data Flow Sequence (end-to-end)
1. User speaks/types sentence
2. Frontend sends audio (or text) to backend via WebSocket
3. Backend: STT (if audio) → RAG retrieval → gloss generation → animation mapping
4. Backend streams status updates + final clip playlist to frontend
5. Frontend renders avatar animation

### B.8 Error Handling & Edge Cases
- Empty/unintelligible audio → prompt re-recording
- Sentence outside retrieval corpus domain → generation proceeds with lower-confidence fallback, logged for future corpus expansion
- Gloss token with no matching clip → skipped with on-screen indicator ("sign unavailable for: X")
- Backend/model load failure → health-check endpoint surfaces status (incl. `database_error`, missing Flan-T5 weights) to frontend before accepting requests
- Avatar WS path traversal in `landmark_file` → rejected before file access; landmark payloads served over HTTP to avoid WS fragmentation
