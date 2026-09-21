# Database Design Document
## RAG-Augmented Speech-to-ISL Avatar Generator

---

## 1. Overview
The system uses two data stores:
- **PostgreSQL** — relational metadata: corpus entries, animation clip registry, session logs, evaluation results
- **FAISS index (file-based)** — vector store for RAG retrieval embeddings, persisted to disk and loaded into memory at backend startup

---

## 2. PostgreSQL Schema

### 2.1 Table: `corpus_entries`
Stores sentence-gloss pairs used for fine-tuning and as the RAG retrieval corpus. Sourced from ISL-CSLTR and CISLR, supplemented by a separately curated retrieval knowledge base (additional sentence-gloss pairs assembled specifically to broaden retrieval coverage beyond the two base corpora — flagged via the `source` column below).

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | SERIAL | PRIMARY KEY | Unique entry ID |
| source_sentence | TEXT | NOT NULL | Original English/Hindi sentence |
| gloss_sequence | TEXT | NOT NULL | Space-separated ISL gloss tokens |
| language | VARCHAR(10) | NOT NULL, DEFAULT 'en' | Source language code |
| source | VARCHAR(20) | NOT NULL | 'ISL-CSLTR' / 'CISLR' / 'curated' — origin corpus for provenance tracking |
| embedding_id | INTEGER | NULLABLE | Reference to FAISS vector index position |
| split | VARCHAR(10) | NOT NULL | 'train' / 'val' / 'test' |
| created_at | TIMESTAMP | DEFAULT now() | |

### 2.2 Table: `gloss_vocabulary`
Registry of known gloss tokens and their animation clip mapping.

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | SERIAL | PRIMARY KEY | |
| gloss_token | VARCHAR(50) | UNIQUE, NOT NULL | Canonical gloss word (uppercase) |
| clip_file_path | TEXT | NOT NULL | Path/identifier for animation clip asset |
| avg_duration_ms | INTEGER | | Clip duration for blend-timing calculations |
| category | VARCHAR(30) | | e.g., 'noun', 'verb', 'pronoun', 'time-marker' |
| notes | TEXT | | Notes on usage/variants |

### 2.3 Table: `sessions`
Tracks a single user interaction session (for demo/eval logging, not persistent user profiles).

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | UUID | PRIMARY KEY | Session identifier |
| started_at | TIMESTAMP | DEFAULT now() | |
| ended_at | TIMESTAMP | NULLABLE | |
| input_mode | VARCHAR(10) | | 'speech' / 'text' |

### 2.4 Table: `translation_logs`
Logs each translation request for debugging and evaluation.

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | SERIAL | PRIMARY KEY | |
| session_id | UUID | FOREIGN KEY → sessions.id | |
| input_text | TEXT | NOT NULL | Transcribed or typed input |
| retrieved_examples | JSONB | | Top-k retrieved sentence-gloss pairs + scores |
| generated_gloss | TEXT | NOT NULL | Model output gloss sequence |
| rag_enabled | BOOLEAN | DEFAULT true | Whether retrieval context was used |
| latency_ms | INTEGER | | End-to-end pipeline latency |
| created_at | TIMESTAMP | DEFAULT now() | |

### 2.5 Table: `evaluation_results`
Stores ablation study results (RAG vs. no-RAG comparisons).

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | SERIAL | PRIMARY KEY | |
| test_sentence_id | INTEGER | FOREIGN KEY → corpus_entries.id | |
| rag_enabled | BOOLEAN | NOT NULL | |
| generated_gloss | TEXT | NOT NULL | |
| reference_gloss | TEXT | NOT NULL | Ground-truth gloss from corpus |
| bleu_score | FLOAT | | |
| notes | TEXT | | Qualitative reviewer notes |

---

## 3. FAISS Vector Store

| Property | Value |
|---|---|
| Index type | `IndexFlatL2` (exact search, suitable for corpus size in this project) |
| Vector dimension | 384 (MiniLM `all-MiniLM-L6-v2` output dimension) |
| Storage | Persisted to disk (`.index` file), loaded into backend memory on startup |
| Mapping | `embedding_id` in FAISS index maps 1:1 to `corpus_entries.embedding_id` in PostgreSQL for retrieving full sentence-gloss text after similarity search |

**Note:** `IndexFlatL2` is chosen over approximate methods (e.g., `IndexIVFFlat`) because the ISL-CSLTR and CISLR corpora scale is small enough that exact search remains fast; approximate indexing is unnecessary complexity at this scale.

---

## 4. Entity Relationship Summary

```text
corpus_entries (1) ──── (1) FAISS vector [via embedding_id]
gloss_vocabulary (independent lookup table, referenced by animation module)
sessions (1) ──── (many) translation_logs
corpus_entries (1) ──── (many) evaluation_results [as test_sentence_id]
```

## 5. Data Retention & Privacy Notes
- Raw audio is not persisted to the database — only the transcribed text is logged, per NFR-7 in the SRS.
- `translation_logs` and `sessions` are intended for development/evaluation use; a production deployment would need explicit user consent and retention limits before logging real user input.
