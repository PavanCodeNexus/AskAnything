# 📄 Product Requirements Document (PRD)
## Project: AskAnything — Multimodal RAG Assistant

**Version:** 2.0  
**Author:** Pavan B C  
**Date:** September 2026  
**Status:** Planning  
**Changelog:** v1.0 → v2.0 — Added Evidence Score, Security, Evaluation, Hybrid Retrieval, MVP phasing, Conflict redesign, Conversational RAG

---

## 1. 🎯 Project Overview

**AskAnything** is a Multimodal Retrieval-Augmented Generation (RAG) system that unifies heterogeneous source ingestion, retrieval, grounded answering, source citation, evidence comparison, multilingual interaction, and optional knowledge visualization within a single workflow.

The system helps users combine PDFs, URLs, YouTube videos, images, diagrams, and notes in one session — then ask questions and receive grounded, cited answers with evidence scores, conflict analysis, and optional voice interaction.

---

## 2. 🧩 Problem Statement

AskAnything is designed to address the following gaps commonly found in existing RAG tools:

- Most tools accept only one input type per session
- Source citations often lack page numbers, timestamps, or URL references
- Contradictions across sources go undetected
- Indian regional language support (Kannada, Telugu, Tamil) is rare
- Voice interaction in regional languages is largely absent
- Visual knowledge mapping is rarely offered
- Insufficient evidence responses are missing — tools hallucinate instead

> **Note:** These are the motivations for building AskAnything. Claims of uniqueness will be validated through a competitive analysis before final release.

---

## 3. 👤 Target Users

| User Type | Use Case |
|---|---|
| Students | Upload textbooks + YouTube lectures + notes and ask questions |
| Researchers | Cross-reference multiple PDFs and detect contradictions |
| Professionals | Query company documents + web sources together |
| Regional language users | Ask questions in Kannada, Hindi, Telugu, Tamil |

---

## 4. 📥 Input Sources

| Input Type | Tool / Library | Limits & Notes |
|---|---|---|
| PDF / Documents | PyMuPDF | Max 50MB per file, up to 500 pages recommended |
| URLs / Websites | BeautifulSoup + requests | Sanitized; SSRF-protected |
| YouTube Video Links | youtube-transcript-api | Graceful fallback if transcript unavailable |
| Images / Photos | Tesseract OCR + Pillow | Text extraction only; not visual understanding |
| Diagrams | Tesseract OCR | Best on printed/clear text diagrams |
| Notes (plain text) | Built-in file reader | .txt, UTF-8 encoding, max 5MB |
| Voice Input | SpeechRecognition | EN, HI, KN, TE, TA (experimental for regional) |

> All inputs can be submitted **simultaneously** in one session (up to 10 documents per session).

---

## 5. ✨ Core Features

### 5.1 RAG Engine (Upgraded Architecture)

```
User Query
    ↓
Language Detection
    ↓
Conversational Query Rewriting (follow-up resolution)
    ↓
Hybrid Retrieval (Vector Search + BM25 Keyword Search)
    ↓
Reranking (cross-encoder reranker)
    ↓
Evidence Filtering (minimum threshold check)
    ↓
LLM Answer Generation (Groq — Llama 3.3 70B)
    ↓
Grounding Verification
    ↓
Citation Validation
    ↓
Final Answer + Evidence Score + Citations
```

**Chunk Metadata (stored per chunk):**
- `document_id`, `session_id`, `source_type`
- `filename` / `title` / `url`
- `page_number` (PDF) or `timestamp` (YouTube)
- `chunk_id`, `section`

**Insufficient Evidence Handling:**
- If retrieved chunks do not meet the minimum relevance threshold, the system responds:
  *"The uploaded sources do not contain enough information to answer this question."*
- No hallucinated answers.

### 5.2 Conversational Query Rewriting
- Follow-up questions are rewritten into standalone queries before retrieval
- Example:
  ```
  Q1: "What is RAG?"
  Q2: "What are its advantages?"       → rewritten: "What are the advantages of RAG?"
  Q3: "What about the second one?"     → rewritten: "What is the second advantage of RAG?"
  ```
- Relevant chat context passed to LLM without sending full history unnecessarily

### 5.3 Source Citation (Strengthened)
Every answer displays citations mapped to actual retrieved chunks:

| Source Type | Citation Format |
|---|---|
| PDF | Filename + Page Number + Section (if available) |
| URL | Page Title + URL + Retrieval Date |
| YouTube | Video Title + Timestamp Range |
| Image | Image filename + OCR region |
| Notes | Filename + Line range |

- Citation validation: UI only shows citations that map to evidence actually used
- No unsupported citations displayed

### 5.4 Evidence Score ⭐ Differentiator
*(Renamed and redesigned from "Confidence Score")*

- **Not** raw cosine similarity — that measures distance, not answer correctness
- Evidence Score is derived from:
  - Retrieval relevance (similarity score)
  - Source coverage (how many sources agree)
  - Answer grounding (does LLM answer stay within retrieved chunks)
  - Contradiction signals (conflicting evidence lowers score)
- Displayed as: **High / Medium / Low** + numeric score (0–100)
- Example: *"Evidence Strength: High (82) — based on 3 aligned sources"*
- Score formula and thresholds documented; validated against test dataset before release

### 5.5 Knowledge Conflict Detector ⭐ Differentiator
*(Redesigned — claim-level workflow)*

```
Retrieved Evidence
    ↓
Claim Extraction (per source)
    ↓
Group Related Claims
    ↓
Compare Claims
    ↓
Classify:
  ✅ Agreement       → sources confirm each other
  ⚠️ Contradiction   → sources make opposite claims
  📊 Different Estimate → numerical differences (not auto-flagged as contradiction)
  ❓ Insufficient    → not enough evidence to compare
    ↓
Show: both claims + source locations + user can inspect evidence
```

- Small numerical differences are NOT automatically flagged as contradictions
- User can expand each conflict to see the raw evidence

### 5.6 Multilingual Input & Output ⭐ Differentiator

**Supported languages:**

| Language | Text Input | Text Output | Voice Input | Voice Output |
|---|---|---|---|---|
| English | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Hindi | ✅ Full | ✅ Full | ✅ Good | ✅ Good |
| Kannada | ✅ Full | ✅ Full | 🧪 Experimental | 🧪 Experimental |
| Telugu | ✅ Full | ✅ Full | 🧪 Experimental | 🧪 Experimental |
| Tamil | ✅ Full | ✅ Full | 🧪 Experimental | 🧪 Experimental |

> Voice accuracy for regional languages is **experimentally measured** — not hardcoded. Benchmarks will be published after testing.

**Pipeline:**
```
Language Detection → Query Normalization → Translation (if needed)
→ RAG Retrieval → Answer Generation → Response Translation → Output
```

- Technical terms (embedding, vector, RAG, chunking) preserved during translation

### 5.7 Optional Mind Map / Knowledge Graph ⭐ Differentiator
- User clicks **"Generate Mind Map"** button after receiving answer
- Visualizes concept relationships from retrieved chunks
- Built with pyvis + networkx
- Interactive: zoom, drag, click nodes
- Not generated automatically — user-triggered only

### 5.8 Voice Chat ⭐ Differentiator

**STT Pipeline:**
```
Microphone → SpeechRecognition → Language Detection → Query Processing → RAG
```

**TTS Pipeline:**
```
Final Answer → Language-specific gTTS → Audio File → Streamlit Audio Player
```

- Voice input and output are separate components
- Latency and transcription accuracy measured during evaluation
- Browser mic permission required (Chrome recommended)

---

## 6. 📦 Document Processing Lifecycle

Every uploaded document passes through these states (shown in UI):

```
Uploaded → Validating → Extracting → Cleaning → Chunking → Embedding → Indexing → ✅ Ready
                                                                                  → ❌ Failed (with retry option)
```

- Failed documents show structured error message
- Retry button available for failed documents
- Temporary files cleaned after processing
- Processing progress shown via Streamlit progress bar

---

## 7. 🖥️ Frontend (Streamlit)

### Layout
```
┌──────────────────────────────────────────────────────┐
│  SIDEBAR                  │  MAIN CHAT AREA          │
│  - Upload PDF             │  - Chat history          │
│  - Paste URL              │  - Q&A display           │
│  - Paste YouTube link     │  - Source cards          │
│  - Upload Image           │  - Evidence Score badge  │
│  - Upload Notes           │  - Conflict alert panel  │
│  - Document Manager       │  - Insufficient evidence │
│    (with processing state)│    message state         │
│  - Session History        │  - Mind map button       │
│  - Language Selector      │  - Voice controls 🎤 🔊  │
│  - Export Q&A button      │  - Expand/collapse       │
│                           │    evidence panel        │
└──────────────────────────────────────────────────────┘
```

### UI Features

| Feature | Details |
|---|---|
| Chat History | Persistent within session via Streamlit session state |
| Document Manager | View / delete docs; shows processing lifecycle state |
| Export Q&A | Download chat as PDF or TXT |
| Loading Indicators | Skeleton loaders during ingestion; spinner during generation |
| Error Handling | Friendly st.error() messages with retry controls |
| Multi-session | Shared ChromaDB collection filtered by session_id |
| Language Selector | Dropdown: English / Hindi / Kannada / Telugu / Tamil |
| Voice Controls | 🎤 mic button (input) + 🔊 speaker toggle (output) |
| Insufficient Evidence | Visible "Not enough info in sources" state — no hallucination |
| Evidence Expand | User can expand source evidence per answer |
| Empty State | First-use guidance shown when no documents uploaded |

---

## 8. 🔧 Backend Architecture

### File Structure
```
askanything/
├── app.py                        # Streamlit UI entry point
├── rag_engine.py                 # Core RAG pipeline
├── vectorstore.py                # ChromaDB setup & hybrid retrieval
├── query_rewriter.py             # Conversational query rewriting
├── grounding.py                  # Grounding verification + evidence score
├── ingest/
│   ├── __init__.py
│   ├── base_loader.py            # Common normalized document schema
│   ├── pdf_loader.py             # PDF text + page metadata
│   ├── url_loader.py             # Web scraping + SSRF protection
│   ├── youtube_loader.py         # Transcript + fallback handling
│   ├── image_loader.py           # OCR for images/diagrams
│   └── notes_loader.py           # Plain text loader
├── features/
│   ├── __init__.py
│   ├── conflict_detector.py      # Claim-level conflict detection
│   ├── translator.py             # Multilingual I/O
│   ├── voice_input.py            # STT pipeline
│   ├── voice_output.py           # TTS pipeline
│   ├── mindmap_generator.py      # Knowledge graph generator
│   └── evidence_score.py         # Evidence Score (redesigned)
├── utils/
│   ├── __init__.py
│   ├── chunker.py                # Text chunking with metadata
│   ├── exporter.py               # Export Q&A as PDF/TXT
│   ├── validator.py              # File type/size/MIME validation
│   └── logger.py                 # Structured logging
├── security/
│   ├── __init__.py
│   ├── file_sanitizer.py         # File validation + sanitization
│   └── url_guard.py              # SSRF protection for URL ingestion
├── evaluation/
│   ├── __init__.py
│   ├── test_dataset.json         # Gold-standard Q&A pairs
│   └── evaluator.py              # Retrieval + generation metrics
├── .env                          # API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

### Data Models

**Chunk Schema (normalized across all sources):**
```python
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "session_id": "uuid",
  "source_type": "pdf|url|youtube|image|notes",
  "filename": "report.pdf",
  "title": "Page title or video title",
  "url": "https://...",
  "page_number": 12,        # PDF only
  "timestamp": "00:03:45",  # YouTube only
  "section": "Introduction",
  "text": "chunk content...",
  "embedding": [...]
}
```

---

## 9. 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | Groq API — Llama 3.3 70B (free) |
| Framework | LangChain |
| Embeddings | HuggingFace sentence-transformers (free, local) |
| Vector Store | ChromaDB (local, session_id filtered) |
| Keyword Search | BM25 (rank-bm25 library) |
| Reranker | cross-encoder/ms-marco (HuggingFace, free) |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Web Scraping | BeautifulSoup4 + requests |
| YouTube | youtube-transcript-api |
| OCR | Tesseract + Pillow |
| Translation | Google Translate API (free tier) |
| Voice Input | SpeechRecognition |
| Voice Output | gTTS (Google Text-to-Speech) |
| Mind Map | pyvis + networkx |
| Export | fpdf2 |
| Config | python-dotenv |
| Logging | Python logging (structured) |

---

## 10. 🔐 Security Requirements

| Requirement | Details |
|---|---|
| API Key Protection | GROQ_API_KEY only in .env — never in frontend code or committed to Git |
| File Validation | Validate file type, extension, MIME type, and size before processing |
| HTML Sanitization | Sanitize all extracted web/HTML content before display |
| SSRF Protection | URL ingestion blocked for internal IPs, localhost, private ranges |
| Prompt Injection Defense | Retrieved document content treated as data — never executed as instructions |
| Context Isolation | System prompt explicitly separates user query from retrieved content |
| Rate Limiting | Limit LLM, embedding, OCR, and transcription calls per session |
| Dependency Scanning | Use pip-audit or safety to scan for vulnerable packages |
| Document Deletion | Deleted documents removed from ChromaDB retrieval immediately |
| .gitignore | .env, chroma_db/, temp files excluded from version control |

---

## 11. 📊 Evaluation & Testing

### Test Dataset
- Small gold-standard Q&A dataset created from own test documents
- Covers: PDF, URL, YouTube, multilingual, conflicting sources, insufficient evidence cases

### Metrics

| Area | Metrics |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR |
| Generation | Faithfulness, Answer Relevance, Context Relevance, Citation Accuracy |
| Conflict Detector | Precision, Recall, F1 on labeled agreement/contradiction cases |
| Evidence Score | Correlation with human-rated answer quality |
| Multilingual | Accuracy for EN, HI, KN, TE, TA (text + voice separately) |
| Voice | Transcription accuracy, end-to-end latency |
| System | Ingestion time, retrieval latency, generation latency, memory usage, failure rate |

### Adversarial Test Cases
- Prompt injection embedded in PDF or webpage content
- Irrelevant sources uploaded (should trigger insufficient evidence)
- Conflicting sources (should trigger conflict detector)
- Empty documents
- YouTube videos with no transcript
- Very large PDFs (stress test)

---

## 12. 🌟 Differentiators Summary

| # | Feature | Differentiator |
|---|---|---|
| 1 | **Knowledge Conflict Detector** | Claim-level contradiction detection across heterogeneous sources |
| 2 | **Multilingual Voice Chat** | Regional language (KN/TE/TA) voice RAG interaction |
| 3 | **Optional Mind Map on demand** | User-triggered interactive knowledge graph from RAG answer |
| 4 | **Evidence Score** | Multi-signal reliability score — not raw cosine similarity |
| 5 | **All input types simultaneously** | PDF + URL + YouTube + Image in one unified session |

> Competitive analysis to be completed before final release to validate differentiator claims.

---

## 13. ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| Image OCR | Text extraction only — does not understand diagram semantics |
| YouTube transcripts | Unavailable if auto-captions disabled; graceful fallback shown |
| Regional voice (KN/TE/TA) | Experimental — accuracy benchmarked during evaluation phase |
| Streamlit mic | Requires browser permission (Chrome recommended) |
| Groq context window | 32K tokens — only top chunks sent, not full document |
| PDF limits | Files above 50MB or 500 pages may have slower processing |
| Translation | Technical AI/CS terms may not translate perfectly in all languages |

---

## 14. 🗓️ Development Roadmap

### MVP (Release 1) — Core RAG
| Phase | Task | Status |
|---|---|---|
| 1 | Project setup, config, logging, error handling | ⏳ Pending |
| 2 | PDF + Notes ingestion + normalized chunk schema | ⏳ Pending |
| 3 | Embeddings + ChromaDB + baseline RAG | ⏳ Pending |
| 4 | Source citations + chunk metadata | ⏳ Pending |
| 5 | Grounding verification + insufficient evidence handling | ⏳ Pending |
| 6 | Conversational query rewriting | ⏳ Pending |
| 7 | Basic Streamlit chat UI + document manager | ⏳ Pending |
| 8 | Basic security + file validation | ⏳ Pending |

### V2 — Multimodal + Advanced Retrieval
| Phase | Task | Status |
|---|---|---|
| 9 | URL + YouTube + Image/OCR ingestion | ⏳ Pending |
| 10 | Hybrid retrieval (BM25 + Vector) + Reranking | ⏳ Pending |
| 11 | Conflict detector (claim-level) | ⏳ Pending |
| 12 | Evidence Score (multi-signal) | ⏳ Pending |

### V3 — Differentiators
| Phase | Task | Status |
|---|---|---|
| 13 | Multilingual text I/O | ⏳ Pending |
| 14 | Voice input + output (STT + TTS) | ⏳ Pending |
| 15 | Knowledge graph / mind map | ⏳ Pending |
| 16 | Export Q&A, session history | ⏳ Pending |
| 17 | Security hardening + evaluation dataset + metrics | ⏳ Pending |
| 18 | README + GitHub release + demo | ⏳ Pending |

---

## 15. 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Never commit `.env` to Git. Add to `.gitignore`.

---

*PRD Version 2.0 — AskAnything Multimodal RAG Assistant*  
*Upgraded from v1.0 based on technical review — September 2026*
