# ⚡ AskAnything — Multimodal RAG Assistant

[![CI / Test Suite](https://github.com/PavanCodeNexus/AskAnything/actions/workflows/ci.yml/badge.svg)](https://github.com/PavanCodeNexus/AskAnything/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AskAnything** is an enterprise-grade Multimodal Retrieval-Augmented Generation (RAG) assistant that unifies heterogeneous source ingestion, hybrid vector + keyword retrieval, cross-encoder reranking, claim-level knowledge conflict detection, autonomous web search fallback with multi-source fusion, ChatGPT & Gemini-style chat history, shareable conversations (WhatsApp/Instagram), and handwritten chalkboard concept maps into a single cohesive platform.

---

## 🌟 Key Features & Differentiators

| # | Feature | Capability |
|---|---|---|
| 1 | **Heterogeneous Ingestion** | Simultaneously upload & cross-reference PDFs, web URLs, YouTube video transcripts, images/diagrams via OCR, and plain text notes in one session. |
| 2 | **ChatGPT & Gemini-Style Chat History** | Chronologically organized history (**📌 Pinned**, **Today**, **Yesterday**, **Previous 7 Days**, **Older**). Supports 1-click **Pin/Unpin**, inline **Rename**, **Delete**, and atomic JSON storage with corrupt-file auto-quarantine. |
| 3 | **Autonomous Web Search & Fusion (< 75%)** | When local documents lack sufficient evidence (relevance score < 75%), AskAnything automatically searches the live web and synthesizes facts from both your uploaded files and live web search results. |
| 4 | **Share Chat via Direct Link** | Generate instant public snapshot links with 1-click sharing to **WhatsApp** and **Instagram Direct**. Recipients can read the conversation or clone it to their workspace. |
| 5 | **Unique Handwritten Chalkboard Mind Map** | Visual concept mind maps styled like study notes on a dark chalkboard graph paper, featuring **multi-sentence handwritten explanations** inside each card and 1-click clear controls. |
| 6 | **Collapsed Verified Sources & Citations** | Clean, non-intrusive Gemini/ChatGPT style: source cards are neatly tucked behind an expandable **`📚 Sources (N)`** button, keeping conversations uncluttered. |
| 7 | **Direct In-Chat Attachment** | Attach and index PDFs, images, and notes directly above the chat box using the inline **`📎 Attach Image or Document`** tool. |
| 8 | **Multi-Signal Evidence Scoring** | Evaluates retrieval relevance (40%), answer grounding (35%), multi-source corroboration (25%), and applies penalties for factual contradictions. |
| 9 | **Claim-Level Conflict Detector** | Dissects multi-source claims and categorizes them into **Agreement**, **Direct Contradiction**, **Different Numerical Estimate** (avoids false alarms on metric variances), or **Insufficient Evidence**. |
| 10 | **Multilingual Voice & Text** | Native support for **English, Hindi, Kannada, Telugu, and Tamil** with technical terminology preservation and regional text-to-speech (TTS). |
| 11 | **Enterprise Security** | SSRF protection (private IP / cloud metadata block), file format & size validation, HTML sanitization, and prompt injection isolation wrappers. |
| 12 | **Document Lifecycle Stepper** | Visual progress indicators showing: `Uploaded ➔ Validating ➔ Extracting ➔ Cleaning ➔ Chunking ➔ Embedding ➔ Indexing ➔ Ready`. |

---

## 🏗️ System Architecture

```
User Query (Text, In-Chat Attachments, or Voice)
         ↓
  Language Detection & Normalization (EN, HI, KN, TE, TA)
         ↓
  Conversational Query Rewriter (Follow-up & pronoun resolution via Groq)
         ↓
  Hybrid Retrieval (ChromaDB Vector + BM25 Keyword Search with RRF)
         ↓
  Cross-Encoder Reranker & Relevance Scoring
         ↓
  Score Evaluation Check:
  ├── If relevance >= 0.75 → Grounded local retrieval
  └── If relevance < 0.75  → Autonomous Live Web Search Fallback & Multi-Source Fusion
         ↓
  Claim-Level Conflict Detector (Agreement / Contradiction / Numerical Variance)
         ↓
  Prompt Injection Defense Isolation Wrapper
         ↓
  LLM Answer Generation (Groq Cloud — Llama 3.3 70B)
         ↓
  Grounding Verification & Verified Citation Validation
         ↓
  Multi-Signal Evidence Score Calculation (High / Medium / Low 0–100)
         ↓
  Atomic Chat Session Persistence (Self-Healing JSON Storage)
         ↓
  Multilingual Output Translation & gTTS Audio Generation
```

---

## 📂 File Structure

```
AskAnything/
├── app.py                            # Streamlit interactive UI entrypoint
├── rag_engine.py                     # Core RAG orchestration pipeline & 75% auto-search fusion
├── vectorstore.py                    # ChromaDB vector store + BM25 hybrid index
├── query_rewriter.py                 # Conversational query rewriting
├── grounding.py                      # Grounding verification & citation validation
├── packages.txt                      # Linux system packages for Streamlit Cloud (Tesseract, FFmpeg)
├── requirements.txt                  # Production Python dependencies
├── Dockerfile                        # Production container image
├── docker-compose.yml                # Docker Compose orchestration
│
├── ingest/                           # Multi-source document loaders
│   ├── __init__.py
│   ├── base_loader.py                # DocumentChunk & IngestedDocument schemas
│   ├── pdf_loader.py                 # PyMuPDF extraction with page & section metadata
│   ├── url_loader.py                 # Web scraper with SSRF protection & auto-URL formatting
│   ├── youtube_loader.py             # YouTube transcript loader with time windowing
│   ├── image_loader.py               # Multilingual OCR & image text extractor
│   └── notes_loader.py               # Plain text & markdown loader
│
├── features/                         # Advanced assistant capabilities
│   ├── __init__.py
│   ├── web_search_fallback.py        # Autonomous DuckDuckGo & Wikipedia search fallback
│   ├── conflict_detector.py          # Claim-level contradiction classification
│   ├── mindmap_generator.py          # Handwritten chalkboard SVG concept map with explanations
│   ├── evidence_score.py             # Multi-signal evidence scoring algorithm
│   ├── translator.py                 # Multilingual translation (EN, HI, KN, TE, TA)
│   ├── voice_input.py                # Speech-to-text input handler
│   └── voice_output.py               # Text-to-speech audio synthesis (gTTS)
│
├── utils/                            # Core utilities
│   ├── __init__.py
│   ├── chat_history_manager.py       # Atomic session persistence, recency grouping, sharing & auto-quarantine
│   ├── chunker.py                    # Metadata-preserving recursive text chunker
│   ├── exporter.py                   # Q&A session export (PDF & TXT)
│   ├── validator.py                  # File size, MIME, and extension validation
│   └── logger.py                     # Structured application logger
│
├── security/                         # Enterprise security guards
│   ├── __init__.py
│   ├── file_sanitizer.py             # Path traversal defense & prompt injection wrapper
│   └── url_guard.py                  # SSRF protection (private CIDR & metadata blocking)
│
├── chat_storage/                     # Persistent session data
│   ├── sessions/                     # Individual session JSON files
│   ├── shared/                       # Public shareable conversation snapshots
│   └── sessions_index.json           # Fast indexed metadata catalog
│
├── evaluation/                       # Benchmark evaluation suite
│   ├── test_dataset.json             # Gold-standard evaluation Q&A pairs
│   └── evaluator.py                  # Recall@K, Precision@K, MRR, Faithfulness runner
│
└── tests/                            # Automated unit & integration tests
    ├── test_system.py                # System-level pipeline verification
    └── test_web_search.py            # Web search fallback & fusion tests
```

---

## 🚀 Quickstart (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/PavanCodeNexus/AskAnything.git
cd AskAnything
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows:
python -m venv venv
.\venv\Scripts\activate

# On Linux/macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Set your **Groq API Key** in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```
*(Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys))*.

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment to Streamlit Community Cloud (Recommended)

AskAnything is pre-configured with `packages.txt`, `requirements.txt`, and cloud secrets bridging for **1-click free deployment**:

1. Log in to **[share.streamlit.io](https://share.streamlit.io)** with GitHub.
2. Click **"New app"** (or **"Create app"**).
3. Set the deployment fields:
   - **Repository**: `PavanCodeNexus/AskAnything`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Expand **Advanced settings** -> **Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
5. Click **Deploy!** Your app will be live at `https://<your-subdomain>.streamlit.app`.

---

## 🐳 Docker Deployment

To run AskAnything inside a production container with pre-installed Tesseract multilingual OCR and audio processing libraries:

```bash
docker compose up --build -d
```
Access the application at `http://localhost:8501`.

---

## 🧪 Testing & Quality Assurance

Run the automated test suites:

```bash
# Run core system tests
python test_system.py

# Run web search fallback and fusion tests
python -m unittest tests/test_web_search.py

# Run evaluation benchmark harness
python evaluation/evaluator.py
```

---

## 🔒 Security Compliance

- **SSRF Defense**: `security/url_guard.py` resolves domain DNS records and strictly rejects loopback, private RFC1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local cloud metadata endpoints (`169.254.169.254`), and non-HTTP/HTTPS protocols.
- **Prompt Injection Defense**: Retrieved chunks are wrapped inside isolated XML boundaries (`<DOCUMENT_DATA_CONTEXT>`) with directive defanging before prompt assembly.
- **Path Traversal Defense**: Uploaded filenames are sanitized via `security/file_sanitizer.py`.
- **Atomic Persistence & Self-Healing**: Session files are written atomically (`.tmp` + `os.replace`) to eliminate truncated files. Corrupt sessions are quarantined automatically without crashing the app.

---

## 📄 License
MIT License. Built with ❤️ by Pavan B C.
