# ⚡ AskAnything — Multimodal RAG Assistant

AskAnything is an enterprise-grade Multimodal Retrieval-Augmented Generation (RAG) assistant that unifies heterogeneous source ingestion, hybrid vector + keyword retrieval, cross-encoder reranking, claim-level knowledge conflict detection, multi-signal evidence scoring, multilingual translation, voice chat (STT/TTS), and dynamic mind map visualization into a single unified workflow.

---

## 🌟 Key Features & Differentiators

| # | Feature | Capability |
|---|---|---|
| 1 | **Heterogeneous Ingestion** | Simultaneously upload & cross-reference PDFs, web URLs, YouTube video transcripts, images/diagrams via OCR, and plain text notes in one session. |
| 2 | **Multi-Signal Evidence Score** | **Not** raw cosine similarity. Evaluates retrieval relevance (40%), answer grounding (35%), multi-source corroboration (25%), and applies penalties for contradictions. |
| 3 | **Claim-Level Conflict Detector** | Dissects multi-source claims and categorizes them into **Agreement**, **Direct Contradiction**, **Different Numerical Estimate** (avoids false-alarm alerts on metric variances), or **Insufficient Evidence**. |
| 4 | **Defensive RAG Pipeline** | Strict grounding verification. If evidence is lacking, explicitly responds with *"The uploaded sources do not contain enough information to answer this question."* — eliminating hallucinations. |
| 5 | **Multilingual Voice & Text** | Native support for **English, Hindi, Kannada, Telugu, and Tamil** with technical terminology preservation and experimental regional voice STT/TTS. |
| 6 | **Interactive Knowledge Graph** | User-triggered interactive mind maps rendered with NetworkX and PyVis directly inside the Streamlit canvas. |
| 7 | **Enterprise Security** | SSRF protection (private IP / cloud metadata block), file format & size validation, HTML sanitization, and prompt injection isolation wrappers. |
| 8 | **Document Lifecycle Stepper** | Visual progress indicators showing: `Uploaded ➔ Validating ➔ Extracting ➔ Cleaning ➔ Chunking ➔ Embedding ➔ Indexing ➔ Ready`. |

---

## 🏗️ System Architecture

```
User Query (Text or Voice)
         ↓
  Language Detection & Normalization (EN, HI, KN, TE, TA)
         ↓
  Conversational Query Rewriter (Pronoun & follow-up resolution via Llama 3.3 70B)
         ↓
  Hybrid Retrieval (ChromaDB Vector + BM25 Keyword Search with RRF)
         ↓
  Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
         ↓
  Relevance Threshold Filter (> 0.30)
         ↓
  Claim-Level Conflict Detector (Agreement / Contradiction / Different Estimate)
         ↓
  Prompt Injection Defense Isolation Wrapper
         ↓
  LLM Answer Generation (Groq Cloud — Llama 3.3 70B)
         ↓
  Grounding Verification & Verified Citation Validation
         ↓
  Multi-Signal Evidence Score Calculation (High / Medium / Low 0–100)
         ↓
  Multilingual Output Translation & gTTS Audio Output
```

---

## 📂 File Structure

```
askanything/
├── app.py                        # Streamlit interactive UI entrypoint
├── rag_engine.py                 # Core RAG orchestration pipeline
├── vectorstore.py                # ChromaDB vector store + BM25 hybrid index
├── query_rewriter.py             # Conversational query rewriting
├── grounding.py                  # Grounding verification & citation validation
├── ingest/
│   ├── __init__.py
│   ├── base_loader.py            # Normalized schema (DocumentChunk & IngestedDocument)
│   ├── pdf_loader.py             # PyMuPDF extraction with page & section metadata
│   ├── url_loader.py             # Web scraping with SSRF guard & HTML cleaner
│   ├── youtube_loader.py         # YouTube transcript loader with time windowing
│   ├── image_loader.py           # Tesseract OCR & Pillow image text extractor
│   └── notes_loader.py           # Plain text (.txt, .md) loader
├── features/
│   ├── __init__.py
│   ├── conflict_detector.py      # Claim extraction & conflict classification
│   ├── translator.py             # Multilingual I/O (EN, HI, KN, TE, TA)
│   ├── voice_input.py            # Speech-to-text (SpeechRecognition)
│   ├── voice_output.py           # Text-to-speech (gTTS)
│   ├── mindmap_generator.py      # Concept graph generator (NetworkX + PyVis)
│   └── evidence_score.py         # Multi-signal evidence scoring algorithm
├── utils/
│   ├── __init__.py
│   ├── chunker.py                # Recursive chunker preserving all metadata
│   ├── exporter.py               # Q&A session export (PDF via fpdf2 & TXT)
│   ├── validator.py              # File size, extension, MIME validation
│   └── logger.py                 # Structured logger
├── security/
│   ├── __init__.py
│   ├── file_sanitizer.py         # Path traversal defense & prompt injection wrapper
│   └── url_guard.py              # SSRF protection (private CIDR & metadata blocking)
├── evaluation/
│   ├── __init__.py
│   ├── test_dataset.json         # Gold-standard evaluation Q&A pairs
│   └── evaluator.py              # Recall@K, Precision@K, MRR, Faithfulness runner
├── Dockerfile                    # Production Docker deployment
├── docker-compose.yml            # Docker compose configuration
├── requirements.txt              # Production dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git exclusions
└── README.md                     # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/askanything.git
cd askanything

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your **Groq API Key**:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(Get your free Groq API key at [console.groq.com/keys](https://console.groq.com/keys))*.

### 4. (Optional) Install Tesseract OCR for Image Ingestion
- **Windows**: Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-kan tesseract-ocr-tam tesseract-ocr-tel
  ```
- **macOS**: `brew install tesseract`

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To run AskAnything as a container with pre-installed Tesseract multilingual packages and audio libraries:

```bash
docker compose up --build -d
```
Access the application at `http://localhost:8501`.

---

## 🧪 Running Evaluation Benchmarks

To run the gold-standard evaluation harness:

```bash
python evaluation/evaluator.py
```

This assesses:
- **Retrieval Quality**: Precision@5, Recall@5, Mean Reciprocal Rank (MRR)
- **Answer Quality**: Keyword Coverage, Faithfulness / Grounding Ratio
- **Refusal Accuracy**: Proper trigger of the insufficient evidence fallback on out-of-domain queries.

---

## 🔒 Security Compliance

- **SSRF Protection**: `security/url_guard.py` resolves domain DNS records and strictly rejects loopback, private RFC1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local metadata endpoints (`169.254.169.254`), and non-HTTP/HTTPS protocols.
- **Prompt Injection Isolation**: Retrieved chunks are wrapped inside isolated XML tags (`<DOCUMENT_DATA_CONTEXT>`) with directive defanging before prompt assembly.
- **Path Traversal Defense**: All uploaded filenames are sanitized via `security/file_sanitizer.py`.
- **Credential Hygiene**: API keys are strictly loaded from `.env` and excluded from git via `.gitignore`.

---

## 📄 License
MIT License. Built by Pavan B C.
