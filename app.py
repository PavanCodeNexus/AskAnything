"""AskAnything — Multimodal RAG Assistant Streamlit Application Entrypoint."""
import html
import os
import sys
from typing import Any, Dict, List, Optional
import urllib.parse
import uuid

import streamlit as st
from dotenv import load_dotenv

# Ensure local imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import features.mindmap_generator
import features.web_search_fallback
import grounding
import ingest.image_loader
import ingest.pdf_loader
import ingest.youtube_loader
import query_rewriter
import rag_engine
import utils.chat_history_manager

importlib.reload(query_rewriter)
importlib.reload(rag_engine)
importlib.reload(grounding)
importlib.reload(features.mindmap_generator)
importlib.reload(features.web_search_fallback)
importlib.reload(ingest.image_loader)
importlib.reload(ingest.pdf_loader)
importlib.reload(ingest.youtube_loader)
importlib.reload(utils.chat_history_manager)

from features.mindmap_generator import MindMapGenerator
from features.translator import LANGUAGE_CODES
from features.voice_output import VoiceOutputHandler
from ingest.base_loader import DocumentChunk, IngestedDocument, ProcessingStatus
from ingest.image_loader import ImageLoader
from ingest.notes_loader import NotesLoader
from ingest.pdf_loader import PDFLoader
from ingest.url_loader import URLLoader
from ingest.youtube_loader import YouTubeLoader
from grounding import render_citation_card
from rag_engine import RAGEngine
from utils.chat_history_manager import ChatHistoryManager
from utils.chunker import DocumentChunker
from utils.exporter import QAExporter
from utils.logger import setup_logger
from vectorstore import HybridVectorStore

load_dotenv()
logger = setup_logger("app")

# Page Configuration
st.set_page_config(
    page_title="AskAnything — Multimodal RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design system
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Caveat:wght@600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }

    /* Evidence Score Badges */
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #A16207;
        border: 1px solid #FDE047;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-web {
        background-color: #EDE9FE;
        color: #5B21B6;
        border: 1px solid #C4B5FD;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Gemini/ChatGPT-style Sources & Citation Cards */
    .citation-card {
        border-left: 3px solid #6366F1;
        background-color: rgba(99, 102, 241, 0.05);
        border-radius: 4px 8px 8px 4px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .citation-title {
        font-weight: 600;
        color: #F8FAFC;
        font-size: 0.9rem;
    }
    .citation-meta {
        color: #94A3B8;
        font-size: 0.8rem;
        margin: 2px 0 6px 0;
    }
    .citation-quote {
        margin: 6px 0 0 0;
        padding-left: 10px;
        border-left: 2px solid #818CF8;
        color: #CBD5E1;
        font-size: 0.84rem;
        font-style: italic;
    }
    .web-source-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF !important;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 6px;
        text-decoration: none;
        margin-top: 6px;
        transition: all 0.2s ease;
        border: none;
    }
    .web-source-btn:hover {
        opacity: 0.92;
        text-decoration: none;
        transform: translateY(-1px);
    }

    /* Conflict callout */
    .conflict-box {
        border: 1px solid #F59E0B;
        background-color: rgba(245, 158, 11, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }

    /* Shared chat banner */
    .shared-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid #6366F1;
        border-radius: 10px;
        padding: 12px 18px;
        color: #F8FAFC;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Sidebar Chat Action Buttons: Ensure emojis/symbols are always fully visible */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 3px !important;
        margin-bottom: 3px !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        padding: 0 !important;
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"] button {
        padding: 2px 2px !important;
        min-height: 32px !important;
        height: 32px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"] button p {
        font-size: 15px !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
    }
    .active-chat-toolbar {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 4px 6px;
        margin-top: 2px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Application Singletons
@st.cache_resource(show_spinner="Initializing Vector Store...")
def get_vector_store() -> HybridVectorStore:
    return HybridVectorStore()


vectorstore = get_vector_store()
rag_engine = RAGEngine(vectorstore=vectorstore)
chunker = DocumentChunker()
mindmap_gen = MindMapGenerator()

# Check for Shared Conversation URL param: ?share=<share_id>
share_id_param = st.query_params.get("share")
shared_data = None
is_shared_view = False

if share_id_param:
    shared_data = ChatHistoryManager.load_shared_session(share_id_param)
    if shared_data:
        is_shared_view = True

# Session State Management
if "session_id" not in st.session_state:
    # Try loading most recent past session or start new
    saved_sessions = ChatHistoryManager.list_all_sessions()
    if saved_sessions and not is_shared_view:
        latest = saved_sessions[0]
        st.session_state.session_id = latest["session_id"]
        loaded = ChatHistoryManager.load_session(latest["session_id"])
        st.session_state.chat_history = loaded.get("chat_history", []) if loaded else []
        st.session_state.documents = loaded.get("documents", {}) if loaded else {}
    else:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.chat_history = []
        st.session_state.documents = {}

if "sessions" not in st.session_state:
    st.session_state.sessions = {st.session_state.session_id: "New chat"}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "target_language" not in st.session_state:
    st.session_state.target_language = "en"

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False

if "active_mindmap_idx" not in st.session_state:
    st.session_state.active_mindmap_idx = None

if "enable_web_search" not in st.session_state:
    st.session_state.enable_web_search = True

if "renaming_session_id" not in st.session_state:
    st.session_state.renaming_session_id = None


def get_runtime_base_url() -> str:
    """Dynamically resolves the active browser base URL for sharing."""
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            headers = st.context.headers or {}
            host = headers.get("x-forwarded-host") or headers.get("host")
            if host:
                proto = headers.get("x-forwarded-proto", "https" if not ("localhost" in host or "127.0.0.1" in host) else "http")
                return f"{proto}://{host}"
    except Exception:
        pass
    return os.getenv("APP_BASE_URL", "http://localhost:8501")


def process_and_index_document(
    loader_func,
    source_payload: Any,
    doc_label: str,
    source_type: str,
    extra_kwargs: Dict[str, Any] = None,
) -> bool:
    """Processes document through the 7 lifecycle states:
    Uploaded -> Validating -> Extracting -> Cleaning -> Chunking -> Embedding -> Indexing -> Ready/Failed
    """
    extra_kwargs = extra_kwargs or {}
    progress_slot = st.sidebar.empty()
    status_text_slot = st.sidebar.empty()
    progress_bar = progress_slot.progress(0.0)

    def on_status_update(status: ProcessingStatus, msg: str):
        state_weights = {
            ProcessingStatus.VALIDATING: 0.15,
            ProcessingStatus.EXTRACTING: 0.35,
            ProcessingStatus.CLEANING: 0.50,
            ProcessingStatus.CHUNKING: 0.65,
            ProcessingStatus.EMBEDDING: 0.85,
            ProcessingStatus.INDEXING: 0.95,
            ProcessingStatus.READY: 1.0,
            ProcessingStatus.FAILED: 1.0,
        }
        val = state_weights.get(status, 0.1)
        progress_bar.progress(val)
        status_text_slot.caption(f"**{status.value}**: {msg}")

    try:
        on_status_update(ProcessingStatus.VALIDATING, f"Validating {doc_label}...")
        ingested_doc: IngestedDocument = loader_func(source_payload, status_callback=on_status_update, **extra_kwargs)

        if ingested_doc.status == ProcessingStatus.FAILED:
            st.session_state.documents[ingested_doc.document_id] = {
                "name": doc_label,
                "type": source_type,
                "status": ProcessingStatus.FAILED,
                "chunks": 0,
                "error": ingested_doc.error_message or "Extraction failed.",
            }
            status_text_slot.error(f"❌ Failed: {ingested_doc.error_message}")
            return False

        on_status_update(ProcessingStatus.CHUNKING, "Generating metadata-rich chunks...")
        chunks: List[DocumentChunk] = chunker.chunk_document(ingested_doc)

        if not chunks:
            st.session_state.documents[ingested_doc.document_id] = {
                "name": doc_label,
                "type": source_type,
                "status": ProcessingStatus.FAILED,
                "chunks": 0,
                "error": "Document produced no text chunks.",
            }
            status_text_slot.error("❌ Document produced no usable chunks.")
            return False

        on_status_update(ProcessingStatus.EMBEDDING, f"Computing embeddings for {len(chunks)} chunks...")
        vectorstore.add_chunks(st.session_state.session_id, chunks)

        on_status_update(ProcessingStatus.READY, f"Indexed {len(chunks)} chunks successfully.")
        st.session_state.documents[ingested_doc.document_id] = {
            "name": doc_label,
            "type": source_type,
            "status": ProcessingStatus.READY,
            "chunks": len(chunks),
            "error": None,
        }
        status_text_slot.success(f"✅ {doc_label} Ready ({len(chunks)} chunks)")

        # Persist session state update
        curr_title = st.session_state.sessions.get(st.session_state.session_id, "Research Workspace")
        ChatHistoryManager.save_session(
            session_id=st.session_state.session_id,
            title=curr_title,
            chat_history=st.session_state.chat_history,
            documents=st.session_state.documents,
        )
        return True

    except Exception as e:
        logger.error("Processing pipeline error: %s", e, exc_info=True)
        status_text_slot.error(f"❌ Pipeline error: {str(e)}")
        return False
    finally:
        progress_slot.empty()
        status_text_slot.empty()


# ---------------------------------------------------------
# SIDEBAR: All Chats History, Ingestion & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 Chats")

    # 1. Sleek ChatGPT/Gemini "+ New chat" Button
    if st.button("➕ New chat", use_container_width=True, key="btn_new_chat"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_id
        st.session_state.sessions[new_id] = "New chat"
        st.session_state.chat_history = []
        st.session_state.documents = {}
        st.session_state.active_mindmap_idx = None
        st.session_state.renaming_session_id = None
        if "share" in st.query_params:
            del st.query_params["share"]
        st.rerun()

    # 2. ChatGPT/Gemini-Style Chronological Chat History List
    all_sessions = ChatHistoryManager.list_all_sessions()
    if all_sessions:
        recency_groups = ChatHistoryManager.group_sessions_by_recency(all_sessions)
        for group_name, group_sessions in recency_groups.items():
            group_label = "📌 Pinned" if group_name == "Pinned" else group_name
            st.caption(f"**{group_label}**")
            for s in group_sessions:
                s_id = s["session_id"]
                s_title = s.get("title") or "New chat"
                # Normalize any legacy 'Session' or 'Chat' generic labels
                if s_title.startswith("Session") or (s_title.startswith("Chat ") and len(s_title) == 13):
                    s_title = "Untitled chat"
                s_pinned = s.get("pinned", False)
                is_active = (s_id == st.session_state.session_id and not is_shared_view)

                # Inline Rename Input
                if st.session_state.get("renaming_session_id") == s_id:
                    with st.container():
                        new_name = st.text_input(
                            "Rename chat",
                            value=s_title,
                            key=f"rename_input_{s_id}",
                            label_visibility="collapsed",
                        )
                        c_r1, c_r2 = st.columns(2)
                        with c_r1:
                            if st.button("✓ Save", key=f"save_r_{s_id}", use_container_width=True):
                                if new_name.strip():
                                    ChatHistoryManager.rename_session(s_id, new_name.strip())
                                    st.session_state.sessions[s_id] = new_name.strip()
                                st.session_state.renaming_session_id = None
                                st.rerun()
                        with c_r2:
                            if st.button("✕ Cancel", key=f"cancel_r_{s_id}", use_container_width=True):
                                st.session_state.renaming_session_id = None
                                st.rerun()
                else:
                    col_main, col_pin, col_ren, col_del = st.columns([5.2, 1.4, 1.4, 1.4])
                    with col_main:
                        prefix = "📌 " if s_pinned else ("● " if is_active else "")
                        short_title = (s_title[:18] + "...") if len(s_title) > 18 else s_title
                        btn_label = f"{prefix}{short_title}"
                        btn_type = "primary" if is_active else "secondary"
                        if st.button(
                            btn_label,
                            key=f"sess_btn_{s_id}",
                            type=btn_type,
                            use_container_width=True,
                            help=s_title,
                        ):
                            loaded = ChatHistoryManager.load_session(s_id)
                            if loaded:
                                st.session_state.session_id = s_id
                                st.session_state.sessions[s_id] = loaded.get("title", s_title)
                                st.session_state.chat_history = loaded.get("chat_history", [])
                                st.session_state.documents = loaded.get("documents", {})
                                st.session_state.active_mindmap_idx = None
                                st.session_state.renaming_session_id = None
                                if "share" in st.query_params:
                                    del st.query_params["share"]
                                st.rerun()
                    with col_pin:
                        pin_icon = "📍" if s_pinned else "📌"
                        pin_help = "Unpin chat" if s_pinned else "Pin chat to top"
                        if st.button(pin_icon, key=f"pin_{s_id}", help=pin_help):
                            ChatHistoryManager.pin_session(s_id)
                            st.rerun()
                    with col_ren:
                        if st.button("✏️", key=f"edit_{s_id}", help="Rename chat"):
                            st.session_state.renaming_session_id = s_id
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_s_{s_id}", help="Delete chat"):
                            ChatHistoryManager.delete_session(s_id)
                            if s_id == st.session_state.session_id:
                                new_id = str(uuid.uuid4())[:8]
                                st.session_state.session_id = new_id
                                st.session_state.sessions[new_id] = "New chat"
                                st.session_state.chat_history = []
                                st.session_state.documents = {}
                            st.rerun()

                    # Highlighted active chat action toolbar for quick 1-click controls
                    if is_active:
                        with st.container():
                            c_a1, c_a2, c_a3 = st.columns([1, 1, 1])
                            with c_a1:
                                pin_lbl = "📍 Unpin" if s_pinned else "📌 Pin"
                                if st.button(pin_lbl, key=f"bar_pin_{s_id}", help="Pin/Unpin", use_container_width=True):
                                    ChatHistoryManager.pin_session(s_id)
                                    st.rerun()
                            with c_a2:
                                if st.button("✏️ Rename", key=f"bar_ren_{s_id}", help="Rename this chat", use_container_width=True):
                                    st.session_state.renaming_session_id = s_id
                                    st.rerun()
                            with c_a3:
                                if st.button("🗑️ Delete", key=f"bar_del_{s_id}", help="Delete this chat", use_container_width=True):
                                    ChatHistoryManager.delete_session(s_id)
                                    new_id = str(uuid.uuid4())[:8]
                                    st.session_state.session_id = new_id
                                    st.session_state.sessions[new_id] = "New chat"
                                    st.session_state.chat_history = []
                                    st.session_state.documents = {}
                                    st.rerun()
    else:
        st.caption("No chats yet. Ask a question to start your first chat!")

    st.markdown("---")

    # 3. Document Ingestion Controls
    st.markdown("#### 📥 Ingest Sources")
    tab_pdf, tab_url, tab_yt, tab_img, tab_notes = st.tabs(["📄 PDF", "🌐 URL", "🎥 YouTube", "🖼️ Image", "📝 Notes"])

    # Tab 1: PDF Upload (Digital + Scanned)
    with tab_pdf:
        uploaded_pdf = st.file_uploader("Upload PDF (max 50MB)", type=["pdf"], key="pdf_uploader")
        if uploaded_pdf:
            st.caption(f"📄 **{uploaded_pdf.name}** ({len(uploaded_pdf.getvalue()) / 1024:.1f} KB)")
            if st.button("📥 Index PDF", key="btn_pdf", use_container_width=True):
                loader = PDFLoader(session_id=st.session_state.session_id)
                success = process_and_index_document(loader.load, uploaded_pdf, uploaded_pdf.name, "pdf")
                if success:
                    st.rerun()

    # Tab 2: URL Scraper (Form with Enter-Key submission & Auto-HTTPS)
    with tab_url:
        with st.form("url_ingest_form", clear_on_submit=False):
            url_input = st.text_input(
                "Web URL (HTTP/HTTPS)",
                placeholder="https://example.com/article",
                key="url_input",
                help="Paste any website link (e.g. example.com or https://wikipedia.org)",
            )
            submit_url = st.form_submit_button("🌐 Ingest Webpage", use_container_width=True)
            if submit_url and url_input.strip():
                clean_url = url_input.strip()
                loader = URLLoader(session_id=st.session_state.session_id)
                success = process_and_index_document(loader.load, clean_url, clean_url, "url")
                if success:
                    st.rerun()

    # Tab 3: YouTube Transcript
    with tab_yt:
        with st.form("yt_ingest_form", clear_on_submit=False):
            yt_input = st.text_input("YouTube Link", placeholder="https://youtube.com/watch?v=...", key="yt_input")
            submit_yt = st.form_submit_button("🎥 Ingest YouTube Video", use_container_width=True)
            if submit_yt and yt_input.strip():
                clean_yt = yt_input.strip()
                loader = YouTubeLoader(session_id=st.session_state.session_id)
                success = process_and_index_document(loader.load, clean_yt, clean_yt, "youtube")
                if success:
                    st.rerun()

    # Tab 4: Image OCR (Tesseract + Windows Native WinOCR)
    with tab_img:
        img_upload = st.file_uploader("Upload Image or Diagram", type=["png", "jpg", "jpeg", "webp"], key="img_uploader")
        if img_upload:
            st.image(img_upload, caption=img_upload.name, use_container_width=True)
        ocr_lang = st.selectbox("OCR Language", ["eng", "hin", "kan", "tam", "tel"], format_func=lambda x: {
            "eng": "English", "hin": "Hindi", "kan": "Kannada", "tam": "Tamil", "tel": "Telugu"
        }.get(x, x))
        if img_upload and st.button("🖼️ Index Image", key="btn_img", use_container_width=True):
            loader = ImageLoader(session_id=st.session_state.session_id)
            success = process_and_index_document(loader.load, img_upload, img_upload.name, "image", extra_kwargs={"ocr_lang": ocr_lang})
            if success:
                st.rerun()

    # Tab 5: Notes / Plain Text
    with tab_notes:
        notes_upload = st.file_uploader("Upload Text (.txt, .md)", type=["txt", "md", "csv"], key="notes_uploader")
        pasted_text = st.text_area("Or Paste Notes directly", placeholder="Paste text here...", height=100)
        if st.button("📝 Ingest Notes", key="btn_notes", use_container_width=True):
            loader = NotesLoader(session_id=st.session_state.session_id)
            if notes_upload:
                success = process_and_index_document(loader.load, notes_upload, notes_upload.name, "notes")
            elif pasted_text.strip():
                success = process_and_index_document(loader.load, pasted_text, "pasted_note.txt", "notes")
            st.rerun()

    # 4. Document Manager
    st.markdown("---")
    st.markdown("#### 📚 Document Manager")
    if not st.session_state.documents:
        st.caption("No documents in this session yet. Upload files above to index.")
    else:
        for doc_id, doc_info in list(st.session_state.documents.items()):
            col_d1, col_d2 = st.columns([4, 1])
            with col_d1:
                status_icon = "✅" if doc_info["status"] == ProcessingStatus.READY else "❌"
                st.markdown(f"**{status_icon} {doc_info['name'][:22]}**")
                st.caption(f"Type: `{doc_info['type']}` | Chunks: `{doc_info['chunks']}`")
                if doc_info.get("error"):
                    st.caption(f":red[{doc_info['error']}]")
            with col_d2:
                if st.button("🗑️", key=f"del_{doc_id}", help="Delete from session"):
                    vectorstore.delete_document(st.session_state.session_id, doc_id)
                    del st.session_state.documents[doc_id]
                    st.rerun()

    # 5. Multilingual, Voice & Web Search Settings
    st.markdown("---")
    st.markdown("#### 🌐 Language, Voice & Web Search")
    lang_labels = {
        "English": "en",
        "Hindi (हिंदी)": "hi",
        "Kannada (ಕನ್ನಡ) 🧪": "kn",
        "Telugu (తెలుగు) 🧪": "te",
        "Tamil (தமிழ்) 🧪": "ta",
    }
    selected_lang_name = st.selectbox("Output Language", list(lang_labels.keys()), index=0)
    st.session_state.target_language = lang_labels[selected_lang_name]
    st.session_state.enable_tts = st.checkbox("🔊 Voice Output (Audio TTS)", value=st.session_state.enable_tts)
    st.session_state.enable_web_search = st.toggle(
        "🌐 Auto Web Search Fallback",
        value=st.session_state.enable_web_search,
        help="If a question cannot be answered from your uploaded PDFs/documents, AskAnything automatically searches the live web and cites open-source material with clickable browser links."
    )

    # 6. Export Q&A History
    st.markdown("---")
    st.markdown("#### 💾 Export Conversation")
    if st.session_state.chat_history:
        txt_data = QAExporter.export_as_txt(st.session_state.session_id, st.session_state.chat_history)
        st.download_button(
            "⬇️ Export as TXT",
            data=txt_data,
            file_name=f"askanything_{st.session_state.session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        try:
            pdf_data = QAExporter.export_as_pdf(st.session_state.session_id, st.session_state.chat_history)
            st.download_button(
                "⬇️ Export as PDF",
                data=pdf_data,
                file_name=f"askanything_{st.session_state.session_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            logger.warning("PDF export error: %s", e)


# ---------------------------------------------------------
# MAIN AREA: Header, Shared View, Chat Stream, & Sharing
# ---------------------------------------------------------

# Shared View Handler
if share_id_param and not shared_data:
    st.error(f"⚠️ Shared conversation link '{html.escape(str(share_id_param))}' was not found or has expired.")
    if st.button("← Return to My Workspace", key="btn_return_invalid_share"):
        if "share" in st.query_params:
            del st.query_params["share"]
        st.rerun()
    active_messages = []
elif is_shared_view and shared_data:
    st.markdown(
        f"""
        <div class="shared-banner">
            <div>
                <strong>🌐 Shared Research Conversation</strong><br>
                <span>Topic: <em>"{shared_data.get('title', 'Conversation')}"</em> (Shared on {shared_data.get('created_at', '')})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_sh1, col_sh2 = st.columns([3, 1])
    with col_sh1:
        if st.button("🚀 Clone to My Workspace (Continue Asking)", use_container_width=True):
            new_id = str(uuid.uuid4())[:8]
            st.session_state.session_id = new_id
            st.session_state.sessions[new_id] = shared_data.get("title", "Cloned Chat")
            st.session_state.chat_history = shared_data.get("chat_history", [])
            st.session_state.documents = shared_data.get("documents", {})
            st.session_state.active_mindmap_idx = None
            ChatHistoryManager.save_session(
                session_id=new_id,
                title=shared_data.get("title", "Cloned Chat"),
                chat_history=st.session_state.chat_history,
                documents=st.session_state.documents,
            )
            if "share" in st.query_params:
                del st.query_params["share"]
            st.rerun()
    with col_sh2:
        if st.button("← Back to My Workspace", use_container_width=True):
            if "share" in st.query_params:
                del st.query_params["share"]
            st.rerun()

    active_messages = shared_data.get("chat_history", [])
else:
    # Normal Active Workspace Header
    col_header, col_share_action = st.columns([5, 2])
    with col_header:
        st.markdown('<div class="main-header">⚡ AskAnything</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Multimodal RAG Assistant with Multi-Signal Evidence Scoring & Conflict Detection</div>', unsafe_allow_html=True)

    with col_share_action:
        st.write("")  # Spacing
        with st.popover("🔗 Share Chat with Friends", use_container_width=True):
            if st.session_state.chat_history:
                curr_title = st.session_state.sessions.get(st.session_state.session_id, "Research Conversation")
                share_id = ChatHistoryManager.create_share_snapshot(
                    session_id=st.session_state.session_id,
                    title=curr_title,
                    chat_history=st.session_state.chat_history,
                    documents=st.session_state.documents,
                )
                runtime_base = get_runtime_base_url()
                share_url = f"{runtime_base}/?share={share_id}"

                st.markdown("### 🔗 Share this Conversation")

                # 1-Click Client-Side Clipboard Copy Button
                copy_html = f"""
                <button id="copy-share-btn" onclick="
                    const urlToCopy = (window.location.origin && !window.location.origin.includes('undefined')) 
                        ? (window.location.origin + window.location.pathname + '?share={share_id}') 
                        : '{share_url}';
                    navigator.clipboard.writeText(urlToCopy).then(() => {{
                        const b = document.getElementById('copy-share-btn');
                        b.innerText = '✅ Link Copied to Clipboard!';
                        b.style.backgroundColor = '#16A34A';
                        setTimeout(() => {{
                            b.innerText = '📋 Copy Share Link';
                            b.style.backgroundColor = '#6366F1';
                        }}, 2500);
                    }}).catch(() => {{
                        prompt('Copy link:', urlToCopy);
                    }});
                " style="
                    width: 100%;
                    background-color: #6366F1;
                    color: white;
                    border: none;
                    padding: 10px 14px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 14px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    margin-bottom: 10px;
                    font-family: inherit;
                    transition: background-color 0.2s ease;
                ">
                    📋 Copy Share Link
                </button>
                """
                st.components.v1.html(copy_html, height=52)

                st.text_input("Direct Public Link:", value=share_url, key="share_url_input")

                # WhatsApp Share Link
                wa_message = f"Check out this research conversation on AskAnything: '{curr_title}'\n\nLink: {share_url}"
                wa_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(wa_message)}"

                st.markdown(
                    f"""
                    <a href="{wa_url}" target="_blank" style="text-decoration: none;">
                        <button style="
                            width: 100%;
                            background-color: #25D366;
                            color: white;
                            border: none;
                            padding: 10px 14px;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 14px;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                            margin-bottom: 8px;
                        ">
                            🟢 Share on WhatsApp
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

                # Instagram Share Link
                ig_url = "https://www.instagram.com/direct/inbox/"
                st.markdown(
                    f"""
                    <a href="{ig_url}" target="_blank" style="text-decoration: none;">
                        <button style="
                            width: 100%;
                            background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
                            color: white;
                            border: none;
                            padding: 10px 14px;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 14px;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                        ">
                            📷 Send in Instagram DM / Stories
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("💡 The link is copied to your clipboard. Send it directly to friends or attach to your story!")
            else:
                st.info("Start a conversation first by asking a question below to create a shareable link!")

    active_messages = st.session_state.chat_history

# Warning if no Groq API Key
if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
    st.warning("⚠️ **GROQ_API_KEY is not set in `.env`**. LLM answers, query rewriting, and conflict detection will use heuristic fallbacks. Add your free key in `.env` to enable Llama 3.3 70B!")

# Render Chat History
for msg_idx, msg in enumerate(active_messages):
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        # Display Assistant Intelligence Badges
        if role == "assistant":
            # 0. Live Web Search / Combined Sources Badge
            if msg.get("is_combined_search"):
                st.markdown(
                    '<div class="badge-web">🌐 Combined Sources: Uploaded Docs + Live Web Search</div>',
                    unsafe_allow_html=True,
                )
            elif msg.get("is_web_search"):
                st.markdown(
                    '<div class="badge-web">🌐 Live Web Search Fallback (Open Source Material)</div>',
                    unsafe_allow_html=True,
                )

            # 1. Evidence Score Badge
            evidence = msg.get("evidence_score")
            if evidence and isinstance(evidence, dict):
                score = evidence.get("score", 0)
                tier = evidence.get("tier", "Low")
                badge_class = f"badge-{tier.lower()}"
                icon = "🟢" if tier == "High" else ("🟡" if tier == "Medium" else "🔴")
                st.markdown(
                    f'<div class="{badge_class}">{icon} {evidence.get("label", "")}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("📊 Evidence Score Breakdown", expanded=False):
                    st.caption(evidence.get("details", ""))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Retrieval", f"{evidence.get('retrieval_component', 0)}%")
                    c2.metric("Grounding", f"{evidence.get('grounding_component', 0)}%")
                    c3.metric("Coverage", f"{evidence.get('source_coverage_component', 0)}%")
                    c4.metric("Conflict Penalty", f"-{evidence.get('conflict_penalty', 0)}%")

            # 2. Conflict Warning Panel
            conflict = msg.get("conflict_data")
            if conflict and isinstance(conflict, dict):
                c_status = conflict.get("overall_classification", "")
                if c_status in {"Contradiction", "Different Estimate"}:
                    alert_icon = "⚠️" if c_status == "Contradiction" else "📊"
                    st.markdown(
                        f"""
                        <div class="conflict-box">
                            <strong>{alert_icon} Conflict Detector Notice ({c_status})</strong><br>
                            {conflict.get('summary', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔍 Inspect Conflicting Claims", expanded=False):
                        for comp in conflict.get("comparisons", []):
                            st.write(f"**Topic:** {comp.get('topic')}")
                            st.write(f"• **{comp.get('source_a', {}).get('source_name')}:** {comp.get('source_a', {}).get('claim')}")
                            st.write(f"• **{comp.get('source_b', {}).get('source_name')}:** {comp.get('source_b', {}).get('claim')}")
                            st.caption(f"*Rationale:* {comp.get('explanation')}")

            # 3. Verified Citations (Hidden by default like ChatGPT & Gemini — click to expand)
            citations = msg.get("citations", [])
            if citations:
                source_count = len(citations)
                source_label = f"📚 Sources ({source_count})" if source_count > 1 else "📚 1 Source"
                with st.expander(source_label, expanded=False):
                    for cit in citations:
                        st.markdown(render_citation_card(cit), unsafe_allow_html=True)

            # 4. Unique Handwritten Mind Map & Clear Toggle
            raw_chunks = msg.get("raw_chunks", [])
            if raw_chunks:
                is_mm_open = (st.session_state.active_mindmap_idx == msg_idx)
                col_mm_btn, col_mm_clear = st.columns([3, 1])

                with col_mm_btn:
                    if not is_mm_open:
                        if st.button("🧠 View Handwritten Mind Map", key=f"open_mm_{msg_idx}"):
                            st.session_state.active_mindmap_idx = msg_idx
                            st.rerun()
                    else:
                        if st.button("✕ Close / Clear Mind Map", key=f"close_mm_{msg_idx}"):
                            st.session_state.active_mindmap_idx = None
                            st.rerun()

                if is_mm_open:
                    with st.spinner("Rendering handwritten chalkboard concept map..."):
                        graph_html = mindmap_gen.generate_html(
                            query=msg.get("query_asked", "Topic"),
                            answer=msg.get("content", ""),
                            chunks=raw_chunks,
                            height="620px",
                        )
                        st.components.v1.html(graph_html, height=640, scrolling=False)

            # 5. Audio Player for Voice Output
            audio_bytes = msg.get("audio_bytes")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")


# ---------------------------------------------------------
# INPUT SECTION: Clean Text Chat with Quick File/Image Attachment
# ---------------------------------------------------------
user_query = ""

# Only show chat input when in interactive mode (not read-only shared view)
if not is_shared_view:
    with st.expander("📎 Attach Image or Document to this chat (PNG, JPG, PDF, TXT)", expanded=False):
        c_att1, c_att2 = st.columns([4, 1])
        with c_att1:
            chat_att = st.file_uploader(
                "Upload Image or Document",
                type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"],
                key="chat_inline_attachment",
                label_visibility="collapsed",
            )
        with c_att2:
            if chat_att and st.button("📥 Attach", key="btn_chat_inline_index", use_container_width=True):
                ext = os.path.splitext(chat_att.name.lower())[1]
                if ext == ".pdf":
                    ldr = PDFLoader(session_id=st.session_state.session_id)
                    process_and_index_document(ldr.load, chat_att, chat_att.name, "pdf")
                elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
                    ldr = ImageLoader(session_id=st.session_state.session_id)
                    process_and_index_document(ldr.load, chat_att, chat_att.name, "image")
                else:
                    ldr = NotesLoader(session_id=st.session_state.session_id)
                    process_and_index_document(ldr.load, chat_att, chat_att.name, "notes")
                st.rerun()

    text_query = st.chat_input("Ask anything across your uploaded PDFs, URLs, videos, images, and notes...")
    if text_query:
        user_query = text_query

# Handle Submission
if user_query and not is_shared_view:
    # Append User Message
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Grounded RAG Response
    with st.chat_message("assistant"):
        with st.spinner("Searching sources, verifying claims & synthesizing answer..."):
            hist_context = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]
            ]

            active_docs = [
                d.get("name", "")
                for d in st.session_state.documents.values()
                if d.get("status") == ProcessingStatus.READY
            ]

            rag_result = rag_engine.answer_query(
                session_id=st.session_state.session_id,
                user_query=user_query,
                chat_history=hist_context,
                top_k=5,
                target_language=st.session_state.target_language,
                active_documents=active_docs,
                enable_web_fallback=st.session_state.enable_web_search,
            )

            answer = rag_result["answer"]
            evidence_score = rag_result["evidence_score"]
            citations = rag_result["citations"]
            conflict_data = rag_result["conflict_data"]
            raw_chunks = rag_result.get("raw_chunks", [])
            is_insufficient = rag_result.get("is_insufficient_evidence", False)
            is_web_search = rag_result.get("is_web_search", False)
            is_combined_search = rag_result.get("is_combined_search", False)

            st.markdown(answer)

            # Generate Speech if TTS is enabled
            audio_bytes = None
            if st.session_state.enable_tts and not is_insufficient:
                with st.spinner("Synthesizing speech..."):
                    audio_bytes, tts_err = VoiceOutputHandler.synthesize_speech(
                        text=answer,
                        language_code=st.session_state.target_language,
                    )
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

            # Intelligence Badges
            if is_combined_search:
                st.markdown(
                    '<div class="badge-web">🌐 Combined Sources: Uploaded Docs + Live Web Search</div>',
                    unsafe_allow_html=True,
                )
            elif is_web_search:
                st.markdown(
                    '<div class="badge-web">🌐 Live Web Search Fallback (Open Source Material)</div>',
                    unsafe_allow_html=True,
                )

            # Evidence Score Badge
            if evidence_score:
                tier = evidence_score.tier
                badge_class = f"badge-{tier.lower()}"
                icon = "🟢" if tier == "High" else ("🟡" if tier == "Medium" else "🔴")
                st.markdown(
                    f'<div class="{badge_class}">{icon} {evidence_score.label}</div>',
                    unsafe_allow_html=True,
                )

            # Citations (Hidden by default like ChatGPT & Gemini — click to expand)
            if citations:
                source_count = len(citations)
                source_label = f"📚 Sources ({source_count})" if source_count > 1 else "📚 1 Source"
                with st.expander(source_label, expanded=False):
                    for cit in citations:
                        st.markdown(render_citation_card(cit), unsafe_allow_html=True)

            ev_dict = None
            if evidence_score:
                ev_dict = {
                    "score": evidence_score.score,
                    "tier": evidence_score.tier,
                    "label": evidence_score.label,
                    "details": evidence_score.details,
                    "retrieval_component": evidence_score.retrieval_component,
                    "grounding_component": evidence_score.grounding_component,
                    "source_coverage_component": evidence_score.source_coverage_component,
                    "conflict_penalty": evidence_score.conflict_penalty,
                }

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "evidence_score": ev_dict,
                "citations": citations,
                "conflict_data": conflict_data,
                "raw_chunks": raw_chunks,
                "query_asked": user_query,
                "audio_bytes": audio_bytes,
                "is_web_search": is_web_search,
                "is_combined_search": is_combined_search,
            })

            # Auto-save session permanently to disk with meaningful title
            current_title = st.session_state.sessions.get(st.session_state.session_id, "")
            is_default = (
                not current_title
                or current_title in {"New chat", "New Conversation", "Default Workspace", "Research Workspace"}
                or current_title.startswith("Session")
                or (current_title.startswith("Chat ") and len(current_title) == 13)
            )
            if is_default:
                clean_q = user_query.strip()
                current_title = clean_q[:32] + ("..." if len(clean_q) > 32 else "")
                st.session_state.sessions[st.session_state.session_id] = current_title

            ChatHistoryManager.save_session(
                session_id=st.session_state.session_id,
                title=current_title,
                chat_history=st.session_state.chat_history,
                documents=st.session_state.documents,
            )
