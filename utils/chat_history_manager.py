"""Chat history and shared session persistence manager for AskAnything."""
from datetime import datetime
import json
import os
import shutil
from typing import Any, Dict, List, Optional
import uuid

from utils.logger import setup_logger

logger = setup_logger("chat_history_manager")

BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_storage")
SESSIONS_DIR = os.path.join(BASE_STORAGE_DIR, "sessions")
SHARED_DIR = os.path.join(BASE_STORAGE_DIR, "shared")
INDEX_FILE = os.path.join(BASE_STORAGE_DIR, "sessions_index.json")


def _ensure_dirs() -> None:
    """Ensures storage directories exist."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(SHARED_DIR, exist_ok=True)


def _load_index() -> Dict[str, Dict[str, Any]]:
    """Loads sessions metadata index."""
    _ensure_dirs()
    if not os.path.exists(INDEX_FILE):
        return {}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading sessions index: %s", e)
        return {}


def _save_index(index_data: Dict[str, Dict[str, Any]]) -> None:
    """Saves sessions metadata index."""
    _ensure_dirs()
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save sessions index: %s", e)


class ChatHistoryManager:
    """Manages chat session lifecycle, persistence, and shareable links."""

    @staticmethod
    def list_all_sessions() -> List[Dict[str, Any]]:
        """Returns all stored sessions sorted by updated_at descending."""
        index = _load_index()
        sessions = list(index.values())
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    @staticmethod
    def load_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session's full chat history and documents from disk."""
        _ensure_dirs()
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    @staticmethod
    def save_session(
        session_id: str,
        title: str,
        chat_history: List[Dict[str, Any]],
        documents: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves or updates a chat session on disk and updates the index."""
        _ensure_dirs()
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")
        index = _load_index()

        existing_meta = index.get(session_id, {})
        created_at = existing_meta.get("created_at", now_iso)

        # Derive auto-title from first user message if title is default
        auto_title = title
        if (not title or title.startswith("Session") or title == "Default Workspace") and chat_history:
            first_user_msg = next((m["content"] for m in chat_history if m.get("role") == "user"), None)
            if first_user_msg:
                auto_title = first_user_msg.strip()[:35] + ("..." if len(first_user_msg.strip()) > 35 else "")

        # Snippet preview from latest assistant message
        preview = ""
        if chat_history:
            last_msg = chat_history[-1]
            content = last_msg.get("content", "")
            preview = (content[:60] + "...") if len(content) > 60 else content

        # Save session payload
        payload = {
            "session_id": session_id,
            "title": auto_title,
            "created_at": created_at,
            "updated_at": now_iso,
            "message_count": len(chat_history),
            "chat_history": chat_history,
            "documents": documents or {},
        }

        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to write session file for %s: %s", session_id, e)
            return

        # Update index
        index[session_id] = {
            "session_id": session_id,
            "title": auto_title,
            "created_at": created_at,
            "updated_at": now_iso,
            "message_count": len(chat_history),
            "preview": preview,
        }
        _save_index(index)

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Deletes a session from disk and index."""
        _ensure_dirs()
        index = _load_index()
        if session_id in index:
            del index[session_id]
            _save_index(index)

        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except Exception as e:
                logger.error("Failed to delete session file %s: %s", session_id, e)
                return False
        return True

    @staticmethod
    def create_share_snapshot(
        session_id: str,
        title: str,
        chat_history: List[Dict[str, Any]],
        documents: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a public share snapshot and returns the unique share_id."""
        _ensure_dirs()
        share_id = f"ask_{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Strip internal memory items (e.g. raw audio bytes) to keep payload clean & fast
        clean_history = []
        for msg in chat_history:
            m_copy = {k: v for k, v in msg.items() if k != "audio_bytes"}
            clean_history.append(m_copy)

        share_data = {
            "share_id": share_id,
            "original_session_id": session_id,
            "title": title or "AskAnything Research Conversation",
            "created_at": now_iso,
            "chat_history": clean_history,
            "documents": documents or {},
        }

        share_file = os.path.join(SHARED_DIR, f"{share_id}.json")
        try:
            with open(share_file, "w", encoding="utf-8") as f:
                json.dump(share_data, f, indent=2, ensure_ascii=False)
            logger.info("Created share snapshot: %s", share_id)
            return share_id
        except Exception as e:
            logger.error("Failed to write share file: %s", e)
            return share_id

    @staticmethod
    def load_shared_session(share_id: str) -> Optional[Dict[str, Any]]:
        """Loads a shared snapshot by share_id."""
        _ensure_dirs()
        share_file = os.path.join(SHARED_DIR, f"{share_id}.json")
        if not os.path.exists(share_file):
            return None
        try:
            with open(share_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load share snapshot %s: %s", share_id, e)
            return None
