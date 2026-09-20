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


def _clean_chunk(chunk: Any) -> Dict[str, Any]:
    """Converts a DocumentChunk (Pydantic model or object) or dict into a JSON-serializable dictionary."""
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    if isinstance(chunk, dict):
        return {k: str(v) if isinstance(v, (bytes, bytearray)) else v for k, v in chunk.items()}
    return {"text": str(chunk)}


def _clean_message_for_storage(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans a single chat message before JSON persistence.

    Removes raw audio_bytes, converts DocumentChunk instances in raw_chunks into serializable dicts,
    and handles complex Pydantic models.
    """
    clean_msg: Dict[str, Any] = {}
    for k, v in msg.items():
        if k == "audio_bytes":
            # Exclude large binary audio payload from persistent JSON
            continue
        elif k == "raw_chunks" and isinstance(v, list):
            clean_msg[k] = [_clean_chunk(c) for c in v]
        elif isinstance(v, (bytes, bytearray)):
            continue
        elif hasattr(v, "model_dump"):
            clean_msg[k] = v.model_dump()
        elif hasattr(v, "dict"):
            clean_msg[k] = v.dict()
        else:
            clean_msg[k] = v
    return clean_msg


def _atomic_write_json(filepath: str, data: Any) -> None:
    """Writes data to a temporary file first, then atomically replaces target."""
    _ensure_dirs()
    temp_filepath = f"{filepath}.tmp_{uuid.uuid4().hex[:6]}"
    try:
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_filepath, filepath)
    except Exception as e:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
        raise e


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
    """Saves sessions metadata index atomically."""
    _ensure_dirs()
    try:
        _atomic_write_json(INDEX_FILE, index_data)
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
        """Loads a session's full chat history and documents from disk with auto-quarantine for corrupted files."""
        _ensure_dirs()
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load session %s: %s. Quarantining corrupted file.", session_id, e)
            try:
                corrupt_path = f"{filepath}.corrupt_{uuid.uuid4().hex[:4]}"
                shutil.move(filepath, corrupt_path)
                index = _load_index()
                if session_id in index:
                    del index[session_id]
                    _save_index(index)
            except Exception as q_err:
                logger.warning("Failed to auto-quarantine corrupted session %s: %s", filepath, q_err)
            return None

    @staticmethod
    def save_session(
        session_id: str,
        title: str,
        chat_history: List[Dict[str, Any]],
        documents: Optional[Dict[str, Any]] = None,
        pinned: Optional[bool] = None,
    ) -> None:
        """Saves or updates a chat session on disk atomically and updates the index."""
        _ensure_dirs()
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")
        index = _load_index()

        existing_meta = index.get(session_id, {})
        created_at = existing_meta.get("created_at", now_iso)
        is_pinned = existing_meta.get("pinned", False) if pinned is None else pinned

        # Derive auto-title from first user message if title is generic
        auto_title = title
        is_generic_title = (
            not title
            or title.startswith("Session")
            or title.startswith("Chat ")
            or title in {"New Conversation", "Default Workspace", "Research Workspace"}
        )
        if is_generic_title and chat_history:
            first_user_msg = next((m["content"] for m in chat_history if m.get("role") == "user"), None)
            if first_user_msg:
                clean_q = first_user_msg.strip()
                auto_title = clean_q[:35] + ("..." if len(clean_q) > 35 else "")

        # Snippet preview from latest assistant message
        preview = ""
        if chat_history:
            last_msg = chat_history[-1]
            content = last_msg.get("content", "")
            preview = (content[:60] + "...") if len(content) > 60 else content

        # Clean messages to sanitize DocumentChunk models and strip raw audio bytes
        clean_history = [_clean_message_for_storage(m) for m in chat_history]

        # Save session payload
        payload = {
            "session_id": session_id,
            "title": auto_title,
            "created_at": created_at,
            "updated_at": now_iso,
            "pinned": is_pinned,
            "message_count": len(chat_history),
            "chat_history": clean_history,
            "documents": documents or {},
        }

        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        try:
            _atomic_write_json(filepath, payload)
        except Exception as e:
            logger.error("Failed to write session file for %s: %s", session_id, e)
            return

        # Update index
        index[session_id] = {
            "session_id": session_id,
            "title": auto_title,
            "created_at": created_at,
            "updated_at": now_iso,
            "pinned": is_pinned,
            "message_count": len(chat_history),
            "preview": preview,
        }
        _save_index(index)

    @staticmethod
    def pin_session(session_id: str, pinned: Optional[bool] = None) -> bool:
        """Toggles or sets the pinned state of a session."""
        _ensure_dirs()
        index = _load_index()
        if session_id not in index:
            return False

        current = index[session_id].get("pinned", False)
        new_val = not current if pinned is None else pinned
        index[session_id]["pinned"] = new_val
        _save_index(index)

        # Update session json
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["pinned"] = new_val
                _atomic_write_json(session_file, data)
            except Exception as e:
                logger.warning("Could not persist pinned status to session file: %s", e)
        return new_val

    @staticmethod
    def rename_session(session_id: str, new_title: str) -> bool:
        """Renames a session in both the index and file."""
        _ensure_dirs()
        clean_title = new_title.strip()
        if not clean_title:
            return False

        index = _load_index()
        if session_id in index:
            index[session_id]["title"] = clean_title
            _save_index(index)

        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["title"] = clean_title
                _atomic_write_json(session_file, data)
                return True
            except Exception as e:
                logger.error("Failed to rename session file %s: %s", session_id, e)
                return False
        return False

    @staticmethod
    def group_sessions_by_recency(sessions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorizes sessions into ChatGPT/Gemini-style recency groups:
        Pinned, Today, Yesterday, Previous 7 Days, Previous 30 Days, and Older.
        """
        from datetime import date, timedelta

        pinned_group = []
        today_group = []
        yesterday_group = []
        last_7_days = []
        last_30_days = []
        older_group = []

        today = date.today()
        yesterday = today - timedelta(days=1)
        seven_days_ago = today - timedelta(days=7)
        thirty_days_ago = today - timedelta(days=30)

        for s in sessions:
            if s.get("pinned", False):
                pinned_group.append(s)
                continue

            updated_str = s.get("updated_at", "")
            try:
                s_date = datetime.strptime(updated_str.split()[0], "%Y-%m-%d").date()
            except Exception:
                s_date = today

            if s_date == today:
                today_group.append(s)
            elif s_date == yesterday:
                yesterday_group.append(s)
            elif s_date >= seven_days_ago:
                last_7_days.append(s)
            elif s_date >= thirty_days_ago:
                last_30_days.append(s)
            else:
                older_group.append(s)

        groups: Dict[str, List[Dict[str, Any]]] = {}
        if pinned_group:
            groups["Pinned"] = pinned_group
        if today_group:
            groups["Today"] = today_group
        if yesterday_group:
            groups["Yesterday"] = yesterday_group
        if last_7_days:
            groups["Previous 7 Days"] = last_7_days
        if last_30_days:
            groups["Previous 30 Days"] = last_30_days
        if older_group:
            groups["Older"] = older_group

        return groups

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

        # Sanitize chat history for sharing (strip audio bytes, convert chunk models)
        clean_history = [_clean_message_for_storage(m) for m in chat_history]

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
            _atomic_write_json(share_file, share_data)
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
