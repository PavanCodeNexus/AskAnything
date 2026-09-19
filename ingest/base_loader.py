"""Base loader and normalized document schema for AskAnything.

All input sources (PDF, Web URL, YouTube, Image/OCR, Text Notes) inherit from BaseLoader
and produce normalized IngestedDocument and DocumentChunk representations.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Document processing lifecycle states."""
    UPLOADED = "Uploaded"
    VALIDATING = "Validating"
    EXTRACTING = "Extracting"
    CLEANING = "Cleaning"
    CHUNKING = "Chunking"
    EMBEDDING = "Embedding"
    INDEXING = "Indexing"
    READY = "Ready"
    FAILED = "Failed"


class DocumentChunk(BaseModel):
    """Normalized chunk schema stored across all modalities."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    session_id: str
    source_type: str  # "pdf" | "url" | "youtube" | "image" | "notes"
    filename: str = ""
    title: str = ""
    url: str = ""
    page_number: Optional[int] = None
    timestamp: Optional[str] = None
    section: str = ""
    text: str
    embedding: Optional[List[float]] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Converts chunk attributes to a flat dictionary suitable for ChromaDB metadata storage.

        ChromaDB requires metadata values to be primitives (str, int, float, bool).
        """
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "session_id": str(self.session_id),
            "source_type": str(self.source_type),
            "filename": str(self.filename or ""),
            "title": str(self.title or ""),
            "url": str(self.url or ""),
            "page_number": int(self.page_number) if self.page_number is not None else -1,
            "timestamp": str(self.timestamp or ""),
            "section": str(self.section or ""),
        }


class IngestedDocument(BaseModel):
    """Normalized representation of a fully ingested document before chunking."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    source_type: str  # "pdf" | "url" | "youtube" | "image" | "notes"
    filename: str = ""
    title: str = ""
    url: str = ""
    status: ProcessingStatus = ProcessingStatus.UPLOADED
    status_message: str = ""
    total_pages: Optional[int] = None
    total_characters: int = 0
    raw_text: str = ""
    # Elements hold page-level or section-level granular items:
    # [{"text": "...", "page_number": 1, "section": "...", "timestamp": "..."}]
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None


class BaseLoader(ABC):
    """Abstract base class for all heterogeneous document loaders."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    @abstractmethod
    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Loads and normalizes the source document.

        Args:
            source: Path, URL, file-like object, or raw payload.
            status_callback: Optional callback to notify lifecycle transitions.
            **kwargs: Loader-specific parameters.

        Returns:
            Normalized IngestedDocument.
        """
        pass
