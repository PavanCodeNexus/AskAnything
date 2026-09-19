"""Text chunking utility maintaining normalized metadata across all source types."""
from typing import List, Optional
import uuid

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:  # type: ignore
            """Pure-Python fallback splitter when LangChain is not installed."""
            def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120, separators=None, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk = text[start:end]
                    chunks.append(chunk)
                    if end >= len(text):
                        break
                    start += self.chunk_size - self.chunk_overlap
                return chunks

from ingest.base_loader import DocumentChunk, IngestedDocument
from utils.logger import setup_logger

logger = setup_logger("chunker")

# Optimal chunking hyperparameters for retrieval balance
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


class DocumentChunker:
    """Splits normalized IngestedDocuments into metadata-rich DocumentChunks."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            length_function=len,
        )

    def chunk_document(self, document: IngestedDocument) -> List[DocumentChunk]:
        """Chunks an IngestedDocument based on its granular elements (pages/sections).

        Args:
            document: Normalized IngestedDocument object.

        Returns:
            List of DocumentChunk instances with preserved metadata.
        """
        chunks: List[DocumentChunk] = []

        # If document has granular elements (e.g. per-page PDF or per-timestamp YouTube), chunk per element
        if document.elements:
            for element in document.elements:
                element_text = element.get("text", "").strip()
                if not element_text:
                    continue

                page_num = element.get("page_number")
                section = element.get("section", "")
                timestamp = element.get("timestamp")

                split_texts = self.splitter.split_text(element_text)
                for split_idx, split_text in enumerate(split_texts):
                    if not split_text.strip():
                        continue

                    chunk_sec = section
                    if len(split_texts) > 1:
                        chunk_sec = f"{section} (Part {split_idx + 1})" if section else f"Part {split_idx + 1}"

                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document.document_id,
                        session_id=document.session_id,
                        source_type=document.source_type,
                        filename=document.filename,
                        title=document.title,
                        url=document.url,
                        page_number=page_num,
                        timestamp=timestamp,
                        section=chunk_sec,
                        text=split_text.strip(),
                    )
                    chunks.append(chunk)
        else:
            # Fallback to splitting raw_text directly
            raw = document.raw_text.strip()
            if raw:
                split_texts = self.splitter.split_text(raw)
                for split_idx, split_text in enumerate(split_texts):
                    if not split_text.strip():
                        continue
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document.document_id,
                        session_id=document.session_id,
                        source_type=document.source_type,
                        filename=document.filename,
                        title=document.title,
                        url=document.url,
                        page_number=None,
                        timestamp=None,
                        section=f"Section {split_idx + 1}",
                        text=split_text.strip(),
                    )
                    chunks.append(chunk)

        logger.info(
            "Document '%s' (%s) partitioned into %d chunks.",
            document.filename or document.title or document.document_id,
            document.source_type,
            len(chunks),
        )
        return chunks
