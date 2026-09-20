"""Text chunking utility maintaining normalized metadata across all source types."""
from typing import List, Optional
import uuid

class RecursiveCharacterTextSplitter:
    """Standalone, robust recursive character text splitter with multi-separator hierarchy."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
        length_function=len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
        self.length_function = length_function

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        _separator = separator

        for s in splits:
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size])
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return [c.strip() for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self.length_function(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                    while total > self.chunk_overlap or (total + _len + len(separator) > self.chunk_size and total > 0):
                        popped = current_doc.pop(0)
                        total -= self.length_function(popped) + (len(separator) if current_doc else 0)
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
        return docs

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._split_text(text, self.separators)

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
