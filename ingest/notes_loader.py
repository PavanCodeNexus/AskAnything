"""Notes and plain text loader (.txt, .md, .csv) with UTF-8 encoding support."""
import os
from typing import Any, Callable, Dict, List, Optional
import uuid

from ingest.base_loader import BaseLoader, IngestedDocument, ProcessingStatus
from security.file_sanitizer import sanitize_filename, sanitize_text_content
from utils.logger import setup_logger
from utils.validator import MAX_TEXT_SIZE_BYTES, validate_file

logger = setup_logger("notes_loader")


class NotesLoader(BaseLoader):
    """Loads plain text, markdown, and code notes with section/line tracking."""

    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Parses a text file into normalized IngestedDocument.

        Args:
            source: File path (str), UploadedFile stream, or raw text string.
            status_callback: Optional lifecycle update callback.

        Returns:
            Normalized IngestedDocument.
        """
        doc_id = str(uuid.uuid4())
        filename = "note.txt"
        file_bytes: Optional[bytes] = None

        def update_status(status: ProcessingStatus, msg: str) -> None:
            if status_callback:
                status_callback(status, msg)

        update_status(ProcessingStatus.VALIDATING, "Validating notes file...")

        if isinstance(source, str):
            if os.path.exists(source):
                filename = os.path.basename(source)
                with open(source, "rb") as f:
                    file_bytes = f.read()
            else:
                # Raw text passed directly
                file_bytes = source.encode("utf-8")
                filename = kwargs.get("filename", "pasted_notes.txt")
        elif hasattr(source, "read") and hasattr(source, "name"):
            filename = source.name
            file_bytes = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        elif isinstance(source, bytes):
            file_bytes = source

        safe_filename = sanitize_filename(filename)

        if not file_bytes:
            update_status(ProcessingStatus.FAILED, "Notes content is empty.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="notes",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message="Notes file is empty.",
            )

        val_result = validate_file(safe_filename, len(file_bytes), file_bytes)
        if not val_result.is_valid:
            update_status(ProcessingStatus.FAILED, val_result.error_message or "Validation failed.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="notes",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=val_result.error_message,
            )

        update_status(ProcessingStatus.EXTRACTING, f"Decoding text for {safe_filename}...")

        # Robust decoding strategy
        raw_text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                raw_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not raw_text:
            update_status(ProcessingStatus.FAILED, "Could not decode text with standard encodings.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="notes",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message="Unable to decode file content as text.",
            )

        update_status(ProcessingStatus.CLEANING, "Sanitizing text content...")
        clean_text = sanitize_text_content(raw_text)

        lines = clean_text.splitlines()
        elements: List[Dict[str, Any]] = []

        # Break text into logical paragraphs or blocks of lines for citation tracking
        line_buffer: List[str] = []
        start_line = 1

        for idx, line in enumerate(lines, start=1):
            if line.strip():
                line_buffer.append(line)
            elif line_buffer:
                section_text = "\n".join(line_buffer)
                end_line = idx - 1
                elements.append({
                    "page_number": None,
                    "section": f"Lines {start_line}-{end_line}",
                    "text": section_text,
                    "timestamp": None,
                })
                line_buffer = []
                start_line = idx + 1

        if line_buffer:
            elements.append({
                "page_number": None,
                "section": f"Lines {start_line}-{len(lines)}",
                "text": "\n".join(line_buffer),
                "timestamp": None,
            })

        title = os.path.splitext(safe_filename)[0].replace("_", " ").title()

        return IngestedDocument(
            document_id=doc_id,
            session_id=self.session_id,
            source_type="notes",
            filename=safe_filename,
            title=title,
            total_pages=1,
            total_characters=len(clean_text),
            raw_text=clean_text,
            elements=elements if elements else [{"page_number": None, "section": "Full Note", "text": clean_text, "timestamp": None}],
            status=ProcessingStatus.READY,
            status_message=f"Processed notes: {len(clean_text)} characters, {len(lines)} lines.",
        )
