"""PDF loader implementation using PyMuPDF (fitz) with page and section extraction."""
import os
from typing import Any, Callable, Dict, List, Optional
import uuid

import fitz  # PyMuPDF

from ingest.base_loader import BaseLoader, IngestedDocument, ProcessingStatus
from security.file_sanitizer import sanitize_filename, sanitize_text_content
from utils.logger import setup_logger
from utils.validator import MAX_PDF_SIZE_BYTES, validate_file

logger = setup_logger("pdf_loader")


class PDFLoader(BaseLoader):
    """Extracts text, page numbers, and structural sections from PDF documents."""

    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Parses a PDF file into normalized IngestedDocument with per-page elements.

        Args:
            source: File path (str) or bytes-like object / Streamlit UploadedFile.
            status_callback: Optional lifecycle update callback.

        Returns:
            IngestedDocument populated with page-by-page text.
        """
        doc_id = str(uuid.uuid4())
        filename = "document.pdf"
        file_bytes: Optional[bytes] = None

        def update_status(status: ProcessingStatus, msg: str) -> None:
            if status_callback:
                status_callback(status, msg)

        # 1. Uploaded / Validating
        update_status(ProcessingStatus.VALIDATING, "Validating PDF format and size...")

        if isinstance(source, str):
            filename = os.path.basename(source)
            if not os.path.exists(source):
                return IngestedDocument(
                    document_id=doc_id,
                    session_id=self.session_id,
                    source_type="pdf",
                    filename=sanitize_filename(filename),
                    status=ProcessingStatus.FAILED,
                    error_message=f"File not found: {source}",
                )
            with open(source, "rb") as f:
                file_bytes = f.read()
        elif hasattr(source, "read") and hasattr(source, "name"):
            filename = source.name
            file_bytes = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        elif isinstance(source, bytes):
            file_bytes = source

        safe_filename = sanitize_filename(filename)

        if not file_bytes:
            update_status(ProcessingStatus.FAILED, "PDF content is empty.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="pdf",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message="Uploaded PDF is empty.",
            )

        val_result = validate_file(safe_filename, len(file_bytes), file_bytes)
        if not val_result.is_valid:
            update_status(ProcessingStatus.FAILED, val_result.error_message or "Validation failed.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="pdf",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=val_result.error_message,
            )

        # 2. Extracting
        update_status(ProcessingStatus.EXTRACTING, f"Extracting pages from {safe_filename}...")
        elements: List[Dict[str, Any]] = []
        combined_text_parts: List[str] = []
        total_pages = 0

        try:
            import io
            from PIL import Image
            from ingest.image_loader import is_winocr_available, get_winocr_language_tag, configure_tesseract_path
            import pytesseract

            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                # Handle blank-password encrypted PDFs
                if pdf.is_encrypted:
                    try:
                        pdf.authenticate("")
                    except Exception as auth_err:
                        logger.warning("PDF authentication note: %s", auth_err)

                total_pages = len(pdf)
                if total_pages == 0:
                    update_status(ProcessingStatus.FAILED, "PDF contains 0 pages.")
                    return IngestedDocument(
                        document_id=doc_id,
                        session_id=self.session_id,
                        source_type="pdf",
                        filename=safe_filename,
                        status=ProcessingStatus.FAILED,
                        error_message="PDF contains 0 pages.",
                    )

                # Fast digital text check: only invoke heavy OCR if document is genuinely a scanned PDF
                total_digital_chars = sum(len(page.get_text("text").strip()) for page in pdf)
                is_scanned_pdf = total_digital_chars < 50

                for page_idx in range(total_pages):
                    page = pdf[page_idx]
                    page_num = page_idx + 1
                    raw_page_text = page.get_text("text")

                    # Heuristic section detection from blocks/lines
                    section_title = f"Page {page_num}"
                    blocks = page.get_text("blocks")
                    if blocks and len(blocks) > 0:
                        first_block_text = blocks[0][4].strip().split("\n")[0]
                        if 3 <= len(first_block_text) <= 80:
                            section_title = first_block_text

                    # 3. Cleaning
                    clean_page_text = sanitize_text_content(raw_page_text)

                    # Only run heavy OCR on pages if the entire PDF has virtually zero digital text (scanned document)
                    if is_scanned_pdf and len(clean_page_text.strip()) < 25:
                        try:
                            pix = page.get_pixmap(dpi=150)
                            img_data = pix.tobytes("png")
                            page_img = Image.open(io.BytesIO(img_data))
                            ocr_text = ""
                            if configure_tesseract_path():
                                try:
                                    ocr_text = pytesseract.image_to_string(page_img, lang="eng")
                                except Exception:
                                    pass
                            if not ocr_text.strip() and is_winocr_available():
                                try:
                                    import winocr
                                    win_res = winocr.recognize_pil_sync(page_img, "en-US")
                                    if isinstance(win_res, dict):
                                        ocr_text = win_res.get("text", "")
                                except Exception:
                                    pass
                            clean_ocr = sanitize_text_content(ocr_text)
                            if clean_ocr.strip():
                                clean_page_text = clean_ocr
                                logger.info("OCR extracted %d characters from scanned page %d of %s", len(clean_ocr), page_num, safe_filename)
                        except Exception as ocr_err:
                            logger.debug("Page %d OCR fallback error: %s", page_num, ocr_err)

                    if clean_page_text:
                        elements.append({
                            "page_number": page_num,
                            "section": section_title,
                            "text": clean_page_text,
                            "timestamp": None,
                        })
                        combined_text_parts.append(clean_page_text)

        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", safe_filename, e, exc_info=True)
            update_status(ProcessingStatus.FAILED, f"PDF extraction error: {str(e)}")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="pdf",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=f"Error reading PDF file: {str(e)}",
            )

        full_text = "\n\n".join(combined_text_parts)
        if not full_text.strip():
            update_status(ProcessingStatus.FAILED, "No readable text found in PDF (may be scanned images).")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="pdf",
                filename=safe_filename,
                total_pages=total_pages,
                status=ProcessingStatus.FAILED,
                error_message="No extractable text found in PDF. If this is a scanned document, please use image OCR.",
            )

        update_status(ProcessingStatus.CLEANING, f"Cleaned {len(elements)} pages successfully.")
        return IngestedDocument(
            document_id=doc_id,
            session_id=self.session_id,
            source_type="pdf",
            filename=safe_filename,
            title=safe_filename.replace(".pdf", "").replace("_", " ").title(),
            total_pages=total_pages,
            total_characters=len(full_text),
            raw_text=full_text,
            elements=elements,
            status=ProcessingStatus.READY,
            status_message=f"Extracted {total_pages} pages, {len(full_text)} characters.",
        )
