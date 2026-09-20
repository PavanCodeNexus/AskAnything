"""Image and diagram OCR loader using Pillow and Tesseract OCR."""
import os
import shutil
from typing import Any, Callable, Dict, List, Optional
import uuid

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from ingest.base_loader import BaseLoader, IngestedDocument, ProcessingStatus
from security.file_sanitizer import sanitize_filename, sanitize_text_content
from utils.logger import setup_logger
from utils.validator import MAX_IMAGE_SIZE_BYTES, validate_file

logger = setup_logger("image_loader")

# Auto-detect Tesseract executable on Windows
def configure_tesseract_path() -> bool:
    """Detects and configures the Tesseract binary location."""
    env_path = os.getenv("TESSERACT_PATH")
    if env_path and os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return True

    # Standard Windows install locations
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in windows_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True

    # Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return True

    return False


def is_winocr_available() -> bool:
    """Checks if Windows native OCR is available without permanently binding early."""
    try:
        import winocr  # noqa: F401
        return True
    except Exception:
        return False


def get_winocr_language_tag(lang_code: str = "eng") -> Optional[str]:
    """Finds best matching Windows OCR language tag."""
    try:
        import winocr
        available = [l.language_tag for l in winocr.OcrEngine.available_recognizer_languages]
        if not available:
            return None
        lang_mapping = {
            "eng": ["en-US", "en-GB", "en"],
            "hin": ["hi-IN", "hi"],
            "kan": ["kn-IN", "kn"],
            "tam": ["ta-IN", "ta"],
            "tel": ["te-IN", "te"],
        }
        for tag in lang_mapping.get(lang_code, [lang_code]):
            for avail in available:
                if avail.lower().startswith(tag.lower()):
                    return avail
        return available[0]
    except Exception as e:
        logger.debug("WinOCR language lookup notice: %s", e)
        return None


class ImageLoader(BaseLoader):
    """Extracts text from images, scanned documents, and diagrams using OCR (Tesseract + Windows Native WinOCR)."""

    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.tesseract_configured = configure_tesseract_path()

    @property
    def has_winocr(self) -> bool:
        return is_winocr_available()

    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Runs OCR on an uploaded image.

        Args:
            source: Image file path, bytes, or UploadedFile.
            status_callback: Optional lifecycle update callback.

        Returns:
            Normalized IngestedDocument.
        """
        doc_id = str(uuid.uuid4())
        filename = "image.png"
        image_obj: Optional[Image.Image] = None
        file_bytes: Optional[bytes] = None

        def update_status(status: ProcessingStatus, msg: str) -> None:
            if status_callback:
                status_callback(status, msg)

        # 1. Validating
        update_status(ProcessingStatus.VALIDATING, "Validating image file...")
        if isinstance(source, str):
            filename = os.path.basename(source)
            if not os.path.exists(source):
                return IngestedDocument(
                    document_id=doc_id,
                    session_id=self.session_id,
                    source_type="image",
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
            update_status(ProcessingStatus.FAILED, "Image file is empty.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="image",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message="Image content is empty.",
            )

        val_result = validate_file(safe_filename, len(file_bytes), file_bytes)
        if not val_result.is_valid:
            update_status(ProcessingStatus.FAILED, val_result.error_message or "Validation failed.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="image",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=val_result.error_message,
            )

        # Check OCR engine availability (Tesseract or WinOCR)
        has_tesseract = self.tesseract_configured or configure_tesseract_path()
        if not has_tesseract and not self.has_winocr:
            msg = (
                "No OCR engine available. Please install Tesseract OCR or enable Windows Media OCR."
            )
            logger.warning(msg)
            update_status(ProcessingStatus.FAILED, "No OCR engine available.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="image",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=msg,
            )

        # 2. Extracting & Preprocessing Image
        update_status(ProcessingStatus.EXTRACTING, f"Preprocessing and running OCR on {safe_filename}...")
        extracted_text = ""
        try:
            import io
            image_obj = Image.open(io.BytesIO(file_bytes))

            # Image enhancement: Grayscale conversion + contrast boost for improved OCR accuracy
            gray_img = image_obj.convert("L")
            enhancer = ImageEnhance.Contrast(gray_img)
            processed_img = enhancer.enhance(1.8)

            ocr_lang = kwargs.get("ocr_lang", "eng")

            # Try Tesseract first if configured
            if has_tesseract:
                try:
                    extracted_text = pytesseract.image_to_string(processed_img, lang=ocr_lang)
                except Exception as tess_err:
                    logger.warning("Tesseract failed on %s: %s; falling back to WinOCR", safe_filename, tess_err)

            # Fallback to Windows native WinOCR if Tesseract was absent or yielded empty text
            if not extracted_text.strip() and self.has_winocr:
                try:
                    import winocr
                    win_tag = get_winocr_language_tag(ocr_lang) or "en-US"
                    win_result = winocr.recognize_pil_sync(image_obj, win_tag)
                    if isinstance(win_result, dict):
                        extracted_text = win_result.get("text", "")
                        logger.info("WinOCR successfully recognized %d chars for %s", len(extracted_text), safe_filename)
                except Exception as win_err:
                    logger.warning("WinOCR fallback failed: %s", win_err)

        except Exception as e:
            logger.error("OCR execution error on %s: %s", safe_filename, e, exc_info=True)
            update_status(ProcessingStatus.FAILED, f"OCR processing failed: {str(e)}")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="image",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message=f"OCR execution error: {str(e)}",
            )

        # 3. Cleaning
        update_status(ProcessingStatus.CLEANING, "Sanitizing extracted OCR text...")
        clean_text = sanitize_text_content(extracted_text)

        if not clean_text.strip():
            update_status(ProcessingStatus.FAILED, "No readable text detected in image.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="image",
                filename=safe_filename,
                status=ProcessingStatus.FAILED,
                error_message="No legible text or diagrams could be extracted from this image.",
            )

        # Build paragraph elements
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if len(p.strip()) > 15]
        elements: List[Dict[str, Any]] = []
        for idx, p in enumerate(paragraphs, start=1):
            elements.append({
                "page_number": 1,
                "section": f"Block {idx}",
                "text": p,
                "timestamp": None,
            })

        if not elements:
            elements.append({"page_number": 1, "section": "Main Image Text", "text": clean_text, "timestamp": None})

        title = os.path.splitext(safe_filename)[0].replace("_", " ").title()

        update_status(ProcessingStatus.READY, f"OCR succeeded: extracted {len(clean_text)} characters.")
        return IngestedDocument(
            document_id=doc_id,
            session_id=self.session_id,
            source_type="image",
            filename=safe_filename,
            title=title,
            total_pages=1,
            total_characters=len(clean_text),
            raw_text=clean_text,
            elements=elements,
            status=ProcessingStatus.READY,
            status_message=f"OCR extracted {len(elements)} blocks ({len(clean_text)} chars).",
        )


def __getattr__(name: str):
    """Module-level attribute getter for backwards compatibility."""
    if name == "HAS_WINOCR":
        return is_winocr_available()
    if name == "winocr":
        try:
            import winocr
            return winocr
        except Exception:
            return None
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
