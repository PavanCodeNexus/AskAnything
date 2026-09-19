"""File validation utilities for type, size, and integrity checks."""
import os
from typing import Dict, List, Optional, Set, Tuple

from utils.logger import setup_logger

logger = setup_logger("validator")

# Supported file extensions and size limits in bytes
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024       # 50 MB
MAX_TEXT_SIZE_BYTES = 5 * 1024 * 1024        # 5 MB
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024     # 20 MB

ALLOWED_EXTENSIONS_MAP: Dict[str, Set[str]] = {
    "pdf": {".pdf"},
    "notes": {".txt", ".md", ".csv", ".json"},
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"},
}

ALL_ALLOWED_EXTENSIONS = set.union(*ALLOWED_EXTENSIONS_MAP.values())


class FileValidationResult:
    """Encapsulates the result of a file validation check."""

    def __init__(self, is_valid: bool, error_message: Optional[str] = None, detected_type: Optional[str] = None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.detected_type = detected_type

    def __repr__(self) -> str:
        return f"FileValidationResult(valid={self.is_valid}, type={self.detected_type}, error={self.error_message})"


def validate_file(filename: str, file_size: int, content: Optional[bytes] = None) -> FileValidationResult:
    """Validates file metadata, extension, size, and non-empty content.

    Args:
        filename: Name of the file with extension.
        file_size: Size in bytes.
        content: Optional raw bytes of the file for integrity inspection.

    Returns:
        FileValidationResult with pass/fail and descriptive error.
    """
    if not filename or not filename.strip():
        return FileValidationResult(False, "Filename cannot be empty.")

    ext = os.path.splitext(filename.lower())[1]
    if not ext:
        return FileValidationResult(False, f"File '{filename}' has no extension.")

    if ext not in ALL_ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALL_ALLOWED_EXTENSIONS))
        return FileValidationResult(
            False,
            f"Unsupported file format '{ext}'. Allowed extensions are: {allowed_str}",
        )

    if file_size <= 0:
        return FileValidationResult(False, f"File '{filename}' is empty (0 bytes).")

    # Determine category
    detected_category: Optional[str] = None
    for category, extensions in ALLOWED_EXTENSIONS_MAP.items():
        if ext in extensions:
            detected_category = category
            break

    # Check size constraints by type
    if detected_category == "pdf" and file_size > MAX_PDF_SIZE_BYTES:
        max_mb = MAX_PDF_SIZE_BYTES // (1024 * 1024)
        return FileValidationResult(
            False,
            f"PDF file size ({file_size / (1024 * 1024):.2f}MB) exceeds limit of {max_mb}MB.",
            detected_category,
        )

    if detected_category == "notes" and file_size > MAX_TEXT_SIZE_BYTES:
        max_mb = MAX_TEXT_SIZE_BYTES // (1024 * 1024)
        return FileValidationResult(
            False,
            f"Text file size ({file_size / (1024 * 1024):.2f}MB) exceeds limit of {max_mb}MB.",
            detected_category,
        )

    if detected_category == "image" and file_size > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        return FileValidationResult(
            False,
            f"Image file size ({file_size / (1024 * 1024):.2f}MB) exceeds limit of {max_mb}MB.",
            detected_category,
        )

    # Inspect magic headers if content is available
    if content:
        if ext == ".pdf" and not content.startswith(b"%PDF"):
            return FileValidationResult(False, "File does not appear to be a valid PDF (corrupt or invalid header).", detected_category)
        if ext in {".png"} and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return FileValidationResult(False, "File does not match PNG signature.", detected_category)
        if ext in {".jpg", ".jpeg"} and not (content.startswith(b"\xff\xd8\xff")):
            return FileValidationResult(False, "File does not match JPEG signature.", detected_category)

    return FileValidationResult(True, None, detected_category)
