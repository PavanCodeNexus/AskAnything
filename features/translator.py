"""Multilingual translation engine supporting English, Hindi, Kannada, Telugu, and Tamil."""
import re
from typing import Dict, List, Optional, Tuple

try:
    from deep_translator import GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    GoogleTranslator = None
    HAS_DEEP_TRANSLATOR = False

from utils.logger import setup_logger

logger = setup_logger("translator")

# Language definitions
LANGUAGE_CODES: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Telugu": "te",
    "Tamil": "ta",
}

CODE_TO_LANGUAGE: Dict[str, str] = {v: k for k, v in LANGUAGE_CODES.items()}

# Technical domain terms to preserve during translation
PROTECTED_TERMS: List[str] = [
    "RAG", "LLM", "AI", "Vector", "Embedding", "Chunking", "ChromaDB",
    "Cross-Encoder", "BM25", "Transformer", "API", "PyMuPDF", "Streamlit",
    "Evidence Score", "Grounding", "Recall", "Precision", "PDF", "YouTube",
]


class MultilingualEngine:
    """Manages language detection, technical term protection, and bidirectional translation."""

    def __init__(self):
        self._translators: Dict[str, GoogleTranslator] = {}

    def _get_translator(self, source: str, target: str):
        if not HAS_DEEP_TRANSLATOR:
            return None
        key = f"{source}->{target}"
        if key not in self._translators:
            self._translators[key] = GoogleTranslator(source=source, target=target)
        return self._translators[key]

    def detect_language(self, text: str) -> str:
        """Detects language based on Unicode script blocks.

        Returns ISO code: 'en', 'hi', 'kn', 'te', 'ta'.
        """
        if not text:
            return "en"

        # Unicode ranges:
        # Devanagari (Hindi): \u0900-\u097F
        # Kannada: \u0C80-\u0CFF
        # Telugu: \u0C00-\u0C7F
        # Tamil: \u0B80-\u0BFF
        devanagari = len(re.findall(r"[\u0900-\u097F]", text))
        kannada = len(re.findall(r"[\u0C80-\u0CFF]", text))
        telugu = len(re.findall(r"[\u0C00-\u0C7F]", text))
        tamil = len(re.findall(r"[\u0B80-\u0BFF]", text))

        counts = {
            "hi": devanagari,
            "kn": kannada,
            "te": telugu,
            "ta": tamil,
        }

        max_lang = max(counts, key=counts.get)
        if counts[max_lang] >= 2:
            return max_lang

        return "en"

    def protect_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replaces domain keywords with numeric placeholders to prevent mistranslation."""
        mapping: Dict[str, str] = {}
        protected_text = text

        for idx, term in enumerate(PROTECTED_TERMS):
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            placeholder = f"__TERM_{idx}__"
            if pattern.search(protected_text):
                mapping[placeholder] = term
                protected_text = pattern.sub(placeholder, protected_text)

        return protected_text, mapping

    def restore_terms(self, text: str, mapping: Dict[str, str]) -> str:
        """Restores original technical keywords from placeholders."""
        restored = text
        for placeholder, original in mapping.items():
            restored = restored.replace(placeholder, original)
        return restored

    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> Tuple[str, str]:
        """Translates regional language text to English for retrieval indexing.

        Returns: (translated_text, detected_or_source_code)
        """
        if not text or not text.strip():
            return "", "en"

        src = source_lang or self.detect_language(text)
        if src == "en":
            return text, "en"

        try:
            protected_text, term_map = self.protect_terms(text)
            translator = self._get_translator(source=src, target="en")
            if translator is None:
                return text, src
            translated = translator.translate(protected_text)
            restored = self.restore_terms(translated, term_map)
            logger.info("Translated [%s -> en]: '%s' -> '%s'", src, text[:40], restored[:40])
            return restored, src
        except Exception as e:
            logger.warning("Translation to English failed: %s. Using original text.", e)
            return text, src

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translates generated English response into the selected target language."""
        if not text or target_lang == "en" or not target_lang:
            return text

        try:
            protected_text, term_map = self.protect_terms(text)
            translator = self._get_translator(source="en", target=target_lang)
            if translator is None:
                return text
            # Break large text into paragraphs if needed
            paragraphs = protected_text.split("\n\n")
            translated_paragraphs = []
            for p in paragraphs:
                if p.strip():
                    translated_paragraphs.append(translator.translate(p))
                else:
                    translated_paragraphs.append("")
            combined = "\n\n".join(translated_paragraphs)
            restored = self.restore_terms(combined, term_map)
            return restored
        except Exception as e:
            logger.warning("Translation to %s failed: %s. Returning English text.", target_lang, e)
            return text
