"""Text-to-Speech (TTS) voice output module using gTTS."""
import io
import os
import re
import tempfile
from typing import Optional, Tuple

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    gTTS = None
    HAS_GTTS = False

from utils.logger import setup_logger

logger = setup_logger("voice_output")

# gTTS language code mapping
TTS_LANGUAGE_CODES = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",  # Experimental
    "te": "te",  # Experimental
    "ta": "ta",  # Experimental
}


def clean_text_for_speech(text: str) -> str:
    """Strips markdown formatting, citations, URLs, and code blocks for clean audio playback."""
    if not text:
        return ""
    # Strip markdown code blocks
    cleaned = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
    # Strip inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Strip markdown links [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Strip URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Strip markdown headers, bold, italics, bullets
    cleaned = re.sub(r"[#*_~>|\\]", " ", cleaned)
    # Remove citation footnotes like [1], [Page 2]
    cleaned = re.sub(r"\[(?:Page|\d+|Source)[\s\w,]*\]", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class VoiceOutputHandler:
    """Synthesizes text into spoken audio MP3 bytes."""

    @classmethod
    def synthesize_speech(
        cls,
        text: str,
        language_code: str = "en",
        slow: bool = False,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Converts response text to MP3 audio bytes.

        Args:
            text: Text to read aloud.
            language_code: Target language ('en', 'hi', 'kn', 'te', 'ta').
            slow: True for slower speech rate.

        Returns:
            Tuple of (mp3_bytes, error_message).
        """
        speech_text = clean_text_for_speech(text)
        if not speech_text:
            return None, "No speakable text found."

        # Truncate very long answers to prevent excessive latency (first 600 chars)
        if len(speech_text) > 600:
            speech_text = speech_text[:600] + "... and more."

        if not HAS_GTTS:
            return None, "gTTS library is not installed. Please run: pip install gTTS"

        lang = TTS_LANGUAGE_CODES.get(language_code, "en")
        try:
            tts = gTTS(text=speech_text, lang=lang, slow=slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            logger.info("Synthesized %d bytes of speech in %s", len(audio_bytes), lang)
            return audio_bytes, None

        except Exception as e:
            logger.error("TTS synthesis error for language %s: %s", lang, e, exc_info=True)
            return None, f"Text-to-speech error: {str(e)}"
