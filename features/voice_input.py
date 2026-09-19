"""Speech-to-Text (STT) voice input module supporting English and Indian regional languages."""
import io
import os
import tempfile
from typing import Optional, Tuple

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    sr = None
    HAS_SR = False

from utils.logger import setup_logger

logger = setup_logger("voice_input")

# BCP-47 language codes for speech recognition
STT_LANGUAGE_CODES = {
    "en": "en-US",
    "hi": "hi-IN",
    "kn": "kn-IN",  # Experimental
    "te": "te-IN",  # Experimental
    "ta": "ta-IN",  # Experimental
}


class VoiceInputHandler:
    """Handles Speech-To-Text audio transcription from microphone or uploaded audio files."""

    def __init__(self):
        self.recognizer = None
        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True

    def transcribe_audio_file(self, audio_bytes: bytes, language_code: str = "en") -> Tuple[Optional[str], Optional[str]]:
        """Transcribes raw audio bytes into text.

        Args:
            audio_bytes: WAV/FLAC audio byte stream.
            language_code: Target language ISO code ('en', 'hi', 'kn', 'te', 'ta').

        Returns:
            Tuple of (transcribed_text, error_message).
        """
        if not HAS_SR or self.recognizer is None:
            return None, "SpeechRecognition library is not installed. Please run: pip install SpeechRecognition"

        if not audio_bytes:
            return None, "Empty audio input provided."

        bcp_code = STT_LANGUAGE_CODES.get(language_code, "en-US")
        is_experimental = language_code in {"kn", "te", "ta"}
        if is_experimental:
            logger.info("Using experimental STT for regional language: %s (%s)", language_code, bcp_code)

        temp_path = None
        try:
            # Normalize audio with soundfile if needed (e.g. from browser WebM/WAV)
            normalized_bytes = audio_bytes
            try:
                import soundfile as sf
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                out_io = io.BytesIO()
                sf.write(out_io, data, samplerate, format="WAV", subtype="PCM_16")
                normalized_bytes = out_io.getvalue()
            except Exception as conv_err:
                logger.debug("Direct audio conversion note: %s; trying raw bytes", conv_err)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(normalized_bytes)
                temp_path = tmp.name

            with sr.AudioFile(temp_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio_data = self.recognizer.record(source)

            # Use Google Speech Recognition API (free tier)
            text = self.recognizer.recognize_google(audio_data, language=bcp_code)
            logger.info("Transcribed speech (%s): '%s'", bcp_code, text)
            return text, None

        except sr.UnknownValueError:
            return None, "Could not understand the audio. Please speak clearly and try again."
        except sr.RequestError as e:
            logger.error("Speech service request error: %s", e)
            return None, f"Speech recognition service unavailable: {e}"
        except Exception as e:
            logger.error("Audio processing exception: %s", e, exc_info=True)
            return None, f"Audio error: {str(e)}"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def record_from_microphone(self, duration_seconds: int = 5, language_code: str = "en") -> Tuple[Optional[str], Optional[str]]:
        """Captures live audio from local microphone hardware (requires PyAudio)."""
        bcp_code = STT_LANGUAGE_CODES.get(language_code, "en-US")
        try:
            with sr.Microphone() as source:
                logger.info("Listening to microphone for %d seconds...", duration_seconds)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=duration_seconds)

            text = self.recognizer.recognize_google(audio_data, language=bcp_code)
            return text, None
        except ImportError:
            return None, "PyAudio is not installed or microphone hardware is not accessible."
        except Exception as e:
            return None, f"Microphone error: {str(e)}"
