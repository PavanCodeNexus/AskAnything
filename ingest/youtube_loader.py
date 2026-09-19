"""YouTube video transcript loader with timestamp mapping and graceful fallback."""
import re
from typing import Any, Callable, Dict, List, Optional
import urllib.parse
import uuid

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from ingest.base_loader import BaseLoader, IngestedDocument, ProcessingStatus
from security.file_sanitizer import sanitize_text_content
from utils.logger import setup_logger

logger = setup_logger("youtube_loader")


def extract_youtube_id(url_or_id: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID from various URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - Raw 11-character video ID
    """
    clean = url_or_id.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", clean):
        return clean

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)

    return None


def format_seconds(seconds: float) -> str:
    """Converts seconds into HH:MM:SS or MM:SS format."""
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class YouTubeLoader(BaseLoader):
    """Fetches and normalizes YouTube video transcripts with timestamped chunks."""

    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Fetches transcripts from a YouTube video URL.

        Args:
            source: YouTube URL or video ID string.
            status_callback: Optional lifecycle update callback.

        Returns:
            Normalized IngestedDocument with timestamped elements.
        """
        doc_id = str(uuid.uuid4())
        raw_input = str(source).strip()

        def update_status(status: ProcessingStatus, msg: str) -> None:
            if status_callback:
                status_callback(status, msg)

        update_status(ProcessingStatus.VALIDATING, "Validating YouTube URL and video ID...")
        video_id = extract_youtube_id(raw_input)
        if not video_id:
            update_status(ProcessingStatus.FAILED, "Invalid YouTube link or video ID.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="youtube",
                url=raw_input,
                status=ProcessingStatus.FAILED,
                error_message="Could not extract a valid YouTube video ID from the provided link.",
            )

        canonical_url = f"https://www.youtube.com/watch?v={video_id}"

        # 2. Extracting Transcripts
        update_status(ProcessingStatus.EXTRACTING, f"Retrieving transcript for video ID: {video_id}...")
        raw_snippets = None
        try:
            # 1. Try instance-based API (youtube-transcript-api >= 0.6.2+)
            try:
                ytt = YouTubeTranscriptApi()
                if hasattr(ytt, "list"):
                    transcript_list = ytt.list(video_id)
                    try:
                        transcript = transcript_list.find_transcript(["en", "en-US", "hi", "kn", "te", "ta"])
                    except Exception:
                        transcript = transcript_list.find_generated_transcript(["en", "hi", "kn", "te", "ta"]) or next(iter(transcript_list))
                    fetched = transcript.fetch()
                    raw_snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
                elif hasattr(ytt, "fetch"):
                    fetched = ytt.fetch(video_id)
                    raw_snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
            except Exception as e_inst:
                logger.info("Instance API attempt note: %s. Trying direct fetch...", e_inst)

            # 2. Fallback to static get_transcript or direct fetch
            if not raw_snippets:
                if hasattr(YouTubeTranscriptApi, "get_transcript"):
                    raw_snippets = YouTubeTranscriptApi.get_transcript(video_id)
                elif hasattr(YouTubeTranscriptApi, "fetch"):
                    fetched = YouTubeTranscriptApi().fetch(video_id)
                    raw_snippets = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)

            if not raw_snippets:
                raise ValueError("No transcript entries could be extracted from video.")

        except TranscriptsDisabled:
            update_status(ProcessingStatus.FAILED, "Subtitles/transcripts are disabled by the video owner.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="youtube",
                url=canonical_url,
                title=f"YouTube Video ({video_id})",
                status=ProcessingStatus.FAILED,
                error_message="Subtitles and transcripts are disabled for this video. AskAnything requires video captions.",
            )
        except NoTranscriptFound:
            update_status(ProcessingStatus.FAILED, "No transcript found in supported languages.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="youtube",
                url=canonical_url,
                title=f"YouTube Video ({video_id})",
                status=ProcessingStatus.FAILED,
                error_message="No transcript was found for this video.",
            )
        except Exception as e:
            logger.error("Failed to load transcript for %s: %s", video_id, e)
            update_status(ProcessingStatus.FAILED, f"Transcript error: {str(e)}")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="youtube",
                url=canonical_url,
                title=f"YouTube Video ({video_id})",
                status=ProcessingStatus.FAILED,
                error_message=f"YouTube transcript error: {str(e)}",
            )

        # 3. Cleaning & Grouping into ~45-second semantic time windows
        update_status(ProcessingStatus.CLEANING, "Aggregating transcript into timestamped segments...")
        elements: List[Dict[str, Any]] = []
        combined_text_parts = []

        window_text: List[str] = []
        window_start: Optional[float] = None
        window_end: float = 0.0

        for item in raw_snippets:
            text_snip = sanitize_text_content(item.get("text", ""))
            start_sec = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
            end_sec = start_sec + duration

            if window_start is None:
                window_start = start_sec

            window_text.append(text_snip)
            window_end = end_sec

            # Group every 45-60 seconds or 400 characters
            if (window_end - window_start >= 45.0) or (len(" ".join(window_text)) >= 450):
                seg_text = " ".join(window_text).strip()
                t_label = f"{format_seconds(window_start)} - {format_seconds(window_end)}"
                elements.append({
                    "page_number": None,
                    "section": f"Time: {t_label}",
                    "text": seg_text,
                    "timestamp": t_label,
                })
                combined_text_parts.append(seg_text)
                window_text = []
                window_start = None

        if window_text and window_start is not None:
            seg_text = " ".join(window_text).strip()
            t_label = f"{format_seconds(window_start)} - {format_seconds(window_end)}"
            elements.append({
                "page_number": None,
                "section": f"Time: {t_label}",
                "text": seg_text,
                "timestamp": t_label,
            })
            combined_text_parts.append(seg_text)

        full_text = "\n\n".join(combined_text_parts)
        video_title = f"YouTube Video ({video_id})"

        update_status(ProcessingStatus.READY, f"Processed {len(elements)} timestamped transcript segments.")
        return IngestedDocument(
            document_id=doc_id,
            session_id=self.session_id,
            source_type="youtube",
            url=canonical_url,
            filename=f"youtube_{video_id}.txt",
            title=video_title,
            total_pages=1,
            total_characters=len(full_text),
            raw_text=full_text,
            elements=elements,
            status=ProcessingStatus.READY,
            status_message=f"Transcript extracted: {len(elements)} segments ({len(full_text)} chars).",
        )
