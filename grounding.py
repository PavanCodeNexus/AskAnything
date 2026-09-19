"""Grounding verification, citation validation, and unsupported claim mitigation."""
import datetime
import html
import re
from typing import Any, Dict, List, Set, Tuple

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

logger = setup_logger("grounding")

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The uploaded sources do not contain enough information to answer this question."
)


def format_citation(chunk: DocumentChunk) -> Dict[str, Any]:
    """Generates a standardized citation object mapped to source type."""
    source_type = (chunk.source_type or "").lower()
    now_str = datetime.date.today().strftime("%Y-%m-%d")

    if source_type == "pdf":
        page_str = f"Page {chunk.page_number}" if chunk.page_number and chunk.page_number != -1 else "Unknown Page"
        section_str = f", {chunk.section}" if chunk.section else ""
        label = f"📄 {chunk.filename or 'PDF'} — {page_str}{section_str}"
        details = f"File: {chunk.filename} | {page_str}"

    elif source_type in ("url", "web"):
        label = f"🌐 {chunk.title or 'Web Source'}"
        link_val = chunk.url or chunk.filename
        details = f"URL: {link_val}"

    elif source_type == "youtube":
        time_str = f"Timestamp: {chunk.timestamp}" if chunk.timestamp else "Video"
        label = f"🎥 {chunk.title or 'YouTube Video'} — {time_str}"
        details = f"Link: {chunk.url} | {time_str}"

    elif source_type == "image":
        label = f"🖼️ {chunk.filename or 'Image'} — OCR Region {chunk.section or 'Main'}"
        details = f"Extracted via OCR from {chunk.filename}"

    elif source_type == "notes":
        sec = chunk.section if chunk.section else "Lines"
        label = f"📝 {chunk.filename or 'Notes'} — {sec}"
        details = f"Text note: {chunk.filename}"

    else:
        label = f"📁 {chunk.filename or chunk.title or 'Source'}"
        details = chunk.section or "Document excerpt"

    link_attr = chunk.url or (chunk.filename if chunk.filename.startswith("http") else "")
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_type": chunk.source_type,
        "label": label,
        "details": details,
        "url": link_attr,
        "link": link_attr,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "timestamp": chunk.timestamp,
        "section": chunk.section,
        "excerpt": chunk.text[:220] + ("..." if len(chunk.text) > 220 else ""),
    }


def render_citation_card(cit: Dict[str, Any]) -> str:
    """Renders an HTML citation card with an optional direct clickable button to open the source in browser."""
    cit_link = cit.get("link") or cit.get("url") or ""
    if not cit_link:
        match = re.search(r'https?://[^\s<>"]+', str(cit.get("details", "")))
        if match:
            cit_link = match.group(0)

    link_html = ""
    if cit_link and cit_link.startswith("http"):
        safe_url = html.escape(cit_link)
        link_html = (
            f'<div style="margin-top: 8px;">'
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="web-source-btn">'
            f'↗ Open Source Material in Browser'
            f'</a>'
            f'</div>'
        )

    title_html = html.escape(str(cit.get("label", "Source")))
    meta_html = html.escape(str(cit.get("details", "")))
    quote_html = html.escape(str(cit.get("excerpt", "")))

    return f"""
    <div class="citation-card">
        <div class="citation-title">{title_html}</div>
        <div class="citation-meta">{meta_html}</div>
        <div class="citation-quote">"{quote_html}"</div>
        {link_html}
    </div>
    """


class GroundingVerifier:
    """Verifies that generated answers are grounded in retrieved chunks and validates citations."""

    def __init__(self, min_token_overlap_threshold: float = 0.30):
        self.min_overlap = min_token_overlap_threshold

    def extract_keywords(self, text: str) -> Set[str]:
        """Extracts content words ignoring common stop words."""
        words = re.findall(r"\b[a-zA-Z0-9_\u0900-\u0DFF]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
            "been", "have", "has", "had", "will", "would", "could", "should", "their",
            "they", "what", "which", "when", "where", "there", "about", "into", "more",
            "also", "these", "those", "such", "than", "then",
        }
        return {w for w in words if w not in stopwords}

    def verify_grounding(
        self,
        answer: str,
        retrieved_chunks: List[DocumentChunk],
    ) -> Tuple[float, List[Dict[str, Any]], bool]:
        """Calculates grounding score and filters out ungrounded citations.

        Args:
            answer: Generated LLM response.
            retrieved_chunks: Chunks provided in the prompt context.

        Returns:
            Tuple of:
            - grounding_score: Float between 0.0 and 1.0
            - verified_citations: List of citation dicts that actually contributed to the answer
            - is_grounded: Boolean indicating if answer is acceptably supported
        """
        if not answer or not retrieved_chunks:
            return 0.0, [], False

        # If LLM responded with insufficient evidence, return 1.0 grounding (no hallucination)
        if INSUFFICIENT_EVIDENCE_MESSAGE.lower() in answer.lower():
            return 1.0, [], True

        answer_keywords = self.extract_keywords(answer)
        if not answer_keywords:
            return 0.5, [format_citation(c) for c in retrieved_chunks[:2]], True

        combined_chunk_text = " ".join([c.text for c in retrieved_chunks])
        chunk_keywords = self.extract_keywords(combined_chunk_text)

        # Grounding ratio: how many content words in the answer exist in the context
        supported_words = answer_keywords.intersection(chunk_keywords)
        grounding_ratio = len(supported_words) / max(len(answer_keywords), 1)

        # Map which specific chunks directly contributed to the answer
        verified_citations: List[Dict[str, Any]] = []
        seen_keys = set()

        for chunk in retrieved_chunks:
            c_keywords = self.extract_keywords(chunk.text)
            overlap = len(answer_keywords.intersection(c_keywords))
            # If chunk shares at least 2 key terms or significant fraction with answer
            if overlap >= 2 or (len(c_keywords) > 0 and (overlap / len(c_keywords)) > 0.15):
                cit = format_citation(chunk)
                key = (cit["source_type"], cit["filename"], cit["page_number"], cit["timestamp"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    verified_citations.append(cit)

        # If no specific chunk passed high bar, fallback to the top 1-2 chunks
        if not verified_citations and retrieved_chunks:
            verified_citations.append(format_citation(retrieved_chunks[0]))

        is_grounded = grounding_ratio >= self.min_overlap
        return round(grounding_ratio, 3), verified_citations, is_grounded
