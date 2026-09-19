"""Web URL loader with SSRF protection, sanitization, and structured article extraction."""
from typing import Any, Callable, Dict, List, Optional
import urllib.parse
import uuid

from bs4 import BeautifulSoup
import requests

from ingest.base_loader import BaseLoader, IngestedDocument, ProcessingStatus
from security.file_sanitizer import sanitize_text_content
from security.url_guard import is_safe_url
from utils.logger import setup_logger

logger = setup_logger("url_loader")

REQUEST_TIMEOUT_SECONDS = 12
DEFAULT_USER_AGENT = "AskAnythingRAGBot/2.0 (Multimodal Research Assistant)"


class URLLoader(BaseLoader):
    """Scrapes web articles and URLs safely with SSRF defense and content cleaning."""

    def load(
        self,
        source: Any,
        status_callback: Optional[Callable[[ProcessingStatus, str], None]] = None,
        **kwargs: Any,
    ) -> IngestedDocument:
        """Fetches and cleans web content from an HTTP/HTTPS URL.

        Args:
            source: Valid HTTP/HTTPS URL string.
            status_callback: Optional lifecycle update callback.

        Returns:
            Normalized IngestedDocument.
        """
        doc_id = str(uuid.uuid4())
        raw_url = str(source).strip()

        def update_status(status: ProcessingStatus, msg: str) -> None:
            if status_callback:
                status_callback(status, msg)

        # 1. Validating with SSRF Guard
        update_status(ProcessingStatus.VALIDATING, f"Checking security and SSRF rules for {raw_url}...")
        is_safe, error_reason = is_safe_url(raw_url)
        if not is_safe:
            update_status(ProcessingStatus.FAILED, f"Security violation: {error_reason}")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="url",
                url=raw_url,
                filename="web_source.html",
                status=ProcessingStatus.FAILED,
                error_message=f"SSRF Check Failed: {error_reason}",
            )

        # 2. Extracting via HTTP GET
        update_status(ProcessingStatus.EXTRACTING, f"Fetching content from {raw_url}...")
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = requests.get(raw_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            update_status(ProcessingStatus.FAILED, "Request timed out while connecting to website.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="url",
                url=raw_url,
                filename="web_source.html",
                status=ProcessingStatus.FAILED,
                error_message="Website request timed out after 12 seconds.",
            )
        except Exception as e:
            logger.error("HTTP fetch error for %s: %s", raw_url, e)
            update_status(ProcessingStatus.FAILED, f"Failed to fetch URL: {str(e)}")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="url",
                url=raw_url,
                filename="web_source.html",
                status=ProcessingStatus.FAILED,
                error_message=f"Network error: {str(e)}",
            )

        # 3. Cleaning and HTML Parsing
        update_status(ProcessingStatus.CLEANING, "Parsing HTML and extracting body text...")
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = "Web Article"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Remove boilerplate tags (scripts, styles, ads, navigation, footer)
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
            tag.decompose()

        # Remove Wikipedia / media wiki infoboxes, sidebars, navboxes, tables of contents, and references
        for bad in soup.find_all(["table", "div"], class_=re.compile(r"infobox|sidebar|navbox|mw-empty-elt|toc|reflist|metadata|noprint", re.I)):
            bad.decompose()

        # Target main content containers if present
        main_content = soup.find("article") or soup.find("main") or soup.find("div", {"id": re.compile(r"content|main|article|mw-content-text", re.I)}) or soup.body
        if not main_content:
            main_content = soup

        # Extract structured paragraphs and headings
        elements: List[Dict[str, Any]] = []
        combined_text_parts = []
        current_section = "Overview"

        for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = element.get_text().strip()
            if not text:
                continue

            if element.name in ["h1", "h2", "h3", "h4"]:
                current_section = text[:80]
            elif element.name in ["p", "li"] and len(text) > 25:
                clean_item = sanitize_text_content(text)
                elements.append({
                    "page_number": None,
                    "section": current_section,
                    "text": clean_item,
                    "timestamp": None,
                })
                combined_text_parts.append(clean_item)

        full_text = "\n\n".join(combined_text_parts)
        if not full_text.strip():
            # Fallback to direct body text
            full_text = sanitize_text_content(soup.get_text(separator="\n"))

        if not full_text.strip():
            update_status(ProcessingStatus.FAILED, "No readable text could be extracted from webpage.")
            return IngestedDocument(
                document_id=doc_id,
                session_id=self.session_id,
                source_type="url",
                url=raw_url,
                title=title,
                status=ProcessingStatus.FAILED,
                error_message="Webpage contained no extractable textual content.",
            )

        parsed_domain = urllib.parse.urlparse(raw_url).netloc
        filename = f"{parsed_domain}.html"

        update_status(ProcessingStatus.READY, f"Extracted {len(full_text)} characters from {parsed_domain}.")
        return IngestedDocument(
            document_id=doc_id,
            session_id=self.session_id,
            source_type="url",
            url=raw_url,
            filename=filename,
            title=title,
            total_pages=1,
            total_characters=len(full_text),
            raw_text=full_text,
            elements=elements if elements else [{"page_number": None, "section": "Article", "text": full_text, "timestamp": None}],
            status=ProcessingStatus.READY,
            status_message=f"Web article extracted ({len(full_text)} chars).",
        )
