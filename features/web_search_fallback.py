"""Live Web Search Fallback module using DuckDuckGo and Wikipedia."""
import json
import os
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

import bs4
import requests

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

logger = setup_logger("web_search")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class WebSearchRetriever:
    """Searches the live web and encyclopedic sources when local documents lack information."""

    @classmethod
    def search(cls, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Executes a live web search for the given query.

        Args:
            query: The search query string.
            max_results: Maximum number of web results to retrieve.

        Returns:
            List of dicts with 'title', 'link', 'snippet', 'source'.
        """
        results: List[Dict[str, str]] = []

        # 1. Try DuckDuckGo HTML search
        try:
            ddg_results = cls._search_duckduckgo(query, max_results=max_results)
            if ddg_results:
                results.extend(ddg_results)
                logger.info("Retrieved %d results from DuckDuckGo for query: '%s'", len(ddg_results), query[:40])
        except Exception as e:
            logger.warning("DuckDuckGo web search encountered issue: %s; trying Wikipedia fallback", e)

        # 2. If fewer than 2 results from DuckDuckGo, supplement with Wikipedia API search
        if len(results) < 2:
            try:
                wiki_results = cls._search_wikipedia(query, max_results=max_results - len(results))
                results.extend(wiki_results)
                logger.info("Retrieved %d results from Wikipedia for query: '%s'", len(wiki_results), query[:40])
            except Exception as e:
                logger.warning("Wikipedia search error: %s", e)

        return results[:max_results]

    @classmethod
    def _search_duckduckgo(cls, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Scrapes DuckDuckGo HTML results cleanly."""
        url = "https://html.duckduckgo.com/html/"
        resp = requests.post(url, data={"q": query}, headers=DEFAULT_HEADERS, timeout=6)
        if resp.status_code != 200:
            return []

        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        items: List[Dict[str, str]] = []

        result_blocks = soup.find_all("div", class_="result")
        for block in result_blocks:
            title_tag = block.find("a", class_="result__a")
            snippet_tag = block.find("a", class_="result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            raw_link = title_tag.get("href", "")

            # Decode DuckDuckGo redirect url (uddg parameter)
            link = raw_link
            if "uddg=" in raw_link:
                try:
                    match = re.search(r"uddg=([^&]+)", raw_link)
                    if match:
                        link = urllib.parse.unquote(match.group(1))
                except Exception:
                    pass

            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if title and snippet and link.startswith("http"):
                items.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "source": "DuckDuckGo Web Search",
                })
                if len(items) >= max_results:
                    break

        return items

    @classmethod
    def _search_wikipedia(cls, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Queries the official Wikipedia API for encyclopedic articles."""
        api_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(query)}"
            "&utf8=&format=json"
        )
        req = urllib.request.Request(api_url, headers=DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
            search_items = data.get("query", {}).get("search", [])

        items: List[Dict[str, str]] = []
        for it in search_items[:max_results]:
            title = it.get("title", "")
            snippet_raw = it.get("snippet", "")
            # Remove HTML highlight tags
            snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

            if title and snippet:
                items.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "source": "Wikipedia",
                })

        return items

    @classmethod
    def convert_to_chunks(cls, web_results: List[Dict[str, str]], session_id: str) -> List[DocumentChunk]:
        """Converts web search hits into normalized DocumentChunk objects for RAG processing."""
        chunks: List[DocumentChunk] = []
        for idx, res in enumerate(web_results, start=1):
            chunk_id = f"web_{idx}_{abs(hash(res['link'])) % 100000}"
            text_content = f"{res['title']}\n{res['snippet']}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id="live_web_search",
                session_id=session_id,
                text=text_content,
                source_type="web",
                page_number=-1,
                title=res["title"],
                filename=res["link"],
                url=res["link"],
                token_count=len(text_content.split()),
            ))
        return chunks
