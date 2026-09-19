"""Tests for Live Web Search Fallback and Clickable Open-Source Material Citations."""
import html
import os
import unittest
from unittest.mock import MagicMock, patch

from features.web_search_fallback import WebSearchRetriever
from grounding import format_citation
from ingest.base_loader import DocumentChunk
from rag_engine import RAGEngine
from vectorstore import HybridVectorStore


class TestWebSearchRetriever(unittest.TestCase):

    def test_convert_to_chunks(self):
        sample_results = [
            {
                "title": "NASA James Webb Space Telescope",
                "link": "https://science.nasa.gov/mission/webb/",
                "snippet": "The James Webb Space Telescope is NASA's premier observatory.",
                "source": "DuckDuckGo Web Search",
            },
            {
                "title": "James Webb Space Telescope - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
                "snippet": "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy.",
                "source": "Wikipedia",
            },
        ]
        chunks = WebSearchRetriever.convert_to_chunks(sample_results, session_id="test_session")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_type, "web")
        self.assertEqual(chunks[0].title, "NASA James Webb Space Telescope")
        self.assertEqual(chunks[0].filename, "https://science.nasa.gov/mission/webb/")
        self.assertEqual(chunks[0].url, "https://science.nasa.gov/mission/webb/")
        self.assertIn("NASA's premier observatory", chunks[0].text)

    def test_format_citation_with_web_chunk(self):
        chunk = DocumentChunk(
            chunk_id="web_1_12345",
            document_id="live_web_search",
            session_id="test_session",
            text="NASA JWST discoveries",
            source_type="web",
            title="NASA Webb Mission",
            filename="https://science.nasa.gov/mission/webb/",
            url="https://science.nasa.gov/mission/webb/",
        )
        cit = format_citation(chunk)
        self.assertEqual(cit["source_type"], "web")
        self.assertIn("NASA Webb Mission", cit["label"])
        self.assertEqual(cit["link"], "https://science.nasa.gov/mission/webb/")
        self.assertEqual(cit["url"], "https://science.nasa.gov/mission/webb/")

    def test_render_citation_card(self):
        from grounding import render_citation_card
        cit = {
            "label": "🌐 NASA Webb Mission",
            "details": "Link: https://science.nasa.gov/mission/webb/",
            "excerpt": "Webb is NASA's flagship telescope.",
            "link": "https://science.nasa.gov/mission/webb/",
        }
        card_html = render_citation_card(cit)
        self.assertIn("NASA Webb Mission", card_html)
        self.assertIn("web-source-btn", card_html)
        self.assertIn('href="https://science.nasa.gov/mission/webb/"', card_html)
        self.assertIn('target="_blank"', card_html)
        self.assertIn('Open Source Material in Browser', card_html)

    def test_web_fallback_flag_disabled(self):
        mock_vs = MagicMock(spec=HybridVectorStore)
        mock_vs.hybrid_search.return_value = []
        mock_vs.get_session_overview_chunks.return_value = []

        engine = RAGEngine(vectorstore=mock_vs, api_key=None)
        res = engine.answer_query(
            session_id="empty_session",
            user_query="What is the launch date of JWST?",
            enable_web_fallback=False,
        )
        self.assertTrue(res["is_insufficient_evidence"])
        self.assertFalse(res.get("is_web_search", False))
        self.assertEqual(len(res["citations"]), 0)


if __name__ == "__main__":
    unittest.main()
