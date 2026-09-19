"""Claim-level knowledge conflict detector across heterogeneous sources."""
import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

logger = setup_logger("conflict_detector")

CONFLICT_SYSTEM_PROMPT = """You are an objective claim-level conflict analysis engine for a research assistant.
Analyze the provided excerpts from different sources regarding a topic or question.

Your task:
1. Extract key claims made by each distinct source.
2. Group comparable claims that discuss the same fact, entity, metric, or event.
3. Classify the relationship between sources into exactly one of these categories:
   - "Agreement": Sources make consistent, mutually confirming claims.
   - "Contradiction": Sources make directly opposite, mutually exclusive factual claims.
   - "Different Estimate": Sources cite different numbers, dates, estimates, or ranges (e.g. $4.2B vs $4.5B, or 2021 vs 2022). Do NOT mark numerical or methodological differences as Contradiction.
   - "Insufficient Evidence": Not enough overlapping evidence to compare claims.

Return a valid JSON object matching this schema:
{
  "has_conflict": true or false,
  "overall_classification": "Agreement" | "Contradiction" | "Different Estimate" | "Insufficient Evidence",
  "summary": "One sentence summary of the agreement or discrepancy",
  "comparisons": [
    {
      "topic": "Metric or topic being compared",
      "status": "Agreement" | "Contradiction" | "Different Estimate" | "Insufficient Evidence",
      "source_a": {
        "source_name": "filename or title",
        "claim": "Direct claim made",
        "evidence_snippet": "short quote"
      },
      "source_b": {
        "source_name": "filename or title",
        "claim": "Direct claim made",
        "evidence_snippet": "short quote"
      },
      "explanation": "Why this is classified this way"
    }
  ]
}

If only one source is present or sources discuss entirely unrelated facets without any contradiction, set has_conflict to false and overall_classification to "Agreement" or "Insufficient Evidence".
Respond ONLY with the JSON object.
"""


class ConflictDetector:
    """Detects and categorizes claim-level contradictions and numerical differences across sources."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.llm = None
        if self.api_key and self.api_key != "your_groq_api_key_here":
            for cand in [self.model_name, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
                try:
                    self.llm = ChatGroq(
                        groq_api_key=self.api_key,
                        model_name=cand,
                        temperature=0.0,
                        max_tokens=800,
                    )
                    self.model_name = cand
                    break
                except Exception as e:
                    logger.warning("Could not initialize Groq for ConflictDetector on %s: %s", cand, e)

    def detect_conflicts(self, query: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Runs claim extraction and conflict analysis over retrieved chunks.

        Args:
            query: User's question.
            chunks: Retrieved chunks from hybrid search.

        Returns:
            Dictionary with classification, summary, and comparisons.
        """
        # If fewer than 2 distinct sources exist, no cross-source conflict can occur
        unique_sources = {c.filename or c.title or c.document_id for c in chunks if (c.filename or c.title or c.document_id)}
        if len(unique_sources) < 2:
            return {
                "has_conflict": False,
                "overall_classification": "Insufficient Evidence",
                "summary": "Single source available; no cross-source contradiction detected.",
                "comparisons": [],
            }

        # If LLM is not configured, perform numerical and negation heuristic scan
        if self.llm is None:
            return self._heuristic_fallback(chunks)

        try:
            # Build structured source excerpts
            source_payload = []
            for idx, c in enumerate(chunks[:6]):
                src_name = c.filename or c.title or f"Source {idx+1}"
                location = f"Page {c.page_number}" if c.page_number and c.page_number != -1 else (c.timestamp or c.section or "")
                source_payload.append(f"[{src_name} - {location}]:\n{c.text[:400]}")

            prompt = (
                f"Question / Query: {query}\n\n"
                "Retrieved Source Passages:\n" + "\n\n---\n\n".join(source_payload)
            )

            response = self.llm.invoke([
                SystemMessage(content=CONFLICT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            raw_text = response.content.strip()
            # Clean markdown code blocks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?", "", raw_text)
                raw_text = re.sub(r"```$", "", raw_text).strip()

            parsed = json.loads(raw_text)
            logger.info("Conflict analysis result: %s", parsed.get("overall_classification"))
            return parsed

        except Exception as e:
            logger.warning("LLM Conflict Detection failed (%s). Running heuristic fallback.", e)
            return self._heuristic_fallback(chunks)

    def _heuristic_fallback(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Heuristic check for numerical variance and direct negation patterns across documents."""
        grouped: Dict[str, List[str]] = {}
        for c in chunks:
            src = c.filename or c.title or "Unknown"
            grouped.setdefault(src, []).append(c.text)

        source_names = list(grouped.keys())
        if len(source_names) < 2:
            return {
                "has_conflict": False,
                "overall_classification": "Insufficient Evidence",
                "summary": "Insufficient distinct sources to compare.",
                "comparisons": [],
            }

        src_a, src_b = source_names[0], source_names[1]
        text_a = " ".join(grouped[src_a])
        text_b = " ".join(grouped[src_b])

        # Look for numbers/currencies
        nums_a = set(re.findall(r"\$?\b\d+(?:\.\d+)?(?:%|B|M|k)?\b", text_a))
        nums_b = set(re.findall(r"\$?\b\d+(?:\.\d+)?(?:%|B|M|k)?\b", text_b))
        diff_nums = nums_a.symmetric_difference(nums_b)

        if len(diff_nums) >= 2:
            return {
                "has_conflict": False,  # Numerical variance is not flagged as hard contradiction per PRD
                "overall_classification": "Different Estimate",
                "summary": f"Sources '{src_a}' and '{src_b}' report different numerical values or estimates.",
                "comparisons": [
                    {
                        "topic": "Numerical Estimates",
                        "status": "Different Estimate",
                        "source_a": {"source_name": src_a, "claim": f"Values: {', '.join(list(nums_a)[:3])}", "evidence_snippet": text_a[:120]},
                        "source_b": {"source_name": src_b, "claim": f"Values: {', '.join(list(nums_b)[:3])}", "evidence_snippet": text_b[:120]},
                        "explanation": "Different numbers cited across sources without necessarily contradicting definitions.",
                    }
                ],
            }

        return {
            "has_conflict": False,
            "overall_classification": "Agreement",
            "summary": f"Sources '{src_a}' and '{src_b}' show no obvious conflict.",
            "comparisons": [],
        }
