"""Conversational query rewriting for multi-turn conversational context resolution."""
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("query_rewriter")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


REWRITE_SYSTEM_PROMPT = """You are a conversational query rewriter for a search retrieval system.
Given the previous chat conversation and the user's latest follow-up question, rewrite the latest question into a fully self-contained, standalone search query.

Rules:
1. Resolve all pronouns (e.g. "it", "they", "its", "the second one", "this") into explicit entities mentioned in conversation history.
2. Maintain the user's intent and language accurately.
3. If the user's question is already completely standalone and needs no context resolution, return it unchanged.
4. Output ONLY the rewritten standalone query string. Do NOT include explanations, quotes, prefixes, or commentary.
"""


class QueryRewriter:
    """Resolves conversational pronouns and references into standalone retrieval queries."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name
        self.llm = None
        if self.api_key and self.api_key != "your_groq_api_key_here":
            for cand in [self.model_name, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
                try:
                    self.llm = ChatGroq(
                        groq_api_key=self.api_key,
                        model_name=cand,
                        temperature=0.0,
                        max_tokens=350,
                    )
                    self.model_name = cand
                    break
                except Exception as e:
                    logger.warning("QueryRewriter could not init model %s: %s", cand, e)

    def rewrite_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        active_documents: Optional[List[str]] = None,
    ) -> str:
        """Rewrites a query into a standalone version if conversational history exists or deictic references are used.

        Args:
            query: The user's latest question.
            chat_history: List of dicts, e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            active_documents: Optional list of active document names/titles in session.

        Returns:
            Standalone search query.
        """
        clean_q = query.strip()
        if not clean_q:
            return ""

        doc_context_str = ", ".join(active_documents) if active_documents else ""

        # Heuristic entity resolution for deictic references ("this", "the document", "the article")
        resolved_q = clean_q
        if doc_context_str:
            # Replace 'this in 500 words' or 'summarize this'
            deictic_pattern = re.compile(r"\b(this document|this article|this video|this paper|this)\b", re.IGNORECASE)
            if deictic_pattern.search(resolved_q):
                resolved_q = deictic_pattern.sub(f"the uploaded source ({doc_context_str})", resolved_q)

        # If no history and no LLM, return resolved query
        if (not chat_history or len(chat_history) < 2) and self.llm is None:
            return resolved_q

        if self.llm is not None:
            try:
                # Build condensed recent context (up to last 6 turns)
                messages = [SystemMessage(content=REWRITE_SYSTEM_PROMPT)]
                if doc_context_str:
                    messages.append(SystemMessage(content=f"Active Documents in user workspace: {doc_context_str}"))

                for turn in (chat_history or [])[-6:]:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        truncated = content[:300] + ("..." if len(content) > 300 else "")
                        messages.append(AIMessage(content=truncated))

                messages.append(HumanMessage(content=f"Latest Question to rewrite: {clean_q}"))

                response = self.llm.invoke(messages)
                rewritten = response.content.strip().strip('"').strip("'")

                if rewritten and len(rewritten) > 2:
                    logger.info("Rewrote query '%s' -> '%s'", clean_q, rewritten)
                    return rewritten
            except Exception as e:
                logger.warning("Query rewriting failed (%s). Falling back to resolved query.", e)

        return resolved_q
