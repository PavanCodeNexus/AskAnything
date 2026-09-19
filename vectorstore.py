"""ChromaDB vector store setup, BM25 keyword search, and hybrid retrieval with reranking."""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("vectorstore")

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class EmbeddingManager:
    """Singleton wrapper for HuggingFace sentence-transformer embeddings."""

    _instance = None
    _model = None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_EMBEDDING_MODEL):
        if cls._model is None:
            logger.info("Loading sentence transformer model: %s", model_name)
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    cls._model = SentenceTransformer(model_name)
                except Exception as net_err:
                    logger.warning("Online HuggingFace load failed (%s). Loading from local cache...", net_err)
                    cls._model = SentenceTransformer(model_name, local_files_only=True)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load SentenceTransformer: %s", e, exc_info=True)
                raise
        return cls._model

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        model = cls.get_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    @classmethod
    def embed_query(cls, text: str) -> List[float]:
        model = cls.get_model()
        embedding = model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        return embedding.tolist()


class RerankerManager:
    """Optional Cross-Encoder reranker for high-precision hybrid ranking."""

    _instance = None
    _reranker = None
    _load_attempted = False

    @classmethod
    def get_reranker(cls):
        if not cls._load_attempted:
            cls._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Initializing CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)...")
                cls._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                logger.info("CrossEncoder reranker initialized.")
            except Exception as e:
                logger.warning("CrossEncoder reranker unavailable (%s). Falling back to RRF ranking.", e)
                cls._reranker = None
        return cls._reranker


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword matching."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridVectorStore:
    """Unifies ChromaDB dense vector search and BM25 sparse keyword retrieval with session isolation."""

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR, collection_name: str = "askanything_knowledge"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # In-memory BM25 index per session: {session_id: {"bm25": BM25Okapi, "chunks": List[DocumentChunk]}}
        self._session_bm25: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized HybridVectorStore at %s", self.persist_dir)

    def add_chunks(self, session_id: str, chunks: List[DocumentChunk]) -> None:
        """Embeds and indexes chunks into ChromaDB and updates BM25 index.

        Args:
            session_id: Target session identifier.
            chunks: List of DocumentChunk instances.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        # Generate embeddings
        logger.info("Generating embeddings for %d chunks (session: %s)...", len(chunks), session_id)
        embeddings = EmbeddingManager.embed_texts(texts)

        # Index in ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info("Indexed %d chunks in ChromaDB.", len(chunks))

        # Rebuild / update BM25 index for the session
        self._rebuild_session_bm25(session_id)

    def _rebuild_session_bm25(self, session_id: str) -> None:
        """Refreshes the BM25 index for a given session from ChromaDB."""
        try:
            results = self.collection.get(
                where={"session_id": session_id},
                include=["metadatas", "documents"],
            )
            docs = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []
            ids = results.get("ids", []) or []

            session_chunks: List[DocumentChunk] = []
            for chunk_id, doc_text, meta in zip(ids, docs, metadatas):
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    session_id=str(meta.get("session_id", session_id)),
                    source_type=str(meta.get("source_type", "")),
                    filename=str(meta.get("filename", "")),
                    title=str(meta.get("title", "")),
                    url=str(meta.get("url", "")),
                    page_number=meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
                    timestamp=str(meta.get("timestamp", "")) or None,
                    section=str(meta.get("section", "")),
                    text=doc_text,
                )
                session_chunks.append(chunk)

            if session_chunks:
                corpus_tokens = [tokenize_text(c.text) for c in session_chunks]
                bm25 = BM25Okapi(corpus_tokens)
                self._session_bm25[session_id] = {
                    "bm25": bm25,
                    "chunks": session_chunks,
                }
                logger.info("BM25 index updated for session %s with %d chunks.", session_id, len(session_chunks))
            else:
                self._session_bm25.pop(session_id, None)
        except Exception as e:
            logger.error("Error updating BM25 index for session %s: %s", session_id, e, exc_info=True)

    def delete_document(self, session_id: str, document_id: str) -> None:
        """Deletes all chunks associated with a document within a session."""
        try:
            # ChromaDB query by document_id and session_id
            self.collection.delete(
                where={
                    "$and": [
                        {"session_id": session_id},
                        {"document_id": document_id},
                    ]
                }
            )
            logger.info("Deleted document %s from session %s in ChromaDB.", document_id, session_id)
            self._rebuild_session_bm25(session_id)
        except Exception as e:
            logger.error("Failed to delete document %s: %s", document_id, e, exc_info=True)

    def get_session_chunks(self, session_id: str) -> List[DocumentChunk]:
        """Retrieves all indexed chunks for a given session."""
        try:
            results = self.collection.get(
                where={"session_id": session_id},
                include=["metadatas", "documents"],
            )
            docs = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []
            ids = results.get("ids", []) or []

            chunks: List[DocumentChunk] = []
            for chunk_id, doc_text, meta in zip(ids, docs, metadatas):
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=str(meta.get("document_id", "")),
                        session_id=session_id,
                        source_type=str(meta.get("source_type", "")),
                        filename=str(meta.get("filename", "")),
                        title=str(meta.get("title", "")),
                        url=str(meta.get("url", "")),
                        page_number=meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
                        timestamp=str(meta.get("timestamp", "")) or None,
                        section=str(meta.get("section", "")),
                        text=doc_text,
                    )
                )
            return chunks
        except Exception as e:
            logger.error("Failed to retrieve session chunks: %s", e)
            return []

    def get_session_overview_chunks(self, session_id: str, top_k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        """Retrieves representative lead and distributed structural chunks for comprehensive summaries."""
        chunks = self.get_session_chunks(session_id)
        if not chunks:
            return []

        grouped: Dict[str, List[DocumentChunk]] = {}
        for c in chunks:
            if len(c.text.strip()) > 80:
                grouped.setdefault(c.document_id, []).append(c)

        if not grouped:
            for c in chunks:
                grouped.setdefault(c.document_id, []).append(c)

        overview: List[Tuple[DocumentChunk, float]] = []
        for doc_id, doc_chunks in grouped.items():
            # Lead section chunks (first 4)
            lead = doc_chunks[:min(4, len(doc_chunks))]
            # Sampled section chunks across the rest of the document
            remaining = doc_chunks[4:]
            sampled = []
            if remaining:
                step = max(1, len(remaining) // 4)
                sampled = remaining[::step][:4]

            combined = lead + sampled
            for c in combined[:top_k]:
                overview.append((c, 0.88))
            if len(overview) >= top_k:
                break

        return overview[:top_k]

    def hybrid_search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Performs hybrid retrieval combining dense vector search and BM25 with cross-encoder reranking.

        Args:
            session_id: Current user session ID.
            query: User or rewritten query.
            top_k: Final number of chunks to return.
            vector_weight: Weight for dense vector search (0.0 to 1.0).
            bm25_weight: Weight for BM25 keyword search.

        Returns:
            List of (DocumentChunk, score) tuples sorted by score descending.
        """
        # 1. Vector Search
        candidate_k = min(top_k * 4, 30)
        query_embedding = EmbeddingManager.embed_query(query)

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where={"session_id": session_id},
            include=["metadatas", "documents", "distances"],
        )

        vector_hits: Dict[str, Tuple[DocumentChunk, float]] = {}
        if vector_results and vector_results.get("ids") and vector_results["ids"][0]:
            ids = vector_results["ids"][0]
            docs = vector_results["documents"][0]
            metas = vector_results["metadatas"][0]
            dists = vector_results["distances"][0]

            for c_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
                # Cosine distance to similarity: similarity = 1 - distance
                sim = max(0.0, min(1.0, 1.0 - float(dist)))
                chunk = DocumentChunk(
                    chunk_id=c_id,
                    document_id=str(meta.get("document_id", "")),
                    session_id=session_id,
                    source_type=str(meta.get("source_type", "")),
                    filename=str(meta.get("filename", "")),
                    title=str(meta.get("title", "")),
                    url=str(meta.get("url", "")),
                    page_number=meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
                    timestamp=str(meta.get("timestamp", "")) or None,
                    section=str(meta.get("section", "")),
                    text=doc_text,
                )
                vector_hits[c_id] = (chunk, sim)

        # 2. BM25 Search
        bm25_hits: Dict[str, Tuple[DocumentChunk, float]] = {}
        session_data = self._session_bm25.get(session_id)
        if session_data is None:
            self._rebuild_session_bm25(session_id)
            session_data = self._session_bm25.get(session_id)

        if session_data and session_data["chunks"]:
            bm25_model = session_data["bm25"]
            session_chunks = session_data["chunks"]
            q_tokens = tokenize_text(query)
            bm25_scores = bm25_model.get_scores(q_tokens)

            # Max score normalization
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            indexed_scores = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:candidate_k]

            for idx, raw_score in indexed_scores:
                if raw_score <= 0:
                    continue
                norm_score = raw_score / max_bm25
                chunk = session_chunks[idx]
                bm25_hits[chunk.chunk_id] = (chunk, norm_score)

        # 3. Reciprocal Rank Fusion (RRF) & Weighted Hybrid Scoring
        combined_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Merge vector candidates
        for rank, (cid, (chunk, v_score)) in enumerate(vector_hits.items()):
            chunk_map[cid] = chunk
            # RRF formula: 1 / (60 + rank) + normalized similarity
            rrf = 1.0 / (60.0 + rank)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (vector_weight * v_score) + (0.5 * rrf)

        # Merge BM25 candidates
        for rank, (cid, (chunk, b_score)) in enumerate(bm25_hits.items()):
            chunk_map[cid] = chunk
            rrf = 1.0 / (60.0 + rank)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (bm25_weight * b_score) + (0.5 * rrf)

        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [chunk_map[cid] for cid, _ in sorted_candidates[:candidate_k]]

        if not top_candidates:
            return []

        # 4. Cross-Encoder Reranking
        reranker = RerankerManager.get_reranker()
        if reranker is not None and len(top_candidates) > 1:
            try:
                pairs = [[query, c.text] for c in top_candidates]
                rerank_scores = reranker.predict(pairs)
                # Convert logits to approximate 0-1 sigmoid probability
                import math
                def sigmoid(x: float) -> float:
                    return 1.0 / (1.0 + math.exp(-max(min(x, 15.0), -15.0)))

                ranked = []
                for chunk, score in zip(top_candidates, rerank_scores):
                    ranked.append((chunk, float(sigmoid(float(score)))))

                ranked.sort(key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                logger.warning("Reranking failed (%s). Using RRF scores.", e)

        # Fallback to normalized hybrid scores
        results = []
        max_score = max([s for _, s in sorted_candidates[:top_k]]) if sorted_candidates else 1.0
        for cid, raw_score in sorted_candidates[:top_k]:
            normalized = min(1.0, raw_score / (max_score if max_score > 0 else 1.0))
            results.append((chunk_map[cid], normalized))

        return results
