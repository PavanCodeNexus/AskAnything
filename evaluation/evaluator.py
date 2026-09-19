"""Evaluation and benchmark test runner for AskAnything RAG pipeline."""
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger("evaluator")


class RAGEvaluator:
    """Evaluates retrieval quality, grounding faithfulness, and edge-case handling."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(__file__), "test_dataset.json")
        self.test_cases = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def evaluate_retrieval(
        self,
        retrieved_documents: List[str],
        expected_documents: List[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """Computes Precision@K, Recall@K, and Reciprocal Rank."""
        top_k = retrieved_documents[:k]
        if not expected_documents:
            return {"precision@k": 1.0, "recall@k": 1.0, "mrr": 1.0}

        relevant_in_top_k = [doc for doc in top_k if doc in expected_documents]
        precision_at_k = len(relevant_in_top_k) / max(len(top_k), 1)
        recall_at_k = len(relevant_in_top_k) / max(len(expected_documents), 1)

        # Reciprocal Rank (RR)
        mrr = 0.0
        for rank, doc in enumerate(top_k, start=1):
            if doc in expected_documents:
                mrr = 1.0 / rank
                break

        return {
            f"precision@{k}": round(precision_at_k, 3),
            f"recall@{k}": round(recall_at_k, 3),
            "mrr": round(mrr, 3),
        }

    def evaluate_generation(
        self,
        actual_answer: str,
        expected_keywords: List[str],
        grounding_score: float,
        is_insufficient_expected: bool,
        actual_is_insufficient: bool,
    ) -> Dict[str, Any]:
        """Assesses answer keyword coverage, refusal accuracy, and grounding score."""
        # Refusal check for insufficient evidence questions
        if is_insufficient_expected:
            refusal_success = actual_is_insufficient or ("not contain enough information" in actual_answer.lower())
            return {
                "keyword_coverage": 1.0 if refusal_success else 0.0,
                "faithfulness": round(grounding_score, 3),
                "insufficient_evidence_refusal_accuracy": 1.0 if refusal_success else 0.0,
            }

        # Normal question check
        answer_lower = actual_answer.lower()
        matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_coverage = len(matched) / max(len(expected_keywords), 1)

        return {
            "keyword_coverage": round(keyword_coverage, 3),
            "faithfulness": round(grounding_score, 3),
            "insufficient_evidence_refusal_accuracy": 1.0,
        }

    def run_benchmark_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates individual evaluation results into a benchmark summary."""
        if not results:
            return {"status": "No test runs available."}

        total = len(results)
        avg_precision = sum(r.get("precision@5", 0.0) for r in results) / total
        avg_recall = sum(r.get("recall@5", 0.0) for r in results) / total
        avg_mrr = sum(r.get("mrr", 0.0) for r in results) / total
        avg_faithfulness = sum(r.get("faithfulness", 0.0) for r in results) / total
        avg_keyword_coverage = sum(r.get("keyword_coverage", 0.0) for r in results) / total

        summary = {
            "total_test_cases": total,
            "mean_precision@5": round(avg_precision, 3),
            "mean_recall@5": round(avg_recall, 3),
            "mean_reciprocal_rank": round(avg_mrr, 3),
            "mean_faithfulness": round(avg_faithfulness, 3),
            "mean_keyword_coverage": round(avg_keyword_coverage, 3),
        }
        logger.info("Evaluation Benchmark Summary: %s", summary)
        return summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    print(f"Loaded {len(evaluator.test_cases)} evaluation test cases.")
    mock_results = [
        {"precision@5": 0.8, "recall@5": 1.0, "mrr": 1.0, "faithfulness": 0.88, "keyword_coverage": 0.85},
        {"precision@5": 1.0, "recall@5": 1.0, "mrr": 1.0, "faithfulness": 0.94, "keyword_coverage": 0.90},
    ]
    summary = evaluator.run_benchmark_summary(mock_results)
    print(json.dumps(summary, indent=2))
