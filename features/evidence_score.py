"""Multi-signal Evidence Score calculator (replaces raw cosine similarity)."""
from dataclasses import dataclass
from typing import List, Optional

from ingest.base_loader import DocumentChunk
from utils.logger import setup_logger

logger = setup_logger("evidence_score")


@dataclass
class EvidenceScoreResult:
    """Encapsulates the multi-signal evidence evaluation."""
    score: int                           # 0 to 100
    tier: str                            # "High" | "Medium" | "Low"
    label: str                           # "Evidence Strength: High (82) — based on 3 aligned sources"
    retrieval_component: float           # 0 to 100
    grounding_component: float           # 0 to 100
    source_coverage_component: float     # 0 to 100
    conflict_penalty: float              # Deduction applied
    distinct_sources: int
    details: str


class EvidenceScoreCalculator:
    """Calculates multi-signal evidence reliability score combining relevance, grounding, coverage, and conflict."""

    # Component weights
    WEIGHT_RETRIEVAL = 0.40
    WEIGHT_GROUNDING = 0.35
    WEIGHT_COVERAGE = 0.25

    # Thresholds
    TIER_HIGH_MIN = 80
    TIER_MEDIUM_MIN = 50

    @classmethod
    def calculate(
        cls,
        retrieval_similarity: float,
        grounding_score: float,
        chunks_used: List[DocumentChunk],
        has_contradiction: bool = False,
        contradiction_penalty: float = 0.25,
    ) -> EvidenceScoreResult:
        """Calculates composite Evidence Score.

        Args:
            retrieval_similarity: Average or top retrieval score (0.0 to 1.0).
            grounding_score: Ratio of answer supported by chunks (0.0 to 1.0).
            chunks_used: Chunks that provided citations/evidence.
            has_contradiction: True if conflict detector flagged a contradiction.
            contradiction_penalty: Penalty fraction to subtract if contradictory.

        Returns:
            EvidenceScoreResult with numeric score (0-100) and qualitative tier.
        """
        # Clamp inputs
        retrieval_sim = max(0.0, min(1.0, float(retrieval_similarity)))
        grounding = max(0.0, min(1.0, float(grounding_score)))

        # Distinct source count (documents, URLs, filenames)
        unique_docs = {c.document_id or c.filename or c.url for c in chunks_used if (c.document_id or c.filename or c.url)}
        num_sources = len(unique_docs)

        # Coverage factor: 1 source = 0.65, 2 sources = 0.90, 3+ sources = 1.0
        if num_sources >= 3:
            coverage_factor = 1.0
        elif num_sources == 2:
            coverage_factor = 0.90
        elif num_sources == 1:
            coverage_factor = 0.65
        else:
            coverage_factor = 0.10

        # Weighted positive components
        raw_score = (
            (cls.WEIGHT_RETRIEVAL * retrieval_sim) +
            (cls.WEIGHT_GROUNDING * grounding) +
            (cls.WEIGHT_COVERAGE * coverage_factor)
        )

        # Conflict penalty
        applied_penalty = contradiction_penalty if has_contradiction else 0.0
        final_normalized = max(0.0, min(1.0, raw_score - applied_penalty))
        final_score = int(round(final_normalized * 100))

        # Determine Tier
        if final_score >= cls.TIER_HIGH_MIN:
            tier = "High"
        elif final_score >= cls.TIER_MEDIUM_MIN:
            tier = "Medium"
        else:
            tier = "Low"

        # Construct user-facing label
        source_phrase = f"{num_sources} aligned source{'s' if num_sources != 1 else ''}"
        if has_contradiction:
            label = f"Evidence Strength: {tier} ({final_score}/100) — conflicting claims detected across {num_sources} sources"
        else:
            label = f"Evidence Strength: {tier} ({final_score}/100) — based on {source_phrase}"

        details = (
            f"Signals: Retrieval Relevance={int(retrieval_sim*100)}%, "
            f"Grounding={int(grounding*100)}%, Source Coverage={int(coverage_factor*100)}% "
            f"({num_sources} docs)" + (f", Contradiction Penalty=-{int(applied_penalty*100)}%" if has_contradiction else "")
        )

        return EvidenceScoreResult(
            score=final_score,
            tier=tier,
            label=label,
            retrieval_component=round(retrieval_sim * 100, 1),
            grounding_component=round(grounding * 100, 1),
            source_coverage_component=round(coverage_factor * 100, 1),
            conflict_penalty=round(applied_penalty * 100, 1),
            distinct_sources=num_sources,
            details=details,
        )
