"""
Confidence Metrics Models

This module defines the data structures for confidence calculation results.
"""

from dataclasses import dataclass


@dataclass
class ConfidenceMetrics:
    """
    Container for confidence calculation metrics.

    This class holds all individual confidence scores and the final
    aggregated confidence value for a RAG response.

    Attributes:
        rerank_score: Confidence from cross-encoder reranking (0-1)
        entity_match_ratio: Ratio of matched entities in docs (0-1)
        source_diversity: Diversity of document sources (0-1)
        context_completeness: Completeness of retrieved context (0-1)
        semantic_coherence: LLM-evaluated semantic quality (0-1)
        final_confidence: Weighted combination of all metrics (0-1)
        reasoning: Textual explanation of confidence assessment
    """

    rerank_score: float
    entity_match_ratio: float
    source_diversity: float
    context_completeness: float
    semantic_coherence: float
    final_confidence: float
    reasoning: str

    def __post_init__(self):
        """Validate that all scores are in valid range [0, 1]"""
        for field_name in [
            'rerank_score', 'entity_match_ratio', 'source_diversity',
            'context_completeness', 'semantic_coherence', 'final_confidence'
        ]:
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{field_name} must be between 0 and 1, got {value}"
                )
