"""
State and Response Models for RAG System

This module defines the data structures used for state management
and API responses in the retrieval pipeline.
"""

from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass, asdict
from langchain_core.documents import Document


class RetrievalState(TypedDict):
    """
    State object passed between graph nodes in the retrieval pipeline.

    Attributes:
        original_query: The user's original query string
        extracted: Dictionary of extracted entities (course codes, titles, lecturers, etc.)
        retrieved_docs: Documents retrieved from vector search
        reranked_docs: Documents after cross-encoder reranking
        final_answer: Generated answer text
        rerank_scores: Confidence scores from reranking
    """
    original_query: str
    extracted: dict
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    final_answer: str
    rerank_scores: List[float]


@dataclass
class RAGResponse:
    """
    Complete response from the RAG system.

    This dataclass encapsulates all information about a query response,
    including the answer, confidence metrics, sources, and metadata.
    """

    query: str
    answer: str
    confidence: float
    sources: List[str]
    generation_mode: str  # "standard", "comparison", or "lecturer"
    processing_time: float
    reasoning_steps: List[str]
    conflicts_detected: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the response
        """
        return asdict(self)

    def to_json_safe(self) -> Dict[str, Any]:
        """
        Convert to a JSON-safe dictionary with safe handling of complex types.

        Returns:
            JSON-serializable dictionary
        """
        data = self.to_dict()
        # Ensure all values are JSON serializable
        if isinstance(data.get('metadata'), dict):
            # Convert any non-serializable metadata values to strings
            for key, value in data['metadata'].items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    data['metadata'][key] = str(value)
        return data
