"""
Document Reranking Module

Cross-encoder based reranking with metadata integration.
"""

import logging
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class DocumentReranker:
    """
    Cross-encoder based document reranking.

    Uses a cross-encoder model to score query-document pairs and rerank
    retrieved documents for improved relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize document reranker.

        Args:
            model_name: Cross-encoder model name
        """
        logger.info(f"Initializing Cross-Encoder: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-Encoder initialized successfully")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return

        Returns:
            Tuple of (reranked documents, scores)
        """
        if not documents:
            return [], []

        # Create query-document pairs with enriched text
        pairs = [(query, self._doc_to_rerank_text(doc)) for doc in documents]

        # Get scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)

        # Create scored tuples and sort by score descending
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top k
        top_docs = scored_docs[:top_k]

        reranked_docs = [doc for _, doc in top_docs]
        rerank_scores = [float(score) for score, _ in top_docs]

        logger.info(f"Reranked {len(documents)} docs, kept top {len(reranked_docs)}")

        return reranked_docs, rerank_scores

    @staticmethod
    def _doc_to_rerank_text(doc: Document) -> str:
        """
        Create enriched text for reranking including metadata.

        Args:
            doc: Document to convert

        Returns:
            Enriched text string
        """
        metadata = doc.metadata or {}
        header = (
            f"[{metadata.get('course_code', '?')}] "
            f"{metadata.get('course_title', '?')} — "
            f"{metadata.get('section_title', '?')} | "
            f"lecturers={metadata.get('lecturers', '?')} | "
            f"file={metadata.get('file_name', '?')}\n"
        )
        return header + (doc.page_content or "")

    def get_scored_docs(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[float, Document]]:
        """
        Get all documents with their scores without filtering.

        Args:
            query: Search query
            documents: Documents to score

        Returns:
            List of (score, document) tuples sorted by score
        """
        if not documents:
            return []

        pairs = [(query, self._doc_to_rerank_text(doc)) for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [(float(s), d) for s, d in scored_docs]
