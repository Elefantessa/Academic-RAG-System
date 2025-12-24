"""
Confidence Calculator Service

Advanced confidence calculation using multiple metrics and LLM evaluation.
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from models.confidence import ConfidenceMetrics
from config.constants import CONFIDENCE_WEIGHTS

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Advanced confidence calculator using multiple metrics.

    Combines:
    - Cross-encoder reranking scores
    - Entity match ratio
    - Source diversity
    - Context completeness
    - Semantic coherence (LLM-evaluated)
    """

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize confidence calculator.

        Args:
            ollama_base_url: URL for Ollama LLM service
        """
        self.confidence_llm = ChatOllama(
            model="llama3.1:latest",
            base_url=ollama_base_url,
            temperature=0,
            format="json"
        )

    def calculate_confidence(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        reranked_docs: List[Document],
        rerank_scores: Optional[List[float]] = None,
        extracted_entities: Optional[Dict[str, Any]] = None,
        generation_mode: str = "standard"
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score.

        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Initially retrieved documents
            reranked_docs: Documents after reranking
            rerank_scores: Scores from cross-encoder
            extracted_entities: Extracted entities from query
            generation_mode: Generation mode (standard/comparison/lecturer)

        Returns:
            ConfidenceMetrics with all scores
        """
        rerank_scores = rerank_scores or []
        extracted_entities = extracted_entities or {}

        # Calculate individual metrics
        rerank_conf = self._calculate_rerank_confidence(rerank_scores)
        entity_conf = self._calculate_entity_match_confidence(
            query, reranked_docs, extracted_entities
        )
        source_conf = self._calculate_source_diversity(reranked_docs, generation_mode)
        complete_conf = self._calculate_context_completeness(
            query, reranked_docs, extracted_entities
        )
        semantic_conf, reasoning = self._calculate_semantic_coherence(
            query, answer, reranked_docs
        )

        # Get weights for this mode
        weights = CONFIDENCE_WEIGHTS.get(generation_mode, CONFIDENCE_WEIGHTS["standard"])

        # Weighted combination
        final = (
            weights["rerank"] * rerank_conf +
            weights["entity"] * entity_conf +
            weights["source"] * source_conf +
            weights["completeness"] * complete_conf +
            weights["semantic"] * semantic_conf
        )

        final = max(0.0, min(1.0, final))

        return ConfidenceMetrics(
            rerank_score=rerank_conf,
            entity_match_ratio=entity_conf,
            source_diversity=source_conf,
            context_completeness=complete_conf,
            semantic_coherence=semantic_conf,
            final_confidence=final,
            reasoning=reasoning
        )

    def _calculate_rerank_confidence(self, rerank_scores: List[float]) -> float:
        """Calculate confidence from reranking scores."""
        if not rerank_scores:
            return 0.5

        # Handle numpy arrays
        if isinstance(rerank_scores, np.ndarray):
            rerank_scores = rerank_scores.tolist()

        try:
            scores = [float(s) for s in rerank_scores if s is not None]
            if not scores:
                return 0.5
        except (ValueError, TypeError):
            return 0.5

        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        # Lower variance = higher confidence
        confidence = (avg_score * 0.7) + (1.0 / (1.0 + variance)) * 0.3

        return max(0.0, min(1.0, confidence))

    def _calculate_entity_match_confidence(
        self,
        query: str,
        docs: List[Document],
        extracted: Dict[str, Any]
    ) -> float:
        """Calculate confidence from entity matching."""
        if not docs:
            return 0.0

        expected = set()
        if extracted.get("course_code"):
            expected.add(extracted["course_code"].lower())
        if extracted.get("course_title"):
            expected.add(extracted["course_title"].lower())
        if extracted.get("lecturers"):
            for lect in extracted["lecturers"]:
                expected.add(lect.lower())

        if not expected:
            return 0.7  # Neutral

        matched = set()
        for doc in docs:
            meta = doc.metadata or {}
            content = (doc.page_content or "").lower()

            if meta.get("course_code", "").lower() in expected:
                matched.add(meta["course_code"].lower())
            if meta.get("course_title", "").lower() in expected:
                matched.add(meta["course_title"].lower())

            for entity in expected:
                if entity in content:
                    matched.add(entity)

        return len(matched) / len(expected)

    def _calculate_source_diversity(
        self,
        docs: List[Document],
        mode: str
    ) -> float:
        """Calculate confidence from source diversity."""
        if not docs:
            return 0.0

        courses = set()
        sections = set()

        for doc in docs:
            meta = doc.metadata or {}
            if meta.get("course_code"):
                courses.add(meta["course_code"])
            if meta.get("section_title"):
                sections.add(meta["section_title"])

        course_div = min(1.0, len(courses) / max(1, len(docs)))
        section_div = min(1.0, len(sections) / max(1, len(docs)))

        if mode == "comparison":
            return course_div * 0.8 + section_div * 0.2
        elif mode == "lecturer":
            return course_div * 0.9 + section_div * 0.1
        else:
            return course_div * 0.5 + section_div * 0.5

    def _calculate_context_completeness(
        self,
        query: str,
        docs: List[Document],
        extracted: Dict[str, Any]
    ) -> float:
        """Calculate context completeness."""
        if not docs:
            return 0.0

        q = query.lower()
        expected = set()

        if any(w in q for w in ["prerequisite", "prereq"]):
            expected.add("prerequisites")
        if any(w in q for w in ["assessment", "exam", "grading"]):
            expected.add("assessment")
        if any(w in q for w in ["learning outcome", "outcome"]):
            expected.add("learning")
        if any(w in q for w in ["teaching", "method"]):
            expected.add("teaching")
        if any(w in q for w in ["content", "summary", "about"]):
            expected.add("content")

        if not expected:
            return 0.8  # Neutral

        present = set()
        for doc in docs:
            section = (doc.metadata or {}).get("section_title", "").lower()
            for exp in expected:
                if exp in section:
                    present.add(exp)

        return len(present) / len(expected)

    def _calculate_semantic_coherence(
        self,
        query: str,
        answer: str,
        docs: List[Document]
    ) -> Tuple[float, str]:
        """Evaluate semantic coherence using LLM."""
        prompt = f"""Evaluate this RAG response. Return ONLY valid JSON.

Query: {query[:100]}...
Answer: {answer[:200]}...
Context: {len(docs)} documents

Return JSON:
{{"confidence_score": 0.75, "reasoning": "Brief explanation"}}"""

        try:
            response = self.confidence_llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON
            start = content.find('{')
            end = content.rfind('}') + 1

            if start != -1 and end > start:
                data = json.loads(content[start:end])
            else:
                data = json.loads(content)

            score = float(data.get("confidence_score", 0.5))
            reasoning = str(data.get("reasoning", "Evaluation completed"))

            return max(0.0, min(1.0, score)), reasoning

        except Exception as e:
            logger.warning(f"Semantic evaluation failed: {e}")
            return self._fallback_evaluation(query, answer, docs), "Fallback evaluation"

    def _fallback_evaluation(
        self,
        query: str,
        answer: str,
        docs: List[Document]
    ) -> float:
        """Heuristic fallback when LLM fails."""
        score = 0.5

        word_count = len(answer.split())
        if 10 <= word_count <= 200:
            score += 0.1
        elif word_count < 5:
            score -= 0.2

        q_terms = set(query.lower().split())
        a_terms = set(answer.lower().split())
        overlap = len(q_terms & a_terms)
        score += min(0.2, overlap * 0.05)

        if docs and any(
            (d.metadata or {}).get("course_code", "").lower() in answer.lower()
            for d in docs[:3]
        ):
            score += 0.1

        return max(0.0, min(1.0, score))
