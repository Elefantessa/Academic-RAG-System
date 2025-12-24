"""
Entity Extraction Module

Multi-stage entity extraction with regex, fuzzy matching, and LLM fallback.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Set

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config.constants import COURSE_CODE_REGEX
from models.catalog import MetadataCatalog
from utils.query_analysis import (
    extract_lecturer_from_query,
    is_comparison_query,
    extract_course_mentions,
)

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Multi-stage entity extraction with regex, fuzzy matching, and LLM fallback.

    Extraction pipeline:
    1. Regex for course codes
    2. Fuzzy matching for course titles
    3. Lecturer name extraction
    4. Multi-course comparison support
    5. LLM extraction as fallback
    """

    def __init__(
        self,
        catalog: MetadataCatalog,
        extraction_llm: Optional[ChatOllama] = None
    ):
        """
        Initialize entity extractor.

        Args:
            catalog: MetadataCatalog for validation
            extraction_llm: Optional LLM for fallback extraction
        """
        self.catalog = catalog
        self.extraction_llm = extraction_llm

    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from query using multi-stage pipeline.

        Args:
            query: User query string

        Returns:
            Dictionary with extracted entities
        """
        extracted: Dict[str, Any] = {}

        # Stage 1: Regex for course codes
        self._extract_course_code(query, extracted)

        # Stage 2: Fuzzy matching for titles if no code found
        if 'course_code' not in extracted:
            self._extract_by_title(query, extracted)

        # Stage 3: Lecturer extraction
        self._extract_lecturer(query, extracted)

        # Stage 4: Multi-course comparison support
        if is_comparison_query(query):
            self._extract_comparison_codes(query, extracted)

        # Stage 5: LLM extraction as fallback
        if self.extraction_llm and not extracted.get('course_code'):
            self._llm_extraction(query, extracted)

        return extracted

    def _extract_course_code(self, query: str, extracted: Dict[str, Any]) -> None:
        """Extract course code using regex."""
        match = re.search(COURSE_CODE_REGEX, query.upper())
        if match:
            code = match.group(0)
            if self.catalog.exists_code(code):
                extracted['course_code'] = code
                extracted['course_title'] = self.catalog.get_title(code)
                logger.info(f"[Extract] Found valid course_code via regex: {code}")

    def _extract_by_title(self, query: str, extracted: Dict[str, Any]) -> None:
        """Extract course code by fuzzy matching title."""
        title_candidate = None

        # Look for quoted text first
        quote_match = re.search(r"'([^']+)'|\"([^\"]+)\"", query)
        if quote_match:
            title_candidate = (quote_match.group(1) or quote_match.group(2) or "").strip()
        else:
            # Multiple patterns to search for title
            patterns = [
                r"(?:course)\s+([A-Za-z][\w\s&\-:]+)",
                r"([A-Za-z][\w\s&\-:]+)\s+course",
                r"for\s+(?:the\s+)?([A-Za-z][\w\s&\-:]+)\s+course",
                r"about\s+(?:the\s+)?([A-Za-z][\w\s&\-:]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, query, flags=re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    candidate = re.sub(r'\b(the|a|an)\b', '', candidate, flags=re.IGNORECASE).strip()
                    if len(candidate) > 2:
                        title_candidate = candidate
                        break

        if title_candidate:
            fuzzy_match = self.catalog.fuzzy_title_to_code(title_candidate, cutoff=0.78)
            if fuzzy_match:
                code, score = fuzzy_match
                extracted['course_code'] = code
                extracted['course_title'] = self.catalog.get_title(code)
                logger.info(f"[Extract] Fuzzy-mapped title '{title_candidate}' → {code} (score={score:.2f})")

    def _extract_lecturer(self, query: str, extracted: Dict[str, Any]) -> None:
        """Extract lecturer name from query."""
        lecturer = extract_lecturer_from_query(query)
        if lecturer:
            extracted['lecturers'] = [lecturer]
            logger.info(f"[Extract] Lecturer parsed from query: {lecturer}")

    def _extract_comparison_codes(self, query: str, extracted: Dict[str, Any]) -> None:
        """Extract multiple course codes for comparison queries."""
        candidates = extract_course_mentions(query)
        codes: List[str] = []
        titles_left: List[str] = []

        # Separate direct codes from potential titles
        for candidate in candidates:
            if re.fullmatch(COURSE_CODE_REGEX, candidate.upper()):
                if self.catalog.exists_code(candidate.upper()):
                    codes.append(candidate.upper())
            else:
                titles_left.append(candidate)

        # Fuzzy match remaining titles
        for title in titles_left:
            title_clean = re.sub(r"[?.!]+$", "", title).strip()
            fuzzy_match = self.catalog.fuzzy_title_to_code(title_clean, cutoff=0.78)
            if fuzzy_match:
                code, _ = fuzzy_match
                codes.append(code)

        # Additional substring matching if needed
        if len(codes) < 2:
            query_lower = query.lower()
            for title in self.catalog.titles_set:
                if title and title.lower() in query_lower:
                    for code, catalog_title in self.catalog.codes_to_titles.items():
                        if catalog_title.lower() == title.lower():
                            codes.append(code)

        # Remove duplicates while preserving order
        seen_codes: Set[str] = set()
        unique_codes: List[str] = []
        for code in codes:
            if code not in seen_codes:
                unique_codes.append(code)
                seen_codes.add(code)

        if len(unique_codes) >= 2:
            extracted['comparison_codes'] = unique_codes
            logger.info(f"[Extract] comparison_codes={unique_codes}")

    def _llm_extraction(self, query: str, extracted: Dict[str, Any]) -> None:
        """Use LLM as fallback for extraction."""
        if not self.extraction_llm:
            return

        try:
            response = self.extraction_llm.invoke(
                f"Extract ONLY from this query: '{query}' as JSON."
            )
            data = json.loads(response.content) if isinstance(response.content, str) else response.content

            if isinstance(data, dict):
                llm_code = (data.get('course_code') or "").strip().upper()
                llm_title = (data.get('course_title') or "").strip()
                llm_lecturers = data.get('lecturers') or []

                # Use LLM lecturers if not already extracted
                if llm_lecturers and not extracted.get('lecturers'):
                    if isinstance(llm_lecturers, list):
                        extracted['lecturers'] = [str(x).strip() for x in llm_lecturers if str(x).strip()]
                    elif isinstance(llm_lecturers, str) and llm_lecturers.strip():
                        extracted['lecturers'] = [llm_lecturers.strip()]

                # Use LLM course code if verified
                if 'course_code' not in extracted and llm_code and self.catalog.exists_code(llm_code):
                    extracted['course_code'] = llm_code
                    extracted['course_title'] = self.catalog.get_title(llm_code)
                    logger.info(f"[Extract] Using LLM course_code verified in catalog: {llm_code}")
                elif 'course_code' not in extracted and llm_title:
                    fuzzy_match = self.catalog.fuzzy_title_to_code(llm_title, cutoff=0.80)
                    if fuzzy_match:
                        code, score = fuzzy_match
                        extracted['course_code'] = code
                        extracted['course_title'] = self.catalog.get_title(code)
                        logger.info(f"[Extract] LLM title fuzzy-mapped → {code} (score={score:.2f})")

        except Exception as e:
            logger.warning(f"[Extract] LLM extraction failed: {e}")
