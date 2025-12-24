"""
Context Expansion Module

Intelligent context expansion for retrieval augmentation.
"""

import logging
from typing import List, Dict, Any, Optional, Set

from langchain_core.documents import Document

from config.constants import SECTION_KEYWORDS
from utils.query_analysis import infer_target_sections

logger = logging.getLogger(__name__)


class ContextExpander:
    """
    Intelligent context expansion to include related document sections.

    Expands the initial retrieved context by fetching related sections
    from the same course to provide more comprehensive information.
    """

    def __init__(self, filter_fetcher=None):
        """
        Initialize context expander.

        Args:
            filter_fetcher: Callable that fetches docs by filter (from retriever)
        """
        self.filter_fetcher = filter_fetcher

    def expand(
        self,
        documents: List[Document],
        query: str,
        focus_code: Optional[str] = None,
        max_additional: int = 3
    ) -> List[Document]:
        """
        Expand context with related document sections.

        Args:
            documents: Initial retrieved documents
            query: User query for section inference
            focus_code: Course code to focus expansion on
            max_additional: Maximum additional docs to add

        Returns:
            Expanded list of documents
        """
        if not documents or not self.filter_fetcher:
            return documents

        # Determine focus course
        if not focus_code:
            focus_code = self._select_focus_code(documents)

        if not focus_code:
            return documents

        # Infer target sections from query
        target_sections = infer_target_sections(query)

        if not target_sections:
            # Default to common informative sections
            target_sections = (
                SECTION_KEYWORDS["contents"] +
                SECTION_KEYWORDS["learning"]
            )

        # Get existing section titles to avoid duplicates
        existing_sections: Set[str] = set()
        for doc in documents:
            section = (doc.metadata or {}).get("section_title", "")
            if section:
                existing_sections.add(section.lower())

        # Fetch additional sections
        additional_docs: List[Document] = []

        for section_title in target_sections:
            if len(additional_docs) >= max_additional:
                break

            if section_title.lower() in existing_sections:
                continue

            try:
                fetched = self.filter_fetcher({
                    "course_code": focus_code,
                    "section_title": section_title
                }, k=1)

                for doc in fetched:
                    if len(additional_docs) < max_additional:
                        additional_docs.append(doc)
                        existing_sections.add(section_title.lower())
                        logger.debug(f"Added section '{section_title}' for {focus_code}")

            except Exception as e:
                logger.warning(f"Failed to fetch section {section_title}: {e}")

        if additional_docs:
            logger.info(f"Expanded context with {len(additional_docs)} additional docs")

        return documents + additional_docs

    def expand_for_comparison(
        self,
        documents: List[Document],
        comparison_codes: List[str],
        axes: List[str]
    ) -> List[Document]:
        """
        Expand context for comparison queries ensuring fair representation.

        Args:
            documents: Initial retrieved documents
            comparison_codes: Course codes to compare
            axes: Comparison axes (section titles)

        Returns:
            Expanded documents with fair representation
        """
        if not self.filter_fetcher or not comparison_codes:
            return documents

        # Track what we have per course
        docs_per_course: Dict[str, List[Document]] = {code: [] for code in comparison_codes}

        for doc in documents:
            code = (doc.metadata or {}).get("course_code", "")
            if code in docs_per_course:
                docs_per_course[code].append(doc)

        # Ensure each course has relevant sections
        additional_docs: List[Document] = []

        for code in comparison_codes:
            existing_sections = {
                (d.metadata or {}).get("section_title", "").lower()
                for d in docs_per_course[code]
            }

            for axis in axes:
                if axis.lower() in existing_sections:
                    continue

                try:
                    fetched = self.filter_fetcher({
                        "course_code": code,
                        "section_title": axis
                    }, k=1)

                    additional_docs.extend(fetched)

                except Exception as e:
                    logger.warning(f"Failed to fetch {axis} for {code}: {e}")

        logger.info(f"Comparison expansion added {len(additional_docs)} docs")

        return documents + additional_docs

    def _select_focus_code(self, documents: List[Document]) -> Optional[str]:
        """Select primary course code from documents."""
        if not documents:
            return None
        return (documents[0].metadata or {}).get("course_code")

    @staticmethod
    def infer_comparison_axes(query: str) -> List[str]:
        """
        Infer which course sections are relevant for comparison.

        Args:
            query: User query

        Returns:
            List of relevant section titles
        """
        base = infer_target_sections(query) or []
        query_lower = query.lower()
        axes: List[str] = []

        def add_if_matches(condition, section_names):
            if condition:
                axes.extend(section_names)

        add_if_matches(
            "prereq" in query_lower or "prerequisite" in query_lower,
            ["Prerequisites"]
        )
        add_if_matches(
            any(w in query_lower for w in ["assessment", "exam", "grading"]),
            ["Assessment method and criteria", "Merged: Assessment method and criteria"]
        )
        add_if_matches(
            "learning outcome" in query_lower or "outcome" in query_lower,
            ["Learning Outcomes"]
        )
        add_if_matches(
            "teaching" in query_lower,
            ["Teaching method and planned learning activities"]
        )
        add_if_matches(
            any(w in query_lower for w in ["content", "summary", "syllabus", "topics"]),
            ["Course Contents", "Course Summary", "Study material"]
        )

        # Remove duplicates while preserving order
        ordered = []
        seen = set()
        all_sections = (
            base + axes +
            SECTION_KEYWORDS["contents"] +
            SECTION_KEYWORDS["learning"] +
            SECTION_KEYWORDS["prereq"] +
            SECTION_KEYWORDS["assessment"]
        )

        for section in all_sections:
            if section not in seen:
                ordered.append(section)
                seen.add(section)

        return ordered
