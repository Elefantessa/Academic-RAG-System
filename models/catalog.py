"""
Metadata Catalog for Course Information

This module provides a catalog of available metadata values for validation
and fuzzy matching of course information.
"""

import difflib
from typing import List, Dict, Set, Optional, Tuple, DefaultDict
from collections import defaultdict
from langchain_core.documents import Document

from config.constants import DEFAULT_FUZZY_MATCH_CUTOFF


class MetadataCatalog:
    """
    Catalog of available metadata values for validation and fuzzy matching.

    This class maintains mappings between course codes, titles, and filenames
    to support robust entity extraction and validation.

    Maintains:
        - course_code -> title mappings
        - course_code -> file_name mappings
        - Set of all titles for fuzzy matching
    """

    def __init__(self, docs: List[Document]) -> None:
        """
        Initialize catalog from a list of documents.

        Args:
            docs: List of LangChain Document objects with metadata
        """
        self.codes_to_titles: Dict[str, str] = {}
        self.codes_to_files: DefaultDict[str, Set[str]] = defaultdict(set)
        self.titles_set: Set[str] = set()

        # Build mappings from document metadata
        for doc in docs:
            metadata = doc.metadata or {}
            code = str(metadata.get("course_code") or "").strip()
            title = str(metadata.get("course_title") or "").strip()
            filename = str(metadata.get("file_name") or "").strip()

            if code:
                if title and code not in self.codes_to_titles:
                    self.codes_to_titles[code] = title
                if filename:
                    self.codes_to_files[code].add(filename)

            if title:
                self.titles_set.add(title)

    def exists_code(self, code: str) -> bool:
        """
        Check if a course code exists in the catalog.

        Args:
            code: Course code to check

        Returns:
            True if the course code exists, False otherwise
        """
        return code in self.codes_to_titles

    def get_title(self, code: str) -> Optional[str]:
        """
        Get the title for a given course code.

        Args:
            code: Course code

        Returns:
            Course title if found, None otherwise
        """
        return self.codes_to_titles.get(code)

    def fuzzy_title_to_code(
        self,
        query_title: str,
        cutoff: float = DEFAULT_FUZZY_MATCH_CUTOFF
    ) -> Optional[Tuple[str, float]]:
        """
        Fuzzy match a course title to a course code.

        Uses sequence matching to find the best matching title in the catalog,
        then returns the corresponding course code if the match quality exceeds
        the cutoff threshold.

        Args:
            query_title: The title to match
            cutoff: Minimum similarity score (0-1) to consider a match

        Returns:
            Tuple of (course_code, similarity_score) if match found above cutoff,
            None otherwise
        """
        if not self.titles_set or not query_title:
            return None

        best_match = None
        best_score = 0.0

        # Find best matching title using sequence matcher
        for title in self.titles_set:
            score = difflib.SequenceMatcher(
                None,
                query_title.lower(),
                title.lower()
            ).ratio()
            if score > best_score:
                best_match, best_score = title, score

        # Return code if match quality exceeds cutoff
        if best_match and best_score >= cutoff:
            # Find the code for this title
            for code, title in self.codes_to_titles.items():
                if title.lower() == best_match.lower():
                    return code, best_score

        return None

    def get_all_codes(self) -> List[str]:
        """
        Get all course codes in the catalog.

        Returns:
            List of all course codes
        """
        return list(self.codes_to_titles.keys())

    def get_all_titles(self) -> List[str]:
        """
        Get all course titles in the catalog.

        Returns:
            List of all course titles
        """
        return list(self.titles_set)

    def get_catalog_stats(self) -> Dict[str, int]:
        """
        Get statistics about the catalog.

        Returns:
            Dictionary with counts of courses, titles, and files
        """
        return {
            "total_course_codes": len(self.codes_to_titles),
            "total_unique_titles": len(self.titles_set),
            "total_files": sum(len(files) for files in self.codes_to_files.values())
        }
