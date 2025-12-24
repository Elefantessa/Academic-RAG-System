"""
Query Analysis Utilities

This module provides helper functions for analyzing and classifying user queries,
extracting entities, and detecting query intent (comparison, lecturer, etc.).
"""

import re
from typing import List, Dict, Any, Optional

from config.constants import (
    COURSE_CODE_REGEX,
    LECTURER_PATTERN,
    SECTION_KEYWORDS,
    LECTURER_QUERY_KEYWORDS,
    COMPARISON_QUERY_KEYWORDS,
)


# ===== Lecturer-related Utilities =====

def normalize_lecturers_field(val: Any) -> str:
    """
    Convert lecturers field to lowercase string for matching.

    Args:
        val: Value from lecturers field (could be string, list, or None)

    Returns:
        Normalized lowercase string representation
    """
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(map(str, val)).lower()
    return str(val).lower()


def lecturer_matches(metadata: Dict[str, Any], target_name_lc: str) -> bool:
    """
    Check if lecturer name matches in document metadata.

    Args:
        metadata: Document metadata dictionary
        target_name_lc: Target lecturer name in lowercase

    Returns:
        True if the lecturer matches, False otherwise
    """
    if not target_name_lc:
        return False
    vals_lc = normalize_lecturers_field((metadata or {}).get("lecturers", ""))
    return target_name_lc in vals_lc


def is_lecturer_query(query: str) -> bool:
    """
    Detect if query is asking about courses taught by a lecturer.

    Args:
        query: User query string

    Returns:
        True if this is a lecturer query, False otherwise
    """
    ql = query.lower()
    return any(keyword in ql for keyword in LECTURER_QUERY_KEYWORDS)


def extract_lecturer_from_query(query: str) -> Optional[str]:
    """
    Extract lecturer name from query text.

    Uses regex pattern matching to find lecturer names in queries like:
    - "taught by John Doe"
    - "courses taught by Jane Smith"
    - "by Dr. Smith"

    Args:
        query: User query string

    Returns:
        Extracted lecturer name if found, None otherwise
    """
    match = LECTURER_PATTERN.search(query)
    if match:
        return match.group(1).strip()
    return None


# ===== Comparison Query Utilities =====

def is_comparison_query(query: str) -> bool:
    """
    Detect comparison intent in query.

    Looks for keywords like: compare, difference, vs, versus, etc.

    Args:
        query: User query string

    Returns:
        True if this is a comparison query, False otherwise
    """
    ql = query.lower()

    # Check for comparison keywords
    if any(keyword in ql for keyword in COMPARISON_QUERY_KEYWORDS):
        return True

    # Check for "vs" or "vs." pattern
    if re.search(r"\bvs\.?\b", ql):
        return True

    return False


def extract_course_mentions(query: str) -> List[str]:
    """
    Extract course titles/codes mentioned in comparison queries.

    Handles multiple patterns:
    1. Text in quotes: 'Course Name' or "Course Name"
    2. Direct course codes: 2001WETGDT
    3. "between X and Y" patterns
    4. Split on conjunctions: and, vs, versus, commas

    Args:
        query: User query string

    Returns:
        List of extracted course mentions (deduplicated, order preserved)
    """
    mentions: List[str] = []

    # 1) Text in quotes
    for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", query):
        text = match[0] or match[1]
        if text and len(text.strip()) >= 2:
            mentions.append(text.strip())

    # 2) Direct course codes
    for match in re.findall(COURSE_CODE_REGEX, query.upper()):
        mentions.append(match.strip())

    # 3) "between X and Y" pattern
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)", query, flags=re.IGNORECASE)
    if match:
        a = re.sub(r"[?.!]+$", "", match.group(1)).strip()
        b = re.sub(r"[?.!]+$", "", match.group(2)).strip()
        if a:
            mentions.append(a)
        if b:
            mentions.append(b)

    # 4) Split on conjunctions as fallback
    if len(mentions) < 2:
        parts = re.split(r"\b(?:and|vs\.?|versus|,|؛|،)\b", query, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip(" ?!.·-–—\"'")
            if len(part) >= 3 and not re.fullmatch(
                r"(compare|between|vs|and|versus|courses?)",
                part,
                flags=re.IGNORECASE
            ):
                mentions.append(part)

    # Remove duplicates while preserving order
    seen = set()
    unique_mentions = []
    for mention in mentions:
        key = mention.lower()
        if key not in seen:
            unique_mentions.append(mention)
            seen.add(key)

    return unique_mentions


# ===== Section Inference =====

def infer_target_sections(query: str) -> Optional[List[str]]:
    """
    Infer target course sections from query phrasing.

    Analyzes query keywords to determine which course sections
    the user is interested in (prerequisites, assessment, etc.).

    Args:
        query: User query string

    Returns:
        List of relevant section titles if detected, None otherwise
    """
    q = query.lower()

    if "prereq" in q or "prerequisites" in q or "prerequisite" in q:
        return SECTION_KEYWORDS["prereq"]

    if "assessment" in q or "exam" in q or "grading" in q:
        return SECTION_KEYWORDS["assessment"]

    if "learning outcome" in q:
        return SECTION_KEYWORDS["learning"]

    if "teaching method" in q or "planned learning" in q:
        return SECTION_KEYWORDS["teaching"]

    if "content" in q or "summary" in q:
        return SECTION_KEYWORDS["contents"]

    return None
