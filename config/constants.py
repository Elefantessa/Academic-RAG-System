"""
Constants and Patterns for Academic RAG System

This module contains all constant values, regex patterns, and keyword mappings
used throughout the application.
"""

import re
from typing import Dict, List

# ===== Regex Patterns =====

# Course code pattern (e.g., 2001WETGDT, 2500WETINT)
COURSE_CODE_REGEX = r"\b\d{4}[A-Z]{3,}[A-Z0-9]*\b"
COURSE_CODE_PATTERN = re.compile(COURSE_CODE_REGEX)

# Lecturer name pattern
LECTURER_PATTERN = re.compile(
    r"(?:taught\s+by|courses\s+taught\s+by|by)\s+([A-Z][A-Za-z .'\-]+)",
    flags=re.IGNORECASE
)

# ===== Section Keywords =====

SECTION_KEYWORDS: Dict[str, List[str]] = {
    "prereq": ["Prerequisites"],
    "assessment": [
        "Assessment method and criteria",
        "Merged: Assessment method and criteria"
    ],
    "learning": ["Learning Outcomes"],
    "teaching": ["Teaching method and planned learning activities"],
    "contents": ["Course Contents", "Course Summary", "Study material"],
}

# ===== Confidence Calculation Weights =====

CONFIDENCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "comparison": {
        "rerank": 0.25,
        "entity": 0.20,
        "source": 0.25,
        "completeness": 0.15,
        "semantic": 0.15
    },
    "lecturer": {
        "rerank": 0.20,
        "entity": 0.30,
        "source": 0.20,
        "completeness": 0.10,
        "semantic": 0.20
    },
    "standard": {
        "rerank": 0.30,
        "entity": 0.20,
        "source": 0.15,
        "completeness": 0.15,
        "semantic": 0.20
    }
}

# ===== Query Type Keywords =====

LECTURER_QUERY_KEYWORDS = [
    "taught by",
    "who teaches",
    "which courses",
    "courses taught"
]

COMPARISON_QUERY_KEYWORDS = [
    "compare",
    "comparison",
    "difference between",
    "differences between",
    "versus",
    "compare between"
]

# ===== Default Values =====

DEFAULT_FUZZY_MATCH_CUTOFF = 0.78
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 12
DEFAULT_LECTURER_K = 40
DEFAULT_RERANK_TOP_K = 5

# ===== Constraint Values =====

MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 1
MAX_CONTEXT_DOCS = 50
MAX_RERANK_CANDIDATES = 100
