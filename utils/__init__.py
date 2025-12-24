"""Utils module for Academic RAG System"""

from .query_analysis import (
    normalize_lecturers_field,
    lecturer_matches,
    is_lecturer_query,
    extract_lecturer_from_query,
    is_comparison_query,
    extract_course_mentions,
    infer_target_sections,
)
from .logging_config import setup_logging, get_logger
from .port_utils import is_port_available, find_available_port

__all__ = [
    # Query analysis
    "normalize_lecturers_field",
    "lecturer_matches",
    "is_lecturer_query",
    "extract_lecturer_from_query",
    "is_comparison_query",
    "extract_course_mentions",
    "infer_target_sections",
    # Logging
    "setup_logging",
    "get_logger",
    # Port utils
    "is_port_available",
    "find_available_port",
]
