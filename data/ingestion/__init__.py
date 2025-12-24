"""
Data Ingestion Module

Provides document chunking and vector store management.
"""

from .chunker import AdvancedAcademicChunker
from .vector_store import LangChainVectorStoreManager, CrossEncoderReRanker

__all__ = [
    "AdvancedAcademicChunker",
    "LangChainVectorStoreManager",
    "CrossEncoderReRanker",
]
