"""
Vector Retrieval Module

Vector search with MMR and metadata filtering.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Vector search with MMR (Maximum Marginal Relevance) and metadata filtering.

    This class provides a clean interface for document retrieval with support
    for different retrieval modes (lecturer, comparison, standard).
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        default_k: int = 12,
        lecturer_k: int = 40
    ):
        """
        Initialize vector retriever.

        Args:
            vectorstore: LangChain VectorStore instance
            default_k: Default number of documents to retrieve
            lecturer_k: Number of documents for lecturer queries
        """
        self.vectorstore = vectorstore
        self.default_k = default_k
        self.lecturer_k = lecturer_k

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k_override: Optional[int] = None,
        search_type: str = "mmr"
    ) -> List[Document]:
        """
        Execute vector search with optional metadata filters.

        Args:
            query: Search query
            filters: Optional metadata filters
            k_override: Override default k value
            search_type: Type of search ("mmr" or "similarity")

        Returns:
            List of retrieved documents
        """
        k = k_override or self.default_k
        search_kwargs: Dict[str, Any] = {'k': k}

        # Process filters
        if filters:
            processed_filter = self._process_filters(filters)
            if processed_filter:
                search_kwargs['filter'] = processed_filter

        logger.info(f"Searching with query='{query[:50]}...', k={k}, filters={filters}")

        try:
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            return retriever.invoke(query)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_for_lecturer(
        self,
        query: str,
        lecturer_name: Optional[str] = None
    ) -> List[Document]:
        """
        Search with larger k for lecturer queries.

        Args:
            query: Search query
            lecturer_name: Optional lecturer name filter

        Returns:
            List of retrieved documents
        """
        filters = None
        if lecturer_name:
            # Note: ChromaDB may require specific filter format for lecturers
            filters = {}  # Lecturer filtering done post-retrieval

        return self.search(query, filters=filters, k_override=self.lecturer_k)

    def filter_only_fetch(
        self,
        filter_dict: Dict[str, Any],
        k: int = 6
    ) -> List[Document]:
        """
        Fetch documents using filters only with neutral query.

        Args:
            filter_dict: Metadata filters
            k: Number of documents to retrieve

        Returns:
            List of filtered documents
        """
        try:
            and_filter = {'$and': [{field: value} for field, value in filter_dict.items()]}
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': k, 'filter': and_filter}
            )
            return retriever.invoke(" ")  # Neutral query
        except Exception as e:
            logger.error(f"Filter-only fetch failed: {e}")
            return []

    def _process_filters(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process and validate metadata filters for ChromaDB.

        Args:
            filters: Raw filter dictionary

        Returns:
            Processed filter dictionary or None
        """
        processed: Dict[str, Any] = {}

        for key, value in filters.items():
            # Skip empty or None values
            if value is None or value == "":
                continue
            # Skip lecturer filter (handled separately)
            if key == 'lecturers':
                continue

            if isinstance(value, list):
                if not value:
                    continue
                processed[key] = {"$eq": value[0]}
            else:
                processed[key] = {"$eq": value}

        if not processed:
            return None

        if len(processed) > 1:
            return {"$and": [{k: v} for k, v in processed.items()]}
        else:
            k, v = next(iter(processed.items()))
            return {k: v}
