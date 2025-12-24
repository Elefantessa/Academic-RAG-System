"""
LangChain Vector Store Manager

Manages ChromaDB vector store with HuggingFace embeddings.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReRanker:
    """
    Cross-Encoder based reranker for document relevance scoring.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Cross-encoder model '{model_name}' loaded.")

    def _sigmoid(self, x):
        """Apply sigmoid to convert logits to probabilities."""
        return 1 / (1 + np.exp(-x))

    def rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: Documents to rerank

        Returns:
            List of (document, score) tuples sorted by score
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]

        raw_scores = self.model.predict(pairs, convert_to_tensor=True, show_progress_bar=False)
        scores = self._sigmoid(raw_scores.cpu().numpy())

        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class LangChainVectorStoreManager:
    """
    Manager for ChromaDB vector store with HuggingFace embeddings.
    """

    def __init__(
        self,
        persist_directory: str,
        model_name: str,
        collection_name: str,
        device: str = "auto"
    ):
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Directory to persist ChromaDB
            model_name: HuggingFace embedding model name
            collection_name: ChromaDB collection name
            device: Device for embeddings (auto/cpu/cuda)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Determine device
        if device != "auto":
            model_kwargs = {'device': device}
        else:
            model_kwargs = {
                'device': 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
            }

        logger.info(f"Using device: {device} for embeddings (Bi-Encoder)")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )

        self.db: Optional[Chroma] = None
        self.reranker = CrossEncoderReRanker()

    def _get_db_instance(self) -> Chroma:
        """Get or create ChromaDB instance."""
        if not self.db:
            logger.info(
                f"Connecting to Chroma DB at '{self.persist_directory}' "
                f"with collection '{self.collection_name}'"
            )
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
        return self.db

    def ingest_documents(self, documents: List[Document], batch_size: int = 2):
        """
        Ingest documents into the vector store.

        Args:
            documents: Documents to ingest
            batch_size: Batch size for ingestion
        """
        if not documents:
            logger.warning("No documents to ingest.")
            return

        logger.info(
            f"Starting ingestion of {len(documents)} documents "
            f"into collection '{self.collection_name}'"
        )

        db = self._get_db_instance()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            db.add_documents(batch)
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            logger.info(f"Ingested batch {batch_num}/{total_batches}")

        logger.info("Ingestion complete.")

    def get_retriever(
        self,
        search_type: str = "similarity",
        score_threshold: float = 0.35,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Get a retriever for the vector store.

        Args:
            search_type: Type of search (similarity, mmr, similarity_score_threshold)
            score_threshold: Score threshold for similarity_score_threshold search
            search_kwargs: Additional search parameters

        Returns:
            LangChain retriever
        """
        if self.db is None:
            logger.info(f"Loading existing DB from {self.persist_directory}")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )

        if search_kwargs is None:
            search_kwargs = {'k': 10}

        if search_type == "similarity_score_threshold":
            search_kwargs['score_threshold'] = score_threshold

        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def retrieve_with_rerank(
        self,
        query: str,
        target_score: float = 0.99,
        filter_dict: Optional[Dict] = None,
        initial_k: int = 25
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve with reranking and cumulative score.

        Args:
            query: Search query
            target_score: Target cumulative score
            filter_dict: Metadata filters
            initial_k: Initial number of candidates

        Returns:
            List of (document, score) tuples
        """
        db = self._get_db_instance()

        logger.info(f"Retrieving top {initial_k} candidates...")
        retriever = db.as_retriever(
            search_kwargs={'k': initial_k, 'filter': filter_dict}
        )
        candidates = retriever.invoke(query)

        if not candidates:
            logger.warning("No candidates found.")
            return []

        logger.info(f"Reranking {len(candidates)} candidates...")
        reranked = self.reranker.rerank(query, candidates)

        # Accumulate to target score
        final_results = []
        cumulative = 0.0

        for doc, score in reranked:
            if score < 0.01:
                continue

            final_results.append((doc, score))
            cumulative += score

            if cumulative >= target_score:
                break

        logger.info(f"Found {len(final_results)} documents with cumulative score {cumulative:.4f}")

        return final_results
