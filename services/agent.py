"""
Main RAG Agent Service

Orchestrates the complete retrieval and generation pipeline.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config.settings import AppSettings
from models.state import RAGResponse
from models.catalog import MetadataCatalog
from core.extractors import EntityExtractor
from core.retriever import VectorRetriever
from core.reranker import DocumentReranker
from core.context_expander import ContextExpander
from core.generator import AnswerGenerator
from services.confidence_calculator import ConfidenceCalculator
from utils.query_analysis import is_lecturer_query, is_comparison_query

logger = logging.getLogger(__name__)


class ContextAwareRetrievalAgent:
    """
    Main orchestrator for the RAG pipeline.

    Coordinates:
    1. Entity extraction
    2. Vector retrieval
    3. Cross-encoder reranking
    4. Context expansion
    5. Answer generation
    6. Confidence calculation
    """

    def __init__(
        self,
        vectorstore,
        all_documents: List[Document],
        settings: Optional[AppSettings] = None,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "llama3.1:latest"
    ):
        """
        Initialize the RAG agent.

        Args:
            vectorstore: LangChain VectorStore instance
            all_documents: All documents for catalog building
            settings: Application settings
            ollama_base_url: Ollama service URL
            model_name: LLM model name
        """
        self.settings = settings or AppSettings()

        # Build catalog
        self.catalog = MetadataCatalog(all_documents)
        logger.info(f"Catalog built: {self.catalog.get_catalog_stats()}")

        # Initialize LLMs
        self.extraction_llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0,
            format="json"
        )

        self.generation_llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0
        )

        # Initialize components
        self.extractor = EntityExtractor(self.catalog, self.extraction_llm)
        self.retriever = VectorRetriever(
            vectorstore,
            default_k=self.settings.default_k,
            lecturer_k=self.settings.lecturer_k
        )
        self.reranker = DocumentReranker(self.settings.rerank_model)
        self.expander = ContextExpander(self.retriever.filter_only_fetch)
        self.generator = AnswerGenerator(self.generation_llm)
        self.confidence = ConfidenceCalculator(ollama_base_url)

        # Stats
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_time": 0.0,
            "mode_usage": {"standard": 0, "comparison": 0, "lecturer": 0}
        }

        logger.info("ContextAwareRetrievalAgent initialized successfully")

    def process_query(self, query: str) -> RAGResponse:
        """
        Process a user query through the complete pipeline.

        Args:
            query: User query string

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        try:
            logger.info(f"Processing: {query[:100]}...")

            # Step 1: Extract entities
            extracted = self.extractor.extract(query)
            logger.info(f"Extracted: {extracted}")

            # Step 2: Determine mode
            mode = self._determine_mode(query, extracted)
            self.stats["mode_usage"][mode] += 1

            # Step 3: Retrieve documents
            docs = self._retrieve(query, extracted, mode)
            logger.info(f"Retrieved: {len(docs)} documents")

            # Step 4: Rerank
            reranked, scores = self.reranker.rerank(
                query, docs, self.settings.top_k_rerank
            )

            # Step 5: Expand context
            if mode == "comparison" and extracted.get("comparison_codes"):
                axes = ContextExpander.infer_comparison_axes(query)
                expanded = self.expander.expand_for_comparison(
                    reranked, extracted["comparison_codes"], axes
                )
            else:
                expanded = self.expander.expand(reranked, query)

            # Step 6: Generate answer
            answer = self.generator.generate(query, expanded, extracted, mode)

            # Step 7: Calculate confidence
            try:
                conf_metrics = self.confidence.calculate_confidence(
                    query=query,
                    answer=answer,
                    retrieved_docs=docs,
                    reranked_docs=reranked,
                    rerank_scores=scores,
                    extracted_entities=extracted,
                    generation_mode=mode
                )
                confidence = conf_metrics.final_confidence
                reasoning_steps = [conf_metrics.reasoning]
            except Exception as e:
                logger.warning(f"Confidence calculation failed: {e}")
                confidence = 0.5
                reasoning_steps = ["Confidence calculation skipped"]

            processing_time = time.time() - start_time
            self.stats["successful_queries"] += 1

            # Update average time
            n = self.stats["successful_queries"]
            self.stats["average_time"] = (
                (self.stats["average_time"] * (n - 1) + processing_time) / n
            )

            return RAGResponse(
                query=query,
                answer=answer,
                confidence=confidence,
                sources=self._extract_sources(expanded),
                generation_mode=mode,
                processing_time=processing_time,
                reasoning_steps=reasoning_steps,
                conflicts_detected=[],
                metadata={
                    "extracted": extracted,
                    "doc_count": len(expanded),
                    "mode": mode
                }
            )

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                sources=[],
                generation_mode="error",
                processing_time=time.time() - start_time,
                reasoning_steps=[f"Error: {str(e)}"],
                conflicts_detected=[],
                metadata={"error": str(e)}
            )

    def _determine_mode(self, query: str, extracted: Dict[str, Any]) -> str:
        """Determine processing mode."""
        if is_lecturer_query(query):
            return "lecturer"
        if is_comparison_query(query) or extracted.get("comparison_codes"):
            return "comparison"
        return "standard"

    def _retrieve(
        self,
        query: str,
        extracted: Dict[str, Any],
        mode: str
    ) -> List[Document]:
        """Retrieve documents based on mode."""
        filters = {}

        if extracted.get("course_code"):
            filters["course_code"] = extracted["course_code"]

        if mode == "lecturer":
            return self.retriever.search_for_lecturer(query)
        else:
            return self.retriever.search(query, filters)

    def _extract_sources(self, docs: List[Document]) -> List[str]:
        """Extract source identifiers from documents."""
        sources = []
        seen = set()

        for doc in docs:
            meta = doc.metadata or {}
            code = meta.get("course_code", "")
            section = meta.get("section_title", "")

            source = f"{code}:{section}" if code else section

            if source and source not in seen:
                sources.append(source)
                seen.add(source)

        return sources

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.stats.copy()
