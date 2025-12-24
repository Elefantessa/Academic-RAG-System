"""
Unit Tests for Models Module
"""

import pytest
from models.state import RAGResponse
from models.confidence import ConfidenceMetrics
from models.catalog import MetadataCatalog
from langchain_core.documents import Document


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_create_response(self):
        response = RAGResponse(
            query="Test query",
            answer="Test answer",
            confidence=0.85,
            sources=["source1", "source2"],
            generation_mode="standard",
            processing_time=1.5,
            reasoning_steps=["step1"],
            conflicts_detected=[],
            metadata={"key": "value"}
        )

        assert response.query == "Test query"
        assert response.confidence == 0.85
        assert response.generation_mode == "standard"

    def test_to_dict(self):
        response = RAGResponse(
            query="Test",
            answer="Answer",
            confidence=0.9,
            sources=[],
            generation_mode="standard",
            processing_time=1.0,
            reasoning_steps=[],
            conflicts_detected=[],
            metadata={}
        )

        result = response.to_dict()
        assert isinstance(result, dict)
        assert result["query"] == "Test"
        assert result["confidence"] == 0.9

    def test_to_json_safe(self):
        response = RAGResponse(
            query="Test",
            answer="Answer",
            confidence=0.9,
            sources=[],
            generation_mode="standard",
            processing_time=1.0,
            reasoning_steps=[],
            conflicts_detected=[],
            metadata={"normal": "value"}
        )

        result = response.to_json_safe()
        assert isinstance(result, dict)


class TestConfidenceMetrics:
    """Tests for ConfidenceMetrics dataclass."""

    def test_create_metrics(self):
        metrics = ConfidenceMetrics(
            rerank_score=0.8,
            entity_match_ratio=0.7,
            source_diversity=0.6,
            context_completeness=0.5,
            semantic_coherence=0.9,
            final_confidence=0.75,
            reasoning="Test reasoning"
        )

        assert metrics.rerank_score == 0.8
        assert metrics.final_confidence == 0.75

    def test_validation_rejects_invalid_range(self):
        with pytest.raises(ValueError):
            ConfidenceMetrics(
                rerank_score=1.5,  # Invalid: > 1
                entity_match_ratio=0.7,
                source_diversity=0.6,
                context_completeness=0.5,
                semantic_coherence=0.9,
                final_confidence=0.75,
                reasoning="Test"
            )

    def test_validation_rejects_negative(self):
        with pytest.raises(ValueError):
            ConfidenceMetrics(
                rerank_score=-0.1,  # Invalid: < 0
                entity_match_ratio=0.7,
                source_diversity=0.6,
                context_completeness=0.5,
                semantic_coherence=0.9,
                final_confidence=0.75,
                reasoning="Test"
            )


class TestMetadataCatalog:
    """Tests for MetadataCatalog class."""

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(
                page_content="Content 1",
                metadata={
                    "course_code": "2001WETGDT",
                    "course_title": "Data Mining",
                    "file_name": "data_mining.pdf"
                }
            ),
            Document(
                page_content="Content 2",
                metadata={
                    "course_code": "2500WETINT",
                    "course_title": "Internet of Things",
                    "file_name": "iot.pdf"
                }
            ),
        ]

    def test_exists_code(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        assert catalog.exists_code("2001WETGDT") is True
        assert catalog.exists_code("INVALID") is False

    def test_get_title(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        assert catalog.get_title("2001WETGDT") == "Data Mining"
        assert catalog.get_title("INVALID") is None

    def test_fuzzy_title_to_code(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        # Exact match
        result = catalog.fuzzy_title_to_code("Data Mining")
        assert result is not None
        assert result[0] == "2001WETGDT"

        # Similar match
        result = catalog.fuzzy_title_to_code("Internet Things")
        assert result is not None
        assert result[0] == "2500WETINT"

    def test_fuzzy_no_match(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        result = catalog.fuzzy_title_to_code("Completely Different")
        assert result is None

    def test_get_all_codes(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        codes = catalog.get_all_codes()
        assert len(codes) == 2
        assert "2001WETGDT" in codes

    def test_get_catalog_stats(self, sample_docs):
        catalog = MetadataCatalog(sample_docs)

        stats = catalog.get_catalog_stats()
        assert stats["total_course_codes"] == 2
        assert stats["total_unique_titles"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
