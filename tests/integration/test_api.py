"""
Integration Tests for API Module
"""

import pytest
from flask import Flask
from api.app import create_app


class MockAgent:
    """Mock agent for testing API without full initialization."""

    def __init__(self):
        self.stats = {
            "total_queries": 5,
            "successful_queries": 4,
            "average_time": 1.5,
            "mode_usage": {"standard": 3, "comparison": 1, "lecturer": 1}
        }

        # Mock catalog
        self.catalog = MockCatalog()

    def process_query(self, query: str):
        from models.state import RAGResponse
        return RAGResponse(
            query=query,
            answer=f"Answer to: {query}",
            confidence=0.85,
            sources=["source1"],
            generation_mode="standard",
            processing_time=1.0,
            reasoning_steps=["processed"],
            conflicts_detected=[],
            metadata={"test": True}
        )

    def get_stats(self):
        return self.stats


class MockCatalog:
    """Mock catalog for testing."""

    def get_catalog_stats(self):
        return {
            "total_course_codes": 10,
            "total_unique_titles": 10,
            "total_files": 10
        }

    def get_all_codes(self):
        return ["2001WETGDT", "2500WETINT"]

    def get_all_titles(self):
        return ["Data Mining", "Internet of Things"]


@pytest.fixture
def app():
    """Create test application."""
    agent = MockAgent()
    app = create_app(agent)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get('/api/health')
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        response = client.get('/api/health')
        data = response.get_json()
        assert data['status'] == 'healthy'


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_returns_200(self, client):
        response = client.get('/api/stats')
        assert response.status_code == 200

    def test_stats_returns_correct_data(self, client):
        response = client.get('/api/stats')
        data = response.get_json()
        assert 'total_queries' in data
        assert 'mode_usage' in data


class TestCatalogEndpoint:
    """Tests for /api/catalog endpoint."""

    def test_catalog_returns_200(self, client):
        response = client.get('/api/catalog')
        assert response.status_code == 200

    def test_catalog_returns_stats(self, client):
        response = client.get('/api/catalog')
        data = response.get_json()
        assert 'stats' in data
        assert 'sample_codes' in data


class TestQueryEndpoint:
    """Tests for /api/query endpoint."""

    def test_query_returns_200(self, client):
        response = client.post('/api/query',
                              json={'query': 'What is IoT?'},
                              content_type='application/json')
        assert response.status_code == 200

    def test_query_returns_answer(self, client):
        response = client.post('/api/query',
                              json={'query': 'Test query'},
                              content_type='application/json')
        data = response.get_json()
        assert 'answer' in data
        assert 'confidence' in data

    def test_query_requires_body(self, client):
        response = client.post('/api/query',
                              content_type='application/json')
        assert response.status_code == 400

    def test_query_requires_query_field(self, client):
        response = client.post('/api/query',
                              json={'other': 'field'},
                              content_type='application/json')
        assert response.status_code == 400

    def test_empty_query_rejected(self, client):
        response = client.post('/api/query',
                              json={'query': '   '},
                              content_type='application/json')
        assert response.status_code == 400


class TestWebInterface:
    """Tests for web interface."""

    def test_index_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_index_returns_html(self, client):
        response = client.get('/')
        assert b'<!DOCTYPE html>' in response.data
        assert b'Academic RAG' in response.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
