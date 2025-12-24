"""
Unit Tests for Config Module
"""

import pytest
import os
from config.settings import AppSettings
from config.constants import (
    COURSE_CODE_REGEX,
    COURSE_CODE_PATTERN,
    SECTION_KEYWORDS,
    CONFIDENCE_WEIGHTS,
)


class TestAppSettings:
    """Tests for AppSettings class."""

    def test_default_values(self):
        settings = AppSettings()

        assert settings.port == 5003
        assert settings.host == "127.0.0.1"
        assert settings.debug is False
        assert settings.ollama_model == "llama3.1:latest"
        assert settings.default_k == 12
        assert settings.lecturer_k == 40

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("RAG_PORT", "8080")
        monkeypatch.setenv("RAG_DEBUG", "true")

        settings = AppSettings()

        assert settings.port == 8080
        assert settings.debug is True


class TestConstants:
    """Tests for constants module."""

    def test_course_code_regex_matches_valid(self):
        import re

        valid_codes = ["2001WETGDT", "2500WETINT", "1234ABCXYZ"]
        for code in valid_codes:
            assert re.match(COURSE_CODE_REGEX, code) is not None

    def test_course_code_regex_rejects_invalid(self):
        import re

        invalid_codes = ["ABC123", "123", "WETGDT", "12WETGDT"]
        for code in invalid_codes:
            assert re.fullmatch(COURSE_CODE_REGEX, code) is None

    def test_section_keywords_structure(self):
        assert "prereq" in SECTION_KEYWORDS
        assert "assessment" in SECTION_KEYWORDS
        assert "learning" in SECTION_KEYWORDS
        assert "teaching" in SECTION_KEYWORDS
        assert "contents" in SECTION_KEYWORDS

        # All values should be lists
        for value in SECTION_KEYWORDS.values():
            assert isinstance(value, list)

    def test_confidence_weights_structure(self):
        assert "comparison" in CONFIDENCE_WEIGHTS
        assert "lecturer" in CONFIDENCE_WEIGHTS
        assert "standard" in CONFIDENCE_WEIGHTS

        # Each mode should have all required weights
        required_keys = ["rerank", "entity", "source", "completeness", "semantic"]
        for mode, weights in CONFIDENCE_WEIGHTS.items():
            for key in required_keys:
                assert key in weights

            # Weights should sum to approximately 1
            total = sum(weights.values())
            assert 0.99 <= total <= 1.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
