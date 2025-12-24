"""
Unit Tests for Query Analysis Module
"""

import pytest
from utils.query_analysis import (
    normalize_lecturers_field,
    lecturer_matches,
    is_lecturer_query,
    extract_lecturer_from_query,
    is_comparison_query,
    extract_course_mentions,
    infer_target_sections,
)


class TestNormalizeLecturersField:
    """Tests for normalize_lecturers_field function."""

    def test_none_returns_empty_string(self):
        assert normalize_lecturers_field(None) == ""

    def test_list_returns_joined_lowercase(self):
        result = normalize_lecturers_field(["John Doe", "Jane Smith"])
        assert result == "john doe jane smith"

    def test_string_returns_lowercase(self):
        assert normalize_lecturers_field("John Doe") == "john doe"

    def test_empty_list_returns_empty(self):
        assert normalize_lecturers_field([]) == ""


class TestLecturerMatches:
    """Tests for lecturer_matches function."""

    def test_matches_when_name_in_metadata(self):
        metadata = {"lecturers": "John Doe, Jane Smith"}
        assert lecturer_matches(metadata, "john doe") is True

    def test_no_match_when_name_not_in_metadata(self):
        metadata = {"lecturers": "John Doe"}
        assert lecturer_matches(metadata, "jane smith") is False

    def test_empty_target_returns_false(self):
        metadata = {"lecturers": "John Doe"}
        assert lecturer_matches(metadata, "") is False

    def test_none_metadata_returns_false(self):
        assert lecturer_matches(None, "john") is False


class TestIsLecturerQuery:
    """Tests for is_lecturer_query function."""

    def test_taught_by_detected(self):
        assert is_lecturer_query("Courses taught by John Doe") is True

    def test_who_teaches_detected(self):
        assert is_lecturer_query("Who teaches Data Mining?") is True

    def test_which_courses_taught_detected(self):
        assert is_lecturer_query("Which courses are taught by Dr. Smith?") is True

    def test_regular_query_not_detected(self):
        assert is_lecturer_query("What are the prerequisites for IoT?") is False

    def test_case_insensitive(self):
        assert is_lecturer_query("TAUGHT BY professor") is True


class TestExtractLecturerFromQuery:
    """Tests for extract_lecturer_from_query function."""

    def test_extracts_name_after_taught_by(self):
        result = extract_lecturer_from_query("Courses taught by John Doe")
        assert result == "John Doe"

    def test_extracts_name_after_by(self):
        result = extract_lecturer_from_query("by Jane Smith")
        assert result == "Jane Smith"

    def test_returns_none_when_no_match(self):
        result = extract_lecturer_from_query("What is IoT about?")
        assert result is None

    def test_handles_complex_names(self):
        result = extract_lecturer_from_query("taught by Dr. John Van Der Berg")
        assert "John Van Der Berg" in result


class TestIsComparisonQuery:
    """Tests for is_comparison_query function."""

    def test_compare_detected(self):
        assert is_comparison_query("Compare IoT and Data Mining") is True

    def test_difference_between_detected(self):
        assert is_comparison_query("What is the difference between A and B?") is True

    def test_vs_detected(self):
        assert is_comparison_query("IoT vs Machine Learning") is True

    def test_versus_detected(self):
        assert is_comparison_query("IoT versus ML") is True

    def test_regular_query_not_comparison(self):
        assert is_comparison_query("Tell me about IoT") is False


class TestExtractCourseMentions:
    """Tests for extract_course_mentions function."""

    def test_extracts_quoted_text(self):
        result = extract_course_mentions("Compare 'Data Mining' and 'IoT'")
        assert "Data Mining" in result
        assert "IoT" in result

    def test_extracts_course_codes(self):
        result = extract_course_mentions("Compare 2001WETGDT and 2500WETINT")
        assert "2001WETGDT" in result
        assert "2500WETINT" in result

    def test_extracts_between_and_pattern(self):
        result = extract_course_mentions("difference between IoT and ML")
        assert len(result) >= 2

    def test_removes_duplicates(self):
        result = extract_course_mentions("'IoT' and 'IoT' vs 'ML'")
        iot_count = sum(1 for m in result if m.lower() == "iot")
        assert iot_count == 1


class TestInferTargetSections:
    """Tests for infer_target_sections function."""

    def test_prereq_detected(self):
        result = infer_target_sections("What are the prerequisites?")
        assert result is not None
        assert "Prerequisites" in result

    def test_assessment_detected(self):
        result = infer_target_sections("How is the exam graded?")
        assert result is not None

    def test_learning_outcomes_detected(self):
        result = infer_target_sections("What are the learning outcomes?")
        assert result is not None

    def test_teaching_detected(self):
        result = infer_target_sections("What is the teaching method?")
        assert result is not None

    def test_content_detected(self):
        result = infer_target_sections("What is the course content?")
        assert result is not None

    def test_general_query_returns_none(self):
        result = infer_target_sections("Tell me about IoT")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
