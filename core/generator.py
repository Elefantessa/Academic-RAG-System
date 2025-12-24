"""
Answer Generation Module

LLM-based answer generation with specialized modes.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from utils.query_analysis import is_lecturer_query, is_comparison_query, lecturer_matches

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    LLM-based answer generation with specialized modes.

    Supports three generation modes:
    - standard: General question answering
    - comparison: Course comparison with structured output
    - lecturer: Deterministic lecturer query responses
    """

    def __init__(self, llm: ChatOllama):
        """
        Initialize answer generator.

        Args:
            llm: ChatOllama instance for generation
        """
        self.llm = llm

    def generate(
        self,
        query: str,
        documents: List[Document],
        extracted: Dict[str, Any],
        mode: Optional[str] = None
    ) -> str:
        """
        Generate answer based on query and context.

        Args:
            query: User query
            documents: Context documents
            extracted: Extracted entities
            mode: Generation mode (auto-detected if None)

        Returns:
            Generated answer string
        """
        if not mode:
            mode = self.determine_mode(query, extracted)

        if mode == "lecturer":
            return self._generate_lecturer_answer(query, documents, extracted)
        elif mode == "comparison":
            return self._generate_comparison_answer(query, documents, extracted)
        else:
            return self._generate_standard_answer(query, documents)

    def determine_mode(self, query: str, extracted: Dict[str, Any]) -> str:
        """Determine the appropriate generation mode."""
        if is_lecturer_query(query):
            return "lecturer"
        if is_comparison_query(query) or extracted.get("comparison_codes"):
            return "comparison"
        return "standard"

    def _generate_standard_answer(
        self,
        query: str,
        documents: List[Document]
    ) -> str:
        """Generate standard answer from context."""
        if not documents:
            return "I don't have enough information to answer this question."

        context = self._build_context(documents)

        prompt = f"""Based on the following course information, answer the question.
Be specific and cite course details when possible.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error generating the answer."

    def _generate_lecturer_answer(
        self,
        query: str,
        documents: List[Document],
        extracted: Dict[str, Any]
    ) -> str:
        """Generate deterministic answer for lecturer queries."""
        lecturers = extracted.get("lecturers", [])

        if not lecturers:
            return "I couldn't identify the lecturer name from your query."

        target = lecturers[0].lower()

        # Filter documents by lecturer
        matching_docs = [
            doc for doc in documents
            if lecturer_matches(doc.metadata, target)
        ]

        if not matching_docs:
            return f"I couldn't find any courses taught by '{lecturers[0]}' in the available data."

        # Extract unique courses
        courses = {}
        for doc in matching_docs:
            code = (doc.metadata or {}).get("course_code", "")
            title = (doc.metadata or {}).get("course_title", "")
            if code and code not in courses:
                courses[code] = title

        if not courses:
            return f"I found documents mentioning '{lecturers[0]}' but couldn't extract specific course information."

        # Format response
        course_list = "\n".join([
            f"- **{code}**: {title}" for code, title in courses.items()
        ])

        return f"""**Courses taught by {lecturers[0]}:**

{course_list}

Found {len(courses)} course(s) in the database."""

    def _generate_comparison_answer(
        self,
        query: str,
        documents: List[Document],
        extracted: Dict[str, Any]
    ) -> str:
        """Generate structured comparison answer."""
        comparison_codes = extracted.get("comparison_codes", [])

        if len(comparison_codes) < 2:
            # Fall back to standard if not enough courses
            return self._generate_standard_answer(query, documents)

        # Group documents by course
        docs_by_course: Dict[str, List[Document]] = {code: [] for code in comparison_codes}

        for doc in documents:
            code = (doc.metadata or {}).get("course_code", "")
            if code in docs_by_course:
                docs_by_course[code].append(doc)

        # Build comparison context
        context_parts = []
        for code in comparison_codes:
            course_docs = docs_by_course.get(code, [])
            if course_docs:
                title = (course_docs[0].metadata or {}).get("course_title", code)
                doc_content = "\n".join([
                    f"[{(d.metadata or {}).get('section_title', 'Section')}]: {d.page_content[:500]}"
                    for d in course_docs[:3]
                ])
                context_parts.append(f"## {title} ({code})\n{doc_content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Compare the following courses based on the user's question.
Provide a structured comparison highlighting key differences and similarities.

{context}

Question: {query}

Comparison:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            return "Sorry, I encountered an error generating the comparison."

    def _build_context(self, documents: List[Document], max_length: int = 4000) -> str:
        """Build context string from documents."""
        context_parts = []
        current_length = 0

        for doc in documents:
            metadata = doc.metadata or {}
            header = (
                f"[{metadata.get('course_code', '?')} - "
                f"{metadata.get('section_title', '?')}]"
            )
            content = f"{header}\n{doc.page_content}"

            if current_length + len(content) > max_length:
                break

            context_parts.append(content)
            current_length += len(content)

        return "\n\n".join(context_parts)
