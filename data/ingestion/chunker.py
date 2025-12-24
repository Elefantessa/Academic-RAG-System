"""
Advanced Academic Chunker

Specialized document chunking for academic course data.
"""

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class AdvancedAcademicChunker:
    """
    A class that specializes in chunking academic course data from JSON.
    It maintains context by merging small sections and creating summaries.
    """

    def __init__(
        self,
        min_chunk_size: int = 250,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        """
        Initialize the chunker.

        Args:
            min_chunk_size: Minimum size for a chunk
            max_chunk_size: Maximum size for a chunk
            chunk_overlap: Overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n- ", "\n\n", "\n", ". ", " "]
        )

    def chunk_course_from_json(self, course_data: Dict[str, Any]) -> List[Document]:
        """
        Process a course JSON object into documents.

        Args:
            course_data: Dictionary containing course information

        Returns:
            List of Document objects
        """
        final_docs: List[Document] = []
        details = course_data.get("course_details", {})

        # Build base metadata
        base_metadata = {
            "course_title": course_data.get("course_title", "N/A"),
            "course_code": details.get("course_code", "N/A"),
            "study_domain": details.get("study_domain", "N/A"),
            "semester": details.get("semester", "N/A"),
            "contact_hours": str(details.get("contact_hours", "N/A")),
            "credits": str(details.get("credits", "N/A")),
            "study_load": str(details.get("study_load", "N/A")),
            "contract_restrictions": details.get("contract_restrictions", "N/A"),
            "language": details.get("language_of_instructions", "N/A"),
            "exam_period": details.get("exam_period", "N/A"),
            "lecturers": ", ".join(details.get("lecturers", [])),
            "file_name": course_data.get("file_name", "N/A")
        }

        # Create summary document
        summary_doc = self._create_summary_document(base_metadata)
        final_docs.append(summary_doc)

        # Process sections
        sections = course_data.get("course_description_sections", {})
        small_sections: List[Dict] = []

        for section_title, content in sections.items():
            if not content or not isinstance(content, str) or not content.strip():
                continue

            content_len = len(content)
            header = (
                f"Regarding the course '{base_metadata['course_title']}' "
                f"({base_metadata['course_code']}). "
                f"This section describes '{section_title}':\n\n"
            )

            if content_len > self.max_chunk_size:
                # Large section: flush buffer and split
                if small_sections:
                    final_docs.extend(self._create_merged_documents(small_sections, base_metadata))
                    small_sections = []

                sub_chunks = self.text_splitter.split_text(content)
                for i, sub in enumerate(sub_chunks):
                    meta = {
                        **base_metadata,
                        "section_title": section_title,
                        "part_of_section": f"{i+1}/{len(sub_chunks)}"
                    }
                    final_docs.append(Document(page_content=header + sub, metadata=meta))

            elif content_len < self.min_chunk_size:
                # Small section: buffer for merging
                small_sections.append({"title": section_title, "content": content})

            else:
                # Normal section
                if small_sections:
                    final_docs.extend(self._create_merged_documents(small_sections, base_metadata))
                    small_sections = []

                meta = {**base_metadata, "section_title": section_title, "part_of_section": "1/1"}
                final_docs.append(Document(page_content=header + content, metadata=meta))

        # Flush remaining small sections
        if small_sections:
            final_docs.extend(self._create_merged_documents(small_sections, base_metadata))

        return final_docs

    def _create_summary_document(self, metadata: Dict[str, Any]) -> Document:
        """Create a summary document for the course."""
        summary_lines = [
            f"This is a summary for the course '{metadata['course_title']}' ({metadata['course_code']}).",
            f" - Study Domain: {metadata['study_domain']}",
            f" - Language: {metadata['language']}",
            f" - ECTS Credits: {metadata['credits']}",
            f" - Semester: {metadata['semester']}",
            f" - Study Load: {metadata['study_load']} hours",
            f" - Contact Hours: {metadata['contact_hours']} hours",
            f" - Lecturers: {metadata['lecturers']}",
            f" - Exam Period: {metadata['exam_period']}",
            f" - Contract Restrictions: {metadata['contract_restrictions']}"
        ]

        meta = {**metadata, "section_title": "Course Summary", "part_of_section": "1/1"}
        return Document(page_content="\n".join(summary_lines), metadata=meta)

    def _create_merged_documents(
        self,
        sections: List[Dict],
        base_metadata: Dict
    ) -> List[Document]:
        """Merge small sections into larger documents."""
        docs = []
        content_acc = ""
        titles_acc = []

        for sec in sections:
            body = f"--- Section: {sec['title']} ---\n{sec['content'].strip()}"

            if len(content_acc) + len(body) > self.max_chunk_size and content_acc:
                docs.append(self._finalize_merged(content_acc, titles_acc, base_metadata))
                content_acc, titles_acc = "", []

            content_acc += body + "\n\n"
            titles_acc.append(sec['title'])

        if content_acc:
            docs.append(self._finalize_merged(content_acc, titles_acc, base_metadata))

        return docs

    def _finalize_merged(
        self,
        content: str,
        titles: List[str],
        base_metadata: Dict
    ) -> Document:
        """Finalize a merged document."""
        merged_titles = ", ".join(titles)
        header = (
            f"This document contains merged sections ({merged_titles}) "
            f"for the course '{base_metadata['course_title']}' "
            f"({base_metadata['course_code']}).\n\n"
        )

        meta = {
            **base_metadata,
            "section_title": f"Merged: {merged_titles}",
            "part_of_section": "1/1"
        }

        return Document(page_content=header + content.strip(), metadata=meta)
