"""
PDF Data Extractor

Extracts structured course information from PDF files.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logger = logging.getLogger(__name__)


# Section headings to split the description
SECTION_HEADINGS = [
    "Is part of the next programmes",
    "Prerequisites",
    "Learning Outcomes",
    "Course Contents",
    "International dimension",
    "Teaching method and planned learning activities",
    "Assessment method and criteria",
    "Study material",
    "Contact information",
    "Tutoring"
]

# Field extraction patterns
FIELD_PATTERNS = {
    "course_code": r"Course Code:\s*(\S+)",
    "study_domain": r"Study Domain:\s*(.+)",
    "semester": r"Semester:\s*(.+)",
    "contact_hours": r"Contact Hours:\s*(\d+)",
    "credits": r"Credits:\s*(\d+)",
    "study_load": r"Study Load \(hours\):\s*(\d+)",
    "contract_restrictions": r"Contract Restrictions:\s*(.+)",
    "language_of_instructions": r"Language of\s*Instructions:\s*([A-Za-z]+)",
    "exam_period": r"Examperiod:\s*(.+)",
    "credit_required_for_degree": r"Credit required to obtain degree:\s*(.+)"
}


class PDFDataExtractor:
    """
    Extracts structured data from academic course PDFs.
    """

    def __init__(self):
        """Initialize extractor and check dependencies."""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    def extract_course_title(self, text: str) -> str:
        """Extract course title from text."""
        for line in text.splitlines():
            line = line.strip()
            if not line or re.match(r"^\d{4}\s*-\s*\d{4}$", line):
                continue
            return line
        return ""

    def extract_key_fields(self, text: str) -> Dict[str, Any]:
        """Extract key course fields using regex patterns."""
        fields = {}
        for key, pattern in FIELD_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                fields[key] = match.group(1).strip()
        return fields

    def extract_lecturer_names(self, text: str) -> List[str]:
        """Extract lecturer names from text."""
        start = text.find("Lecturer(s):")
        if start < 0:
            return []

        snippet = text[start + len("Lecturer(s):"):]
        end = re.search(r"(?m)^Examperiod:", snippet)
        block = snippet[:end.start()] if end else snippet

        names = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("-") or line.lower().startswith("http"):
                continue
            if re.match(r"^M\d{7}", line):
                continue

            match = re.match(r'^[TCM]\s+(.+)', line)
            if match:
                names.append(match.group(1).strip())

        return names

    def extract_non_table_text(self, pdf_path: str) -> str:
        """Extract text excluding table content."""
        def in_bbox(x, y, bbox):
            return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

        lines = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                bboxes = [t.bbox for t in page.find_tables()]
                words = page.extract_words()

                # Filter out words inside tables
                keep = [
                    w for w in words
                    if not any(
                        in_bbox((w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2, b)
                        for b in bboxes
                    )
                ]

                # Sort by position
                keep.sort(key=lambda w: (round(w["top"]), w["x0"]))

                # Group into lines
                buffer, current_y = [], None
                for w in keep:
                    y = round(w["top"])
                    if current_y is None or abs(y - current_y) <= 3:
                        buffer.append(w["text"])
                        current_y = y
                    else:
                        lines.append(" ".join(buffer))
                        buffer = [w["text"]]
                        current_y = y

                if buffer:
                    lines.append(" ".join(buffer))

        return "\n".join(lines)

    def split_description_sections(self, description: str) -> Dict[str, str]:
        """Split description into sections."""
        patterns = {
            h: re.compile(rf"(?mi)^(?:\d+\.\s*)?{re.escape(h)}\b", re.IGNORECASE)
            for h in SECTION_HEADINGS
        }

        sections = {h: "" for h in SECTION_HEADINGS}
        current = None

        for line in description.splitlines():
            for heading, pattern in patterns.items():
                if pattern.match(line):
                    current = heading
                    break
            else:
                if current:
                    sections[current] += line + "\n"

        return {h: sections[h].strip() for h in SECTION_HEADINGS}

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted course data
        """
        logger.info(f"Processing: {pdf_path}")

        # Read full text with PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        # Extract key fields and lecturers
        details = self.extract_key_fields(full_text)
        details["lecturers"] = self.extract_lecturer_names(full_text)

        # Extract non-table text for description
        non_table_text = self.extract_non_table_text(pdf_path)

        return {
            "file_name": os.path.basename(pdf_path),
            "course_title": self.extract_course_title(full_text),
            "course_details": details,
            "course_description_sections": self.split_description_sections(non_table_text),
            "course_description": non_table_text
        }

    def process_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.

        Args:
            input_dir: Directory containing PDF files

        Returns:
            List of extracted course data
        """
        results = []

        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(input_dir, filename)
            try:
                result = self.process_pdf(full_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        logger.info(f"Processed {len(results)} PDFs")
        return results
