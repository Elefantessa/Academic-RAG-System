#!/usr/bin/env python3
"""
PDF Extraction Script

Extracts course data from PDF files and creates JSON output.
Use this when you want to:
- Process new PDF files
- Recreate the JSON data from scratch
- Update the data after adding new PDFs

Usage:
    python scripts/extract_pdfs.py                     # Use defaults
    python scripts/extract_pdfs.py --input-dir path    # Custom input
    python scripts/extract_pdfs.py --output path       # Custom output
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.extractors import PDFDataExtractor
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Default paths
DEFAULT_INPUT_DIR = "/project_antwerp/pdf_pipline/data/upload"
DEFAULT_OUTPUT_FILE = "/project_antwerp/pdf_pipline/data/processed/all_extracted.json"


def main():
    parser = argparse.ArgumentParser(
        description="Extract course data from PDF files"
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_INPUT_DIR})"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📄 Academic RAG - PDF Extraction Script")
    print("=" * 60)

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory not found: {args.input_dir}")
        return 1

    # Count PDFs
    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')]
    print(f"\n📂 Input directory: {args.input_dir}")
    print(f"   Found {len(pdf_files)} PDF files")

    if not pdf_files:
        logger.error("❌ No PDF files found!")
        return 1

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process PDFs
    print(f"\n🔄 Processing PDFs...")

    extractor = PDFDataExtractor()
    results = extractor.process_directory(args.input_dir)

    # Save results
    print(f"\n💾 Saving to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Done
    print("\n" + "=" * 60)
    print("✅ Extraction Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   - PDFs processed: {len(results)}")
    print(f"   - Output file: {args.output}")
    print(f"\n🚀 Next step - ingest into database:")
    print(f"   python scripts/ingest_data.py --json-file {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
