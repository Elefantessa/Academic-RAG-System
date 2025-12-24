#!/usr/bin/env python3
"""
Data Ingestion Script

This script creates/recreates the ChromaDB vector database from scratch.
Use this when you want to:
- Create a new database
- Rebuild the database after data changes
- Clear and reingest all documents

Usage:
    python scripts/ingest_data.py                    # Ingest to existing DB
    python scripts/ingest_data.py --clean            # Clear and reingest
    python scripts/ingest_data.py --json-file path   # Use custom data file
"""

import argparse
import json
import os
import sys
import shutil
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from data.ingestion import AdvancedAcademicChunker, LangChainVectorStoreManager
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest course data into ChromaDB vector database"
    )

    parser.add_argument(
        "--json-file",
        default=settings.json_file,
        help=f"Path to JSON data file (default: {settings.json_file})"
    )

    parser.add_argument(
        "--persist-dir",
        default=settings.persist_dir,
        help=f"ChromaDB directory (default: {settings.persist_dir})"
    )

    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"Collection name (default: {settings.collection_name})"
    )

    parser.add_argument(
        "--embed-model",
        default=settings.embed_model,
        help=f"Embedding model (default: {settings.embed_model})"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing database and create fresh"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for ingestion (default: 2)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Academic RAG - Data Ingestion Script")
    print("=" * 60)

    # Validate input file
    if not os.path.exists(args.json_file):
        logger.error(f"❌ Data file not found: {args.json_file}")
        return 1

    # Handle --clean flag
    if args.clean and os.path.exists(args.persist_dir):
        print(f"\n⚠️  Cleaning existing database at: {args.persist_dir}")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(args.persist_dir)
            logger.info(f"✅ Deleted existing database")
        else:
            logger.info("Aborted.")
            return 0

    # Create persist directory if needed
    os.makedirs(args.persist_dir, exist_ok=True)

    # Step 1: Load JSON data
    print(f"\n📂 Loading data from: {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    print(f"   Found {len(courses)} courses")

    # Step 2: Chunk documents
    print(f"\n📄 Chunking documents...")
    chunker = AdvancedAcademicChunker()
    all_documents = []

    for course in courses:
        docs = chunker.chunk_course_from_json(course)
        all_documents.extend(docs)

    print(f"   Generated {len(all_documents)} document chunks")

    # Step 3: Initialize vector store
    print(f"\n🗄️  Initializing ChromaDB...")
    print(f"   Directory: {args.persist_dir}")
    print(f"   Collection: {args.collection}")
    print(f"   Model: {args.embed_model}")

    manager = LangChainVectorStoreManager(
        persist_directory=args.persist_dir,
        model_name=args.embed_model,
        collection_name=args.collection
    )

    # Step 4: Ingest documents
    print(f"\n📥 Ingesting {len(all_documents)} documents...")
    print(f"   Batch size: {args.batch_size}")

    manager.ingest_documents(all_documents, batch_size=args.batch_size)

    # Done
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   - Courses processed: {len(courses)}")
    print(f"   - Documents created: {len(all_documents)}")
    print(f"   - Database location: {args.persist_dir}")
    print(f"   - Collection: {args.collection}")
    print(f"\n🚀 You can now run the main application:")
    print(f"   python main.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
