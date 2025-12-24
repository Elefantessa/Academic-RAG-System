#!/usr/bin/env python3
"""
Academic RAG System - Main Entry Point

Refactored modular version of the Academic RAG system.
"""

import argparse
import json
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from utils.logging_config import setup_logging, get_logger
from utils.port_utils import find_available_port


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Academic RAG System (Refactored)")

    # Data arguments
    parser.add_argument("--json-file", default=settings.json_file,
                       help="Path to course data JSON")
    parser.add_argument("--persist-dir", default=settings.persist_dir,
                       help="ChromaDB persist directory")

    # Model arguments
    parser.add_argument("--ollama-model", default=settings.ollama_model,
                       help="Ollama model name")
    parser.add_argument("--ollama-url", default=settings.ollama_base_url,
                       help="Ollama base URL")
    parser.add_argument("--device", default=settings.device,
                       help="Device: auto | cpu | cuda:N")

    # Server arguments
    parser.add_argument("--host", default=settings.host, help="Server host")
    parser.add_argument("--port", type=int, default=settings.port, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Flags
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke tests")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    print("🚀 Starting Academic RAG System (Refactored)")
    print("=" * 60)

    # Validate data file
    if not os.path.exists(args.json_file):
        logger.error(f"Data file not found: {args.json_file}")
        return 1

    # Load data
    logger.info(f"Loading data from {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)

    # Import here to avoid circular imports
    from data.ingestion.chunker import AdvancedAcademicChunker
    from data.ingestion.vector_store import LangChainVectorStoreManager
    from services.agent import ContextAwareRetrievalAgent
    from api.app import create_app

    # Process documents
    logger.info("Processing documents...")
    chunker = AdvancedAcademicChunker()
    all_documents = []
    for course in courses:
        all_documents.extend(chunker.chunk_course_from_json(course))
    logger.info(f"Processed {len(all_documents)} document chunks")

    # Setup device
    if args.device.startswith('cuda:'):
        device_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        logger.info(f"Set CUDA_VISIBLE_DEVICES={device_id}")

    # Initialize vector store
    logger.info(f"Initializing ChromaDB at {args.persist_dir}")
    db_manager = LangChainVectorStoreManager(
        model_name=settings.embed_model,
        collection_name=settings.collection_name,
        persist_directory=args.persist_dir
    )

    # Initialize agent
    logger.info("Initializing RAG Agent...")

    # Get retriever and extract vectorstore
    retriever = db_manager.get_retriever(search_type="mmr", search_kwargs={'k': 10})
    vectorstore = retriever.vectorstore

    agent = ContextAwareRetrievalAgent(
        vectorstore=vectorstore,
        all_documents=all_documents,
        ollama_base_url=args.ollama_url,
        model_name=args.ollama_model
    )
    logger.info("✅ Agent initialized successfully")

    # Smoke tests
    if args.smoke_test:
        print("\n🧪 Running smoke tests...")
        test_queries = [
            "What are the prerequisites for IoT?",
            "Who teaches Data Mining?",
            "Compare IoT and Machine Learning"
        ]
        for i, q in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {q}")
            try:
                resp = agent.process_query(q)
                print(f"✅ Mode: {resp.generation_mode} | Time: {resp.processing_time:.2f}s")
                print(f"   Answer: {resp.answer[:150]}...")
            except Exception as e:
                print(f"❌ Failed: {e}")

    # Find available port
    try:
        port = find_available_port(args.port)
        if port != args.port:
            logger.info(f"Port {args.port} busy, using {port}")
    except RuntimeError:
        port = args.port

    # Create and run app
    app = create_app(agent)

    print(f"\n{'=' * 60}")
    print("🎉 Academic RAG System Ready!")
    print(f"📱 Web Interface: http://{args.host}:{port}")
    print(f"🔗 API Endpoint: http://{args.host}:{port}/api/query")
    print(f"❤️  Health Check: http://{args.host}:{port}/api/health")
    print(f"📊 Statistics: http://{args.host}:{port}/api/stats")
    print(f"📚 Catalog: http://{args.host}:{port}/api/catalog")
    print("=" * 60)

    try:
        app.run(host=args.host, port=port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
