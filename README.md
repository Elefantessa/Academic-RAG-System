# 🎓 Academic RAG System

> **Retrieval-Augmented Generation (RAG) system for Uantwerp academic course information**

An advanced, production-ready RAG system that combines vector similarity search with cross-encoder reranking to provide accurate, context-aware answers about academic courses. Built with a modular architecture for maintainability and extensibility.

---

## ✨ Features

- **Multi-Stage Retrieval Pipeline**: Vector search → Cross-encoder reranking → Context expansion
- **Intelligent Query Understanding**: Automatic detection of query types (comparison, lecturer, standard)
- **Advanced Entity Extraction**: Regex + Fuzzy matching + LLM fallback
- **Confidence Scoring**: Multi-dimensional confidence calculation with LLM evaluation
- **Modern Web Interface**: Interactive chat UI with real-time responses
- **RESTful API**: Complete API for integration with other systems
- **Modular Architecture**: Clean separation of concerns across 7 modules

---

## 🏗️ Architecture

```
pdf_pipline/
├── config/              # Configuration & constants
│   ├── settings.py      # Pydantic Settings with env var support
│   └── constants.py     # Regex patterns, keywords, weights
│
├── models/              # Data models
│   ├── state.py         # RetrievalState, RAGResponse
│   ├── confidence.py    # ConfidenceMetrics
│   └── catalog.py       # MetadataCatalog with fuzzy matching
│
├── utils/               # Utility functions
│   ├── query_analysis.py    # Query classification & entity extraction
│   ├── logging_config.py    # Centralized logging
│   └── port_utils.py        # Port management
│
├── core/                # Core retrieval logic
│   ├── extractors.py        # Multi-stage entity extraction
│   ├── retriever.py         # Vector search with MMR
│   ├── reranker.py          # Cross-encoder reranking
│   ├── context_expander.py  # Intelligent context expansion
│   └── generator.py         # LLM answer generation
│
├── services/            # Business logic
│   ├── confidence_calculator.py  # Advanced confidence scoring
│   └── agent.py                  # Main RAG orchestrator
│
├── api/                 # Web layer
│   ├── app.py           # Flask application factory
│   └── routes.py        # API endpoints + web UI
│
├── data/                # Data processing
│   ├── extractors/      # PDF extraction
│   │   └── pdf_extractor.py
│   └── ingestion/       # Document chunking & vector store
│       ├── chunker.py
│       └── vector_store.py
│
├── scripts/             # CLI scripts
│   ├── extract_pdfs.py  # PDF → JSON extraction
│   └── ingest_data.py   # JSON → ChromaDB ingestion
│
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
│
└── main.py              # Application entry point
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install pydantic-settings PyMuPDF pdfplumber langchain-chroma sentence-transformers
```

### Option 1: Run with Existing Database

```bash
python main.py
```

### Option 2: Build from Scratch

```bash
# Step 1: Extract data from PDFs
python scripts/extract_pdfs.py

# Step 2: Create ChromaDB database
python scripts/ingest_data.py

# Step 3: Start the application
python main.py
```

### Access the Application

- **Web Interface**: http://127.0.0.1:5003
- **API Endpoint**: http://127.0.0.1:5003/api/query
- **Health Check**: http://127.0.0.1:5003/api/health

---

## 📖 Usage

### Web Interface

Navigate to `http://127.0.0.1:5003` and start asking questions:

- *"What are the prerequisites for Internet of Things?"*
- *"Who teaches Data Mining?"*
- *"Compare Machine Learning and Deep Learning courses"*

### API Usage

```bash
# Query the system
curl -X POST http://127.0.0.1:5003/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is IoT about?"}'

# Response
{
  "answer": "...",
  "confidence": 0.85,
  "sources": ["2500WETINT:Course Contents"],
  "generation_mode": "standard",
  "processing_time": 2.5
}
```

### Command Line Options

```bash
python main.py --help

Options:
  --json-file PATH     Path to course data JSON
  --persist-dir PATH   ChromaDB directory
  --ollama-model NAME  LLM model name (default: llama3.1:latest)
  --host HOST          Server host (default: 127.0.0.1)
  --port PORT          Server port (default: 5003)
  --debug              Enable debug mode
  --smoke-test         Run smoke tests before starting
```

---

## ⚙️ Configuration

### Environment Variables

All settings can be overridden via environment variables with `RAG_` prefix:

```bash
export RAG_PORT=5010
export RAG_DEVICE=cuda:0
export RAG_OLLAMA_MODEL=llama3.1:latest
export RAG_DEBUG=true
```

### Configuration File (.env)

```env
RAG_PORT=5003
RAG_HOST=127.0.0.1
RAG_DEVICE=cuda:4
RAG_OLLAMA_MODEL=llama3.1:latest
RAG_EMBED_MODEL=Salesforce/SFR-Embedding-Mistral
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 📊 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/api/query` | Process a query |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/catalog` | Course catalog info |

### Query Request

```json
{
  "query": "What are the prerequisites for IoT?"
}
```

### Query Response

```json
{
  "answer": "The prerequisites for IoT include...",
  "confidence": 0.87,
  "sources": ["2500WETINT:Prerequisites"],
  "generation_mode": "standard",
  "processing_time": 2.34,
  "metadata": {
    "extracted": {"course_code": "2500WETINT"},
    "doc_count": 8,
    "mode": "standard"
  }
}
```

---

## 🔧 Data Pipeline

### 1. PDF Extraction

```bash
python scripts/extract_pdfs.py --input-dir /path/to/pdfs --output data.json
```

Extracts structured data from academic course PDFs:
- Course title and code
- Lecturer information
- Content sections (prerequisites, learning outcomes, etc.)

### 2. Document Ingestion

```bash
python scripts/ingest_data.py --json-file data.json --clean
```

Processes JSON into ChromaDB:
- Intelligent chunking with section awareness
- Metadata preservation
- Vector embedding generation

### Data Flow

```
📂 PDF Files
    ↓  extract_pdfs.py
📄 JSON Data
    ↓  ingest_data.py (chunking + embedding)
🗄️ ChromaDB
    ↓  main.py
🌐 Web Application
```

---

## 🧪 Testing

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 🛠️ Development

### Project Structure

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `config` | Configuration management | `AppSettings` |
| `models` | Data structures | `RAGResponse`, `MetadataCatalog` |
| `utils` | Helper functions | Query analysis, logging |
| `core` | Retrieval pipeline | `EntityExtractor`, `VectorRetriever`, `DocumentReranker` |
| `services` | Business logic | `ContextAwareRetrievalAgent` |
| `api` | Web layer | Flask routes |
| `data` | Data processing | `PDFDataExtractor`, `AdvancedAcademicChunker` |

### Adding New Features

1. **New Query Type**: Add detection in `utils/query_analysis.py`, handling in `core/generator.py`
2. **New Data Source**: Implement extractor in `data/extractors/`
3. **New API Endpoint**: Add route in `api/routes.py`

---

## 📋 Requirements

### Core Dependencies

- Python 3.10+
- LangChain ecosystem
- Sentence Transformers
- ChromaDB
- Flask
- Pydantic

### External Services

- **Ollama**: LLM service (default: llama3.1)
- **GPU** (optional): CUDA for faster embeddings

### Full Requirements

See `requirements_refactored.txt` for complete list.

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :5003

# Kill process
kill -9 <PID>

# Or use different port
python main.py --port 5010
```

### Database Issues

```bash
# Recreate database from scratch
python scripts/ingest_data.py --clean
```

### Missing Dependencies

```bash
pip install pydantic-settings PyMuPDF pdfplumber
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Average query time | ~2-5 seconds |
| Embedding model | Salesforce/SFR-Embedding-Mistral |
| Reranking model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | ChromaDB |

---

## 👥 Contributors

[Hala Alramli]

---

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Sentence Transformers for embedding models
- ChromaDB for vector storage
- Ollama for local LLM inference
