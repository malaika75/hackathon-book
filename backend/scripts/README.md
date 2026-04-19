# RAG Ingestion Pipeline - Backend Scripts

This directory contains the Python scripts for ingesting the Physical AI & Humanoid Robotics Textbook into a Qdrant vector database.

## Structure

```
scripts/
├── __init__.py
├── config.py                 # Configuration and environment variable loading
├── cohere_client.py         # Cohere API client for embeddings
├── qdrant_client.py        # Qdrant vector database client
├── document_processor.py   # File scanning and reading
├── ingest_book.py          # Main ingestion script
├── parsers/
│   └── markdown_parser.py  # Markdown parsing
├── extractors/
│   └── metadata_extractor.py  # Metadata extraction
├── chunkers/
│   └── text_chunker.py     # Text chunking
├── loaders/
│   └── document_loader.py # Document loading coordination
├── generators/
│   └── embedding_generator.py  # Embedding generation
├── uploaders/
│   └── qdrant_uploader.py # Vector upload to Qdrant
├── loggers/
│   └── progress_logger.py # Progress logging
├── reporters/
│   └── statistics_reporter.py # Statistics reporting
└── testers/
    └── test_query_runner.py # Test queries
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Basic Usage

```bash
cd backend/scripts
python ingest_book.py
```

### With Custom Options

```bash
python ingest_book.py --docs-path ../docusaurus/docs --collection my-textbook
```

### With Verbose Output

```bash
python ingest_book.py -v
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|--------------|
| `COHERE_API_KEY` | Yes | Your Cohere API key |
| `QDRANT_URL` | Yes | Qdrant server URL |
| `QDRANT_API_KEY` | No | Qdrant API key (if using cloud) |
| `QDRANT_COLLECTION` | No | Collection name (default: physical-ai-textbook) |
| `DOCS_PATH` | No | Path to docs folder (default: ./docs) |

## Quick Start

1. Install dependencies
2. Configure environment variables
3. Run: `python ingest_book.py`

## Dependencies

- cohere
- qdrant-client
- langchain
- langchain-community
- python-dotenv
- tiktoken
