# Quickstart: RAG Ingestion Pipeline

## Prerequisites

- Python 3.10+
- Cohere API account
- Qdrant cloud account (or local Docker)

## Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install cohere qdrant-client langchain langchain-community python-dotenv tiktoken
   ```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys:
   ```bash
   # Required
   COHERE_API_KEY="your-cohere-api-key"

   # Qdrant Cloud (recommended)
   QDRANT_URL="https://your-cluster.qdrant.cloud"
   QDRANT_API_KEY="your-qdrant-api-key"

   # Or Qdrant Local (alternative)
   # QDRANT_URL="http://localhost:6333"
   # QDRANT_API_KEY=""
   ```

## Usage

### Basic Usage

```bash
cd scripts
python ingest_book.py
```

### With Custom Options

```bash
cd scripts
python ingest_book.py --docs-path ../docusaurus/docs --collection physical-ai-textbook
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--docs-path` | `../docusaurus/docs` | Path to Docusaurus docs folder |
| `--collection` | `physical-ai-textbook` | Qdrant collection name |
| `--chunk-size` | `1000` | Tokens per chunk |
| `--chunk-overlap` | `200` | Token overlap between chunks |
| `--batch-size` | `100` | Cohere API batch size |
| `-v, --verbose` | False | Show detailed progress |

## Expected Output

```
Scanning ../docusaurus/docs/ directory...
Found 15 markdown files
Processing module1/chapter1-intro-architecture.md (1/15)
Processing module1/chapter2-cli-packages.md (2/15)
...
✓ Extracted 15 documents
✓ Created 127 chunks
✓ Generated 127 embeddings
✓ Stored 127 vectors in Qdrant

=== Summary ===
Documents processed: 15
Chunks created: 127
Vectors stored: 127
Collection: physical-ai-textbook

=== Verification Query ===
Query: "What is ROS 2?"
Results: 3 relevant chunks found
✓ Ingestion complete!
```

## Verification

After successful ingestion, verify by checking the Qdrant dashboard or running:

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='your-qdrant-url', api_key='your-qdrant-key')
info = client.get_collection('physical-ai-textbook')
print(f'Vectors in collection: {info.vectors_count}')
"
```

## Troubleshooting

### API Key Issues

- Ensure `COHERE_API_KEY` is set correctly in `.env`
- Check Cohere dashboard for key status

### Qdrant Connection

- Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- Check network connectivity

### Empty Results

- Verify collection was created in Qdrant dashboard
- Check that documents were actually processed
- Run with `-v` flag for detailed debug output