# Implementation Plan: RAG Ingestion Pipeline - Phase 1

**Branch**: `001-rag-ingestion-pipeline` | **Date**: 2026-04-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-rag-ingestion-pipeline/spec.md`

## Summary

Create an ingestion pipeline that extracts content from the Docusaurus textbook, generates embeddings using Cohere, and stores vectors in Qdrant. This is Phase 1 of a 4-phase RAG chatbot development.

## Technical Context

| Item | Value |
|------|-------|
| **Language/Version** | Python 3.10+ |
| **Primary Dependencies** | langchain-core, qdrant-client, cohere |
| **Storage** | Qdrant (vector database) |
| **Testing** | pytest |
| **Target Platform** | Linux server / local machine |
| **Project Type** | CLI script (single project) in `backend/` |
| **Performance Goals** | Process ~15 textbook chapters, generate embeddings efficiently |
| **Constraints** | Handle Cohere API rate limits, support idempotent runs |
| **Scale/Scope** | ~15 chapters, each chunked into ~5-10 segments |

## Constitution Check

*Gate: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Code must be testable | PASS | Python with pytest |
| Reproducible setup | PASS | Env variables for API keys |
| Error handling defined | PASS | Edge cases in spec |

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-ingestion-pipeline/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (if needed)
└── tasks.md             # Phase 2 output (NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── scripts/
    ├── ingest_book.py  # Main ingestion script
    ├── config.py       # Configuration loader
    ├── cohere_client.py # Cohere API client
    ├── qdrant_client.py # Qdrant client
    ├── parsers/        # Markdown parsing
    ├── extractors/     # Metadata extraction
    ├── chunkers/       # Text chunking
    ├── loaders/        # Document loading
    ├── generators/     # Embedding generation
    ├── uploaders/      # Vector upload
    ├── loggers/        # Progress logging
    ├── reporters/      # Statistics reporting
    └── testers/        # Test queries
```

**Structure Decision**: Modular Python package in `backend/` - clean separation from frontend (docusaurus).

---

## Phase 0: Research

### Unknowns to Research

1. **Cohere API specifics**: Embedding model versioning, rate limits, batching requirements
2. **Qdrant setup**: Collection creation, indexing options, filtering capabilities
3. **Chunking strategy**: Token estimation for markdown content, semantic boundary handling

### Research Findings

#### Decision 1: Cohere Embedding Model

- **Selected**: `embed-english-v3.0` (or latest)
- **Rationale**: Best quality for semantic search, widely used in RAG applications
- **Alternatives**: OpenAI text-embedding-3-small, sentence-transformers

#### Decision 2: Qdrant Configuration

- **Selected**: Cloud-hosted Qdrant with API key
- **Rationale**: Easy setup, proper indexing for filtering by module/chapter
- **Alternatives**: Local Qdrant with Docker

#### Decision 3: Chunking Approach

- **Selected**: LangChain RecursiveCharacterTextSplitter with custom separators
- **Rationale**: Preserves semantic boundaries, handles markdown structure
- **Parameters**: chunk_size=1000, chunk_overlap=200

---

## Phase 1: Design & Contracts

### Data Model

#### Document Entity

| Field | Type | Description |
|-------|------|-------------|
| file_path | string | Path to markdown file |
| module | string | Module name (1-4) |
| chapter_title | string | Title of chapter |
| headings | list[string] | H1-H3 headings |
| url | string | Relative URL path |
| content | string | Full text content |

#### Chunk Entity

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | string | Unique identifier |
| document_id | string | Reference to source document |
| content | string | Text chunk content |
| token_count | int | Estimated tokens |
| vector | float[] | Cohere embedding |

#### Metadata Payload

| Field | Type | Description |
|-------|------|-------------|
| module | string | Module name |
| chapter | string | Chapter title |
| section | string | Current section heading |
| url | string | Page URL |
| hierarchy_level | int | Document depth |

### API Contracts

#### Ingestion Script Interface

```bash
python scripts/ingest_book.py \
  --docs-path ./docs \
  --cohere-key $COHERE_API_KEY \
  --qdrant-url $QDRANT_URL \
  --qdrant-key $QDRANT_API_KEY \
  --collection physical-ai-textbook
```

#### Output

- Progress: "Processing document X of Y..."
- Summary: "Ingested X documents, Y chunks, Z vectors"
- Test query: 2-3 sample retrievals

### Quickstart

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables
3. Run: `python scripts/ingest_book.py`
4. Verify: Check console output shows successful ingestion

---

## Complexity Tracking

No violations requiring justification.

---

## Next Steps

Proceed to `/sp.tasks` to generate task breakdown for implementation.
