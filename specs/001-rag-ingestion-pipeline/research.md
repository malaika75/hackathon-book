# Research: RAG Ingestion Pipeline

**Feature**: RAG Ingestion Pipeline - Phase 1
**Date**: 2026-04-15

## Research Questions

### 1. Cohere API Embedding Generation

**Question**: Which Cohere embedding model to use and how to handle batching/rate limits?

**Decision**: Use `embed-english-v3.0` with batch processing (max 100 chunks per batch)
**Rationale**: Best quality for semantic search, efficient batching reduces API calls
**Alternatives considered**:
- embed-english-v2.0: Older, slightly lower quality
- embed-multilingual-v3.0: Not needed for English-only content
- OpenAI text-embedding-3-small: Good alternative, but Cohere specified in requirements

### 2. Qdrant Vector Database Configuration

**Question**: How to set up Qdrant for filtering by module/chapter?

**Decision**: Use Qdrant cloud with payload indexing on module and chapter fields
**Rationale**: Built-in filtering support, proper indexing for fast retrieval
**Alternatives considered**:
- Local Qdrant (Docker): More complex setup
- Weaviate: Good but Qdrant specified in requirements

### 3. Text Chunking Strategy

**Question**: How to chunk markdown content while preserving semantic boundaries?

**Decision**: Use LangChain's RecursiveCharacterTextSplitter with custom markdown-aware separators
**Rationale**: Handles code blocks, lists, and paragraphs as atomic units
**Parameters**:
- chunk_size: 1000 tokens
- chunk_overlap: 200 tokens
- separators: ["\n\n", "\n", "```\n", " ", ""]

### 4. Idempotency Implementation

**Question**: How to ensure script can run multiple times without duplicating data?

**Decision**: Use document hash-based deduplication - check if document already exists before processing
**Rationale**: Simple and effective for content that doesn't change frequently
**Implementation**: Generate hash from file_path + last_modified_time, store in metadata

---

## Findings Summary

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Embedding Model | embed-english-v3.0 | Best quality for RAG |
| Batch Size | 100 per request | Cohere rate limit safe |
| Qdrant Setup | Cloud with payload index | Filtering support |
| Chunking | RecursiveCharacterTextSplitter | Semantic boundary preservation |
| Deduplication | Hash-based | Simple idempotent design |

---

## References

- Cohere API Documentation: https://docs.cohere.com/
- Qdrant Documentation: https://qdrant.tech/documentation/
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
