# Data Model: RAG Ingestion Pipeline

## Entities

### 1. Document

Represents a single markdown file from the textbook.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file_path | string | Yes | Absolute path to markdown file |
| module | string | Yes | Module name (e.g., "Module 1", "Module 2") |
| chapter_title | string | Yes | Title extracted from first H1 |
| headings | list[string] | Yes | All H1-H3 headings in document |
| url | string | Yes | Relative URL for the page |
| hierarchy_level | int | Yes | Document depth in hierarchy |
| last_modified | datetime | No | File last modified timestamp |
| content | string | Yes | Full extracted text content |

### 2. Chunk

A segment of document content suitable for embedding.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| chunk_id | string | Yes | Unique identifier (UUID) |
| document_id | string | Yes | Reference to source Document |
| content | string | Yes | Text chunk content |
| token_count | int | Yes | Estimated token count |
| chunk_index | int | Yes | Order within document |
| source_heading | string | No | Nearest heading above chunk |

### 3. Vector

Embedding vector stored in Qdrant.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| vector_id | string | Yes | Unique identifier |
| chunk_id | string | Yes | Reference to Chunk |
| embedding | float[] | Yes | Cohere embedding vector (1536 dims) |
| model | string | Yes | Embedding model used |

### 4. Metadata (Payload)

Structured information stored with each vector in Qdrant.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| module | string | Yes | Module name for filtering |
| chapter | string | Yes | Chapter title for filtering |
| section | string | No | Current section heading |
| url | string | Yes | Page URL |
| file_path | string | Yes | Source file path |
| chunk_index | int | Yes | Position in document |
| content | string | Yes | Original text chunk |

## Relationships

```
Document 1───* Chunk
Chunk 1───1 Vector
Vector *──1 Metadata (payload)
```

## Validation Rules

1. **Document**:
   - file_path must exist and be readable
   - module must be one of: "Module 1", "Module 2", "Module 3", "Module 4"
   - content must not be empty

2. **Chunk**:
   - token_count must be between 900-1200 (target)
   - content must not be empty
   - chunk_index must be >= 0

3. **Vector**:
   - embedding must have exactly 1536 dimensions (embed-english-v3.0)
   - model must be "embed-english-v3.0"

## State Transitions

```
UNPROCESSED → EXTRACTING → CHUNKING → EMBEDDING → STORING → COMPLETE
                                              ↓
                                           FAILED
```

## Indexing (Qdrant)

Create payload index on:
- `module` (keyword) - for filtering by module
- `chapter` (keyword) - for filtering by chapter
- `url` (keyword) - for exact URL lookups
