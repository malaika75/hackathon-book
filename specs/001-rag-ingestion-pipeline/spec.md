# Feature Specification: RAG Ingestion Pipeline - Phase 1

**Feature Branch**: `001-rag-ingestion-pipeline`
**Created**: 2026-04-15
**Status**: Draft
**Input**: User description: "RAG Ingestion Pipeline Specification (Phase 1) - Create a complete ingestion pipeline that extracts content from the entire Docusaurus textbook, generates high-quality embeddings using Cohere, and stores them in Qdrant vector database."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Document Extraction & Ingestion (Priority: P1)

A developer needs to automatically ingest all textbook content into a vector database so that users can query the knowledge base.

**Why this priority**: This is the foundational capability required before any RAG queries can work. Without successful ingestion, the entire chatbot system fails.

**Independent Test**: Can be tested by running the ingestion script and verifying that all documents from the docs/ folder are processed and stored in Qdrant with proper metadata.

**Acceptance Scenarios**:

1. **Given** the Docusaurus docs folder contains markdown files, **When** the ingestion script runs, **Then** all .md and .mdx files should be scanned and extracted
2. **Given** extracted documents, **When** processing completes, **Then** each document should have metadata including module name, chapter title, section headings, and URL
3. **Given** successful extraction, **When** chunks are created, **Then** each chunk should contain 900-1200 tokens with 200-token overlap

---

### User Story 2 - Embedding Generation & Storage (Priority: P1)

The system needs to generate embeddings for all content chunks and store them in Qdrant for semantic search.

**Why this priority**: Without embeddings, the system cannot perform semantic search. This is essential for the RAG chatbot to understand user queries.

**Independent Test**: Can be tested by querying Qdrant for sample vectors and verifying they return relevant content.

**Acceptance Scenarios**:

1. **Given** text chunks are created, **When** embedding generation runs, **Then** each chunk should have a Cohere embedding vector
2. **Given** embeddings are generated, **When** storage completes, **Then** Qdrant should contain the collection "physical-ai-textbook" with all vectors
3. **Given** vectors are stored, **When** a query is executed, **Then** relevant documents should be retrievable

---

### User Story 3 - Validation & Verification (Priority: P2)

The system needs to provide clear output showing ingestion progress and final statistics to confirm successful completion.

**Why this priority**: Users need to verify the ingestion completed successfully and understand how much content was processed.

**Independent Test**: Can be tested by running the script and verifying the console output shows correct statistics.

**Acceptance Scenarios**:

1. **Given** the script runs, **When** ingestion starts, **Then** progress should be displayed showing documents processed
2. **Given** ingestion completes, **When** summary is printed, **Then** it should show total documents, total chunks, and total vectors stored
3. **Given** storage is complete, **When** the verification query runs, **Then** it should return 2-3 sample vectors confirming storage

---

### Edge Cases

- What happens when an empty or unreadable markdown file is encountered?
- How does the system handle files with special characters in filenames?
- What happens if Qdrant is not available or connection fails?
- How does the system handle rate limiting from Cohere API?
- What happens when running the script twice (idempotency)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST scan and read all .md and .mdx files from the Docusaurus docs/ folder
- **FR-002**: System MUST extract clean text content while preserving structural elements
- **FR-003**: System MUST capture metadata including: module name, chapter title, section headings (H1-H3), full page URL, document hierarchy level
- **FR-004**: System MUST chunk content into 900-1200 token segments with 200-token overlap
- **FR-005**: System MUST maintain semantic boundaries and not cut paragraphs or code blocks mid-content
- **FR-006**: System MUST generate embeddings using Cohere (embed-english-v3.0 model)
- **FR-007**: System MUST store vectors in Qdrant collection named "physical-ai-textbook"
- **FR-008**: System MUST store each chunk with vector and payload (metadata + original text)
- **FR-009**: System MUST support filtering by module and chapter in queries
- **FR-010**: System MUST read configuration from environment variables (COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY)
- **FR-011**: System MUST be idempotent - can run multiple times without duplicating data
- **FR-012**: System MUST print summary statistics after ingestion (documents processed, chunks created, vectors stored)

### Key Entities

- **Document**: Represents a single markdown file from the textbook with its extracted content and metadata
- **Chunk**: A segment of document content suitable for embedding (900-1200 tokens)
- **Embedding**: Vector representation of a chunk generated by Cohere
- **Metadata**: Structured information about each document/chapter (module, title, headings, URL)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 4 modules of the textbook are successfully ingested (verified by document count)
- **SC-002**: Qdrant collection "physical-ai-textbook" is populated with vectors and metadata (verified by collection existence)
- **SC-003**: No major errors occur during ingestion (verified by error-free execution)
- **SC-004**: Clear console output shows progress and final statistics (verified by output readability)
- **SC-005**: Sample queries return relevant content from stored vectors (verified by test query)

---

**Next Phase**: Once this spec is complete, we will move to Spec 2: Retrieval pipeline and testing.
