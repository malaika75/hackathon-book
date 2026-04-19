# Tasks: RAG Ingestion Pipeline - Phase 1

**Input**: Design documents from `/specs/001-rag-ingestion-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md

**Tests**: Not explicitly requested in spec - tests will be validation through quickstart.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create scripts directory at docusaurus/scripts/
- [X] T002 Create requirements.txt with dependencies (cohere, qdrant-client, langchain, langchain-community, python-dotenv)
- [X] T003 [P] Create .env.example file with required environment variables
- [X] T004 Create main __init__.py in scripts/ directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create config.py in scripts/ with environment variable loading
- [X] T006 [P] Create document_processor.py in scripts/ for file scanning and reading
- [X] T007 [P] Create qdrant_client.py in scripts/ for Qdrant connection setup
- [X] T008 Create cohere_client.py in scripts/ for Cohere embedding generation

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Document Extraction & Ingestion (Priority: P1) 🎯 MVP

**Goal**: Scan all textbook markdown files, extract content with metadata, and create chunks

**Independent Test**: Run the ingestion script and verify all docs/ folder files are processed with proper metadata

### Implementation for User Story 1

- [X] T009 [P] [US1] Implement markdown_parser.py in scripts/parsers/ to extract headings and content
- [X] T010 [P] [US1] Implement metadata_extractor.py in scripts/extractors/ to capture module, chapter, URL
- [X] T011 [US1] Implement text_chunker.py in scripts/chunkers/ using RecursiveCharacterTextSplitter
- [X] T012 [US1] Implement document_loader.py in scripts/loaders/ to coordinate document processing
- [ ] T013 [US1] Create test_document_extraction.py to verify document processing (optional validation)

**Checkpoint**: User Story 1 complete - all documents should be extracted and chunked

---

## Phase 4: User Story 2 - Embedding Generation & Storage (Priority: P1)

**Goal**: Generate embeddings for all chunks and store in Qdrant vector database

**Independent Test**: Query Qdrant for sample vectors and verify they return relevant content

### Implementation for User Story 2

- [X] T014 [P] [US2] Implement embedding_generator.py in scripts/generators/ using Cohere client
- [X] T015 [US2] Implement qdrant_uploader.py in scripts/uploaders/ to store vectors with payload
- [X] T016 [US2] Implement idempotency_check.py in scripts/ to prevent duplicate ingestion
- [X] T017 [US2] Integrate chunk generation with embedding storage in main ingest_book.py

**Checkpoint**: User Story 2 complete - all embeddings should be stored in Qdrant

---

## Phase 5: User Story 3 - Validation & Verification (Priority: P2)

**Goal**: Provide clear progress output and final statistics, verify storage with test queries

**Independent Test**: Run script and verify console shows correct statistics and test queries work

### Implementation for User Story 3

- [X] T018 [US3] Implement progress_logger.py in scripts/loggers/ for progress output
- [X] T019 [US3] Implement statistics_reporter.py in scripts/reporters/ for summary output
- [X] T020 [US3] Implement test_query_runner.py in scripts/testers/ for verification queries
- [X] T021 [US3] Add verbose mode flag to show detailed progress (-v/--verbose)
- [X] T022 [US3] Update quickstart.md with test query script path

**Checkpoint**: All user stories complete - full pipeline functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T023 [P] Add error handling for empty/unreadable files in document_processor.py
- [X] T024 [P] Add logging configuration for production use
- [X] T025 Create README.md in scripts/ with usage documentation
- [ ] T026 Run quickstart.md validation steps to verify complete ingestion

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can proceed sequentially (P1 → P2)
- **Polish (Final Phase)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Depends on US1 completion (needs chunks to embed)
- **User Story 3 (P2)**: Depends on US2 completion (needs stored vectors to verify)

### Within Each User Story

- Models/splitters before chunk generation
- Chunk generation before embedding
- Embedding generation before storage
- Storage before validation

### Parallel Opportunities

- Phase 1 setup tasks T001-T004 can run in parallel
- Phase 2 foundational tasks T005-T008 can run in parallel
- US1 tasks T009-T011 can run in parallel (different files)

---

## Parallel Example: User Story 1

```bash
# Launch metadata and parsing tasks in parallel:
Task: "Implement markdown_parser.py in scripts/parsers/"
Task: "Implement metadata_extractor.py in scripts/extractors/"
```

---

## Implementation Strategy

### MVP First (User Story 1 + User Story 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. Complete Phase 4: User Story 2
5. **STOP and VALIDATE**: Run full ingestion and verify Qdrant has vectors
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Verify document extraction works
3. Add User Story 2 → Verify embeddings stored in Qdrant
4. Add User Story 3 → Verify summary output works
5. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
