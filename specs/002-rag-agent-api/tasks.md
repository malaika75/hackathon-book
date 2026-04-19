# Tasks: RAG Backend & Agent

**Feature**: 002-rag-agent-api
**Generated**: 2026-04-16

## Phase 1: Setup

**Goal**: Project initialization and dependencies

- [X] T001 Initialize project structure per implementation plan in backend/src/api, backend/src/agents, backend/src/services, backend/src/models
- [X] T002 Create requirements.txt with dependencies: fastapi, uvicorn, openai, qdrant-client, cohere, python-dotenv, pydantic
- [X] T003 Create .env.example file with OPENAI_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY placeholders

## Phase 2: Foundational

**Goal**: Core infrastructure - MUST complete before user stories

- [X] T004 Create Pydantic request/response schemas in backend/src/models/schemas.py (ChatRequest, ContextualChatRequest, ChatResponse, Citation, ErrorResponse)
- [X] T005 Create Cohere embedding client in backend/src/services/embedding.py (embed_query function)
- [X] T006 Create Qdrant retrieval service in backend/src/services/retrieval.py (retrieve_chunks function)
- [X] T007 Create config module to load environment variables in backend/src/config.py

## Phase 3: User Story 1 - Normal Chat Query

**Goal**: Primary chat endpoint (/chat) - Priority P1

**Independent Test**: Send question to /chat endpoint, verify answer with citations

- [X] T008 [P] [US1] Implement main FastAPI application in backend/src/main.py with app instance
- [X] T009 [P] [US1] Implement health check endpoint GET /health in backend/src/api/routes.py
- [X] T010 [US1] Implement RAG agent logic in backend/src/agents/rag_agent.py (process_normal_chat function)
- [X] T011 [US1] Implement POST /chat endpoint in backend/src/api/routes.py
- [X] T012 [US1] Test /chat endpoint integration with curl test

## Phase 4: User Story 2 - Contextual Chat with Selected Text

**Goal**: Selected text chat endpoint (/chat-with-context) - Priority P1

**Independent Test**: Send question + selectedText to /chat-with-context, verify combined context retrieval

- [X] T013 [US2] Extend RAG agent to handle selected text context in backend/src/agents/rag_agent.py
- [X] T014 [US2] Implement POST /chat-with-context endpoint in backend/src/api/routes.py
- [X] T015 [US2] Test /chat-with-context endpoint with curl test

## Phase 5: User Story 3 - Backend Reliability & Error Handling

**Goal**: Graceful error handling - Priority P2

**Independent Test**: Trigger error conditions, verify user-friendly messages

- [X] T016 Add input validation for empty/witespace questions in backend/src/models/schemas.py
- [X] T017 Add error handlers for Qdrant connection failures in backend/src/services/retrieval.py
- [X] T018 Add error handlers for OpenAI API failures in backend/src/agents/rag_agent.py
- [X] T019 Add error handlers for Cohere embedding failures in backend/src/services/embedding.py

## Phase 6: User Story 4 - Frontend Integration Support

**Goal**: CORS and frontend compatibility - Priority P2

**Independent Test**: Make cross-origin request, verify CORS headers present

- [X] T020 Configure CORS middleware in backend/src/main.py for Docusaurus frontend
- [X] T021 Test CORS headers with preflight request

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Final refinements

- [X] T022 Add logging for debugging in all services
- [X] T023 Verify "not found" response when no relevant chunks retrieved
- [X] T024 Verify all citations include module, chapter, section

---

## Implementation Strategy

### MVP Scope (User Story 1 only)

Focus on Phase 3 tasks first - the basic /chat endpoint:
- T004, T005, T006, T007 (foundational)
- T008, T009, T010, T011 (User Story 1)

### Incremental Delivery

1. **Increment 1** (Phase 1 + Phase 2): Project setup, schemas, services
2. **Increment 2** (Phase 3): Basic /chat endpoint
3. **Increment 3** (Phase 4): /chat-with-context endpoint
4. **Increment 4** (Phase 5): Error handling
5. **Increment 5** (Phase 6): CORS for frontend
6. **Increment 6** (Phase 7): Polish

## Dependency Graph

```
Phase 1 (Setup)
    └── Phase 2 (Foundational)
            ├── T004: schemas.py → Phase 3
            ├── T005: embedding.py → Phase 3, Phase 4
            ├── T006: retrieval.py → Phase 3, Phase 4
            └── T007: config.py → Phase 3, Phase 4
                    │
                    ▼
Phase 3 (US1 - Normal Chat) ← Phase 4 can start after T010
Phase 4 (US2 - Contextual Chat) ← depends on Phase 3 complete
Phase 5 (US3 - Error Handling) ← depends on Phase 2
Phase 6 (US4 - Frontend CORS) ← depends on Phase 3
Phase 7 (Polish) ← depends on all phases
```

## Parallel Opportunities

- T008, T009: Can run in parallel (main.py, routes.py health endpoint are independent)
- T011, T014: /chat and /chat-with-context are independent endpoints
- T016, T017, T018, T019: Error handling tasks are independent

## File Paths Summary

| Task ID | File Path |
|---------|-----------|
| T001 | backend/src/api, backend/src/agents, backend/src/services, backend/src/models |
| T002 | backend/requirements.txt |
| T003 | backend/.env.example |
| T004 | backend/src/models/schemas.py |
| T005 | backend/src/services/embedding.py |
| T006 | backend/src/services/retrieval.py |
| T007 | backend/src/config.py |
| T008 | backend/src/main.py |
| T009 | backend/src/api/routes.py |
| T010 | backend/src/agents/rag_agent.py |
| T011 | backend/src/api/routes.py |
| T012 | tests/integration/ |
| T013 | backend/src/agents/rag_agent.py |
| T014 | backend/src/api/routes.py |
| T015 | tests/integration/ |
| T016 | backend/src/models/schemas.py |
| T017 | backend/src/services/retrieval.py |
| T018 | backend/src/agents/rag_agent.py |
| T019 | backend/src/services/embedding.py |
| T020 | backend/src/main.py |
| T021 | tests/integration/ |
| T022 | All service files |
| T023 | backend/src/agents/rag_agent.py |
| T024 | backend/src/agents/rag_agent.py |