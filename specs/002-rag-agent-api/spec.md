# Feature Specification: RAG Backend & Agent

**Feature Branch**: `002-rag-agent-api`
**Created**: 2026-04-16
**Status**: Draft
**Input**: User description: "Build a FastAPI backend with an intelligent RAG Agent using OpenAI Agents SDK that can retrieve relevant information from Qdrant and generate accurate answers."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Normal Chat Query (Priority: P1)

A user sends a question to the chatbot and receives an educational answer with citations from the textbook.

**Why this priority**: This is the primary use case - users should be able to ask questions about the textbook content and get accurate, cited answers.

**Independent Test**: Can be tested by sending a question to `/chat` endpoint and verifying a helpful response with source citations is returned.

**Acceptance Scenarios**:

1. **Given** the backend is running and Qdrant contains textbook content, **When** a user sends a question to `/chat`, **Then** the system returns an answer based on relevant chunks from the vector store
2. **Given** relevant chunks are found, **When** response is generated, **Then** it should include citations referencing the source (module, chapter, section)
3. **Given** no relevant content exists in the vector store, **When** a question is asked, **Then** the system politely indicates the information was not found in the textbook

---

### User Story 2 - Contextual Chat with Selected Text (Priority: P1)

A user selects text in the frontend and asks a question about that specific context, receiving an answer that combines the selected text with related content from the textbook.

**Why this priority**: This enables deeper exploration of specific sections, allowing users to get answers that build upon their current reading.

**Independent Test**: Can be tested by sending a request to `/chat-with-context` with selected text and verifying the response incorporates both the selected context and retrieved related content.

**Acceptance Scenarios**:

1. **Given** a user has selected text in the UI, **When** they send it to `/chat-with-context` along with a question, **Then** the system uses the selected text as additional context for retrieval
2. **Given** the selected text is provided, **When** retrieval runs, **Then** the system finds chunks semantically related to both the question and the selected text
3. **Given** related chunks are found, **When** answer is generated, **Then** it should reference the selected text context and cite sources

---

### User Story 3 - Backend Reliability & Error Handling (Priority: P2)

The backend handles errors gracefully and provides meaningful feedback to users when issues occur.

**Why this priority**: Users should not see cryptic errors; they should receive helpful messages even when something goes wrong.

**Independent Test**: Can be tested by triggering various error conditions (invalid requests, service unavailability) and verifying appropriate error responses.

**Acceptance Scenarios**:

1. **Given** the backend receives an invalid request, **When** processing fails, **Then** it returns a clear error message with appropriate HTTP status code
2. **Given** Qdrant is unavailable, **When** a user sends a chat request, **Then** the system returns a friendly error indicating the service is temporarily unavailable
3. **Given** the OpenAI API is unavailable, **When** a user sends a chat request, **Then** the system handles the error gracefully and informs the user

---

### User Story 4 - Frontend Integration Support (Priority: P2)

The backend is configured to work seamlessly with the Docusaurus frontend, including CORS and proper response formatting.

**Why this priority**: The ultimate goal is to integrate with the Docusaurus frontend for the chatbot feature to work in the browser.

**Independent Test**: Can be tested by making cross-origin requests from a browser and verifying the response includes appropriate CORS headers.

**Acceptance Scenarios**:

1. **Given** a browser makes a request to the backend, **When** CORS is configured, **Then** the response includes appropriate Access-Control headers
2. **Given** the frontend sends a JSON request, **When** the backend processes it, **Then** the response is properly formatted JSON that the frontend can consume

---

### Edge Cases

- What happens when the user sends an empty question?
- How does the system handle very long questions that exceed token limits?
- What happens when Qdrant returns no results for a query?
- How does the system handle rate limiting from external APIs?
- What happens when embedding generation fails for a query?
- How does the system handle special characters or malformed input?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a FastAPI backend running on localhost
- **FR-002**: System MUST expose a POST endpoint `/chat` that accepts a JSON body with a "question" field
- **FR-003**: System MUST expose a POST endpoint `/chat-with-context` that accepts a JSON body with "question" and "selectedText" fields
- **FR-004**: System MUST use the OpenAI Agents SDK to create an intelligent RAG agent
- **FR-005**: System MUST integrate Qdrant as a retrieval tool within the agent workflow
- **FR-006**: System MUST generate embeddings for user queries using Cohere at runtime
- **FR-007**: System MUST retrieve top relevant chunks from Qdrant based on query embedding
- **FR-008**: System MUST generate well-structured, educational answers using the retrieved context
- **FR-009**: System MUST include citations in responses referencing module, chapter, and section
- **FR-010**: System MUST read all API keys from environment variables (OPENAI_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY)
- **FR-011**: System MUST enable CORS to allow requests from the Docusaurus frontend
- **FR-012**: System MUST implement proper error handling with meaningful error messages
- **FR-013**: System MUST reduce hallucination by relying primarily on retrieved context for answers
- **FR-014**: System MUST politely indicate when answer is not found in the textbook content
- **FR-015**: System MUST include logging for debugging and monitoring
- **FR-016**: System MUST support async operations where applicable for better performance

### Key Entities

- **ChatRequest**: Represents a user question sent to the chat endpoint
- **ContextualChatRequest**: Represents a question with selected text context
- **ChatResponse**: The system's answer including text and citations
- **RetrievedChunk**: A piece of content retrieved from Qdrant with metadata
- **Citation**: Source reference including module, chapter, and section information

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Backend runs successfully on localhost without errors
- **SC-002**: `/chat` endpoint returns accurate answers based on textbook content
- **SC-003**: `/chat-with-context` endpoint correctly incorporates selected text as additional context
- **SC-004**: Responses include source citations (module, chapter, section) for verification
- **SC-005**: System indicates when information is not found in the textbook rather than hallucinating
- **SC-006**: CORS is properly configured to allow frontend integration
- **SC-007**: Error responses are user-friendly and provide actionable information

---

**Next Phase**: Once this spec is complete, we will move to frontend integration with the Docusaurus chatbot component.
