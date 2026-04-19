# Data Model: RAG Backend & Agent

**Feature**: 002-rag-agent-api

## Request Models

### ChatRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | User's question about the textbook |

**Validation**:
- Must be 1-1000 characters
- Must not be empty/whitespace only

### ContextualChatRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | User's question |
| selectedText | string | Yes | Text selected by user in the UI |

**Validation**:
- question: 1-1000 characters
- selectedText: 1-5000 characters

## Response Models

### ChatResponse

| Field | Type | Description |
|-------|------|-------------|
| answer | string | Generated response to user's question |
| citations | Citation[] | Source references |

### Citation

| Field | Type | Description |
|-------|------|-------------|
| module | string | Module name (e.g., "Module 1") |
| chapter | string | Chapter title |
| section | string | Section heading |
| url | string | Link to source document |

## Entity Relationships

```
ChatRequest/ContextualChatRequest
    │
    ├── question (string)
    └── selectedText (optional)
              │
              ▼
         RAG Agent
              │
              ├── Embed Query (Cohere)
              │         │
              │         ▼
              │    Query Vector
              │         │
              ▼         ▼
         Qdrant Retrieval
              │
              ▼
         RetrievedChunk[]
              │
              ▼
         LLM Generation
              │
              ▼
         ChatResponse
              │
              ├── answer (string)
              └── citations (Citation[])
```

## Error Models

| Field | Type | Description |
|-------|------|-------------|
| detail | string | Error message |

## State Transitions

1. **Idle** → **Processing**: User sends request
2. **Processing** → **Retrieving**: Agent embeds query
3. **Retrieving** → **Generating**: Chunks retrieved
4. **Generating** → **Complete**: Response generated
5. **Any Error State** → **Error**: Exception occurred