# Data Model: RAG Chatbot Frontend

**Feature**: 003-rag-chatbot-frontend

## Entities

### ChatMessage

Represents a single message in the chat conversation.

| Field | Type | Description |
|-------|------|-------------|
| id | string (UUID) | Unique identifier for the message |
| role | "user" \| "assistant" | Whether the message is from user or AI |
| content | string | The text content of the message |
| timestamp | ISO 8601 | When the message was sent |
| sources | SourceCitation[] | Optional sources for AI responses |

**Validation**:
- content: required, non-empty, max 4000 characters
- role: required, must be "user" or "assistant"

### SourceCitation

Represents a reference to textbook content used in the answer.

| Field | Type | Description |
|-------|------|-------------|
| documentId | string | Identifier for the source document |
| title | string | Display title of the source |
| section | string | Section/chapter reference |
| relevanceScore | number (0-1) | How relevant this source is to the answer |
| url | string (optional) | Link to the source section |

**Validation**:
- title: required, non-empty
- relevanceScore: 0-1 range

### ChatSession

Represents the current chat session state.

| Field | Type | Description |
|-------|------|-------------|
| messages | ChatMessage[] | All messages in the conversation |
| isLoading | boolean | Whether a request is in progress |
| error | string \| null | Current error message if any |
| isOpen | boolean | Whether chat window is visible |

**State Transitions**:
- `isOpen`: false → true (user clicks button)
- `isOpen`: true → false (user closes window)
- `isLoading`: false → true (sending message) → false (response received)

### UserPreferences

User settings for the chatbot (stored in sessionStorage).

| Field | Type | Description |
|-------|------|-------------|
| lastBackendUrl | string | Last used backend URL |
| theme | "light" \| "dark" \| "system" | Preferred theme override |

## API Contract (Frontend ↔ Backend)

### POST /chat

**Request**:
```json
{
  "question": "string"
}
```

**Response**:
```json
{
  "answer": "string",
  "sources": [
    {
      "documentId": "string",
      "title": "string",
      "section": "string",
      "relevanceScore": 0.95
    }
  ]
}
```

### POST /chat-with-context

**Request**:
```json
{
  "question": "string",
  "selectedText": "string"
}
```

**Response**: Same as `/chat`

## Component State Machine

```
┌─────────────┐
│  CLOSED     │──click──> OPEN
└─────────────┘               │
     ▲                         │
     │──────close──────────────┘
           │
           ▼
    ┌──────────────┐
    │ OPEN         │
    │              │
    │ ┌──────────┐ │
    │ │ LOADING  │ │──response──> IDLE
    │ └──────────┘ │
    └──────────────┘
```

- **CLOSED**: Chat window hidden, floating button visible
- **OPEN**: Chat window visible
- **LOADING**: Waiting for backend response
- **IDLE**: Ready for user input