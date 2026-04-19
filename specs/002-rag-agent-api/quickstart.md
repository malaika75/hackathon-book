# Quickstart: RAG Backend & Agent

**Feature**: 002-rag-agent-api

## Prerequisites

- Python 3.11+
- Qdrant running with "physical-ai-textbook" collection
- API keys in `.env` file

## Setup

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn openai qdrant-client cohere
   ```

2. **Configure environment**:
   Create `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   COHERE_API_KEY=your_cohere_key
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_key
   ```

3. **Verify Qdrant collection**:
   Ensure Phase 1 ingestion was completed successfully.

## Running the Server

```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

## API Endpoints

### POST /chat

Normal chat query:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS2?"}'
```

### POST /chat-with-context

Contextual chat with selected text:
```bash
curl -X POST http://localhost:8000/chat-with-context \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain this", "selectedText": "ROS2 is the next generation of ROS"}'
```

### GET /health

Health check:
```bash
curl http://localhost:8000/health
```

## Testing

```bash
pytest tests/
```

## Common Issues

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Check API keys in `.env` |
| Qdrant connection error | Ensure Qdrant is running |
| Empty response | Check textbook content was ingested |