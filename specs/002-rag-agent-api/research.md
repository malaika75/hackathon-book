# Research: RAG Backend & Agent

**Date**: 2026-04-16
**Feature**: 002-rag-agent-api

## Technical Decisions

### Decision 1: OpenAI Agents SDK

**Chosen**: OpenAI Agents SDK for Python (aka "openai-agents" or "Agents SDK")

**Rationale**: The specification explicitly requires OpenAI Agents SDK. This SDK provides built-in support for:
- Agent orchestration with tools
- Function calling capabilities
- Streaming responses
- Structured output handling

**Alternatives considered**:
- LangChain Agents: More mature but adds abstraction layer
- Direct OpenAI Chat Completions API: Would require manual tool handling

### Decision 2: FastAPI for REST API

**Chosen**: FastAPI

**Rationale**:
- Existing project structure uses Python
- Built-in async support
- Automatic OpenAPI schema generation
- Easy CORS configuration for frontend integration

### Decision 3: Qdrant Integration

**Chosen**: Qdrant Python client

**Rationale**:
- Already configured from Phase 1 (ingestion pipeline)
- Native async support available
- Easy filtering by payload metadata
- Existing collection "physical-ai-textbook"

### Decision 4: Cohere for Query Embeddings

**Chosen**: Cohere SDK (embed-english-v3.0)

**Rationale**:
- Consistent with Phase 1 (used same embeddings for ingestion)
- Fast and cost-effective
- High quality embeddings
- Already have API key configured

## Key Integration Patterns

### Agent Tool Pattern

The RAG Agent uses tools pattern:
1. **Embed query**: Convert user question to vector using Cohere
2. **Search Qdrant**: Retrieve top-k relevant chunks
3. **Generate response**: Use LLM with context to generate answer

### Citation Format

Include in response:
- Module name (e.g., "Module 1: Introduction to Robotics")
- Chapter title (e.g., "Chapter 2: CLI Packages")
- Section heading (e.g., "2.3 Installing ROS2")

## References

- FastAPI: https://fastapi.tiangolo.com/
- Qdrant: https://qdrant.tech/documentation/
- Cohere: https://docs.cohere.com/
- OpenAI Python SDK: https://github.com/openai/openai-python