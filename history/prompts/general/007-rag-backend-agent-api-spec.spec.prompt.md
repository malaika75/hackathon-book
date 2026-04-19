---
id: "007"
title: "RAG Backend Agent API Spec"
stage: "spec"
date: "2026-04-16"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "002-rag-agent-api"
user: "Hassan"
command: "/sp.specify"
labels: ["rag", "backend", "agent", "api"]
links:
  spec: "specs/002-rag-agent-api/spec.md"
  ticket: "null"
  adr: "null"
  pr: "null"
files:
  - "specs/002-rag-agent-api/spec.md"
  - "specs/002-rag-agent-api/checklists/requirements.md"
tests:
  - "none"
---

## Prompt

# RAG Backend & Agent Specification

Spec Name: rag-backend-agent-spec
Version: 1.0
Phase: 2 of 2 (Remaining Work)
Status: To Be Implemented

## Objective
Build a FastAPI backend with an intelligent RAG Agent using OpenAI Agents SDK that can retrieve relevant information from Qdrant and generate accurate answers.

## Requirements

### 1. FastAPI Backend Setup
- Create FastAPI application
- Add necessary endpoints:
  - POST /chat → Normal chat query
  - POST /chat-with-context → For selected text feature
- Use environment variables for all keys
- Proper error handling and CORS support (for Docusaurus frontend)

### 2. RAG Agent using OpenAI Agents SDK
- Create an intelligent agent with OpenAI Agents SDK / ChatKit
- Integrate Qdrant as retrieval tool
- The agent should:
  - Take user question
  - Retrieve top relevant chunks from Qdrant (using Cohere embeddings)
  - Generate well-structured, educational answer
  - Include citations (chapter name + section)

### 3. Two Query Modes Support
- Mode 1 (Normal Chat): User ka normal sawal → full book se retrieve karke jawab do
- Mode 2 (Contextual Chat): User ne koi text select kiya ho → us selected text ko extra context ke saath bhejna aur related chunks bhi retrieve karna

### 4. Response Quality
- Answers should be clear, educational and easy to understand
- Always try to cite the source (Module → Chapter → Section)
- Reduce hallucination as much as possible
- If answer not found in book, politely say so

### 5. Technical Requirements
- Use openai-agents SDK
- Qdrant client for retrieval
- Cohere for embedding user queries (at runtime)
- Proper async support where needed
- Logging for debugging

## Success Criteria
- Backend runs successfully on localhost
- /chat endpoint se sahi jawab aata hai
- Selected text wala mode bhi kaam karta hai
- Responses mein sources dikhte hain

## Next Step
After this spec is complete, we will move to frontend integration.

## Response snapshot

Created feature spec at specs/002-rag-agent-api/spec.md with:
- 4 user stories (Normal Chat, Contextual Chat, Error Handling, Frontend Integration)
- 16 functional requirements covering all aspects
- 7 success criteria (measurable, technology-agnostic)
- Specification quality checklist with all items passing

## Outcome

- Impact: Defined RAG Backend & Agent specification with clear user scenarios and requirements
- Tests: None - spec phase only
- Files: specs/002-rag-agent-api/spec.md, specs/002-rag-agent-api/checklists/requirements.md
- Next prompts: /sp.plan (architecture planning), /sp.implement (build)
- Reflection: Spec is complete and ready for planning. No [NEEDS CLARIFICATION] markers were required.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment: N/A