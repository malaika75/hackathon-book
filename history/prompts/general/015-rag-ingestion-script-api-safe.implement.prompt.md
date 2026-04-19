---
id: "015"
title: "Create API-limit-safe Qdrant ingestion script"
stage: implement
date: 2026-04-18
surface: agent
model: oc/minimax-m2.5-free
feature: none
branch: 003-rag-chatbot-frontend
user: claude
command: python scripts/ingest_qdrant.py
labels: [rag, ingestion, qdrant, cohere, api-limit]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - scripts/ingest_qdrant.py
tests:
  - Syntax check passed
  - Import check passed
  - Env variables load verified
---

## Prompt

please work on it smartly Create a robust, API-limit-safe ingestion script `scripts/ingest_qdrant.py` for my RAG backend.

### Requirements:
1. **Qdrant Setup**: Connect via `.env`. Auto-create collection `physical-ai-textbook` if missing (`dim=1024`, `distance=COSINE` for Cohere v3).
2. **Safe Cohere Embedding**:
   - Read PDFs/TXTs from `./data/docs/`, chunk using LangChain (500 tokens, 50 overlap)
   - Embed in batches of 15: `co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")`
   - Add `time.sleep(1.5)` between batches + custom retry for `429` errors (3 attempts, exponential backoff)
   - Track processed chunk IDs in `./data/ingestion_progress.json` to skip already-embedded chunks on re-run
3. **Qdrant Upload**: Use `client.upload_points()` with `id=str(uuid5(NAMESPACE_DNS, chunk_text))`, `vector=embedding`, `payload={"text": chunk, "source": filename}`
4. **Logging**:
   - Start: "🚀 Starting ingestion..."
   - Progress: "📦 Skipped X chunks | ✅ Uploaded Y/Z to Qdrant"
   - End: "🎉 Ingestion

## Response snapshot

Created `backend/scripts/ingest_qdrant.py` with all requirements:
- Qdrant collection auto-creation with dim=1024, COSINE distance
- LangChain text chunking (500 tokens, 50 overlap)
- Batch embedding (15 at a time) with 1.5s rate limiting
- 429 retry logic (3 attempts, exponential backoff)
- Progress tracking in `./data/ingestion_progress.json`
- Deterministic UUIDs using uuid5(NAMESPACE_DNS, chunk_text)
- Proper logging with emojis as specified

## Outcome

- ✅ Impact: New ingestion script handles API rate limits safely with retry logic
- 🧪 Tests: Syntax, imports, and env loading verified
- 📁 Files: `backend/scripts/ingest_qdrant.py` created (~300 lines)
- 🔁 Next prompts: None
- 🧠 Reflection: Leveraged existing RAG pipeline structure but added robustness with rate limiting and progress tracking

## Evaluation notes (flywheel)

- Failure modes observed: None (syntax verified)
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Test with actual document files