#!/usr/bin/env python3
"""
API-limit-safe ingestion script for Qdrant RAG backend.

Features:
- Qdrant collection auto-creation (dim=1024, COSINE for Cohere v3)
- Safe Cohere embedding with batch processing
- Rate limiting (batch of 15, 1.5s delay between batches)
- Retry logic for 429 errors (3 attempts, exponential backoff)
- Progress tracking to skip already-embedded chunks

Usage:
    python ingest_qdrant.py [--docs-path ./data/docs] [--collection physical-ai-textbook]
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Windows-friendly output - use ASCII-safe emojis
import codecs
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# Constants
COLLECTION_NAME = "physical-ai-textbook"
VECTOR_DIM = 1024  # Cohere embed-english-v3.0
BATCH_SIZE = 15
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
API_DELAY = 1.5  # seconds between batches
MAX_RETRIES = 3
NAMESPACE_DNS = uuid.NAMESPACE_DNS
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROGRESS_FILE = DATA_DIR / "ingestion_progress.json"


def get_default_docs_paths():
    """Get robust default docs paths with fallbacks."""
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    root_dir = backend_dir.parent
    return [
        backend_dir / "data" / "docs",
        backend_dir / "docs",
        root_dir / "docusaurus" / "docs",
    ]


def load_env() -> dict:
    """Load environment variables."""
    load_dotenv()
    return {
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection": os.getenv("QDRANT_COLLECTION", COLLECTION_NAME),
    }


def find_docs_dir(custom_path: str | None = None) -> Path | None:
    """Find docs directory with fallback search."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists() and any(p.iterdir()):
            print(f"  📁 Loading documents from: {p}")
            return p
        print(f"  ⚠️ Custom path not found or empty: {p}")

    # Try default paths
    for p in get_default_docs_paths():
        if p.exists() and any(p.iterdir()):
            print(f"  📁 Loading documents from: {p}")
            return p

    # Log attempted paths
    print("  🔍 Searched paths:")
    for p in get_default_docs_paths():
        print(f"     - {p}")
    print(f"     - (custom: {custom_path})")

    return None


def ensure_collection_exists(client: QdrantClient, collection_name: str) -> None:
    """Create collection if it doesn't exist."""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        print(f"  Creating collection '{collection_name}' with dim={VECTOR_DIM}, COSINE")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"  Collection '{collection_name}' created successfully")
    else:
        print(f"  Collection '{collection_name}' already exists")


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create LangChain text splitter for chunking."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "```\n", " ", ""],
        keep_separator=True,
    )


def load_documents(docs_path: Path) -> list[tuple[str, str]]:
    """
    Load PDF, TXT and MD files from docs directory.

    Returns:
        List of (filename, content) tuples
    """
    documents = []

    if not docs_path.exists():
        print(f"  ⚠️ Docs path does not exist: {docs_path}")
        return documents

    print(f"  📂 Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    for file_path in docs_path.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.is_file():
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                print(f"  ⏭️  Skipped (>{MAX_FILE_SIZE//1024//1024}MB): {file_path.name}")
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append((file_path.name, content))
                print(f"  ✅ Loaded: {file_path.name} ({len(content)} chars)")
            except Exception as e:
                print(f"  ❌ Error reading {file_path.name}: {e}")

    return documents


def chunk_documents(
    documents: list[tuple[str, str]], splitter: RecursiveCharacterTextSplitter
) -> list[dict[str, Any]]:
    """
    Chunk documents into smaller pieces.

    Returns:
        List of dicts with: filename, chunk_index, chunk_text
    """
    chunks = []

    for filename, content in documents:
        texts = splitter.split_text(content)
        for i, text in enumerate(texts):
            chunks.append({
                "filename": filename,
                "chunk_index": i,
                "chunk_text": text.strip(),
            })

    return chunks


def generate_chunk_id(chunk_text: str) -> str:
    """Generate deterministic UUID for chunk text."""
    return str(uuid.uuid5(NAMESPACE_DNS, chunk_text))


def load_progress() -> dict:
    """Load ingestion progress from JSON file."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(progress: dict) -> None:
    """Save ingestion progress to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def embed_with_retry(
    cohere_client: cohere.Client, texts: list[str]
) -> tuple[Optional[list[list[float]]], bool]:
    """
    Embed texts with retry logic for 429 errors.

    Returns:
        Tuple of (embeddings or None if failed, success flag)
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document",
            )
            return response.embeddings, True
        except cohere.core.TooManyRequestsError:
            wait_time = (2 ** attempt) * API_DELAY
            print(f"    429rate limit, retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.1f}s")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    Error: {e}")
            if attempt == MAX_RETRIES - 1:
                return None, False
            time.sleep(API_DELAY * (2 ** attempt))

    return None, False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="API-limit-safe Qdrant ingestion script"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Path to documents folder (default: auto-detect)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )
    args = parser.parse_args()

    # Load environment
    env = load_env()

    print("🚀 Starting ingestion...")

    # Initialize Qdrant client
    print("  Connecting to Qdrant...")
    qdrant_client = QdrantClient(
        url=env["qdrant_url"],
        api_key=env["qdrant_api_key"],
    )

    # Ensure collection exists
    ensure_collection_exists(qdrant_client, args.collection)

    # Initialize Cohere client
    print("  Connecting to Cohere...")
    cohere_client = cohere.Client(api_key=env["cohere_api_key"])

    # Find docs directory
    docs_path = find_docs_dir(args.docs_dir)

    # Load documents
    print("  Loading documents...")
    if not docs_path:
        # No documents found - improved error message
        exts = ", ".join(SUPPORTED_EXTENSIONS)
        print(f"  ❌ No .{exts} files found")
        print(f"  💡 Add your documents to a docs folder, or use --docs-dir to specify location")
        print("🎉 Ingestion complete (0 chunks)")
        return

    documents = load_documents(docs_path)
    if not documents:
        exts = ", ".join(SUPPORTED_EXTENSIONS)
        print(f"  ❌ No .{exts} files found in {docs_path}")
        print(f"  💡 Add your documents there, or use --docs-dir to specify location")
        print("🎉 Ingestion complete (0 chunks)")
        return

    # Chunk documents
    print("  Chunking documents...")
    splitter = get_text_splitter()
    all_chunks = chunk_documents(documents, splitter)
    total_chunks = len(all_chunks)
    print(f"  Created {total_chunks} chunks")

    # Load progress
    progress = load_progress()
    processed_ids = set(progress.get("processed_ids", []))

    # Filter already processed chunks
    chunks_to_process = []
    for chunk in all_chunks:
        chunk_id = generate_chunk_id(chunk["chunk_text"])
        if chunk_id not in processed_ids:
            chunks_to_process.append(chunk)

    skipped = total_chunks - len(chunks_to_process)
    print(f"  {skipped} chunks already processed, {len(chunks_to_process)} to process")

    # Embed and upload
    uploaded = 0
    for i in range(0, len(chunks_to_process), BATCH_SIZE):
        batch = chunks_to_process[i : i + BATCH_SIZE]
        batch_texts = [c["chunk_text"] for c in batch]

        # Embed with retry
        embeddings, success = embed_with_retry(cohere_client, batch_texts)
        if not success:
            print(f"  Failed to embed batch starting at {i}")
            continue

        # Upload to Qdrant
        points = []
        for chunk, embedding in zip(batch, embeddings):
            chunk_id = generate_chunk_id(chunk["chunk_text"])
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "text": chunk["chunk_text"],
                        "source": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "ingested_at": datetime.now().isoformat(),
                    },
                )
            )

        qdrant_client.upsert(
            collection_name=args.collection,
            points=points,
        )

        # Track progress
        for c in batch:
            processed_ids.add(generate_chunk_id(c["chunk_text"]))
        save_progress({
            "processed_ids": list(processed_ids),
            "last_updated": datetime.now().isoformat(),
        })

        uploaded += len(batch)
        print(f"  📦 Skipped {skipped} chunks | ✅ Uploaded {uploaded}/{len(chunks_to_process)} to Qdrant")

        # Rate limiting delay
        if i + BATCH_SIZE < len(chunks_to_process):
            time.sleep(API_DELAY)

    print("🎉 Ingestion complete")


if __name__ == "__main__":
    main()