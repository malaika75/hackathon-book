#!/usr/bin/env python3
"""
Main ingestion script for RAG pipeline.

Orchestrates the complete document ingestion process:
1. Load documents from docs/
2. Generate embeddings with Cohere
3. Store vectors in Qdrant

Usage:
    python ingest_book.py [--docs-path ./docs] [--collection collection_name]
"""

import argparse
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from scripts.config import load_config
from scripts.document_processor import DocumentProcessor
from scripts.parsers.markdown_parser import MarkdownParser
from scripts.extractors.metadata_extractor import MetadataExtractor
from scripts.chunkers.text_chunker import TextChunker, TextChunk
from scripts.loaders.document_loader import DocumentLoader
from scripts.generators.embedding_generator import EmbeddingGenerator
from scripts.uploaders.qdrant_uploader import QdrantUploader
from scripts.cohere_client import CohereEmbeddingClient
from scripts.qdrant_vec_client import QdrantVectorStore
from scripts.loggers.progress_logger import ProgressLogger
from scripts.reporters.statistics_reporter import StatisticsReporter
from scripts.testers.test_query_runner import TestQueryRunner


class IngestionPipeline:
    """Main ingestion pipeline coordinator."""

    def __init__(self, docs_path: Path, collection_name: str, verbose: bool = False):
        self.docs_path = docs_path
        self.collection_name = collection_name
        self.verbose = verbose
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
        }

    def run(self):
        """Execute the full ingestion pipeline."""
        print("=" * 60)
        print("RAG Ingestion Pipeline - Phase 1")
        print("=" * 60)

        # Load configuration
        print("\n[1/6] Loading configuration...")
        config = load_config()
        print(f"  Docs path: {config.docs_path}")
        print(f"  Collection: {config.qdrant_collection}")

        # Initialize clients
        print("\n[2/6] Initializing clients...")
        cohere_client = CohereEmbeddingClient(config.cohere_api_key)
        qdrant_client = QdrantVectorStore(
            config.qdrant_url,
            config.qdrant_api_key,
            self.collection_name,
        )

        # Load documents
        print("\n[3/6] Loading documents...")
        loader = DocumentLoader(self.docs_path)
        documents = loader.load_all()
        self.stats["documents_processed"] = len(documents)
        print(f"  Loaded {len(documents)} documents")

        # Count total chunks
        total_chunks = sum(len(doc.chunks) for doc in documents)
        self.stats["chunks_created"] = total_chunks
        print(f"  Created {total_chunks} chunks")

        # Generate embeddings
        print("\n[4/6] Generating embeddings...")
        embedding_gen = EmbeddingGenerator(cohere_client)

        all_embeddings = []
        all_chunks = []
        for doc in documents:
            for chunk in doc.chunks:
                all_chunks.append(chunk)

        # Batch generate embeddings
        embeddings = embedding_gen.generate_embeddings(all_chunks)
        all_embeddings.extend(embeddings)
        print(f"  Generated {len(all_embeddings)} embeddings")

        # Upload to Qdrant
        print("\n[5/6] Uploading to Qdrant...")
        uploader = QdrantUploader(qdrant_client)
        vectors_stored = uploader.upload_documents(documents, all_embeddings)
        self.stats["vectors_stored"] = vectors_stored
        print(f"  Stored {vectors_stored} vectors")

        # Test query (optional verification)
        print("\n[6/6] Verifying storage...")
        test_runner = TestQueryRunner(qdrant_client)
        test_results = test_runner.run_verification_query()
        print(f"  Test query returned {len(test_results)} results")

        # Print summary
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {self.stats['documents_processed']}")
        print(f"Chunks created: {self.stats['chunks_created']}")
        print(f"Vectors stored: {self.stats['vectors_stored']}")
        print(f"Collection: {self.collection_name}")
        print("=" * 60)

        return self.stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest textbook content into Qdrant vector database"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default="./docs",
        help="Path to docs folder (default: ./docs)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="physical-ai-textbook",
        help="Qdrant collection name (default: physical-ai-textbook)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Load .env file if exists
    load_dotenv()

    # Run pipeline
    pipeline = IngestionPipeline(
        docs_path=Path(args.docs_path),
        collection_name=args.collection,
        verbose=args.verbose,
    )

    try:
        pipeline.run()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
