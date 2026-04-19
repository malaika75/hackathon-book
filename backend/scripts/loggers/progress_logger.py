"""
Progress logger module for displaying ingestion progress.

Provides progress output during the ingestion process.
"""

import sys
from typing import Optional


class ProgressLogger:
    """
    Logs progress during ingestion.

    Displays progress messages and updates.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize progress logger.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.total_docs = 0
        self.current_doc = 0

    def set_total_documents(self, total: int):
        """
        Set the total number of documents.

        Args:
            total: Total document count
        """
        self.total_docs = total
        print(f"Processing {total} documents...")

    def log_document(self, file_path: str, doc_num: int):
        """
        Log processing of a document.

        Args:
            file_path: Path to the document
            doc_num: Current document number
        """
        self.current_doc = doc_num
        if self.verbose:
            print(f"  [{doc_num}/{self.total_docs}] Processing: {file_path}")

    def log_chunk(self, chunk_num: int, total_chunks: int):
        """
        Log chunk processing.

        Args:
            chunk_num: Current chunk number
            total_chunks: Total chunks
        """
        if self.verbose and chunk_num % 10 == 0:
            print(f"    Chunks: {chunk_num}/{total_chunks}")

    def log_embedding(self, batch_num: int, total_batches: int):
        """
        Log embedding generation progress.

        Args:
            batch_num: Current batch number
            total_batches: Total batches
        """
        if self.verbose:
            print(f"    Generating embeddings: {batch_num}/{total_batches}")

    def log_upload(self, vectors_uploaded: int):
        """
        Log upload progress.

        Args:
            vectors_uploaded: Number of vectors uploaded
        """
        if self.verbose:
            print(f"    Uploaded {vectors_uploaded} vectors")

    def info(self, message: str):
        """
        Log an info message.

        Args:
            message: Message to log
        """
        print(f"  INFO: {message}")

    def warning(self, message: str):
        """
        Log a warning message.

        Args:
            message: Warning message
        """
        print(f"  WARNING: {message}", file=sys.stderr)

    def error(self, message: str):
        """
        Log an error message.

        Args:
            message: Error message
        """
        print(f"  ERROR: {message}", file=sys.stderr)
