"""
Qdrant uploader module for storing vectors.

Uploads embeddings and payloads to Qdrant vector database.
"""

import uuid
from typing import List

from qdrant_client.models import PointStruct

from scripts.qdrant_vec_client import QdrantVectorStore
from scripts.chunkers.text_chunker import TextChunk
from scripts.loaders.document_loader import LoadedDocument


class QdrantUploader:
    """
    Uploads vectors to Qdrant.

    Handles embedding storage with metadata payloads.
    """

    def __init__(self, qdrant_client: QdrantVectorStore):
        """
        Initialize Qdrant uploader.

        Args:
            qdrant_client: Initialized Qdrant client
        """
        self.qdrant_client = qdrant_client
        self.vector_size = 1536  # Cohere embed-english-v3.0

    def upload_documents(
        self, documents: List[LoadedDocument], embeddings: List[List[float]]
    ) -> int:
        """
        Upload documents with their embeddings to Qdrant.

        Args:
            documents: List of LoadedDocument objects
            embeddings: Corresponding list of embedding vectors

        Returns:
            Number of vectors uploaded
        """
        # Create collection if needed
        self.qdrant_client.create_collection_if_not_exists(
            vector_size=self.vector_size
        )

        points = []
        chunk_idx = 0

        for doc in documents:
            for chunk in doc.chunks:
                if chunk_idx >= len(embeddings):
                    break

                embedding = embeddings[chunk_idx]

                # Create payload with metadata
                payload = {
                    "module": doc.module,
                    "chapter": doc.chapter,
                    "section": doc.section,
                    "url": doc.url,
                    "file_path": str(doc.file_path),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                }

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
                points.append(point)
                chunk_idx += 1

        return self.qdrant_client.upsert_vectors(points)

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the collection.

        Returns:
            Vector count
        """
        return self.qdrant_client.get_count()
