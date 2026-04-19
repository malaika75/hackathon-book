"""
Embedding generator module using Cohere.

Generates embeddings for text chunks using Cohere API.
"""

from typing import List

from scripts.cohere_client import CohereEmbeddingClient
from scripts.chunkers.text_chunker import TextChunk


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.

    Uses Cohere's embed-english-v3.0 model.
    """

    def __init__(self, cohere_client: CohereEmbeddingClient):
        """
        Initialize embedding generator.

        Args:
            cohere_client: Initialized Cohere client
        """
        self.cohere_client = cohere_client
        self.embedding_dim = 1536  # embed-english-v3.0 dimension

    def generate_embeddings(self, chunks: List[TextChunk]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of embedding vectors
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.cohere_client.generate_embeddings(texts)
        return embeddings

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.cohere_client.generate_query_embedding(query)
