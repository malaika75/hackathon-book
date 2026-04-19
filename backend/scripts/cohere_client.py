"""
Cohere client module for embedding generation.

Handles embedding text using Cohere's embedding API.
"""

from typing import List

import cohere


class CohereEmbeddingClient:
    """
    Generates embeddings using Cohere's API.

    Handles text-to-vector conversion using embed-english-v3.0 model.
    """

    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        """
        Initialize Cohere client.

        Args:
            api_key: Cohere API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key=api_key)

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query",
        )
        return response.embeddings[0]
