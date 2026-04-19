"""Cohere embedding service."""

import logging
from typing import Optional

import cohere

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Cohere."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Cohere client.

        Args:
            api_key: Optional API key override
        """
        self.api_key = api_key or settings.cohere_api_key
        if not self.api_key:
            raise ValueError("Cohere API key is required")
        self.client = cohere.AsyncClient(self.api_key)
        logger.info("EmbeddingService initialized")

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector

        Raises:
            cohere.errors.CohereAPIError: On API errors
        """
        try:
            response = await self.client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_query",
                truncate="END"
            )
            embedding = response.embeddings[0]
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        except cohere.errors.CohereAPIError as e:
            logger.error(f"Cohere API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise


# Global service instance (lazy initialization)
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service