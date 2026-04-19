"""Qdrant retrieval service with retry logic."""

import logging
import time
from functools import wraps
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import settings
from .embedding import get_embedding_service
from ..models.schemas import Citation

logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds


def retry_on_qdrant_error(max_retries: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF):
    """Retry decorator for Qdrant operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = backoff ** attempt
                        logger.warning(f"Qdrant error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Qdrant error after {max_retries} attempts: {e}")
            # Return empty on all retries failed
            return last_error
        return wrapper
    return decorator


class RetrievedChunk:
    """Represents a chunk retrieved from Qdrant."""

    def __init__(self, id: int, score: float, text: str, module: str, chapter: str, section: str, url: Optional[str]):
        self.id = id
        self.score = score
        self.text = text
        self.module = module
        self.chapter = chapter
        self.section = section
        self.url = url

    def to_citation(self) -> Citation:
        """Convert to Citation model."""
        return Citation(
            module=self.module,
            chapter=self.chapter,
            section=self.section,
            url=self.url
        )


class RetrievalService:
    """Service for retrieving content from Qdrant."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the Qdrant client.

        Args:
            url: Optional URL override
            api_key: Optional API key override
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection = settings.qdrant_collection

        # Detailed logging for debugging 403 errors
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 8 else "NOT_SET"
        logger.info(f"Qdrant config: url={self.url}, api_key={masked_key}, collection={self.collection}")

        # Create client with fixes
        if self.api_key:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
                prefer_grpc=False,
                check_compatibility=False
            )
        else:
            self.client = QdrantClient(
                url=self.url,
                timeout=30,
                prefer_grpc=False,
                check_compatibility=False
            )

        logger.info(f"RetrievalService initialized for collection: {self.collection}")

    async def retrieve_chunks(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from Qdrant.

        Args:
            query_text: The search query
            top_k: Number of results to return

        Returns:
            List of retrieved chunks
        """
        try:
            # Get embedding for query
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_query(query_text)

            # Search in Qdrant with retry
            chunks = await self._search_with_retry(query_embedding, top_k)

            logger.info(f"Retrieved {len(chunks)} chunks for query")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []  # Return empty, allow LLM-only mode

    @retry_on_qdrant_error(max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF)
    async def _search_with_retry(self, query_embedding: list, top_k: int) -> list[RetrievedChunk]:
        """Search with retry logic."""
        try:
            # Use query_points (v1.5+ syntax)
            search_result = self.client.query_points(
                collection_name=self.collection,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            # Handle both v1.5+ (.points) and older versions (.result)
            points = getattr(search_result, 'points', None) or getattr(search_result, 'result', [])
            if not points:
                logger.warning("No chunks retrieved from Qdrant")
                return []

            # Convert to RetrievedChunk objects
            chunks = []
            for result in points:
                payload = result.payload
                chunk = RetrievedChunk(
                    id=result.id,
                    score=result.score,
                    text=payload.get("text", ""),
                    module=payload.get("module", ""),
                    chapter=payload.get("chapter", ""),
                    section=payload.get("section", ""),
                    url=payload.get("url")
                )
                chunks.append(chunk)
                logger.debug(f"Retrieved chunk {chunk.id} with score {chunk.score:.3f}")

            return chunks

        except UnexpectedResponse as e:
            status_code = e.status_code if hasattr(e, 'status_code') else "unknown"
            if status_code == 403:
                logger.error(f"Auth failed: Verify API key has access to collection '{self.collection}'")
            raise
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                logger.error(f"Qdrant 403 Forbidden: Auth failed. Verify API key has access to collection '{self.collection}'")
            elif "timeout" in error_msg.lower():
                logger.error(f"Qdrant timeout: Check network connectivity to {self.url}")
            else:
                logger.error(f"Qdrant search error: {e}")
            raise


# Global service instance (lazy initialization)
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get the global retrieval service instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


def check_qdrant_health() -> dict:
    """Check Qdrant connection health."""
    try:
        service = get_retrieval_service()
        # Use v1.5+ syntax
        collections = service.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        return {
            "status": "healthy",
            "url": service.url,
            "collection": service.collection,
            "available_collections": collection_names
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }