"""
Qdrant client module for vector database operations.

Handles connection to Qdrant and vector storage operations.
"""

from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QdrantVectorStore:
    """
    Manages Qdrant vector database operations.

    Handles connection, collection creation, and vector storage.
    """

    def __init__(self, url: str, api_key: str, collection_name: str):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Name of the collection to use
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection_if_not_exists(
        self, vector_size: int = 1536, force_recreate: bool = False
    ) -> bool:
        """
        Create the collection if it doesn't exist.

        Args:
            vector_size: Dimension of the embedding vectors
            force_recreate: If True, delete and recreate the collection

        Returns:
            True if collection was created
        """
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if force_recreate and collection_exists:
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            return True

        return False

    def get_collection_info(self) -> Optional[Dict]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection info or None if not found
        """
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return None

    def upsert_vectors(
        self, vectors: List[PointStruct], batch_size: int = 100
    ) -> int:
        """
        Insert or update vectors in the collection.

        Args:
            vectors: List of PointStruct objects to insert
            batch_size: Number of vectors to insert per batch

        Returns:
            Number of vectors inserted
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=vectors,
            batch_size=batch_size,
        )
        return len(vectors)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_module: Optional[str] = None,
        filter_chapter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            filter_module: Optional module to filter by
            filter_chapter: Optional chapter to filter by

        Returns:
            List of search results with payload
        """
        search_filters = []

        if filter_module:
            search_filters.append(
                FieldCondition(key="module", match=MatchValue(value=filter_module))
            )

        if filter_chapter:
            search_filters.append(
                FieldCondition(
                    key="chapter", match=MatchValue(value=filter_chapter)
                )
            )

        filter_query = (
            Filter(must=search_filters) if search_filters else None
        )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_query,
            with_payload=True,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def get_count(self) -> int:
        """
        Get the number of vectors in the collection.

        Returns:
            Count of vectors
        """
        info = self.get_collection_info()
        return info.vectors_count if info else 0

    def delete_collection(self) -> bool:
        """
        Delete the collection.

        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception:
            return False
