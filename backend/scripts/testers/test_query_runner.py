"""
Test query runner module for verifying vector storage.

Runs test queries to verify embeddings were stored correctly.
"""

from typing import List, Dict

from scripts.qdrant_vec_client import QdrantVectorStore


class TestQueryRunner:
    """
    Runs verification queries against Qdrant.

    Tests that vectors are properly stored and searchable.
    """

    def __init__(self, qdrant_client: QdrantVectorStore):
        """
        Initialize test query runner.

        Args:
            qdrant_client: Initialized Qdrant client
        """
        self.qdrant_client = qdrant_client

    def run_verification_query(self, limit: int = 3) -> List[Dict]:
        """
        Run a verification query to test storage.

        Args:
            limit: Number of results to return

        Returns:
            List of search results
        """
        # Use a sample query related to the textbook content
        test_queries = [
            "What is ROS 2?",
            "robot simulation",
            "navigation and path planning",
        ]

        all_results = []

        for query in test_queries:
            try:
                # Generate query embedding
                from scripts.generators.embedding_generator import EmbeddingGenerator
                from scripts.cohere_client import CohereEmbeddingClient

                # Get config
                from scripts.config import get_config

                config = get_config()
                cohere_client = CohereEmbeddingClient(config.cohere_api_key)
                embedding_gen = EmbeddingGenerator(cohere_client)

                query_embedding = embedding_gen.generate_query_embedding(query)

                # Search
                results = self.qdrant_client.search(
                    query_vector=query_embedding, limit=limit
                )

                if results:
                    all_results.extend(results)

            except Exception as e:
                print(f"  Warning: Test query failed: {e}")

        return all_results

    def verify_module_filter(self, module: str, limit: int = 3) -> List[Dict]:
        """
        Verify that filtering by module works.

        Args:
            module: Module name to filter by
            limit: Number of results

        Returns:
            List of filtered search results
        """
        # Get any existing vector to use as reference
        # This is a simplified check
        try:
            # Search with module filter
            # Note: This requires having at least one vector in the collection
            results = self.qdrant_client.search(
                query_vector=[0.0] * 1536,  # Placeholder
                limit=limit,
                filter_module=module,
            )
            return results
        except Exception:
            return []

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics from Qdrant.

        Returns:
            Dictionary with storage stats
        """
        info = self.qdrant_client.get_collection_info()
        if info:
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "collection_name": info.name,
            }
        return {}
