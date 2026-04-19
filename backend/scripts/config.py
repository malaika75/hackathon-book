"""
Configuration module for RAG Ingestion Pipeline.

Loads environment variables and provides configuration to all other modules.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration settings for the ingestion pipeline."""

    # Cohere settings
    cohere_api_key: str

    # Qdrant settings
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str

    # Paths
    docs_path: Path

    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 100


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Configuration object with all settings

    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()

    # Required: Cohere
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is required")

    # Required: Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable is required")

    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "physical-ai-textbook")

    # Optional: Docs path
    docs_path_str = os.getenv("DOCS_PATH", "./docs")
    docs_path = Path(docs_path_str)

    return Config(
        cohere_api_key=cohere_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        docs_path=docs_path,
    )


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


_config: Optional[Config] = None
