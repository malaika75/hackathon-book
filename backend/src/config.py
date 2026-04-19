"""Configuration module for RAG Backend API."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Try parent directory
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)


class Settings:
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Google Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Groq (for Grok)
    groq_api_key: str = os.getenv("GROK-API-KEY", "") or os.getenv("GROQ_API_KEY", "")

    # Cohere
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "physical-ai-textbook")

    # Application
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "5"))

    def validate(self) -> list[str]:
        """Validate required settings and return list of missing keys."""
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        return missing


# Global settings instance
settings = Settings()