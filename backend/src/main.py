"""FastAPI application for RAG Backend."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("Starting RAG Backend API...")
    # Validate settings
    missing = settings.validate()
    if missing:
        logger.warning(f"Missing required settings: {', '.join(missing)}")
    yield
    logger.info("Shutting down RAG Backend API...")


# Create FastAPI application
app = FastAPI(
    title="RAG Backend API",
    description="FastAPI backend with RAG Agent for Physical AI Textbook Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )