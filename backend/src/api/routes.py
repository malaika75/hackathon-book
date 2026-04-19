"""API routes for RAG Backend."""

import logging
from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    ContextualChatRequest,
    ErrorResponse,
    HealthResponse
)
from ..agents.rag_agent import get_rag_agent
from ..services.retrieval import check_qdrant_health

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@router.get("/health/qdrant", tags=["health"])
async def qdrant_health_check():
    """Qdrant connection health check."""
    return check_qdrant_health()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["chat"]
)
async def chat(request: ChatRequest):
    """Normal chat endpoint - answer questions about the textbook."""
    try:
        agent = get_rag_agent()
        response = await agent.process_normal_chat(request.question)
        return response
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process your question. Please try again.")


@router.post(
    "/chat-with-context",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["chat"]
)
async def chat_with_context(request: ContextualChatRequest):
    """Contextual chat endpoint - answer questions with selected text context."""
    try:
        agent = get_rag_agent()
        response = await agent.process_normal_chat(
            request.question,
            selected_text=request.selected_text
        )
        return response
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process your question. Please try again.")