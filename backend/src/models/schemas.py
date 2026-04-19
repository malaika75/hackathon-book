"""Pydantic schemas for request/response models."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for normal chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question about the textbook"
    )

    @field_validator("question", mode="before")
    @classmethod
    def validate_question_not_empty(cls, v: str) -> str:
        """Validate that question is not just whitespace."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Question cannot be empty or whitespace")
        return v


class ContextualChatRequest(BaseModel):
    """Request model for contextual chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question"
    )
    selected_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text selected by user in the UI"
    )

    @field_validator("question", "selected_text", mode="before")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that field is not just whitespace."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Field cannot be empty or whitespace")
        return v


class Citation(BaseModel):
    """Citation for source reference."""

    module: str = Field(..., description="Module name")
    chapter: str = Field(..., description="Chapter title")
    section: str = Field(..., description="Section heading")
    url: Optional[str] = Field(None, description="Link to source document")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    answer: str = Field(..., description="Generated response to user's question")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source references"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")