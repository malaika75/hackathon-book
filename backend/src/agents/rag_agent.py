"""RAG Agent implementation using Grok (via Groq)."""

import logging
from typing import Optional
import os

from groq import Groq

from ..config import settings
from ..models.schemas import ChatResponse, Citation
from ..services.retrieval import get_retrieval_service, RetrievedChunk

logger = logging.getLogger(__name__)


class RAGAgent:
    """RAG Agent for generating answers from textbook content."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the RAG Agent."""
        self.api_key = api_key or settings.groq_api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("Groq or Google API key is required")

        # Use sync Groq client
        self.client = Groq(api_key=self.api_key)
        self.retrieval_service = get_retrieval_service()
        logger.info("RAGAgent initialized with Grok (sync)")

    async def process_normal_chat(
        self,
        question: str,
        selected_text: Optional[str] = None
    ) -> ChatResponse:
        """Process a chat request."""
        logger.info(f"Processing chat question: {question[:50]}...")

        context_parts = []
        if selected_text:
            context_parts.append(f"Selected text from the document:\n{selected_text}\n")

        top_k = settings.top_k
        chunks = await self.retrieval_service.retrieve_chunks(question, top_k=top_k)

        if chunks:
            for i, chunk in enumerate(chunks):
                context_parts.append(f"[{i+1}] {chunk.text}")
                logger.debug(f"Chunk {i+1}: score={chunk.score:.3f}, {chunk.chapter}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful teaching assistant for a textbook on Physical AI and Robotics.
Your task is to answer the user's question based on the provided context from the textbook.

Instructions:
- Use the context to answer the question accurately
- If the context doesn't contain relevant information, say so honestly
- Include citations when referencing specific sections
- Be educational and clear in your explanations

Context from the textbook:
{context}

User's question: {question}

Your answer:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {len(answer)} chars")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ChatResponse(
                answer=f"I'm having trouble responding right now. Please try again. (Error: {type(e).__name__})",
                citations=[]
            )

        citations = [chunk.to_citation() for chunk in chunks]

        return ChatResponse(
            answer=answer,
            citations=citations
        )


_rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get the global RAG agent instance."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent