"""
Text chunker module for splitting documents into chunks.

Uses LangChain's RecursiveCharacterTextSplitter for semantic chunking.
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    """Represents a chunk of text."""

    content: str
    chunk_index: int
    token_count: int


class TextChunker:
    """
    Splits text into semantic chunks.

    Uses RecursiveCharacterTextSplitter with markdown-aware separators.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                "```\n", # Code blocks
                " ",     # Words
                "",
            ],
            keep_separator=True,
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Plain text to chunk

        Returns:
            List of TextChunk objects
        """
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4

        # If text is small enough, return as single chunk
        if estimated_tokens <= self.chunk_size:
            return [
                TextChunk(
                    content=text,
                    chunk_index=0,
                    token_count=estimated_tokens,
                )
            ]

        # Split the text
        chunks = self.splitter.split_text(text)

        return [
            TextChunk(
                content=chunk,
                chunk_index=i,
                token_count=len(chunk) // 4,  # Rough estimate
            )
            for i, chunk in enumerate(chunks)
        ]

    def chunk_markdown(self, content: str) -> List[TextChunk]:
        """
        Split markdown content while trying to preserve structure.

        Args:
            content: Markdown content

        Returns:
            List of TextChunk objects
        """
        # Use same logic - splitter handles markdown structure
        return self.chunk_text(content)