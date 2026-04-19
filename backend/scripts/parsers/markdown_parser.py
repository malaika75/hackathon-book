"""
Markdown parser module for extracting headings and content from markdown files.

Handles parsing markdown files to extract structure and content.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class ParsedDocument:
    """Represents a parsed markdown document."""

    file_path: Path
    title: str
    headings: List[Tuple[int, str]]  # (level, text)
    content: str
    plain_text: str


class MarkdownParser:
    """
    Parses markdown files and extracts structure.

    Extracts headings (H1-H3) and content while preserving structure.
    """

    # Regex patterns for markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def __init__(self):
        """Initialize the markdown parser."""
        pass

    def parse(self, file_path: Path, content: str) -> ParsedDocument:
        """
        Parse markdown content.

        Args:
            file_path: Path to the markdown file
            content: Raw markdown content

        Returns:
            ParsedDocument with extracted structure
        """
        # Extract title from first H1
        headings = self.extract_headings(content)
        title = self.extract_title(headings)

        # Get plain text (strip markdown syntax)
        plain_text = self.markdown_to_plain_text(content)

        return ParsedDocument(
            file_path=file_path,
            title=title,
            headings=headings,
            content=content,
            plain_text=plain_text,
        )

    def extract_headings(self, content: str) -> List[Tuple[int, str]]:
        """
        Extract all headings from markdown content.

        Args:
            content: Markdown content

        Returns:
            List of tuples (level, heading_text)
        """
        headings = []
        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append((level, text))
        return headings

    def extract_title(self, headings: List[Tuple[int, str]]) -> str:
        """
        Extract the title from headings (first H1).

        Args:
            headings: List of headings

        Returns:
            Title string or "Untitled"
        """
        for level, text in headings:
            if level == 1:
                return text
        return "Untitled"

    def markdown_to_plain_text(self, content: str) -> str:
        """
        Convert markdown to plain text.

        Args:
            content: Markdown content

        Returns:
            Plain text
        """
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", content)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove heading markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove bold/italic
        text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
        text = re.sub(r"_+([^_]+)_+", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text
