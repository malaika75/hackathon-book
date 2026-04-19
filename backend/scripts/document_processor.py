"""
Document processor module for scanning and reading markdown files.

This module handles file system operations for finding and reading documents.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """Represents a document from the filesystem."""

    file_path: Path
    content: str
    module: str
    chapter_title: str
    headings: List[str]
    url: str
    hierarchy_level: int


class DocumentProcessor:
    """
    Scans directories and reads markdown files.

    Handles finding all .md and .mdx files and reading their content.
    """

    def __init__(self, docs_path: Path):
        """
        Initialize the document processor.

        Args:
            docs_path: Root path to the docs directory
        """
        self.docs_path = docs_path
        self.markdown_extensions = {".md", ".mdx"}

    def scan_documents(self) -> List[Path]:
        """
        Scan the docs directory for all markdown files.

        Returns:
            List of paths to markdown files
        """
        markdown_files = []

        for root, dirs, files in os.walk(self.docs_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.markdown_extensions:
                    markdown_files.append(file_path)

        return sorted(markdown_files)

    def read_document(self, file_path: Path) -> str:
        """
        Read the content of a markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            Content of the file as string
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read {file_path}: {e}")

    def is_readable(self, file_path: Path) -> bool:
        """
        Check if a file is readable.

        Args:
            file_path: Path to check

        Returns:
            True if file exists and is readable
        """
        return file_path.exists() and file_path.is_file() and os.access(file_path, os.R_OK)
