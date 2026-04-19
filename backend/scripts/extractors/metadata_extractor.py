"""
Metadata extractor module for capturing document metadata.

Extracts module name, chapter, URL, and other metadata from document paths.
"""

from pathlib import Path
from typing import Dict, List


class MetadataExtractor:
    """
    Extracts metadata from document file paths and content.

    Determines module, chapter, URL, and hierarchy level.
    """

    # Map folder names to module names
    MODULE_MAP = {
        "module1": "Module 1: ROS 2 Fundamentals",
        "module2": "Module 2: Robotics Simulation",
        "module3": "Module 3: AI-Robot Brain",
        "module4": "Module 4: VLA Models",
    }

    def __init__(self, base_docs_path: Path):
        """
        Initialize metadata extractor.

        Args:
            base_docs_path: Root path to docs directory
        """
        self.base_docs_path = base_docs_path

    def extract(self, file_path: Path, title: str, headings: List) -> Dict:
        """
        Extract metadata from a document.

        Args:
            file_path: Path to the markdown file
            title: Document title from first H1
            headings: List of (level, text) tuples

        Returns:
            Dictionary of metadata
        """
        relative_path = file_path.relative_to(self.base_docs_path)
        parts = relative_path.parts

        # Determine module
        module = self._extract_module(parts)

        # Determine chapter (use title or folder name)
        chapter = self._extract_chapter(parts, title)

        # Determine URL path
        url = self._extract_url(relative_path)

        # Determine hierarchy level
        hierarchy_level = self._extract_hierarchy_level(parts)

        # Extract current section (nearest heading)
        section = self._extract_section(headings)

        return {
            "module": module,
            "chapter": chapter,
            "section": section,
            "url": url,
            "hierarchy_level": hierarchy_level,
            "file_path": str(file_path),
        }

    def _extract_module(self, parts: tuple) -> str:
        """Extract module name from path parts."""
        for part in parts:
            if part.lower() in self.MODULE_MAP:
                return self.MODULE_MAP[part.lower()]
        return "Unknown Module"

    def _extract_chapter(self, parts: tuple, title: str) -> str:
        """Extract chapter name from path or title."""
        if len(parts) >= 2:
            # Use folder name as chapter if available
            chapter_folder = parts[1]
            if chapter_folder.startswith("chapter"):
                return chapter_folder.replace("-", " ").title()
        return title

    def _extract_url(self, relative_path: Path) -> str:
        """Generate URL path from file path."""
        # Remove .md/.mdx extension and convert to URL format
        path_str = str(relative_path.with_suffix(""))
        return "/" + path_str.replace("\\", "/")

    def _extract_hierarchy_level(self, parts: tuple) -> int:
        """Determine hierarchy level from path depth."""
        # Base docs folder is level 1
        return len(parts)

    def _extract_section(self, headings: List) -> str:
        """Extract current section (first H2 after title)."""
        for level, text in headings:
            if level == 2:
                return text
        return ""
