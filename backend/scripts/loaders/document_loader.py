"""
Document loader module to coordinate document processing.

Coordinates parsing, metadata extraction, and chunking.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from scripts.document_processor import DocumentProcessor
from scripts.parsers.markdown_parser import MarkdownParser
from scripts.extractors.metadata_extractor import MetadataExtractor
from scripts.chunkers.text_chunker import TextChunk


@dataclass
class LoadedDocument:
    """Represents a loaded and processed document."""

    file_path: Path
    module: str
    chapter: str
    section: str
    url: str
    hierarchy_level: int
    chunks: List[TextChunk]


class DocumentLoader:
    """
    Coordinates document loading pipeline.

    Handles scanning, parsing, metadata extraction, and chunking.
    """

    def __init__(self, docs_path: Path):
        """
        Initialize document loader.

        Args:
            docs_path: Path to docs directory
        """
        self.docs_path = docs_path
        self.processor = DocumentProcessor(docs_path)
        self.parser = MarkdownParser()
        self.metadata_extractor = MetadataExtractor(docs_path)

    def load_all(self) -> List[LoadedDocument]:
        """
        Load and process all documents.

        Returns:
            List of LoadedDocument objects
        """
        documents = []

        # Scan for markdown files
        markdown_files = self.processor.scan_documents()

        for file_path in markdown_files:
            try:
                doc = self.load_document(file_path)
                if doc and doc.chunks:
                    documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

        return documents

    def load_document(self, file_path: Path) -> LoadedDocument:
        """
        Load and process a single document.

        Args:
            file_path: Path to markdown file

        Returns:
            LoadedDocument object

        Raises:
            IOError: If file cannot be read
        """
        # Read content
        content = self.processor.read_document(file_path)

        # Parse markdown
        parsed = self.parser.parse(file_path, content)

        # Extract metadata
        metadata = self.metadata_extractor.extract(
            file_path, parsed.title, parsed.headings
        )

        # Create text chunker and chunk the plain text
        from scripts.chunkers.text_chunker import TextChunker

        chunker = TextChunker()
        chunks = chunker.chunk_text(parsed.plain_text)

        return LoadedDocument(
            file_path=file_path,
            module=metadata["module"],
            chapter=metadata["chapter"],
            section=metadata["section"],
            url=metadata["url"],
            hierarchy_level=metadata["hierarchy_level"],
            chunks=chunks,
        )