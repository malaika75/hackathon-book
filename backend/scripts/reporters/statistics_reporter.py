"""
Statistics reporter module for displaying final ingestion statistics.

Provides summary output after ingestion completes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IngestionStats:
    """Statistics from ingestion."""

    documents_processed: int
    chunks_created: int
    vectors_stored: int
    errors_count: int = 0
    warnings_count: int = 0


class StatisticsReporter:
    """
    Reports final statistics after ingestion.

    Displays formatted summary of ingestion results.
    """

    def __init__(self):
        """Initialize statistics reporter."""
        self.stats: Optional[IngestionStats] = None

    def set_stats(self, stats: IngestionStats):
        """
        Set statistics to report.

        Args:
            stats: IngestionStats object
        """
        self.stats = stats

    def print_summary(self):
        """Print the summary report."""
        if not self.stats:
            return

        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Documents processed: {self.stats.documents_processed}")
        print(f"Chunks created:     {self.stats.chunks_created}")
        print(f"Vectors stored:     {self.stats.vectors_stored}")

        if self.stats.errors_count > 0:
            print(f"\nErrors:             {self.stats.errors_count}")

        if self.stats.warnings_count > 0:
            print(f"Warnings:           {self.stats.warnings_count}")

        print("=" * 50)

    def print_detailed(self):
        """Print detailed statistics."""
        if not self.stats:
            return

        self.print_summary()

        # Calculate some derived stats
        if self.stats.documents_processed > 0:
            avg_chunks = self.stats.chunks_created / self.stats.documents_processed
            print(f"Average chunks per document: {avg_chunks:.1f}")

        if self.stats.chunks_created > 0:
            coverage = (
                self.stats.vectors_stored / self.stats.chunks_created * 100
            )
            print(f"Embedding coverage: {coverage:.1f}%")

    def to_dict(self) -> dict:
        """
        Convert statistics to dictionary.

        Returns:
            Dictionary representation
        """
        if not self.stats:
            return {}

        return {
            "documents_processed": self.stats.documents_processed,
            "chunks_created": self.stats.chunks_created,
            "vectors_stored": self.stats.vectors_stored,
            "errors_count": self.stats.errors_count,
            "warnings_count": self.stats.warnings_count,
        }
