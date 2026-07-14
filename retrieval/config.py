"""Define configuration values for the evidence-building pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceBuilderConfig:
    """Store settings used while constructing the knowledge base.

    Attributes:
        ingestion_root (Path): Directory containing ingested documents.
        output_root (Path): Directory receiving knowledge-base files.
        embedding_model (str): Sentence Transformer model identifier.
        max_chunk_tokens (int): Maximum tokens in a prose chunk.
        small_table_max_rows (int): Maximum rows for a single table chunk.
        small_table_max_tokens (int): Maximum tokens for a single table chunk.
        table_window_rows (int): Rows included in each large-table window.
        table_window_overlap (int): Rows shared between table windows.
        embedding_batch_size (int): Number of chunks embedded per batch.
        enable_figure_ocr (bool): Whether Tesseract should inspect figures.
    """

    ingestion_root: Path = Path("/data/documents")
    output_root: Path = Path("/data/retrieval/current")

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    max_chunk_tokens: int = 256
    small_table_max_rows: int = 12
    small_table_max_tokens: int = 220
    table_window_rows: int = 6
    table_window_overlap: int = 1
    embedding_batch_size: int = 32
    enable_figure_ocr: bool = True