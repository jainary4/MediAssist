"""Define evidence records shared by the retrieval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceChunk:
    """Represent one independently searchable piece of evidence.

    Attributes:
        chunk_id (str): Stable unique identifier.
        document_id (str): Source ingestion document identifier.
        content_type (str): text, table, table_window, figure, or reference.
        title (str): Human-readable document title.
        section (str): Closest section heading.
        search_text (str): Contextualized text sent to search indexes.
        display_text (str): Evidence text later shown to the assistant.
        page_numbers (list[int]): Source PDF pages.
        source_refs (list[str]): Docling JSON references.
        parent_id (str | None): Parent evidence identifier.
        asset_path (str | None): Table or figure image path.
        ingestion_quality (str): Ingestion quality label.
        metadata (dict[str, Any]): Additional evidence metadata.
    """

    chunk_id: str
    document_id: str
    content_type: str
    title: str
    section: str
    search_text: str
    display_text: str
    page_numbers: list[int]
    source_refs: list[str]
    parent_id: str | None = None
    asset_path: str | None = None
    ingestion_quality: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the evidence chunk to a serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the chunk.
        """
        return asdict(self)


@dataclass
class EvidenceBundle:
    """Collect all evidence produced from all ingested documents.

    Attributes:
        documents (list[dict[str, Any]]): Document metadata records.
        chunks (list[EvidenceChunk]): Searchable evidence chunks.
        tables (list[dict[str, Any]]): Table parent records.
        table_rows (list[dict[str, Any]]): Structured table rows.
        figures (list[dict[str, Any]]): Figure metadata records.
        references (list[dict[str, Any]]): Cross-document links.
    """

    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[EvidenceChunk] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    table_rows: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)