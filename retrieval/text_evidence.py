"""Create prose evidence chunks from a Docling document."""

from __future__ import annotations

from typing import Any

from docling_core.transforms.chunker.hybrid_chunker import (
    HybridChunker,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer

from retrieval.config import EvidenceBuilderConfig
from retrieval.models import EvidenceChunk
from retrieval.utils import (
    item_pages,
    label_value,
    normalize_text,
    source_reference,
    stable_id,
)


EXCLUDED_LABELS = {
    "table",
    "picture",
    "chart",
    "page_header",
    "page_footer",
}


class TextEvidenceBuilder:
    """Create token-aware prose chunks while preserving document structure."""

    def __init__(self, config: EvidenceBuilderConfig) -> None:
        """Initialize the Docling hybrid chunker.

        Args:
            config (EvidenceBuilderConfig): Evidence builder settings.
        """
        self.config = config

        huggingface_tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model
        )

        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=huggingface_tokenizer,
            max_tokens=config.max_chunk_tokens,
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
            repeat_table_header=True,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer.

        Args:
            text (str): Candidate evidence text.

        Returns:
            int: Token count.
        """
        return int(self.tokenizer.count_tokens(text=text))

    def build(
        self,
        document: Any,
        document_record: dict[str, Any],
    ) -> list[EvidenceChunk]:
        """Build prose chunks from one Docling document.

        Args:
            document (Any): Loaded DoclingDocument.
            document_record (dict[str, Any]): Source document metadata.

        Returns:
            list[EvidenceChunk]: Searchable prose chunks.
        """
        evidence: list[EvidenceChunk] = []

        for position, chunk in enumerate(
            self.chunker.chunk(dl_doc=document),
            start=1,
        ):
            doc_items = list(
                getattr(chunk.meta, "doc_items", []) or []
            )

            labels = {
                label_value(item)
                for item in doc_items
            }

            if labels.intersection(EXCLUDED_LABELS):
                continue

            display_text = normalize_text(
                getattr(chunk, "text", "")
            )

            if not display_text:
                continue

            contextualized_text = normalize_text(
                self.chunker.contextualize(chunk=chunk)
            )

            headings = [
                normalize_text(str(heading))
                for heading in (
                    getattr(chunk.meta, "headings", []) or []
                )
                if normalize_text(str(heading))
            ]

            section = headings[-1] if headings else ""

            pages = sorted({
                page
                for item in doc_items
                for page in item_pages(item)
            })

            source_refs = [
                reference
                for reference in (
                    source_reference(item)
                    for item in doc_items
                )
                if reference
            ]

            content_type = (
                "footnote"
                if labels and labels == {"footnote"}
                else "text"
            )

            chunk_id = stable_id(
                "text",
                document_record["document_id"],
                position,
                "|".join(source_refs),
                display_text[:120],
            )

            search_text = self._build_search_text(
                title=document_record["title"],
                section=section,
                text=contextualized_text,
            )

            evidence.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=document_record["document_id"],
                    content_type=content_type,
                    title=document_record["title"],
                    section=section,
                    search_text=search_text,
                    display_text=display_text,
                    page_numbers=pages,
                    source_refs=source_refs,
                    ingestion_quality=document_record[
                        "ingestion_quality"
                    ],
                    metadata={
                        "headings": headings,
                        "token_count": self.count_tokens(
                            contextualized_text
                        ),
                    },
                )
            )

        return evidence

    @staticmethod
    def _build_search_text(
        title: str,
        section: str,
        text: str,
    ) -> str:
        """Attach document context to prose before indexing.

        Args:
            title (str): Source document title.
            section (str): Closest section heading.
            text (str): Contextualized chunk text.

        Returns:
            str: Search-ready text.
        """
        parts = [f"Document: {title}"]

        if section:
            parts.append(f"Section: {section}")

        parts.append(text)
        return "\n".join(parts)