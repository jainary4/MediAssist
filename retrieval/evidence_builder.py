"""Orchestrate evidence extraction and knowledge-base construction."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import (
    DoclingDocument,
)

from retrieval.config import EvidenceBuilderConfig
from retrieval.figure_evidence import (
    FigureEvidenceBuilder,
)
from retrieval.keyword_store import KeywordStore
from retrieval.metadata_evidence import (
    build_document_record,
    build_reference_evidence,
    resolve_reference_targets,
)
from retrieval.models import (
    EvidenceBundle,
    EvidenceChunk,
)
from retrieval.semantic_store import SemanticStore
from retrieval.structured_store import StructuredStore
from retrieval.table_evidence import (
    TableEvidenceBuilder,
)
from retrieval.text_evidence import TextEvidenceBuilder


def build_knowledge_base(
    config: EvidenceBuilderConfig,
) -> dict[str, Any]:
    """Build the complete retrieval knowledge base.

    Args:
        config (EvidenceBuilderConfig): Builder settings.

    Returns:
        dict[str, Any]: Build manifest.

    Raises:
        FileNotFoundError: If no verified documents exist.
        RuntimeError: If every document fails.
    """
    final_directories = discover_verified_documents(
        config.ingestion_root
    )

    if not final_directories:
        raise FileNotFoundError(
            "No verified documents found in "
            f"{config.ingestion_root}"
        )

    _prepare_output_directory(
        config.output_root
    )

    bundle = EvidenceBundle()

    text_builder = TextEvidenceBuilder(config)

    table_builder = TableEvidenceBuilder(
        config=config,
        token_counter=text_builder.count_tokens,
    )

    figure_builder = FigureEvidenceBuilder(
        config
    )

    processing_errors: list[dict[str, str]] = []
    figure_audit: list[dict[str, Any]] = []

    for final_directory in final_directories:
        try:
            document_id = (
                final_directory.parent.name
            )

            document = (
                DoclingDocument.load_from_json(
                    final_directory
                    / "document.json"
                )
            )

            document_record = (
                build_document_record(
                    document=document,
                    final_directory=(
                        final_directory
                    ),
                    document_id=document_id,
                )
            )

            text_chunks = text_builder.build(
                document=document,
                document_record=document_record,
            )

            (
                table_records,
                table_rows,
                table_chunks,
            ) = table_builder.build(
                document=document,
                document_record=document_record,
                final_directory=final_directory,
            )

            (
                figure_records,
                figure_chunks,
                document_figure_audit,
            ) = figure_builder.build(
                document=document,
                document_record=document_record,
                final_directory=final_directory,
                table_records=table_records,
            )

            _synchronize_table_chunk_assets(
                table_records=table_records,
                table_chunks=table_chunks,
            )

            (
                reference_records,
                reference_chunks,
            ) = build_reference_evidence(
                document=document,
                document_record=document_record,
            )

            bundle.documents.append(
                document_record
            )

            bundle.chunks.extend(
                text_chunks
            )

            bundle.tables.extend(
                table_records
            )

            bundle.table_rows.extend(
                table_rows
            )

            bundle.chunks.extend(
                table_chunks
            )

            bundle.figures.extend(
                figure_records
            )

            bundle.chunks.extend(
                figure_chunks
            )

            bundle.references.extend(
                reference_records
            )

            bundle.chunks.extend(
                reference_chunks
            )

            figure_audit.extend(
                document_figure_audit
            )

        except Exception as error:
            processing_errors.append({
                "directory": str(
                    final_directory
                ),
                "error_type": type(
                    error
                ).__name__,
                "message": str(error),
            })

    if not bundle.documents:
        raise RuntimeError(
            "Every document failed during "
            "evidence extraction."
        )

    resolve_reference_targets(
        references=bundle.references,
        documents=bundle.documents,
    )

    _update_reference_chunk_targets(
        chunks=bundle.chunks,
        references=bundle.references,
    )

    bundle.chunks = _deduplicate_chunks(
        bundle.chunks
    )

    database_path = (
        config.output_root
        / "retrieval.sqlite"
    )

    StructuredStore(
        database_path
    ).write(bundle)

    KeywordStore(
        database_path
    ).build(bundle.chunks)

    SemanticStore(
        model_name=config.embedding_model,
        index_path=(
            config.output_root
            / "chunks.faiss"
        ),
        mapping_path=(
            config.output_root
            / "vector_mapping.json"
        ),
        batch_size=(
            config.embedding_batch_size
        ),
    ).build(bundle.chunks)

    _write_chunks_jsonl(
        chunks=bundle.chunks,
        output_path=(
            config.output_root
            / "chunks.jsonl"
        ),
    )

    _write_json(
        path=(
            config.output_root
            / "figure_audit.json"
        ),
        value=figure_audit,
    )

    manifest = _build_manifest(
        config=config,
        bundle=bundle,
        processing_errors=processing_errors,
        figure_audit=figure_audit,
    )

    _write_json(
        path=(
            config.output_root
            / "build_manifest.json"
        ),
        value=manifest,
    )

    return manifest


def discover_verified_documents(
    ingestion_root: Path,
) -> list[Path]:
    """Find final directories containing usable Docling JSON.

    Args:
        ingestion_root (Path): Ingestion output root.

    Returns:
        list[Path]: Verified final directories.
    """
    verified: list[Path] = []

    for document_json in sorted(
        ingestion_root.glob(
            "*/final/document.json"
        )
    ):
        final_directory = (
            document_json.parent
        )

        manifest_path = (
            final_directory
            / "manifest.json"
        )

        if not manifest_path.is_file():
            verified.append(
                final_directory
            )
            continue

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if manifest.get("status") == "pass":
            verified.append(
                final_directory
            )

    return verified


def _synchronize_table_chunk_assets(
    table_records: list[dict[str, Any]],
    table_chunks: list[EvidenceChunk],
) -> None:
    """Copy table visual paths into their search chunks.

    The figure builder may discover that a PictureItem is
    actually a visual representation of a structured table.

    Args:
        table_records (list[dict[str, Any]]): Updated tables.
        table_chunks (list[EvidenceChunk]): Table search chunks.
    """
    table_assets = {
        record["table_id"]: record.get(
            "asset_path"
        )
        for record in table_records
        if record.get("asset_path")
    }

    for chunk in table_chunks:
        if not chunk.parent_id:
            continue

        asset_path = table_assets.get(
            chunk.parent_id
        )

        if asset_path:
            chunk.asset_path = asset_path
            chunk.metadata[
                "has_picture_visual"
            ] = True


def _update_reference_chunk_targets(
    chunks: list[EvidenceChunk],
    references: list[dict[str, Any]],
) -> None:
    """Add resolved target document IDs to reference chunks.

    Args:
        chunks (list[EvidenceChunk]): Search chunks.
        references (list[dict[str, Any]]): Resolved references.
    """
    target_by_reference_id = {
        reference["reference_id"]: reference.get(
            "target_document_id"
        )
        for reference in references
    }

    for chunk in chunks:
        if chunk.content_type != "reference":
            continue

        reference_id = chunk.metadata.get(
            "reference_id"
        )

        chunk.metadata[
            "target_document_id"
        ] = target_by_reference_id.get(
            reference_id
        )


def _prepare_output_directory(
    output_root: Path,
) -> None:
    """Replace only generated retrieval output.

    Args:
        output_root (Path): Retrieval output directory.
    """
    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def _deduplicate_chunks(
    chunks: list[EvidenceChunk],
) -> list[EvidenceChunk]:
    """Remove duplicate chunk identifiers.

    Args:
        chunks (list[EvidenceChunk]): Candidate chunks.

    Returns:
        list[EvidenceChunk]: Deduplicated chunks.
    """
    seen: set[str] = set()
    result: list[EvidenceChunk] = []

    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue

        seen.add(chunk.chunk_id)
        result.append(chunk)

    return result


def _write_chunks_jsonl(
    chunks: list[EvidenceChunk],
    output_path: Path,
) -> None:
    """Write one JSON object per evidence chunk.

    Args:
        chunks (list[EvidenceChunk]): Evidence chunks.
        output_path (Path): JSONL output path.
    """
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_json(
    path: Path,
    value: Any,
) -> None:
    """Write a formatted JSON file.

    Args:
        path (Path): Output path.
        value (Any): Serializable value.
    """
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_manifest(
    config: EvidenceBuilderConfig,
    bundle: EvidenceBundle,
    processing_errors: list[dict[str, str]],
    figure_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final knowledge-base manifest.

    Args:
        config (EvidenceBuilderConfig): Builder settings.
        bundle (EvidenceBundle): Extracted evidence.
        processing_errors (list[dict[str, str]]): Errors.
        figure_audit (list[dict[str, Any]]): Picture decisions.

    Returns:
        dict[str, Any]: Build manifest.
    """
    indexed_figure_audits = [
        record
        for record in figure_audit
        if record["decision"]
        == "indexed_as_figure"
    ]

    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "ingestion_root": str(
            config.ingestion_root
        ),
        "output_root": str(
            config.output_root
        ),
        "embedding_model": (
            config.embedding_model
        ),
        "maximum_chunk_tokens": (
            config.max_chunk_tokens
        ),
        "counts": {
            "documents": len(
                bundle.documents
            ),
            "chunks": len(bundle.chunks),
            "text_chunks": _count_type(
                bundle.chunks,
                {"text", "footnote"},
            ),
            "table_chunks": _count_type(
                bundle.chunks,
                {
                    "table",
                    "table_parent",
                    "table_window",
                },
            ),
            "figure_chunks": _count_type(
                bundle.chunks,
                {"figure"},
            ),
            "reference_chunks": _count_type(
                bundle.chunks,
                {"reference"},
            ),
            "tables": len(bundle.tables),
            "structured_table_rows": len(
                bundle.table_rows
            ),
            "figures": len(bundle.figures),
            "document_references": len(
                bundle.references
            ),
            "resolved_document_references": sum(
                1
                for reference
                in bundle.references
                if reference.get(
                    "target_document_id"
                )
            ),
            "processing_errors": len(
                processing_errors
            ),
        },
        "figure_audit": {
            "picture_candidates": len(
                figure_audit
            ),
            "indexed_as_figures": len(
                indexed_figure_audits
            ),
            "merged_into_tables": sum(
                1
                for record in figure_audit
                if record["decision"]
                == "merged_into_table"
            ),
            "rejected": sum(
                1
                for record in figure_audit
                if record["decision"]
                == "rejected"
            ),
            "figure_assets_found": sum(
                1
                for record
                in indexed_figure_audits
                if record.get(
                    "asset_found"
                )
            ),
            "figure_ocr_attempted": sum(
                1
                for record
                in indexed_figure_audits
                if record.get(
                    "ocr_attempted"
                )
            ),
            "figure_ocr_passed": sum(
                1
                for record
                in indexed_figure_audits
                if record.get(
                    "ocr_quality"
                ) == "pass"
            ),
            "figure_manual_review": sum(
                1
                for record
                in indexed_figure_audits
                if record.get(
                    "requires_visual_check"
                )
            ),
        },
        "processing_errors": processing_errors,
        "files": {
            "structured_and_keyword_database": (
                "retrieval.sqlite"
            ),
            "semantic_index": "chunks.faiss",
            "vector_mapping": (
                "vector_mapping.json"
            ),
            "human_readable_chunks": (
                "chunks.jsonl"
            ),
            "figure_audit": (
                "figure_audit.json"
            ),
        },
    }


def _count_type(
    chunks: list[EvidenceChunk],
    content_types: set[str],
) -> int:
    """Count chunks matching selected types.

    Args:
        chunks (list[EvidenceChunk]): Evidence chunks.
        content_types (set[str]): Types to count.

    Returns:
        int: Matching chunk count.
    """
    return sum(
        1
        for chunk in chunks
        if chunk.content_type
        in content_types
    )