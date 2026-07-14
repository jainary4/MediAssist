"""Extract document metadata and valid cross-document references."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from retrieval.models import EvidenceChunk
from retrieval.utils import (
    item_pages,
    label_value,
    normalize_text,
    source_reference,
    stable_id,
)


REGISTRY_PATTERN = re.compile(
    r"\b(?:registry\s+code\s*)?(CDR-\d+)\b",
    flags=re.IGNORECASE,
)

LINKING_LANGUAGE_PATTERN = re.compile(
    r"""
    \b(
        consult
        |
        refer(?:red)?\s+to
        |
        see\s+(?:the\s+)?(?:monograph|document|appendix)
        |
        use\s+(?:the\s+)?(?:values|dosing|information)\s+from
        |
        found\s+in
        |
        available\s+in
    )\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

LOOKUP_REQUEST_PATTERN = re.compile(
    r"""
    for\s+the\s+
    (?P<requested_field>.+?)
    \s+of\s+
    (?P<entity>[^,;]+?)
    \s*,?\s*
    do\s+not\s+use
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

TARGET_TITLE_PATTERN = re.compile(
    r"""
    monograph\s+
    (?P<target_title>.+?)
    \s*\(
    \s*registry\s+code\s+
    CDR-\d+
    \s*\)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

IGNORED_REFERENCE_LABELS = {
    "page_header",
    "page_footer",
    "title",
}

STRUCTURAL_REGISTRY_PATTERNS = (
    re.compile(
        r"^CDR-\d+\s*\|\s*Rev\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^CONFIDENTIAL\s*[-—].*Clinical Document Registry",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^Clinical Reference Monograph\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Key facts\s+)?Registry:\s*CDR-\d+",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"maintained under registry code CDR-\d+",
        flags=re.IGNORECASE,
    ),
)


def build_document_record(
    document: Any,
    final_directory: Path,
    document_id: str,
) -> dict[str, Any]:
    """Build canonical metadata for one ingested document.

    Args:
        document (Any): Loaded DoclingDocument.
        final_directory (Path): Final ingestion output directory.
        document_id (str): Stable ingestion document identifier.

    Returns:
        dict[str, Any]: Canonical document metadata record.
    """
    manifest = _load_json(final_directory / "manifest.json")
    quality_report = _load_json(
        final_directory / "quality_report.json"
    )

    title = _find_document_title(document)
    metadata = _extract_metadata_tables(document)

    registry_code = _find_registry_code(
        title=title,
        metadata=metadata,
        document=document,
    )

    page_count_method = getattr(document, "num_pages", None)
    page_count = (
        int(page_count_method())
        if callable(page_count_method)
        else 0
    )

    return {
        "document_id": document_id,
        "title": title,
        "registry_code": registry_code,
        "population_scope": _population_scope(title),
        "page_count": page_count,
        "selected_pipeline": manifest.get(
            "selected_pipeline",
            "unknown",
        ),
        "ingestion_quality": _quality_label(
            manifest=manifest,
            quality_report=quality_report,
        ),
        "source_json_path": str(
            final_directory / "document.json"
        ),
        "source_markdown_path": str(
            final_directory / "document.md"
        ),
        "metadata": metadata,
    }


def build_reference_evidence(
    document: Any,
    document_record: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[EvidenceChunk],
]:
    """Build validated and deduplicated cross-document links.

    A registry-code mention becomes a link only when:

    1. It is not inside a page header, footer, or title.
    2. It is not ordinary registry metadata.
    3. The target registry differs from the source registry.
    4. The sentence contains explicit linking language.

    Args:
        document (Any): Loaded DoclingDocument.
        document_record (dict[str, Any]): Source metadata.

    Returns:
        tuple: Valid reference records and searchable reference chunks.
    """
    source_registry = str(
        document_record.get("registry_code") or ""
    ).upper()

    deduplicated: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}

    for text_item in getattr(document, "texts", []) or []:
        item_label = label_value(text_item)

        if item_label in IGNORED_REFERENCE_LABELS:
            continue

        text = normalize_text(
            getattr(text_item, "text", "")
        )

        if not text:
            continue

        if _is_structural_registry_text(text):
            continue

        if not LINKING_LANGUAGE_PATTERN.search(text):
            continue

        registry_matches = list(
            REGISTRY_PATTERN.finditer(text)
        )

        if not registry_matches:
            continue

        source_ref = source_reference(text_item)
        pages = item_pages(text_item)

        requested_field, entity = _extract_lookup_request(
            text
        )

        target_section = _target_section(text)
        target_title = _extract_target_title(text)

        for registry_match in registry_matches:
            target_registry = (
                registry_match.group(1).upper()
            )

            if (
                source_registry
                and target_registry == source_registry
            ):
                continue

            deduplication_key = (
                target_registry,
                target_section.lower(),
                entity.lower(),
                requested_field.lower(),
            )

            existing = deduplicated.get(
                deduplication_key
            )

            if existing is not None:
                existing["source_page_numbers"] = sorted(
                    set(
                        existing["source_page_numbers"]
                        + pages
                    )
                )

                if (
                    source_ref
                    and source_ref
                    not in existing["source_refs"]
                ):
                    existing["source_refs"].append(
                        source_ref
                    )

                continue

            reference_id = stable_id(
                "reference",
                document_record["document_id"],
                target_registry,
                target_section,
                entity,
                requested_field,
            )

            deduplicated[deduplication_key] = {
                "reference_id": reference_id,
                "reference_type": _reference_type(
                    target_section
                ),
                "source_document_id": document_record[
                    "document_id"
                ],
                "source_ref": source_ref,
                "source_refs": (
                    [source_ref] if source_ref else []
                ),
                "source_page_numbers": pages,
                "reference_text": text,
                "entity": entity,
                "requested_field": requested_field,
                "target_registry_code": target_registry,
                "target_title": target_title,
                "target_document_id": None,
                "target_section": target_section,
            }

    references = list(deduplicated.values())
    chunks = [
        _reference_to_chunk(
            reference=reference,
            document_record=document_record,
        )
        for reference in references
    ]

    return references, chunks


def resolve_reference_targets(
    references: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> None:
    """Resolve target registry codes to document identifiers.

    Args:
        references (list[dict[str, Any]]): References to update.
        documents (list[dict[str, Any]]): Available documents.
    """
    registry_to_document = {
        str(record["registry_code"]).upper(): record[
            "document_id"
        ]
        for record in documents
        if record.get("registry_code")
    }

    for reference in references:
        reference["target_document_id"] = (
            registry_to_document.get(
                reference[
                    "target_registry_code"
                ].upper()
            )
        )


def _reference_to_chunk(
    reference: dict[str, Any],
    document_record: dict[str, Any],
) -> EvidenceChunk:
    """Convert a validated hard link into searchable evidence.

    Args:
        reference (dict[str, Any]): Valid reference record.
        document_record (dict[str, Any]): Source metadata.

    Returns:
        EvidenceChunk: Searchable reference chunk.
    """
    search_parts = [
        f"Document: {document_record['title']}",
        (
            "Cross-document instruction: "
            f"{reference['reference_text']}"
        ),
        (
            "Target registry code: "
            f"{reference['target_registry_code']}"
        ),
    ]

    if reference["target_title"]:
        search_parts.append(
            f"Target document: {reference['target_title']}"
        )

    if reference["target_section"]:
        search_parts.append(
            f"Target section: {reference['target_section']}"
        )

    if reference["entity"]:
        search_parts.append(
            f"Lookup entity: {reference['entity']}"
        )

    if reference["requested_field"]:
        search_parts.append(
            "Requested information: "
            f"{reference['requested_field']}"
        )

    return EvidenceChunk(
        chunk_id=stable_id(
            "reference-chunk",
            reference["reference_id"],
        ),
        document_id=document_record["document_id"],
        content_type="reference",
        title=document_record["title"],
        section=reference["target_section"],
        search_text="\n".join(search_parts),
        display_text=reference["reference_text"],
        page_numbers=reference[
            "source_page_numbers"
        ],
        source_refs=reference["source_refs"],
        parent_id=reference["reference_id"],
        ingestion_quality=document_record[
            "ingestion_quality"
        ],
        metadata={
            "reference_id": reference[
                "reference_id"
            ],
            "reference_type": reference[
                "reference_type"
            ],
            "entity": reference["entity"],
            "requested_field": reference[
                "requested_field"
            ],
            "target_title": reference[
                "target_title"
            ],
            "target_registry_code": reference[
                "target_registry_code"
            ],
            "target_section": reference[
                "target_section"
            ],
        },
    )


def _is_structural_registry_text(text: str) -> bool:
    """Check whether text is registry metadata, not a link.

    Args:
        text (str): Candidate text.

    Returns:
        bool: True when the text is structural metadata.
    """
    return any(
        pattern.search(text)
        for pattern in STRUCTURAL_REGISTRY_PATTERNS
    )


def _extract_lookup_request(
    text: str,
) -> tuple[str, str]:
    """Extract the requested field and clinical entity.

    Args:
        text (str): Cross-document instruction.

    Returns:
        tuple[str, str]: Requested field and entity.
    """
    match = LOOKUP_REQUEST_PATTERN.search(text)

    if not match:
        return "", ""

    requested_field = normalize_text(
        match.group("requested_field")
    )

    entity = normalize_text(
        match.group("entity")
    )

    return requested_field, entity


def _extract_target_title(text: str) -> str:
    """Extract the destination monograph title.

    Args:
        text (str): Cross-document instruction.

    Returns:
        str: Target title or an empty string.
    """
    match = TARGET_TITLE_PATTERN.search(text)

    if not match:
        return ""

    return normalize_text(
        match.group("target_title")
    )


def _reference_type(target_section: str) -> str:
    """Classify a valid document relationship.

    Args:
        target_section (str): Destination section.

    Returns:
        str: Relationship type.
    """
    if "formulary" in target_section.lower():
        return "formulary_lookup"

    return "document_lookup"


def _target_section(text: str) -> str:
    """Extract a referenced section or appendix.

    Args:
        text (str): Reference sentence.

    Returns:
        str: Target section or an empty string.
    """
    match = re.search(
        r"""
        \b
        (?:
            Formulary\s+
            |
            Scanned\s+
        )?
        (?:
            Appendix
            |
            Section
            |
            Table
            |
            Figure
        )
        \s+[A-Z0-9.-]+
        \b
        """,
        text,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return (
        normalize_text(match.group(0))
        if match
        else ""
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load an optional JSON object.

    Args:
        path (Path): JSON path.

    Returns:
        dict[str, Any]: Parsed object or an empty dictionary.
    """
    if not path.is_file():
        return {}

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _find_document_title(document: Any) -> str:
    """Find the best title in a Docling document.

    Args:
        document (Any): Loaded DoclingDocument.

    Returns:
        str: Document title.
    """
    fallback = normalize_text(
        getattr(document, "name", "")
    )

    for item in getattr(document, "texts", []) or []:
        if label_value(item) == "title":
            text = normalize_text(
                getattr(item, "text", "")
            )

            if text:
                return text

    for item in getattr(document, "texts", []) or []:
        if label_value(item) == "section_header":
            text = normalize_text(
                getattr(item, "text", "")
            )

            if text:
                return text

    return fallback or "Untitled clinical document"


def _extract_metadata_tables(
    document: Any,
) -> dict[str, str]:
    """Extract simple key-value metadata from tables.

    Args:
        document (Any): Loaded DoclingDocument.

    Returns:
        dict[str, str]: Metadata fields.
    """
    metadata: dict[str, str] = {}

    for table in getattr(document, "tables", []) or []:
        data = getattr(table, "data", None)

        if data is None:
            continue

        row_values: dict[int, dict[int, str]] = {}

        for cell in (
            getattr(data, "table_cells", []) or []
        ):
            row_index = int(
                getattr(
                    cell,
                    "start_row_offset_idx",
                    0,
                )
                or 0
            )

            column_index = int(
                getattr(
                    cell,
                    "start_col_offset_idx",
                    0,
                )
                or 0
            )

            text = normalize_text(
                getattr(cell, "text", "")
            )

            if text:
                row_values.setdefault(
                    row_index,
                    {},
                )[column_index] = text

        for columns in row_values.values():
            ordered = [
                value
                for _, value in sorted(
                    columns.items()
                )
            ]

            if len(ordered) != 2:
                continue

            key = normalize_text(
                ordered[0]
            ).lower()

            value = normalize_text(ordered[1])

            if key and value and len(key) <= 80:
                metadata.setdefault(key, value)

    return metadata


def _find_registry_code(
    title: str,
    metadata: dict[str, str],
    document: Any,
) -> str | None:
    """Find the source document's registry code.

    Args:
        title (str): Document title.
        metadata (dict[str, str]): Extracted metadata.
        document (Any): Loaded DoclingDocument.

    Returns:
        str | None: Registry code.
    """
    for key, value in metadata.items():
        if "registry" not in key:
            continue

        match = REGISTRY_PATTERN.search(value)

        if match:
            return match.group(1).upper()

    title_match = REGISTRY_PATTERN.search(title)

    if title_match:
        return title_match.group(1).upper()

    for item in getattr(document, "texts", []) or []:
        text = normalize_text(
            getattr(item, "text", "")
        )

        match = REGISTRY_PATTERN.search(text)

        if match:
            return match.group(1).upper()

    return None


def _population_scope(title: str) -> str:
    """Infer a broad patient population from the title.

    Args:
        title (str): Document title.

    Returns:
        str: children, adults, or general.
    """
    lowered = title.lower()

    if any(
        term in lowered
        for term in (
            "children",
            "childhood",
            "paediatric",
            "pediatric",
        )
    ):
        return "children"

    if "adult" in lowered:
        return "adults"

    return "general"


def _quality_label(
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
) -> str:
    """Read an ingestion quality label.

    Args:
        manifest (dict[str, Any]): Ingestion manifest.
        quality_report (dict[str, Any]): Quality report.

    Returns:
        str: Quality label.
    """
    for source in (quality_report, manifest):
        for key in (
            "quality",
            "quality_label",
            "overall_quality",
            "grade",
        ):
            value = source.get(key)

            if value:
                return str(value)

    if manifest.get("status") == "pass":
        return "pass"

    return "unknown"