"""Provide shared utility functions for evidence extraction."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def normalize_text(value: str | None) -> str:
    """Normalize whitespace without changing textual meaning.

    Args:
        value (str | None): Text to normalize.

    Returns:
        str: Cleaned text.
    """
    if not value:
        return ""

    return re.sub(r"\s+", " ", value).strip()


def stable_id(prefix: str, *parts: object) -> str:
    """Create a repeatable identifier from source values.

    Args:
        prefix (str): Human-readable identifier prefix.
        *parts (object): Values that uniquely identify the record.

    Returns:
        str: Stable identifier containing a short SHA-256 digest.
    """
    raw_value = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def label_value(item: Any) -> str:
    """Read an item's Docling label safely.

    Args:
        item (Any): Docling document item.

    Returns:
        str: Lowercase label name or an empty string.
    """
    label = getattr(item, "label", None)
    value = getattr(label, "value", label)
    return str(value or "").lower()


def source_reference(item: Any) -> str:
    """Read a Docling item's self-reference.

    Args:
        item (Any): Docling document item.

    Returns:
        str: Docling JSON reference or an empty string.
    """
    return str(getattr(item, "self_ref", "") or "")


def item_pages(item: Any) -> list[int]:
    """Collect unique PDF page numbers from Docling provenance.

    Args:
        item (Any): Docling document item.

    Returns:
        list[int]: Sorted one-based page numbers.
    """
    pages: set[int] = set()

    for provenance in getattr(item, "prov", []) or []:
        page_number = getattr(provenance, "page_no", None)

        if page_number is not None:
            pages.add(int(page_number))

    return sorted(pages)


def first_existing_path(paths: list[Path]) -> Path | None:
    """Return the first path that exists.

    Args:
        paths (list[Path]): Candidate paths.

    Returns:
        Path | None: First existing path or None.
    """
    for path in paths:
        if path.is_file():
            return path

    return None