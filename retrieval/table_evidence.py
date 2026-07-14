"""Create searchable and structured evidence from Docling tables."""

from __future__ import annotations

from typing import Any, Callable

from retrieval.config import EvidenceBuilderConfig
from retrieval.models import EvidenceChunk
from retrieval.utils import (
    item_pages,
    normalize_text,
    source_reference,
    stable_id,
)


class TableEvidenceBuilder:
    """Convert Docling tables into parent, window, and row records."""

    def __init__(
        self,
        config: EvidenceBuilderConfig,
        token_counter: Callable[[str], int],
    ) -> None:
        """Initialize table chunking settings.

        Args:
            config (EvidenceBuilderConfig): Builder configuration.
            token_counter (Callable[[str], int]): Token-count function.
        """
        self.config = config
        self.token_counter = token_counter

    def build(
        self,
        document: Any,
        document_record: dict[str, Any],
        final_directory: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[EvidenceChunk],
    ]:
        """Build all table evidence for one document.

        Args:
            document (Any): Loaded DoclingDocument.
            document_record (dict[str, Any]): Source metadata.
            final_directory (Any): Document's final ingestion directory.

        Returns:
            tuple: Table records, structured rows, and search chunks.
        """
        table_records: list[dict[str, Any]] = []
        structured_rows: list[dict[str, Any]] = []
        chunks: list[EvidenceChunk] = []

        for table_number, table in enumerate(
            getattr(document, "tables", []) or [],
            start=1,
        ):
            matrix, header_row_indexes = self._table_matrix(table)

            if not matrix:
                continue

            title = self._table_title(
                table=table,
                document=document,
                matrix=matrix,
                table_number=table_number,
            )

            headers, data_start = self._select_headers(
                matrix=matrix,
                header_row_indexes=header_row_indexes,
            )

            data_rows = matrix[data_start:]
            pages = item_pages(table)
            source_ref = source_reference(table)

            table_id = stable_id(
                "table",
                document_record["document_id"],
                source_ref or table_number,
            )

            image_path = (
                final_directory
                / "tables"
                / f"table_{table_number}"
                / "table.png"
            )

            asset_path = (
                str(image_path) if image_path.is_file() else None
            )

            row_records = self._build_structured_rows(
                table_id=table_id,
                document_id=document_record["document_id"],
                page_numbers=pages,
                headers=headers,
                rows=data_rows,
            )

            requires_visual_check = any(
                record["requires_visual_check"]
                for record in row_records
            )

            table_markdown = self._to_markdown(
                headers=headers,
                rows=data_rows,
            )

            table_records.append({
                "table_id": table_id,
                "document_id": document_record["document_id"],
                "title": title,
                "page_numbers": pages,
                "source_ref": source_ref,
                "headers": headers,
                "row_count": len(data_rows),
                "column_count": len(headers),
                "markdown": table_markdown,
                "asset_path": asset_path,
                "requires_visual_check": requires_visual_check,
            })

            structured_rows.extend(row_records)

            chunks.extend(
                self._build_table_chunks(
                    table_id=table_id,
                    document_record=document_record,
                    title=title,
                    headers=headers,
                    rows=data_rows,
                    pages=pages,
                    source_ref=source_ref,
                    asset_path=asset_path,
                    requires_visual_check=requires_visual_check,
                )
            )

        return table_records, structured_rows, chunks

    @staticmethod
    def _table_matrix(
        table: Any,
    ) -> tuple[list[list[str]], set[int]]:
        """Reconstruct a two-dimensional matrix from Docling cells.

        Args:
            table (Any): Docling table item.

        Returns:
            tuple: Cell matrix and detected column-header row indexes.
        """
        data = getattr(table, "data", None)

        if data is None:
            return [], set()

        row_count = int(getattr(data, "num_rows", 0) or 0)
        column_count = int(getattr(data, "num_cols", 0) or 0)

        if row_count <= 0 or column_count <= 0:
            return [], set()

        matrix = [
            ["" for _ in range(column_count)]
            for _ in range(row_count)
        ]

        header_rows: set[int] = set()

        for cell in getattr(data, "table_cells", []) or []:
            start_row = int(
                getattr(cell, "start_row_offset_idx", 0) or 0
            )
            end_row = int(
                getattr(
                    cell,
                    "end_row_offset_idx",
                    start_row + 1,
                )
                or start_row + 1
            )

            start_column = int(
                getattr(cell, "start_col_offset_idx", 0) or 0
            )
            end_column = int(
                getattr(
                    cell,
                    "end_col_offset_idx",
                    start_column + 1,
                )
                or start_column + 1
            )

            text = normalize_text(getattr(cell, "text", ""))

            for row_index in range(start_row, min(end_row, row_count)):
                for column_index in range(
                    start_column,
                    min(end_column, column_count),
                ):
                    if not matrix[row_index][column_index]:
                        matrix[row_index][column_index] = text

            if bool(getattr(cell, "column_header", False)):
                header_rows.update(
                    range(start_row, min(end_row, row_count))
                )

        return matrix, header_rows

    @staticmethod
    def _table_title(
        table: Any,
        document: Any,
        matrix: list[list[str]],
        table_number: int,
    ) -> str:
        """Choose a useful table title.

        Args:
            table (Any): Docling table item.
            document (Any): Parent DoclingDocument.
            matrix (list[list[str]]): Reconstructed table.
            table_number (int): One-based table number.

        Returns:
            str: Table title.
        """
        caption_method = getattr(table, "caption_text", None)

        if callable(caption_method):
            caption = normalize_text(caption_method(document))

            if caption:
                return caption

        if matrix:
            first_row_values = [
                value for value in matrix[0] if value
            ]

            if len(set(first_row_values)) == 1:
                return first_row_values[0]

        return f"Table {table_number}"

    @staticmethod
    def _select_headers(
        matrix: list[list[str]],
        header_row_indexes: set[int],
    ) -> tuple[list[str], int]:
        """Choose table headers and the first data row.

        Args:
            matrix (list[list[str]]): Reconstructed table matrix.
            header_row_indexes (set[int]): Docling header row indexes.

        Returns:
            tuple: Unique header names and first data-row index.
        """
        if header_row_indexes:
            header_index = max(header_row_indexes)
        elif (
            len(matrix) > 1
            and len({value for value in matrix[0] if value}) == 1
        ):
            header_index = 1
        else:
            header_index = 0

        raw_headers = matrix[header_index]
        headers: list[str] = []
        used_headers: dict[str, int] = {}

        for column_index, raw_header in enumerate(raw_headers, start=1):
            base_header = normalize_text(raw_header)

            if not base_header:
                base_header = f"column_{column_index}"

            occurrence = used_headers.get(base_header, 0) + 1
            used_headers[base_header] = occurrence

            if occurrence > 1:
                headers.append(f"{base_header}_{occurrence}")
            else:
                headers.append(base_header)

        return headers, header_index + 1

    @staticmethod
    def _build_structured_rows(
        table_id: str,
        document_id: str,
        page_numbers: list[int],
        headers: list[str],
        rows: list[list[str]],
    ) -> list[dict[str, Any]]:
        """Turn each table row into a structured dictionary.

        Args:
            table_id (str): Parent table identifier.
            document_id (str): Source document identifier.
            page_numbers (list[int]): Source pages.
            headers (list[str]): Column names.
            rows (list[list[str]]): Table data rows.

        Returns:
            list[dict[str, Any]]: Structured table-row records.
        """
        records: list[dict[str, Any]] = []

        for row_number, row in enumerate(rows, start=1):
            padded_row = (
                row + [""] * max(0, len(headers) - len(row))
            )[:len(headers)]

            values = {
                header: normalize_text(value)
                for header, value in zip(headers, padded_row)
            }

            nonempty_values = [
                value for value in values.values() if value
            ]

            if not nonempty_values:
                continue

            requires_visual_check = (
                not padded_row[0]
                and any(padded_row[1:])
            )

            records.append({
                "row_id": stable_id(
                    "row",
                    table_id,
                    row_number,
                    "|".join(padded_row),
                ),
                "table_id": table_id,
                "document_id": document_id,
                "row_number": row_number,
                "page_numbers": page_numbers,
                "values": values,
                "searchable_text": " | ".join(
                    f"{header}: {value}"
                    for header, value in values.items()
                    if value
                ),
                "requires_visual_check": requires_visual_check,
            })

        return records

    def _build_table_chunks(
        self,
        table_id: str,
        document_record: dict[str, Any],
        title: str,
        headers: list[str],
        rows: list[list[str]],
        pages: list[int],
        source_ref: str,
        asset_path: str | None,
        requires_visual_check: bool,
    ) -> list[EvidenceChunk]:
        """Create small-table or windowed-table search chunks.

        Args:
            table_id (str): Parent table identifier.
            document_record (dict[str, Any]): Source metadata.
            title (str): Table title.
            headers (list[str]): Column names.
            rows (list[list[str]]): Data rows.
            pages (list[int]): Source pages.
            source_ref (str): Docling JSON reference.
            asset_path (str | None): Extracted table image.
            requires_visual_check (bool): Whether extraction looks incomplete.

        Returns:
            list[EvidenceChunk]: Searchable table chunks.
        """
        full_markdown = self._to_markdown(headers, rows)
        full_text = self._table_search_text(
            document_title=document_record["title"],
            table_title=title,
            markdown=full_markdown,
        )

        is_small_table = (
            len(rows) <= self.config.small_table_max_rows
            and self.token_counter(full_text)
            <= self.config.small_table_max_tokens
        )

        common_metadata = {
            "table_id": table_id,
            "headers": headers,
            "row_count": len(rows),
            "requires_visual_check": requires_visual_check,
        }

        if is_small_table:
            return [
                EvidenceChunk(
                    chunk_id=stable_id("table-chunk", table_id, "full"),
                    document_id=document_record["document_id"],
                    content_type="table",
                    title=document_record["title"],
                    section=title,
                    search_text=full_text,
                    display_text=full_markdown,
                    page_numbers=pages,
                    source_refs=[source_ref] if source_ref else [],
                    parent_id=table_id,
                    asset_path=asset_path,
                    ingestion_quality=document_record[
                        "ingestion_quality"
                    ],
                    metadata={
                        **common_metadata,
                        "window_start": 1,
                        "window_end": len(rows),
                    },
                )
            ]

        chunks: list[EvidenceChunk] = []

        parent_summary = (
            f"Document: {document_record['title']}\n"
            f"Table: {title}\n"
            f"Columns: {', '.join(headers)}\n"
            f"Number of data rows: {len(rows)}"
        )

        chunks.append(
            EvidenceChunk(
                chunk_id=stable_id(
                    "table-parent-chunk",
                    table_id,
                ),
                document_id=document_record["document_id"],
                content_type="table_parent",
                title=document_record["title"],
                section=title,
                search_text=parent_summary,
                display_text=parent_summary,
                page_numbers=pages,
                source_refs=[source_ref] if source_ref else [],
                parent_id=table_id,
                asset_path=asset_path,
                ingestion_quality=document_record[
                    "ingestion_quality"
                ],
                metadata=common_metadata,
            )
        )

        window_size = self.config.table_window_rows
        step = max(
            1,
            window_size - self.config.table_window_overlap,
        )

        for start in range(0, len(rows), step):
            window_rows = rows[start:start + window_size]

            if not window_rows:
                continue

            end = start + len(window_rows)
            markdown = self._to_markdown(headers, window_rows)

            chunks.append(
                EvidenceChunk(
                    chunk_id=stable_id(
                        "table-window",
                        table_id,
                        start + 1,
                        end,
                    ),
                    document_id=document_record["document_id"],
                    content_type="table_window",
                    title=document_record["title"],
                    section=title,
                    search_text=self._table_search_text(
                        document_title=document_record["title"],
                        table_title=title,
                        markdown=markdown,
                    ),
                    display_text=markdown,
                    page_numbers=pages,
                    source_refs=(
                        [source_ref] if source_ref else []
                    ),
                    parent_id=table_id,
                    asset_path=asset_path,
                    ingestion_quality=document_record[
                        "ingestion_quality"
                    ],
                    metadata={
                        **common_metadata,
                        "window_start": start + 1,
                        "window_end": end,
                    },
                )
            )

            if end >= len(rows):
                break

        return chunks

    @staticmethod
    def _table_search_text(
        document_title: str,
        table_title: str,
        markdown: str,
    ) -> str:
        """Attach document and table context to table data.

        Args:
            document_title (str): Source document title.
            table_title (str): Table title.
            markdown (str): Markdown-formatted rows.

        Returns:
            str: Search-ready table text.
        """
        return (
            f"Document: {document_title}\n"
            f"Table: {table_title}\n"
            f"{markdown}"
        )

    @staticmethod
    def _to_markdown(
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """Serialize table rows into Markdown.

        Args:
            headers (list[str]): Column names.
            rows (list[list[str]]): Table data.

        Returns:
            str: Markdown table.
        """
        def escape(value: str) -> str:
            """Escape Markdown table delimiters.

            Args:
                value (str): Table cell value.

            Returns:
                str: Escaped cell value.
            """
            return normalize_text(value).replace("|", r"\|")

        header_line = "| " + " | ".join(
            escape(header) for header in headers
        ) + " |"

        separator_line = "| " + " | ".join(
            "---" for _ in headers
        ) + " |"

        row_lines = []

        for row in rows:
            padded = (
                row + [""] * max(0, len(headers) - len(row))
            )[:len(headers)]

            row_lines.append(
                "| " + " | ".join(
                    escape(value) for value in padded
                ) + " |"
            )

        return "\n".join(
            [header_line, separator_line, *row_lines]
        )