"""Store canonical evidence and exact table data in SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from retrieval.models import EvidenceBundle


class StructuredStore:
    """Write canonical evidence records into a SQLite database."""

    def __init__(self, database_path: Path) -> None:
        """Initialize the structured store.

        Args:
            database_path (Path): SQLite output path.
        """
        self.database_path = database_path

    def write(self, bundle: EvidenceBundle) -> None:
        """Create the schema and insert every evidence record.

        Args:
            bundle (EvidenceBundle): Complete extracted evidence.
        """
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.database_path.exists():
            self.database_path.unlink()

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")

            self._create_schema(connection)
            self._insert_documents(connection, bundle.documents)
            self._insert_chunks(connection, bundle)
            self._insert_tables(connection, bundle.tables)
            self._insert_table_rows(connection, bundle.table_rows)
            self._insert_figures(connection, bundle.figures)
            self._insert_references(connection, bundle.references)
            connection.commit()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create all canonical knowledge-base tables.

        Args:
            connection (sqlite3.Connection): Open database connection.
        """
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                registry_code TEXT,
                population_scope TEXT,
                page_count INTEGER,
                selected_pipeline TEXT,
                ingestion_quality TEXT,
                source_json_path TEXT NOT NULL,
                source_markdown_path TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT,
                search_text TEXT NOT NULL,
                display_text TEXT NOT NULL,
                page_numbers_json TEXT NOT NULL,
                source_refs_json TEXT NOT NULL,
                parent_id TEXT,
                asset_path TEXT,
                ingestion_quality TEXT,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(document_id)
            );

            CREATE TABLE table_records (
                table_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT,
                page_numbers_json TEXT NOT NULL,
                source_ref TEXT,
                headers_json TEXT NOT NULL,
                row_count INTEGER,
                column_count INTEGER,
                markdown TEXT,
                asset_path TEXT,
                requires_visual_check INTEGER NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(document_id)
            );

            CREATE TABLE table_rows (
                row_id TEXT PRIMARY KEY,
                table_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                page_numbers_json TEXT NOT NULL,
                values_json TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                requires_visual_check INTEGER NOT NULL,
                FOREIGN KEY(table_id)
                    REFERENCES table_records(table_id),
                FOREIGN KEY(document_id)
                    REFERENCES documents(document_id)
            );

            CREATE TABLE figures (
                figure_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                figure_number INTEGER,
                caption TEXT,
                section TEXT,
                nearby_text TEXT,
                ocr_text TEXT,
                page_numbers_json TEXT NOT NULL,
                source_ref TEXT,
                asset_path TEXT,
                requires_visual_check INTEGER NOT NULL,
                FOREIGN KEY(document_id)
                    REFERENCES documents(document_id)
            );

            CREATE TABLE document_references (
                reference_id TEXT PRIMARY KEY,
                source_document_id TEXT NOT NULL,
                source_ref TEXT,
                source_page_numbers_json TEXT NOT NULL,
                reference_text TEXT NOT NULL,
                target_registry_code TEXT,
                target_document_id TEXT,
                target_section TEXT,
                FOREIGN KEY(source_document_id)
                    REFERENCES documents(document_id)
            );

            CREATE INDEX idx_chunks_document
                ON chunks(document_id);

            CREATE INDEX idx_chunks_type
                ON chunks(content_type);

            CREATE INDEX idx_rows_table
                ON table_rows(table_id);

            CREATE INDEX idx_documents_registry
                ON documents(registry_code);

            CREATE INDEX idx_references_target
                ON document_references(target_document_id);
            """
        )

    @staticmethod
    def _insert_documents(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> None:
        """Insert document metadata.

        Args:
            connection (sqlite3.Connection): Open database connection.
            records (list[dict[str, Any]]): Document records.
        """
        connection.executemany(
            """
            INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record["document_id"],
                    record["title"],
                    record.get("registry_code"),
                    record.get("population_scope"),
                    record.get("page_count", 0),
                    record.get("selected_pipeline"),
                    record.get("ingestion_quality"),
                    record["source_json_path"],
                    record.get("source_markdown_path"),
                    json.dumps(record.get("metadata", {})),
                )
                for record in records
            ],
        )

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection,
        bundle: EvidenceBundle,
    ) -> None:
        """Insert searchable chunks.

        Args:
            connection (sqlite3.Connection): Open database connection.
            bundle (EvidenceBundle): Evidence bundle.
        """
        connection.executemany(
            """
            INSERT INTO chunks VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content_type,
                    chunk.title,
                    chunk.section,
                    chunk.search_text,
                    chunk.display_text,
                    json.dumps(chunk.page_numbers),
                    json.dumps(chunk.source_refs),
                    chunk.parent_id,
                    chunk.asset_path,
                    chunk.ingestion_quality,
                    json.dumps(chunk.metadata),
                )
                for chunk in bundle.chunks
            ],
        )

    @staticmethod
    def _insert_tables(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> None:
        """Insert table parent records.

        Args:
            connection (sqlite3.Connection): Open database connection.
            records (list[dict[str, Any]]): Table records.
        """
        connection.executemany(
            """
            INSERT INTO table_records VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    record["table_id"],
                    record["document_id"],
                    record["title"],
                    json.dumps(record["page_numbers"]),
                    record.get("source_ref"),
                    json.dumps(record["headers"]),
                    record["row_count"],
                    record["column_count"],
                    record["markdown"],
                    record.get("asset_path"),
                    int(record["requires_visual_check"]),
                )
                for record in records
            ],
        )

    @staticmethod
    def _insert_table_rows(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> None:
        """Insert exact structured table rows.

        Args:
            connection (sqlite3.Connection): Open database connection.
            records (list[dict[str, Any]]): Table-row records.
        """
        connection.executemany(
            """
            INSERT INTO table_rows VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    record["row_id"],
                    record["table_id"],
                    record["document_id"],
                    record["row_number"],
                    json.dumps(record["page_numbers"]),
                    json.dumps(record["values"]),
                    record["searchable_text"],
                    int(record["requires_visual_check"]),
                )
                for record in records
            ],
        )

    @staticmethod
    def _insert_figures(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> None:
        """Insert figure records.

        Args:
            connection (sqlite3.Connection): Open database connection.
            records (list[dict[str, Any]]): Figure records.
        """
        connection.executemany(
            """
            INSERT INTO figures VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    record["figure_id"],
                    record["document_id"],
                    record["figure_number"],
                    record["caption"],
                    record["section"],
                    record["nearby_text"],
                    record["ocr_text"],
                    json.dumps(record["page_numbers"]),
                    record["source_ref"],
                    record.get("asset_path"),
                    int(record["requires_visual_check"]),
                )
                for record in records
            ],
        )

    @staticmethod
    def _insert_references(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
    ) -> None:
        """Insert explicit document-reference links.

        Args:
            connection (sqlite3.Connection): Open database connection.
            records (list[dict[str, Any]]): Reference records.
        """
        connection.executemany(
            """
            INSERT INTO document_references VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    record["reference_id"],
                    record["source_document_id"],
                    record["source_ref"],
                    json.dumps(record["source_page_numbers"]),
                    record["reference_text"],
                    record["target_registry_code"],
                    record.get("target_document_id"),
                    record.get("target_section"),
                )
                for record in records
            ],
        )