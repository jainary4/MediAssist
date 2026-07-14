"""Build the SQLite FTS5 keyword-search index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from retrieval.models import EvidenceChunk


class KeywordStore:
    """Index evidence chunks for exact words and BM25 search."""

    def __init__(self, database_path: Path) -> None:
        """Initialize the keyword store.

        Args:
            database_path (Path): Existing SQLite database.
        """
        self.database_path = database_path

    def build(self, chunks: list[EvidenceChunk]) -> None:
        """Create and populate the FTS5 search table.

        Args:
            chunks (list[EvidenceChunk]): Searchable evidence chunks.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "DROP TABLE IF EXISTS chunks_fts"
            )

            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    content_type UNINDEXED,
                    title,
                    section,
                    search_text,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )

            connection.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id,
                    document_id,
                    content_type,
                    title,
                    section,
                    search_text
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.content_type,
                        chunk.title,
                        chunk.section,
                        chunk.search_text,
                    )
                    for chunk in chunks
                ],
            )

            connection.execute(
                "INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')"
            )

            connection.commit()