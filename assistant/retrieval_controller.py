from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
from contextvars import ContextVar

from assistant.models import (
    EvidenceItem,
    EvidencePack,
    RetrievalTrace,
    RoutePlan,
)

from assistant.query_router import (
    analyse_query as build_route_plan,
)

"""Provide deterministic hybrid retrieval over SQLite and FAISS."""
"""
This is the deterministic retrieval controller.
It is normal Python code. It:
loads the same MiniLM model used to build FAISS;
always runs semantic and keyword search;
adds table, figure, metadata, reference or corpus queries based on deterministic rules;
never allows arbitrary SQL;
records every selected evidence item;
follows explicit cross-document links;
creates corpus-wide results for questions such as “every Tier 3 condition.”
"""

RRF_CONSTANT = 60

STOP_WORDS = {
    "about",
    "according",
    "another",
    "answer",
    "are",
    "does",
    "every",
    "from",
    "given",
    "have",
    "how",
    "into",
    "listed",
    "monograph",
    "question",
    "that",
    "their",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "whose",
    "with",
}


class RetrievalController:
    """Search the fixed clinical knowledge base using deterministic code."""

    def __init__(
        self,
        database_path: Path,
        index_path: Path,
        mapping_path: Path,
        embedding_model: str,
    ) -> None:
        """Load the retrieval database, FAISS index and query embedder.

        Args:
            database_path (Path): SQLite knowledge-base path.
            index_path (Path): FAISS index path.
            mapping_path (Path): Vector-to-chunk mapping path.
            embedding_model (str): Sentence Transformer model identifier.

        Raises:
            FileNotFoundError: If a required retrieval artifact is missing.
            ValueError: If the vector mapping is inconsistent with FAISS.
        """
        for path in (
            database_path,
            index_path,
            mapping_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required retrieval artifact not found: {path}"
                )

        self.database_path = database_path

        self.index = faiss.read_index(
            str(index_path)
        )

        self.embedder = SentenceTransformer(
            embedding_model
        )

        mapping = json.loads(
            mapping_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            mapping["embedding_model"]
            != embedding_model
        ):
            raise ValueError(
                "The query embedder does not match the model used "
                "to build the FAISS index."
            )

        if (
            int(mapping["vector_count"])
            != int(self.index.ntotal)
        ):
            raise ValueError(
                "FAISS vector count does not match "
                "vector_mapping.json."
            )

        self.vector_mapping = {
            int(record["vector_id"]): record
            for record in mapping["vectors"]
        }

        self.documents = (
            self._load_documents()
        )

        self._active_evidence_context: ContextVar[
            dict[str, EvidenceItem] | None
        ] = ContextVar(
            "clinical_qa_active_evidence",
            default=None,
        )

        self._embedding_lock = (
            threading.Lock()
        )

    @property
    def active_evidence(
        self,
    ) -> dict[str, EvidenceItem]:
        """Return evidence belonging to the current request context.

        Returns:
            dict[str, EvidenceItem]: Request-local evidence registry.
        """
        registry = (
            self._active_evidence_context.get()
        )

        if registry is None:
            registry = {}

            self._active_evidence_context.set(
                registry
            )

        return registry

    @active_evidence.setter
    def active_evidence(
        self,
        value: dict[str, EvidenceItem],
    ) -> None:
        """Replace evidence for the current request context.

        Args:
            value (dict[str, EvidenceItem]): New evidence registry.
        """
        self._active_evidence_context.set(
            value
        )

    def _connect(self) -> sqlite3.Connection:
        """Open the knowledge base in immutable read-only mode.

        Returns:
            sqlite3.Connection: Read-only SQLite connection.
        """
        uri = (
            f"file:{self.database_path}"
            "?mode=ro&immutable=1"
        )

        connection = sqlite3.connect(
            uri,
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _load_documents(self) -> list[dict[str, Any]]:
        """Load document identifiers and titles for title matching.

        Returns:
            list[dict[str, Any]]: Document metadata rows.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    title,
                    registry_code,
                    population_scope,
                    ingestion_quality
                FROM documents
                ORDER BY title
                """
            ).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize text for conservative title matching.

        Args:
            value (str): Text to normalize.

        Returns:
            str: Lowercase alphanumeric text.
        """
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            value.casefold(),
        ).strip()

    def resolve_document_ids(
        self,
        document_title: str,
    ) -> list[str]:
        """Resolve a human-readable title to matching document IDs.

        Args:
            document_title (str): Full or partial document title.

        Returns:
            list[str]: Matching document IDs.
        """
        target = self._normalize(document_title)

        if not target:
            return []

        exact = [
            record["document_id"]
            for record in self.documents
            if self._normalize(record["title"]) == target
        ]

        if exact:
            return exact

        return [
            record["document_id"]
            for record in self.documents
            if target in self._normalize(record["title"])
            or self._normalize(record["title"]) in target
        ]

    def analyse_query(self, question: str) -> RoutePlan:
        """Build a deterministic retrieval plan from question wording.

        Args:
            question (str): User question.

        Returns:
            RoutePlan: Intent-specific retrieval plan.
        """
        return build_route_plan(
            question=question,
            documents=self.documents,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 12,
    ) -> EvidencePack:
        """Run all retrieval routes required for one question.

        Args:
            question (str): User's question.
            top_k (int): Maximum selected evidence records.

        Returns:
            EvidencePack: Selected evidence and diagnostic trace.
        """
        self.active_evidence = {}

        route = self.analyse_query(question)

        trace = RetrievalTrace(route=route)

        semantic_items = self.search_semantic(
            query=question,
            top_k=12,
            document_ids=route.named_document_ids or None,
        )
        trace.semantic_result_count = len(semantic_items)

        keyword_items = self.search_keywords(
            query=question,
            top_k=12,
            document_ids=route.named_document_ids or None,
        )
        trace.keyword_result_count = len(keyword_items)

        if route.requires_metadata:
            metadata_items = self.lookup_metadata(
                query=question,
                document_ids=route.named_document_ids or None,
            )
            trace.structured_result_count += len(metadata_items)

        if route.requires_structured_table:
            table_items = self.lookup_table_rows(
                query=question,
                document_ids=route.named_document_ids or None,
                limit=20,
            )
            trace.structured_result_count += len(table_items)
        
        if "footnote" in route.retrieval_channels:
            footnote_items = self.lookup_footnotes(
                document_ids=(
                    route.named_document_ids or None
                ),
            )
            trace.footnote_result_count = len(
                footnote_items
            )

        if route.requires_figure:
            figure_items = self.lookup_figures(
                query=question,
                document_ids=route.named_document_ids or None,
            )
            trace.figure_result_count = len(figure_items)

        if route.requires_cross_document:
            reference_items = self.follow_references(
                query=question,
                source_document_ids=(
                    route.named_document_ids or None
                ),
            )
            trace.reference_result_count = len(reference_items)

            self._retrieve_reference_targets(
                question=question,
                references=reference_items,
                trace=trace,
            )

        if route.requires_corpus_aggregation:
            aggregation_items = self.aggregate_corpus(
                question
            )
            trace.aggregation_result_count = len(
                aggregation_items
            )

        selected = sorted(
            self.active_evidence.values(),
            key=lambda item: (
                item.fusion_score,
                item.extraction_quality,
            ),
            reverse=True,
        )[:top_k]

        trace.selected_evidence_ids = [
            item.evidence_id
            for item in selected
        ]

        return EvidencePack(
            question=question,
            route=route,
            evidence=selected,
            trace=trace,
        )

    

    def search_semantic(self,query: str,top_k: int = 10,document_ids: list[str] | None = None,) -> list[EvidenceItem]:
        """Search FAISS using the same embedder used during indexing.

        The embedding model and FAISS search are protected by a lock so
        concurrent Modal requests cannot use these shared objects at the
        same time.

        Args:
            query (str): Semantic search query.
            top_k (int): Maximum number of results to return.
            document_ids (list[str] | None): Optional document filter.

        Returns:
            list[EvidenceItem]: Semantic search evidence.
        """
        with self._embedding_lock:
            if hasattr(
                self.embedder,
                "encode_query",
            ):
                vector = self.embedder.encode_query(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            else:
                vector = self.embedder.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            vector = np.asarray(
                vector,
                dtype="float32",
            )

            pool_size = min(
                max(top_k * 10, 50),
                int(self.index.ntotal),
            )

            scores, vector_ids = self.index.search(
                vector,
                pool_size,
            )

        candidates: list[tuple[str, float]] = []

        allowed_documents = (
            set(document_ids)
            if document_ids
            else None
        )

        for score, vector_id in zip(
            scores[0],
            vector_ids[0],
        ):
            if int(vector_id) < 0:
                continue

            mapping = self.vector_mapping.get(
                int(vector_id)
            )

            if mapping is None:
                continue

            if (
                allowed_documents is not None
                and mapping["document_id"]
                not in allowed_documents
            ):
                continue

            candidates.append(
                (
                    mapping["chunk_id"],
                    float(score),
                )
            )

            if len(candidates) >= top_k:
                break

        rows = self._fetch_chunks(
            [chunk_id for chunk_id, _ in candidates]
        )

        row_by_id = {
            row["chunk_id"]: row
            for row in rows
        }

        results: list[EvidenceItem] = []

        for rank, (chunk_id, score) in enumerate(
            candidates,
            start=1,
        ):
            row = row_by_id.get(chunk_id)

            if row is None:
                continue

            item = self._chunk_to_evidence(row)

            self._register(
                item=item,
                channel="semantic",
                rank=rank,
                raw_score=score,
            )

            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results
        

    def search_keywords(
        self,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Search the SQLite FTS5 keyword index.

        Args:
            query (str): Keyword search query.
            top_k (int): Maximum results.
            document_ids (list[str] | None): Optional document filter.

        Returns:
            list[EvidenceItem]: Keyword search evidence.
        """
        fts_query = self._build_fts_query(query)

        if not fts_query:
            return []

        parameters: list[Any] = [fts_query]
        document_filter = ""

        if document_ids:
            placeholders = ",".join(
                "?" for _ in document_ids
            )
            document_filter = (
                f"AND c.document_id IN ({placeholders})"
            )
            parameters.extend(document_ids)

        parameters.append(top_k)

        sql = f"""
            SELECT
                c.*,
                bm25(chunks_fts) AS keyword_score
            FROM chunks_fts
            JOIN chunks AS c
                ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            {document_filter}
            ORDER BY keyword_score
            LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        results: list[EvidenceItem] = []

        for rank, row in enumerate(rows, start=1):
            item = self._chunk_to_evidence(row)

            self._register(
                item=item,
                channel="keyword",
                rank=rank,
                raw_score=float(row["keyword_score"]),
            )
            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    def lookup_metadata(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Look up exact document metadata without generative SQL.

        Args:
            query (str): User query.
            document_ids (list[str] | None): Optional document filter.

        Returns:
            list[EvidenceItem]: Matching metadata evidence.
        """
        registry_match = re.search(
            r"\bCDR-\d+\b",
            query,
            flags=re.IGNORECASE,
        )

        clauses: list[str] = []
        parameters: list[Any] = []

        if registry_match:
            clauses.append(
                "UPPER(registry_code) = ?"
            )
            parameters.append(
                registry_match.group(0).upper()
            )

        if document_ids:
            placeholders = ",".join(
                "?" for _ in document_ids
            )
            clauses.append(
                f"document_id IN ({placeholders})"
            )
            parameters.extend(document_ids)

        if not clauses:
            return []

        sql = f"""
            SELECT *
            FROM documents
            WHERE {" AND ".join(clauses)}
            ORDER BY title
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        results: list[EvidenceItem] = []

        for rank, row in enumerate(rows, start=1):
            metadata = json.loads(
                row["metadata_json"] or "{}"
            )

            text = (
                f"Document title: {row['title']}\n"
                f"Registry code: {row['registry_code']}\n"
                f"Population scope: {row['population_scope']}\n"
                f"Additional metadata: "
                f"{json.dumps(metadata, ensure_ascii=False)}"
            )

            item = EvidenceItem(
                evidence_id=(
                    f"document-metadata:{row['document_id']}"
                ),
                document_id=row["document_id"],
                document_title=row["title"],
                content_type="document_metadata",
                text=text,
                ingestion_quality=(
                    row["ingestion_quality"] or "unknown"
                ),
                extraction_quality=0.95,
                metadata=metadata,
            )

            self._register(
                item=item,
                channel="metadata",
                rank=rank,
                raw_score=1.0,
                bonus=0.04,
            )
            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    
    def lookup_footnotes(self,document_ids: list[str] | None = None,) -> list[EvidenceItem]:
        """Retrieve approval footnotes from named documents.

        The lookup accepts both dedicated footnote chunks and ordinary
        text chunks containing explicit approval language.

        Args:
            document_ids (list[str] | None): Optional document filter.

        Returns:
            list[EvidenceItem]: Matching footnote evidence.
        """
        clauses = [
            """
            (
                LOWER(content_type) = 'footnote'
                OR LOWER(display_text)
                    LIKE '%reviewed and approved by%'
            )
            """
        ]
        parameters: list[Any] = []

        if document_ids:
            placeholders = ",".join(
                "?" for _ in document_ids
            )
            clauses.append(
                f"document_id IN ({placeholders})"
            )
            parameters.extend(document_ids)

        sql = f"""
            SELECT *
            FROM chunks
            WHERE {" AND ".join(clauses)}
            ORDER BY document_id, chunk_id
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        results: list[EvidenceItem] = []

        for rank, row in enumerate(
            rows,
            start=1,
        ):
            item = self._chunk_to_evidence(row)

            self._register(
                item=item,
                channel="footnote",
                rank=rank,
                raw_score=1.0,
                bonus=0.08,
            )

            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    def lookup_table_rows(
        self,
        query: str,
        document_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[EvidenceItem]:
        """Search canonical structured table rows.

        Args:
            query (str): Table lookup query.
            document_ids (list[str] | None): Optional document filter.
            limit (int): Maximum table rows.

        Returns:
            list[EvidenceItem]: Matching structured table rows.
        """
        parameters: list[Any] = []
        document_filter = ""

        if document_ids:
            placeholders = ",".join(
                "?" for _ in document_ids
            )
            document_filter = (
                f"WHERE r.document_id IN ({placeholders})"
            )
            parameters.extend(document_ids)

        sql = f"""
            SELECT
                r.*,
                t.title AS table_title,
                t.headers_json,
                t.source_ref,
                t.asset_path,
                d.title AS document_title,
                d.ingestion_quality
            FROM table_rows AS r
            JOIN table_records AS t
                ON t.table_id = r.table_id
            JOIN documents AS d
                ON d.document_id = r.document_id
            {document_filter}
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        tokens = self._important_tokens(query)

        scored_rows: list[
            tuple[int, sqlite3.Row]
        ] = []

        for row in rows:
            searchable = row["searchable_text"].casefold()

            score = sum(
                1
                for token in tokens
                if token.casefold() in searchable
            )

            if score > 0:
                scored_rows.append((score, row))

        scored_rows.sort(
            key=lambda value: (
                value[0],
                not bool(value[1]["requires_visual_check"]),
            ),
            reverse=True,
        )

        results: list[EvidenceItem] = []

        for rank, (match_score, row) in enumerate(
            scored_rows[:limit],
            start=1,
        ):
            values = json.loads(
                row["values_json"]
            )
            pages = json.loads(
                row["page_numbers_json"]
            )
            requires_visual_check = bool(
                row["requires_visual_check"]
            )

            item = EvidenceItem(
                evidence_id=f"table-row:{row['row_id']}",
                document_id=row["document_id"],
                document_title=row["document_title"],
                content_type="structured_table_row",
                section=row["table_title"] or "",
                page_numbers=pages,
                source_refs=(
                    [row["source_ref"]]
                    if row["source_ref"]
                    else []
                ),
                text=(
                    f"Table: {row['table_title']}\n"
                    f"Row values: "
                    f"{json.dumps(values, ensure_ascii=False)}"
                ),
                parent_id=row["table_id"],
                asset_path=row["asset_path"],
                ingestion_quality=(
                    row["ingestion_quality"] or "unknown"
                ),
                extraction_quality=(
                    0.55
                    if requires_visual_check
                    else 0.97
                ),
                requires_visual_check=(
                    requires_visual_check
                ),
                metadata={
                    "row_number": row["row_number"],
                    "headers": json.loads(
                        row["headers_json"]
                    ),
                    "values": values,
                    "match_score": match_score,
                },
            )

            self._register(
                item=item,
                channel="structured_table",
                rank=rank,
                raw_score=float(match_score),
                bonus=0.04,
            )
            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    def lookup_figures(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Search figure captions, nearby text and OCR.

        Args:
            query (str): Figure-related query.
            document_ids (list[str] | None): Optional document filter.

        Returns:
            list[EvidenceItem]: Matching figure evidence.
        """
        parameters: list[Any] = []
        document_filter = ""

        if document_ids:
            placeholders = ",".join(
                "?" for _ in document_ids
            )
            document_filter = (
                f"WHERE f.document_id IN ({placeholders})"
            )
            parameters.extend(document_ids)

        sql = f"""
            SELECT
                f.*,
                d.title AS document_title,
                d.ingestion_quality
            FROM figures AS f
            JOIN documents AS d
                ON d.document_id = f.document_id
            {document_filter}
            ORDER BY d.title, f.figure_number
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        tokens = self._important_tokens(query)
        figure_number_match = re.search(
            r"figure\s+(\d+)",
            query,
            flags=re.IGNORECASE,
        )

        scored_rows: list[
            tuple[int, sqlite3.Row]
        ] = []

        for row in rows:
            combined = " ".join([
                row["caption"] or "",
                row["section"] or "",
                row["nearby_text"] or "",
                row["ocr_text"] or "",
            ]).casefold()

            score = sum(
                1
                for token in tokens
                if token.casefold() in combined
            )

            if (
                figure_number_match
                and row["figure_number"]
                == int(figure_number_match.group(1))
            ):
                score += 5

            if document_ids or score > 0:
                scored_rows.append((score, row))

        scored_rows.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        results: list[EvidenceItem] = []

        for rank, (match_score, row) in enumerate(
            scored_rows[:10],
            start=1,
        ):
            requires_visual_check = bool(
                row["requires_visual_check"]
            )

            item = EvidenceItem(
                evidence_id=f"figure:{row['figure_id']}",
                document_id=row["document_id"],
                document_title=row["document_title"],
                content_type="figure",
                section=row["section"] or "",
                page_numbers=json.loads(
                    row["page_numbers_json"]
                ),
                source_refs=(
                    [row["source_ref"]]
                    if row["source_ref"]
                    else []
                ),
                text=(
                    f"Caption: {row['caption'] or ''}\n"
                    f"Figure OCR: {row['ocr_text'] or ''}\n"
                    f"Nearby text: {row['nearby_text'] or ''}"
                ),
                asset_path=row["asset_path"],
                ingestion_quality=(
                    row["ingestion_quality"] or "unknown"
                ),
                extraction_quality=(
                    0.60
                    if requires_visual_check
                    else 0.78
                ),
                requires_visual_check=(
                    requires_visual_check
                ),
                metadata={
                    "figure_number": (
                        row["figure_number"]
                    ),
                    "match_score": match_score,
                },
            )

            self._register(
                item=item,
                channel="figure",
                rank=rank,
                raw_score=float(match_score),
                bonus=0.035,
            )
            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    def follow_references(
        self,
        query: str,
        source_document_ids: list[str] | None = None,
    ) -> list[EvidenceItem]:
        """Retrieve explicit cross-document reference records.

        Args:
            query (str): Cross-document question.
            source_document_ids (list[str] | None): Optional source filter.

        Returns:
            list[EvidenceItem]: Matching document-reference evidence.
        """
        parameters: list[Any] = []
        source_filter = ""

        if source_document_ids:
            placeholders = ",".join(
                "?" for _ in source_document_ids
            )
            source_filter = (
                "WHERE r.source_document_id "
                f"IN ({placeholders})"
            )
            parameters.extend(source_document_ids)

        sql = f"""
            SELECT
                r.*,
                source.title AS source_title,
                target.title AS target_title
            FROM document_references AS r
            JOIN documents AS source
                ON source.document_id = r.source_document_id
            LEFT JOIN documents AS target
                ON target.document_id = r.target_document_id
            {source_filter}
            ORDER BY source.title
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        query_tokens = self._important_tokens(query)
        scored_rows: list[
            tuple[int, sqlite3.Row]
        ] = []

        for row in rows:
            reference_text = (
                row["reference_text"] or ""
            ).casefold()

            score = sum(
                1
                for token in query_tokens
                if token.casefold() in reference_text
            )

            if source_document_ids or score > 0:
                scored_rows.append((score, row))

        scored_rows.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        results: list[EvidenceItem] = []

        for rank, (match_score, row) in enumerate(
            scored_rows[:10],
            start=1,
        ):
            item = EvidenceItem(
                evidence_id=(
                    f"reference:{row['reference_id']}"
                ),
                document_id=row["source_document_id"],
                document_title=row["source_title"],
                content_type="document_reference",
                section=row["target_section"] or "",
                page_numbers=json.loads(
                    row["source_page_numbers_json"]
                ),
                source_refs=(
                    [row["source_ref"]]
                    if row["source_ref"]
                    else []
                ),
                text=(
                    f"Reference instruction: "
                    f"{row['reference_text']}\n"
                    f"Target registry code: "
                    f"{row['target_registry_code']}\n"
                    f"Target document: "
                    f"{row['target_title'] or ''}\n"
                    f"Target section: "
                    f"{row['target_section'] or ''}"
                ),
                extraction_quality=0.95,
                metadata={
                    "target_registry_code": (
                        row["target_registry_code"]
                    ),
                    "target_document_id": (
                        row["target_document_id"]
                    ),
                    "target_document_title": (
                        row["target_title"]
                    ),
                    "target_section": row["target_section"],
                    "match_score": match_score,
                },
            )

            self._register(
                item=item,
                channel="document_reference",
                rank=rank,
                raw_score=float(match_score),
                bonus=0.05,
            )
            results.append(
                self.active_evidence[item.evidence_id]
            )

        return results

    def _retrieve_reference_targets(
        self,
        question: str,
        references: list[EvidenceItem],
        trace: RetrievalTrace,
    ) -> None:
        """Search target documents named by explicit reference records.

        Args:
            question (str): Original user question.
            references (list[EvidenceItem]): Resolved references.
            trace (RetrievalTrace): Mutable retrieval trace.
        """
        for reference in references:
            target_document_id = reference.metadata.get(
                "target_document_id"
            )

            if not target_document_id:
                continue

            agent_match = re.search(
                r"induction dosing of\s+([^,;]+)",
                reference.text,
                flags=re.IGNORECASE,
            )

            agent_name = (
                agent_match.group(1).strip()
                if agent_match
                else ""
            )

            target_query = " ".join(
                value
                for value in (
                    agent_name,
                    "induction dose",
                    question,
                )
                if value
            )

            semantic_items = self.search_semantic(
                query=target_query,
                top_k=6,
                document_ids=[target_document_id],
            )

            keyword_items = self.search_keywords(
                query=target_query,
                top_k=6,
                document_ids=[target_document_id],
            )

            table_items = self.lookup_table_rows(
                query=target_query,
                document_ids=[target_document_id],
                limit=10,
            )

            trace.semantic_result_count += len(
                semantic_items
            )
            trace.keyword_result_count += len(
                keyword_items
            )
            trace.structured_result_count += len(
                table_items
            )

    def aggregate_corpus(
        self,
        question: str,
    ) -> list[EvidenceItem]:
        """Execute supported corpus-wide structured lookups.

        Args:
            question (str): Corpus-wide question.

        Returns:
            list[EvidenceItem]: Synthetic aggregation evidence.
        """
        normalized = self._normalize(question)
        members: list[dict[str, Any]] = []
        label = ""

        tier_match = re.search(
            r"monitoring tier\s+(\d+)",
            normalized,
        )

        if tier_match:
            tier = tier_match.group(1)
            label = f"Monitoring Tier {tier}"

            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT DISTINCT
                        d.document_id,
                        d.title,
                        t.page_numbers_json,
                        t.source_ref
                    FROM table_rows AS r
                    JOIN table_records AS t
                        ON t.table_id = r.table_id
                    JOIN documents AS d
                        ON d.document_id = r.document_id
                    WHERE LOWER(t.title)
                        LIKE '%document control record%'
                    AND LOWER(r.searchable_text)
                        LIKE ?
                    ORDER BY d.title
                    """,
                    (f"%tier {tier}%",),
                ).fetchall()

            members = [
                {
                    "document_id": row["document_id"],
                    "document_title": row["title"],
                    "page_numbers": json.loads(
                        row["page_numbers_json"]
                    ),
                    "source_ref": row["source_ref"],
                }
                for row in rows
            ]

        agent_match = re.search(
            r"which conditions list\s+(.+?)\s+as their "
            r"formulary agent",
            question,
            flags=re.IGNORECASE,
        )

        if agent_match:
            agent_name = agent_match.group(1).strip()
            label = f"Formulary agent: {agent_name}"

            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT DISTINCT
                        d.document_id,
                        d.title,
                        t.page_numbers_json,
                        t.source_ref
                    FROM table_rows AS r
                    JOIN table_records AS t
                        ON t.table_id = r.table_id
                    JOIN documents AS d
                        ON d.document_id = r.document_id
                    WHERE LOWER(r.searchable_text) LIKE ?
                    ORDER BY d.title
                    """,
                    (f"%{agent_name.casefold()}%",),
                ).fetchall()

            members = [
                {
                    "document_id": row["document_id"],
                    "document_title": row["title"],
                    "page_numbers": json.loads(
                        row["page_numbers_json"]
                    ),
                    "source_ref": row["source_ref"],
                }
                for row in rows
            ]

        if not members:
            return []

        item = EvidenceItem(
            evidence_id=(
                "corpus-aggregation:"
                + re.sub(
                    r"[^a-z0-9]+",
                    "-",
                    label.casefold(),
                ).strip("-")
            ),
            document_id="corpus",
            document_title="Corpus-wide structured lookup",
            content_type="corpus_aggregation",
            text=(
                f"Aggregation rule: {label}\n"
                "Matching conditions:\n"
                + "\n".join(
                    f"- {member['document_title']}"
                    for member in members
                )
            ),
            extraction_quality=0.90,
            metadata={
                "aggregation_label": label,
                "source_members": members,
                "match_count": len(members),
            },
        )

        self._register(
            item=item,
            channel="corpus_aggregation",
            rank=1,
            raw_score=1.0,
            bonus=0.08,
        )

        return [
            self.active_evidence[item.evidence_id]
        ]

    def search_more(
        self,
        query: str,
        document_title: str = "",
        top_k: int = 6,
    ) -> list[EvidenceItem]:
        """Run a bounded additional hybrid search for an agent tool call.

        Args:
            query (str): Refined search query.
            document_title (str): Optional document title filter.
            top_k (int): Maximum returned evidence items.

        Returns:
            list[EvidenceItem]: Additional evidence.
        """
        document_ids = (
            self.resolve_document_ids(document_title)
            if document_title
            else None
        )

        results = self.search_semantic(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )

        results.extend(
            self.search_keywords(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
        )

        unique = {
            item.evidence_id: item
            for item in results
        }

        return sorted(
            unique.values(),
            key=lambda item: item.fusion_score,
            reverse=True,
        )[:top_k]

    def get_evidence_registry(
        self,
    ) -> dict[str, EvidenceItem]:
        """Return all evidence retrieved during the current request.

        Returns:
            dict[str, EvidenceItem]: Evidence keyed by evidence ID.
        """
        return dict(self.active_evidence)

    def _register(
        self,
        item: EvidenceItem,
        channel: str,
        rank: int,
        raw_score: float,
        bonus: float = 0.0,
    ) -> None:
        """Merge one retrieval result into the active evidence registry.

        Args:
            item (EvidenceItem): Retrieved evidence.
            channel (str): Retrieval method name.
            rank (int): Rank within that retrieval method.
            raw_score (float): Original retrieval score.
            bonus (float): Priority bonus for exact structured evidence.
        """
        reciprocal_rank = 1.0 / (
            RRF_CONSTANT + rank
        )

        existing = self.active_evidence.get(
            item.evidence_id
        )

        if existing is None:
            item.retrieval_channels = [channel]
            item.retrieval_scores = {
                channel: raw_score
            }
            item.fusion_score = (
                reciprocal_rank + bonus
            )
            self.active_evidence[item.evidence_id] = item
            return

        if channel not in existing.retrieval_channels:
            existing.retrieval_channels.append(channel)

        existing.retrieval_scores[channel] = raw_score
        existing.fusion_score += (
            reciprocal_rank + bonus
        )

    def _fetch_chunks(
        self,
        chunk_ids: list[str],
    ) -> list[sqlite3.Row]:
        """Load canonical chunk rows by ID.

        Args:
            chunk_ids (list[str]): Chunk IDs.

        Returns:
            list[sqlite3.Row]: Matching chunk rows.
        """
        if not chunk_ids:
            return []

        placeholders = ",".join(
            "?" for _ in chunk_ids
        )

        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()

    def _chunk_to_evidence(
        self,
        row: sqlite3.Row,
    ) -> EvidenceItem:
        """Convert a canonical chunk row into evidence.

        Args:
            row (sqlite3.Row): SQLite chunks row.

        Returns:
            EvidenceItem: Normalized evidence.
        """
        metadata = json.loads(
            row["metadata_json"] or "{}"
        )
        content_type = row["content_type"]
        requires_visual_check = bool(
            metadata.get("requires_visual_check", False)
        )

        extraction_quality = 0.88

        if content_type in {
            "table",
            "table_parent",
            "table_window",
        }:
            extraction_quality = (
                0.55
                if requires_visual_check
                else 0.95
            )
        elif content_type == "figure":
            extraction_quality = 0.75
        elif content_type == "reference":
            extraction_quality = 0.95

        return EvidenceItem(
            evidence_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row["title"],
            content_type=content_type,
            section=row["section"] or "",
            page_numbers=json.loads(
                row["page_numbers_json"]
            ),
            source_refs=json.loads(
                row["source_refs_json"]
            ),
            text=row["display_text"],
            parent_id=row["parent_id"],
            asset_path=row["asset_path"],
            ingestion_quality=(
                row["ingestion_quality"] or "unknown"
            ),
            extraction_quality=extraction_quality,
            requires_visual_check=(
                requires_visual_check
            ),
            metadata=metadata,
        )

    @staticmethod
    def _important_tokens(
        text: str,
    ) -> list[str]:
        """Extract useful retrieval terms without SQL syntax.

        Args:
            text (str): Query text.

        Returns:
            list[str]: Unique important tokens.
        """
        tokens = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            text,
        )

        result: list[str] = []

        for token in tokens:
            normalized = token.casefold()

            if (
                len(normalized) < 3
                or normalized in STOP_WORDS
                or normalized in result
            ):
                continue

            result.append(normalized)

        return result[:20]

    def _build_fts_query(
        self,
        text: str,
    ) -> str:
        """Build a safe FTS5 OR query.

        Args:
            text (str): User search text.

        Returns:
            str: Escaped FTS5 query.
        """
        tokens = self._important_tokens(text)

        return " OR ".join(
            f'"{token.replace(chr(34), "")}"'
            for token in tokens[:12]
        )