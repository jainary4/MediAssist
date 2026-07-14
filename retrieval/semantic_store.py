"""Build an exact FAISS semantic-search index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.models import EvidenceChunk


class SemanticStore:
    """Convert evidence chunks to normalized vectors and index them."""

    def __init__(
        self,
        model_name: str,
        index_path: Path,
        mapping_path: Path,
        batch_size: int,
    ) -> None:
        """Initialize semantic-index settings.

        Args:
            model_name (str): Sentence Transformer model name.
            index_path (Path): FAISS output path.
            mapping_path (Path): Vector-ID mapping output path.
            batch_size (int): Embedding batch size.
        """
        self.model_name = model_name
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.batch_size = batch_size

    def build(self, chunks: list[EvidenceChunk]) -> None:
        """Embed chunks and build an exact cosine-similarity index.

        Args:
            chunks (list[EvidenceChunk]): Searchable evidence chunks.

        Raises:
            ValueError: If no searchable chunks were produced.
        """
        if not chunks:
            raise ValueError(
                "Cannot build a semantic index with zero chunks."
            )

        model = SentenceTransformer(self.model_name)

        texts = [chunk.search_text for chunk in chunks]

        vectors = model.encode_document(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        vectors = np.asarray(vectors, dtype="float32")
        dimension = int(vectors.shape[1])

        vector_ids = np.arange(
            len(chunks),
            dtype="int64",
        )

        base_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap2(base_index)
        index.add_with_ids(vectors, vector_ids)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(index, str(self.index_path))

        mapping = {
            "embedding_model": self.model_name,
            "dimension": dimension,
            "metric": "cosine_via_normalized_inner_product",
            "vector_count": len(chunks),
            "vectors": [
                {
                    "vector_id": int(vector_id),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content_type": chunk.content_type,
                }
                for vector_id, chunk in zip(vector_ids, chunks)
            ],
        }

        self.mapping_path.write_text(
            json.dumps(mapping, indent=2),
            encoding="utf-8",
        )