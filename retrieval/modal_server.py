"""Run the retrieval evidence builder as a private Modal service."""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "clinical-qa-evidence-builder"
INGESTION_VOLUME_NAME = "clinical-qa-ingestion-data"
MODEL_CACHE_VOLUME_NAME = (
    "clinical-qa-retrieval-model-cache"
)


app = modal.App(APP_NAME)

ingestion_volume = modal.Volume.from_name(
    INGESTION_VOLUME_NAME,
    create_if_missing=False,
)

model_cache_volume = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "tesseract-ocr",
        "libgomp1",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
    "docling-core[chunking]>=2.45,<3.0",
    "transformers>=4.45,<6.0",
    "sentence-transformers>=5.0,<6.0",
    "faiss-cpu>=1.8,<2.0",
    "numpy>=1.26,<3.0",
    "Pillow>=10.0,<13.0",
    "pytesseract>=0.3.13,<1.0",
)
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/root/retrieval",
    )
)


@app.function(
    image=image,
    volumes={
        "/data": ingestion_volume,
        "/root/.cache/huggingface": model_cache_volume,
    },
    cpu=4.0,
    memory=16384,
    timeout=7200,
    max_containers=1,
)
def build_knowledge_base_remote() -> dict:
    """Build and save the knowledge base on the Modal Volume.

    Returns:
        dict: Build manifest and record counts.
    """
    from retrieval.config import EvidenceBuilderConfig
    from retrieval.evidence_builder import (
        build_knowledge_base,
    )

    config = EvidenceBuilderConfig(
        ingestion_root=Path("/data/documents"),
        output_root=Path("/data/retrieval/current"),
    )

    manifest = build_knowledge_base(config)

    ingestion_volume.commit()
    model_cache_volume.commit()

    return manifest