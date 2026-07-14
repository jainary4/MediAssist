"""Call the private Modal Docling ingestion service from the local computer."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import modal


DEFAULT_APP_NAME = "clinical-qa-docling"


@dataclass(frozen=True)
class IngestionResult:
    """Store the response returned by the Modal ingestion service.

    Attributes:
        document_id (str): Stable ID for this PDF in the Modal Volume.
        status (str): Either "pass" or "manual_review".
        selected_pipeline (str): Either "standard" or "full_page_ocr".
        retrieval_ready (bool): Whether verified final output exists.
        volume_path (str): Remote Modal Volume path for the document.
    """

    document_id: str
    status: str
    selected_pipeline: str
    retrieval_ready: bool
    volume_path: str


def submit_pdf(pdf_path: Path) -> IngestionResult:
    """Send one local PDF to the private Modal ingestion service.

    Args:
        pdf_path (Path): Local path to the source PDF.

    Returns:
        IngestionResult: Result returned by the Modal ingestion service.

    Raises:
        FileNotFoundError: If the local PDF does not exist.
        ValueError: If the supplied file is not a PDF.
    """
    _validate_pdf_path(pdf_path)

    app_name = os.getenv(
        "MODAL_INGESTION_APP_NAME",
        DEFAULT_APP_NAME,
    )

    remote_function = modal.Function.from_name(
        app_name,
        "process_pdf",
    )

    response = remote_function.remote(
        pdf_path.read_bytes(),
        pdf_path.name,
    )

    return IngestionResult(
        document_id=str(response["document_id"]),
        status=str(response["status"]),
        selected_pipeline=str(response["selected_pipeline"]),
        retrieval_ready=bool(response["retrieval_ready"]),
        volume_path=str(response["volume_path"]),
    )


def get_ingestion_status(document_id: str) -> dict[str, Any]:
    """Read the saved final or manual-review status from Modal.

    Args:
        document_id (str): Document ID returned by submit_pdf.

    Returns:
        dict[str, Any]: Saved Modal manifest for the PDF.
    """
    app_name = os.getenv(
        "MODAL_INGESTION_APP_NAME",
        DEFAULT_APP_NAME,
    )

    remote_function = modal.Function.from_name(
        app_name,
        "get_status",
    )

    return remote_function.remote(document_id)


def download_final_output(
    document_id: str,
    output_directory: Path,
) -> Path:
    """Download and safely extract verified output from Modal.

    Args:
        document_id (str): Document ID returned by submit_pdf.
        output_directory (Path): Local directory receiving extracted files.

    Returns:
        Path: Local output directory containing final JSON, Markdown, and assets.

    Raises:
        ValueError: If Modal returns an invalid or unsafe ZIP archive.
    """
    app_name = os.getenv(
        "MODAL_INGESTION_APP_NAME",
        DEFAULT_APP_NAME,
    )

    remote_function = modal.Function.from_name(
        app_name,
        "download_final_output",
    )

    archive_bytes = remote_function.remote(document_id)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _extract_zip_safely(
        archive_bytes=archive_bytes,
        output_directory=output_directory,
    )

    return output_directory


def _validate_pdf_path(pdf_path: Path) -> None:
    """Check that a supplied local path points to a PDF file.

    Args:
        pdf_path (Path): Candidate local file path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not .pdf.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {pdf_path.name}"
        )


def _extract_zip_safely(
    archive_bytes: bytes,
    output_directory: Path,
) -> None:
    """Extract a ZIP archive while blocking path-traversal attacks.

    Args:
        archive_bytes (bytes): ZIP bytes returned by Modal.
        output_directory (Path): Local extraction destination.

    Returns:
        None.

    Raises:
        ValueError: If the archive is invalid or contains an unsafe path.
    """
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            output_root = output_directory.resolve()

            for member in archive.infolist():
                destination = (
                    output_directory / member.filename
                ).resolve()

                is_inside_output_directory = (
                    output_root in destination.parents
                    or destination == output_root
                )

                if not is_inside_output_directory:
                    raise ValueError(
                        f"Unsafe file path in ZIP: {member.filename}"
                    )

                archive.extract(member, output_directory)

    except zipfile.BadZipFile as error:
        raise ValueError(
            "Modal did not return a valid ZIP archive."
        ) from error