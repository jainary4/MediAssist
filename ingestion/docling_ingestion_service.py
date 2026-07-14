"""Run the Docling ingestion pipeline remotely on Modal.

The service stores original PDFs, raw Docling output, quality reports, and
verified final output in a Modal Volume. The local computer only submits PDFs
and optionally downloads verified output.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import modal


APP_NAME = "clinical-qa-docling"
VOLUME_NAME = "clinical-qa-ingestion-data"
VOLUME_PATH = "/data"
IMAGE_SCALE = 2.0
MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024

app = modal.App(APP_NAME)

ingestion_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)

docling_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "docling[easyocr]",
        "pypdf",
        "pandas",
        "tabulate",
    )
)


@app.function(
    image=docling_image,
    volumes={VOLUME_PATH: ingestion_volume},
    cpu=4,
    memory=8192,
    timeout=1800,
    max_containers=1,
)
def process_pdf(pdf_bytes: bytes, source_filename: str) -> dict[str, Any]:
    """Ingest one PDF using standard Docling and an OCR retry when required.

    Args:
        pdf_bytes (bytes): Complete PDF file sent by the local client.
        source_filename (str): Original PDF filename.

    Returns:
        dict[str, Any]: Document ID, quality status, selected pipeline, and
        Modal Volume location.

    Raises:
        ValueError: If the uploaded data is not a valid PDF payload.
    """
    _validate_pdf(pdf_bytes, source_filename)

    document_id = _make_document_id(source_filename, pdf_bytes)
    document_root = Path(VOLUME_PATH) / "documents" / document_id
    source_path = document_root / "source" / "original.pdf"

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(pdf_bytes)

    source_profile = _profile_source_pdf(source_path)

    standard_directory = document_root / "runs" / "standard"
    _reset_directory(standard_directory)

    _run_docling(
        source_path=source_path,
        output_directory=standard_directory,
        source_profile=source_profile,
        force_full_page_ocr=False,
    )

    standard_quality = _check_output(
        output_directory=standard_directory,
        source_profile=source_profile,
        force_full_page_ocr=False,
    )

    _write_json(
        standard_directory / "quality_report.json",
        standard_quality,
    )

    selected_directory = standard_directory
    selected_quality = standard_quality

    if standard_quality["needs_full_page_ocr"]:
        ocr_directory = document_root / "runs" / "full_page_ocr"
        _reset_directory(ocr_directory)

        _run_docling(
            source_path=source_path,
            output_directory=ocr_directory,
            source_profile=source_profile,
            force_full_page_ocr=True,
        )

        selected_quality = _check_output(
            output_directory=ocr_directory,
            source_profile=source_profile,
            force_full_page_ocr=True,
        )

        _write_json(
            ocr_directory / "quality_report.json",
            selected_quality,
        )

        selected_directory = ocr_directory

    final_directory = document_root / "final"

    if selected_quality["passed"]:
        _reset_directory(final_directory)
        shutil.copytree(
            selected_directory,
            final_directory,
            dirs_exist_ok=True,
        )

        _write_json(
            final_directory / "ingestion_manifest.json",
            {
                "document_id": document_id,
                "status": "pass",
                "selected_pipeline": selected_directory.name,
                "needs_vlm": False,
                "main_files": {
                    "json": "document.json",
                    "markdown": "document.md",
                    "tables": "tables/",
                    "figures": "figures/",
                    "pages": "pages/",
                },
            },
        )

        status = "pass"
    else:
        review_directory = document_root / "review"
        _reset_directory(review_directory)

        _write_json(
            review_directory / "ingestion_manifest.json",
            {
                "document_id": document_id,
                "status": "manual_review",
                "selected_pipeline": selected_directory.name,
                "needs_vlm": selected_quality["needs_vlm"],
                "issues": selected_quality["issues"],
            },
        )

        status = "manual_review"

    ingestion_volume.commit()

    return {
        "document_id": document_id,
        "status": status,
        "selected_pipeline": selected_directory.name,
        "retrieval_ready": status == "pass",
        "volume_path": str(document_root),
    }


@app.function(
    volumes={VOLUME_PATH: ingestion_volume},
    timeout=300,
)
def get_status(document_id: str) -> dict[str, Any]:
    """Read the saved ingestion status for a document from the Modal Volume.

    Args:
        document_id (str): Document ID returned by process_pdf.

    Returns:
        dict[str, Any]: Saved final or manual-review ingestion manifest.

    Raises:
        FileNotFoundError: If no status is saved for the document.
    """
    ingestion_volume.reload()

    document_root = _get_document_root(document_id)

    final_manifest = document_root / "final" / "ingestion_manifest.json"
    review_manifest = document_root / "review" / "ingestion_manifest.json"

    if final_manifest.exists():
        return json.loads(final_manifest.read_text(encoding="utf-8"))

    if review_manifest.exists():
        return json.loads(review_manifest.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"No ingestion result found for {document_id}.")


@app.function(
    volumes={VOLUME_PATH: ingestion_volume},
    timeout=300,
)
def download_final_output(document_id: str) -> bytes:
    """Create a ZIP file containing verified final ingestion output.

    Args:
        document_id (str): Document ID returned by process_pdf.

    Returns:
        bytes: ZIP archive containing final JSON, Markdown, tables, figures,
        pages, and the ingestion manifest.

    Raises:
        FileNotFoundError: If the document did not pass quality checks.
    """
    ingestion_volume.reload()

    final_directory = _get_document_root(document_id) / "final"

    if not final_directory.exists():
        raise FileNotFoundError(
            "This document has no verified final output. Check its status first."
        )

    return _zip_directory(final_directory)


def _validate_pdf(pdf_bytes: bytes, source_filename: str) -> None:
    """Validate the basic type, size, and PDF signature of an upload.

    Args:
        pdf_bytes (bytes): PDF file bytes from the local client.
        source_filename (str): Original uploaded filename.

    Raises:
        ValueError: If the file is invalid, too large, or not a PDF.
    """
    if Path(source_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed.")

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not look like a PDF.")

    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise ValueError("The PDF is larger than 100 MB.")


def _make_document_id(source_filename: str, pdf_bytes: bytes) -> str:
    """Create a stable and filesystem-safe ID for one PDF.

    Args:
        source_filename (str): Original PDF filename.
        pdf_bytes (bytes): PDF bytes used to create a content hash.

    Returns:
        str: Stable document ID, such as short_bowel_syndrome-a1b2c3d4e5f6.
    """
    raw_name = Path(source_filename).stem.lower()
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in raw_name
    ).strip("_")

    content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

    return f"{safe_name[:60] or 'document'}-{content_hash}"


def _get_document_root(document_id: str) -> Path:
    """Return the safe Modal Volume path for one document.

    Args:
        document_id (str): Document ID returned by process_pdf.

    Returns:
        Path: Root directory for this document in the Modal Volume.

    Raises:
        ValueError: If the ID contains unsafe filesystem characters.
    """
    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyz0123456789_-"
    )

    if not document_id or any(
        character not in allowed_characters
        for character in document_id
    ):
        raise ValueError("Invalid document ID.")

    return Path(VOLUME_PATH) / "documents" / document_id


def _reset_directory(directory: Path) -> None:
    """Delete and recreate one exact run directory.

    Args:
        directory (Path): Directory to recreate.
    """
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def _profile_source_pdf(source_path: Path) -> dict[str, Any]:
    """Inspect the source PDF before Docling runs.

    This is intentionally simple. It identifies likely scanned pages by checking
    whether a page has very little native selectable text.

    Args:
        source_path (Path): PDF stored in the Modal Volume.

    Returns:
        dict[str, Any]: Source page count and likely scanned page numbers.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(source_path))

    scan_candidate_pages: list[int] = []
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        native_text = page.extract_text() or ""

        pages.append(
            {
                "page_number": page_number,
                "native_text_characters": len(native_text.strip()),
                "rotation": int(page.get("/Rotate", 0)) % 360,
            }
        )

        if len(native_text.strip()) < 40:
            scan_candidate_pages.append(page_number)

    return {
        "source_page_count": len(reader.pages),
        "scan_candidate_pages": scan_candidate_pages,
        "pages": pages,
    }


def _run_docling(
    source_path: Path,
    output_directory: Path,
    source_profile: dict[str, Any],
    force_full_page_ocr: bool,
) -> None:
    """Run one Docling conversion mode and save all output files.

    Args:
        source_path (Path): PDF input path in the Modal Volume.
        output_directory (Path): Directory receiving this run's output.
        source_profile (dict[str, Any]): Source-PDF inspection metadata.
        force_full_page_ocr (bool): False for standard extraction and True for
            the stronger full-page OCR retry.
    """
    converter = _build_converter(force_full_page_ocr)
    result = converter.convert(source_path)

    _save_docling_output(
        result=result,
        output_directory=output_directory,
        source_profile=source_profile,
        force_full_page_ocr=force_full_page_ocr,
    )


def _build_converter(force_full_page_ocr: bool) -> Any:
    """Build Docling's standard or full-page OCR PDF converter.

    Args:
        force_full_page_ocr (bool): Whether to OCR every page image.

    Returns:
        Any: Configured Docling DocumentConverter.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=force_full_page_ocr,
        do_table_structure=True,
        generate_page_images=True,
        generate_picture_images=True,
        generate_table_images=True,
        images_scale=IMAGE_SCALE,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )


def _save_docling_output(
    result: Any,
    output_directory: Path,
    source_profile: dict[str, Any],
    force_full_page_ocr: bool,
) -> None:
    """Save Docling JSON, Markdown, page images, tables, figures, and metadata.

    Args:
        result (Any): Completed Docling conversion result.
        output_directory (Path): Directory receiving saved output files.
        source_profile (dict[str, Any]): Source PDF inspection result.
        force_full_page_ocr (bool): Pipeline mode used for this output.
    """
    from docling_core.types.doc import ImageRefMode, PictureItem

    pages_directory = output_directory / "pages"
    tables_directory = output_directory / "tables"
    figures_directory = output_directory / "figures"

    pages_directory.mkdir()
    tables_directory.mkdir()
    figures_directory.mkdir()

    result.document.save_as_json(
        output_directory / "document.json",
        indent=2,
    )

    result.document.save_as_markdown(
        output_directory / "document.md",
        image_mode=ImageRefMode.REFERENCED,
    )

    for page_number, page in result.document.pages.items():
        if page.image is not None:
            page.image.pil_image.save(
                pages_directory / f"page_{page_number:03d}.png"
            )

    for table_number, table in enumerate(
        result.document.tables,
        start=1,
    ):
        stem = f"table_{table_number:03d}"
        dataframe = table.export_to_dataframe(doc=result.document)

        dataframe.to_json(
            tables_directory / f"{stem}.json",
            orient="split",
            indent=2,
        )
        dataframe.to_csv(
            tables_directory / f"{stem}.csv",
            index=False,
        )
        dataframe.to_markdown(
            tables_directory / f"{stem}.md",
            index=False,
        )

        table_image = table.get_image(result.document)

        if table_image is not None:
            table_image.save(
                tables_directory / f"{stem}.png",
                "PNG",
            )

    figure_number = 0

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            figure_number += 1
            figure_image = element.get_image(result.document)

            if figure_image is not None:
                figure_image.save(
                    figures_directory / f"figure_{figure_number:03d}.png",
                    "PNG",
                )

    _write_json(
        output_directory / "source_metadata.json",
        {
            **source_profile,
            "source_filename": result.input.file.name,
            "full_page_ocr": force_full_page_ocr,
            "docling_confidence": _get_confidence(result),
        },
    )


def _get_confidence(result: Any) -> dict[str, Any]:
    """Convert Docling confidence data into JSON-safe information.

    Args:
        result (Any): Completed Docling conversion result.

    Returns:
        dict[str, Any]: Confidence data, or an empty dictionary if unavailable.
    """
    confidence = getattr(result, "confidence", None)

    if confidence is None:
        return {}

    model_dump = getattr(confidence, "model_dump", None)

    if callable(model_dump):
        return model_dump(mode="json")

    return {}


def _check_output(
    output_directory: Path,
    source_profile: dict[str, Any],
    force_full_page_ocr: bool,
) -> dict[str, Any]:
    """Inspect saved output files and decide whether a stronger OCR retry is needed.

    Args:
        output_directory (Path): Saved Docling output directory.
        source_profile (dict[str, Any]): Expected source-page information.
        force_full_page_ocr (bool): Whether this is already the OCR retry run.

    Returns:
        dict[str, Any]: Pass/fail result, issues, and VLM recommendation flag.
    """
    issues: list[str] = []

    document_json_path = output_directory / "document.json"
    markdown_path = output_directory / "document.md"
    metadata_path = output_directory / "source_metadata.json"

    if not document_json_path.exists():
        issues.append("document.json is missing.")

    if not markdown_path.exists():
        issues.append("document.md is missing.")

    markdown_text = (
        markdown_path.read_text(encoding="utf-8")
        if markdown_path.exists()
        else ""
    )

    document_json = (
        json.loads(document_json_path.read_text(encoding="utf-8"))
        if document_json_path.exists()
        else {}
    )

    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    expected_page_count = source_profile["source_page_count"]
    document_pages = document_json.get("pages", {})
    extracted_page_count = len(document_pages)

    page_images = list(
        (output_directory / "pages").glob("page_*.png")
    )

    if extracted_page_count != expected_page_count:
        issues.append(
            "Docling JSON page count does not match the source PDF page count."
        )

    if len(page_images) != expected_page_count:
        issues.append(
            "Saved page-image count does not match the source PDF page count."
        )

    for table_json_path in (output_directory / "tables").glob("table_*.json"):
        table_png_path = table_json_path.with_suffix(".png")

        if not table_png_path.exists():
            issues.append(
                f"Missing source crop for {table_json_path.name}."
            )

    low_grade = str(
        metadata.get("docling_confidence", {}).get("low_grade", "")
    ).upper()

    scan_candidates = source_profile["scan_candidate_pages"]

    markdown_is_too_small = (
        len(markdown_text.strip()) < expected_page_count * 100
    )

    needs_full_page_ocr = (
        not force_full_page_ocr
        and bool(scan_candidates)
        and (
            low_grade in {"POOR", "FAIR"}
            or markdown_is_too_small
        )
    )

    has_structural_failure = bool(issues)

    needs_vlm = (
        force_full_page_ocr
        and has_structural_failure
    )

    passed = (
        not has_structural_failure
        and not needs_full_page_ocr
    )

    return {
        "passed": passed,
        "needs_full_page_ocr": needs_full_page_ocr,
        "needs_vlm": needs_vlm,
        "issues": issues,
        "metrics": {
            "expected_page_count": expected_page_count,
            "extracted_page_count": extracted_page_count,
            "saved_page_images": len(page_images),
            "scan_candidate_pages": scan_candidates,
            "low_grade": low_grade,
            "markdown_characters": len(markdown_text.strip()),
        },
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to a UTF-8 file.

    Args:
        path (Path): JSON destination path.
        data (dict[str, Any]): JSON-serializable dictionary to write.
    """
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _zip_directory(directory: Path) -> bytes:
    """Create an in-memory ZIP archive from all files in a directory.

    Args:
        directory (Path): Directory containing verified final output.

    Returns:
        bytes: Complete ZIP archive.
    """
    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                archive.write(
                    file_path,
                    file_path.relative_to(directory),
                )

    return buffer.getvalue()