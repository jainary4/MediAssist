"""Submit every PDF in a local folder to the deployed Modal ingestion service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import modal


DEFAULT_APP_NAME = "clinical-qa-docling"
DEFAULT_MAX_CONTAINERS = 5


def find_pdf_files(
    pdf_directory: Path,
    limit: int | None,
) -> list[Path]:
    """Find PDF files recursively inside a local directory.

    Args:
        pdf_directory (Path): Root directory containing source PDFs.
        limit (int | None): Optional maximum number of PDFs to return.

    Returns:
        list[Path]: Sorted local PDF paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no PDF files are found.
    """
    if not pdf_directory.is_dir():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_directory}"
        )

    pdf_paths = sorted(
        path
        for path in pdf_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    if not pdf_paths:
        raise ValueError(
            f"No PDF files found in: {pdf_directory}"
        )

    return pdf_paths


def submit_pdf_batch(
    pdf_paths: list[Path],
    max_containers: int,
) -> list[dict[str, Any]]:
    """Submit multiple PDFs to Modal using parallel function inputs.

    Args:
        pdf_paths (list[Path]): Local PDFs to ingest.
        max_containers (int): Maximum Modal containers processing at once.

    Returns:
        list[dict[str, Any]]: One result record for each submitted PDF.

    Raises:
        ValueError: If max_containers is outside the safe range.
    """
    if max_containers < 1 or max_containers > 5:
        raise ValueError(
            "Use between 1 and 5 containers with the current Modal Volume. "
            "More concurrent writers require changing to a Volume v2 design."
        )

    app_name = os.getenv(
        "MODAL_INGESTION_APP_NAME",
        DEFAULT_APP_NAME,
    )

    remote_function = modal.Function.from_name(
        app_name,
        "process_pdf",
    )

    remote_function.update_autoscaler(
        max_containers=max_containers,
        scaledown_window=600,
    )

    inputs = (
        (pdf_path.read_bytes(), pdf_path.name)
        for pdf_path in pdf_paths
    )

    remote_results = remote_function.starmap(
        inputs,
        order_outputs=True,
        return_exceptions=True,
    )

    batch_results: list[dict[str, Any]] = []

    for pdf_path, result in zip(pdf_paths, remote_results):
        if isinstance(result, BaseException):
            batch_results.append(
                {
                    "filename": pdf_path.name,
                    "local_path": str(pdf_path),
                    "status": "remote_error",
                    "error_type": type(result).__name__,
                    "error": str(result),
                }
            )
            continue

        batch_results.append(
            {
                "filename": pdf_path.name,
                "local_path": str(pdf_path),
                "status": result.get("status"),
                "document_id": result.get("document_id"),
                "selected_pipeline": result.get(
                    "selected_pipeline"
                ),
                "retrieval_ready": result.get(
                    "retrieval_ready"
                ),
                "volume_path": result.get("volume_path"),
            }
        )

    return batch_results


def write_batch_results(
    results: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Save batch-ingestion results as a local JSON report.

    Args:
        results (list[dict[str, Any]]): Result records from the batch.
        output_file (Path): Local JSON report destination.

    Returns:
        None.
    """
    passed_count = sum(
        result["status"] == "pass"
        for result in results
    )

    manual_review_count = sum(
        result["status"] == "manual_review"
        for result in results
    )

    error_count = sum(
        result["status"] == "remote_error"
        for result in results
    )

    report = {
        "total_pdfs": len(results),
        "passed": passed_count,
        "manual_review": manual_review_count,
        "remote_errors": error_count,
        "results": results,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for batch ingestion.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: Configured command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Submit a folder of PDFs to Modal ingestion."
    )

    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Folder containing source PDFs.",
    )

    parser.add_argument(
        "--max-containers",
        type=int,
        default=DEFAULT_MAX_CONTAINERS,
        help="Parallel Modal containers. Keep this at 5 for now.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of PDFs to process for a test run.",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("batch_ingestion_results.json"),
        help="Local JSON file receiving the batch report.",
    )

    return parser


def main() -> None:
    """Run the batch ingestion command.

    Args:
        None.

    Returns:
        None.
    """
    parser = build_parser()
    arguments = parser.parse_args()

    pdf_paths = find_pdf_files(
        pdf_directory=arguments.pdf_directory,
        limit=arguments.limit,
    )

    print(
        f"Submitting {len(pdf_paths)} PDF files "
        f"with up to {arguments.max_containers} containers..."
    )

    results = submit_pdf_batch(
        pdf_paths=pdf_paths,
        max_containers=arguments.max_containers,
    )

    write_batch_results(
        results=results,
        output_file=arguments.results_file,
    )

    print(
        f"Finished. Report saved to: "
        f"{arguments.results_file.resolve()}"
    )

    for result in results:
        print(
            f"{result['filename']}: "
            f"{result['status']}"
        )


if __name__ == "__main__":
    main()