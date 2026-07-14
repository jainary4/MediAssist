"""Provide terminal commands for submitting and downloading ingested PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docling_ingestion_client import (
    download_final_output,
    get_ingestion_status,
    submit_pdf,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for ingestion actions.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: Configured parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Submit PDFs to the Modal Docling ingestion pipeline.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Upload and ingest one local PDF.",
    )
    submit_parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the source PDF.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Read a saved ingestion result.",
    )
    status_parser.add_argument(
        "document_id",
        help="Document ID returned by the submit command.",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download verified final JSON, Markdown, and assets.",
    )
    download_parser.add_argument(
        "document_id",
        help="Document ID returned by the submit command.",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloaded_output"),
        help="Local directory that will receive extracted output.",
    )

    return parser


def main() -> None:
    """Run the selected ingestion command.

    Args:
        None.

    Returns:
        None.
    """
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "submit":
        result = submit_pdf(arguments.pdf_path)

        print(
            json.dumps(
                {
                    "document_id": result.document_id,
                    "status": result.status,
                    "selected_pipeline": result.selected_pipeline,
                    "retrieval_ready": result.retrieval_ready,
                    "volume_path": result.volume_path,
                },
                indent=2,
            )
        )
        return

    if arguments.command == "status":
        result = get_ingestion_status(arguments.document_id)
        print(json.dumps(result, indent=2))
        return

    if arguments.command == "download":
        output_directory = download_final_output(
            document_id=arguments.document_id,
            output_directory=arguments.output,
        )
        print(f"Downloaded output to: {output_directory.resolve()}")
        return


if __name__ == "__main__":
    main()