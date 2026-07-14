"""Trigger the deployed Modal evidence builder from a local computer."""

from __future__ import annotations

import json
import os

import modal


DEFAULT_APP_NAME = "clinical-qa-evidence-builder"


def run_remote_build() -> dict:
    """Call the deployed Modal evidence-building function.

    Returns:
        dict: Remote build manifest.
    """
    app_name = os.getenv(
        "MODAL_RETRIEVAL_APP_NAME",
        DEFAULT_APP_NAME,
    )

    remote_function = modal.Function.from_name(
        app_name,
        "build_knowledge_base_remote",
    )

    return remote_function.remote()


def main() -> None:
    """Run the remote build and print its manifest."""
    result = run_remote_build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()