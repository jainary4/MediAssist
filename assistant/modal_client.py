from __future__ import annotations

import argparse
import json
from typing import Any

import modal


APP_NAME = "clinical-qa-assistant"
BACKEND_CLASS_NAME = "ClinicalQAAgentBackend"

"""Call the deployed Modal clinical QA agent for one question."""

def ask_question(
    question: str,
    session_id: str,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Send one question to the deployed Modal backend.

    Args:
        question (str): Clinical question.
        session_id (str): Conversation session identifier.
        include_diagnostics (bool): Include retrieval traces.

    Returns:
        dict[str, Any]: Structured assistant response.
    """
    backend_class = modal.Cls.from_name(
        APP_NAME,
        BACKEND_CLASS_NAME,
    )
    backend = backend_class()

    return backend.answer.remote(
        question=question,
        session_id=session_id,
        user_id="local-user",
        include_diagnostics=include_diagnostics,
    )


def main() -> None:
    """Parse CLI arguments and print one structured response."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask the deployed clinical QA assistant a question."
        )
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask.",
    )
    parser.add_argument(
        "--session-id",
        default="local-test-session",
        help="Conversation session ID.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include retrieval diagnostics.",
    )

    arguments = parser.parse_args()

    result = ask_question(
        question=arguments.question,
        session_id=arguments.session_id,
        include_diagnostics=arguments.debug,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()