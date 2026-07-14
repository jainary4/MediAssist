from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal


APP_NAME = "clinical-qa-assistant"
BACKEND_CLASS_NAME = "ClinicalQAAgentBackend"
SAVE_FUNCTION_NAME = "save_batch_results"


def load_questions(
    question_file: Path,
) -> list[tuple[int, str]]:
    """Parse numbered questions from a Markdown file.

    Args:
        question_file (Path): Markdown file containing numbered questions.

    Returns:
        list[tuple[int, str]]: Question number and question text pairs.

    Raises:
        FileNotFoundError: If the question file does not exist.
        ValueError: If no numbered questions are found.
    """
    if not question_file.is_file():
        raise FileNotFoundError(
            f"Question file not found: {question_file}"
        )

    question_pattern = re.compile(
        r"^\s*(\d+)\.\s+(.+?)\s*$"
    )

    questions: list[tuple[int, str]] = []

    for line in question_file.read_text(
        encoding="utf-8"
    ).splitlines():
        match = question_pattern.match(line)

        if match:
            questions.append(
                (
                    int(match.group(1)),
                    match.group(2),
                )
            )

    if not questions:
        raise ValueError(
            "No numbered questions were found."
        )

    return questions


def run_one_question(
    backend: Any,
    question_number: int,
    question: str,
    include_diagnostics: bool,
) -> dict[str, Any]:
    """Run one question as an isolated Modal request.

    Args:
        backend (Any): Instantiated Modal backend class handle.
        question_number (int): Number assigned to the question.
        question (str): Question sent to the assistant.
        include_diagnostics (bool): Whether to include retrieval traces.

    Returns:
        dict[str, Any]: Full backend result or captured error.
    """
    session_id = (
        f"evaluation-{question_number:02d}-"
        f"{uuid4().hex[:8]}"
    )

    try:
        response = backend.answer.remote(
            question=question,
            session_id=session_id,
            user_id="evaluation-runner",
            include_diagnostics=include_diagnostics,
        )

        return {
            "question_number": question_number,
            "question": question,
            "status": "completed",
            "session_id": session_id,
            "response": response,
        }

    except Exception as error:
        return {
            "question_number": question_number,
            "question": question,
            "status": "error",
            "session_id": session_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_batch(
    question_file: Path,
    workers: int,
    include_diagnostics: bool,
) -> list[dict[str, Any]]:
    """Run all candidate questions concurrently.

    Args:
        question_file (Path): Candidate-question Markdown file.
        workers (int): Maximum number of concurrent requests.
        include_diagnostics (bool): Whether to include retrieval traces.

    Returns:
        list[dict[str, Any]]: Results ordered by question number.
    """
    questions = load_questions(
        question_file
    )

    backend_class = modal.Cls.from_name(
        APP_NAME,
        BACKEND_CLASS_NAME,
    )
    backend = backend_class()

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:
        futures = {
            executor.submit(
                run_one_question,
                backend,
                question_number,
                question,
                include_diagnostics,
            ): question_number
            for question_number, question in questions
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                f"Question {result['question_number']}: "
                f"{result['status']}"
            )

    results.sort(
        key=lambda result: result["question_number"]
    )

    return results


def serialize_backend_responses(
    results: list[dict[str, Any]],
) -> str:
    """Serialize complete backend results as JSONL.

    Args:
        results (list[dict[str, Any]]): Ordered evaluation results.

    Returns:
        str: One lossless JSON object per line.
    """
    return "".join(
        json.dumps(
            result,
            ensure_ascii=False,
        )
        + "\n"
        for result in results
    )


def render_agent_answers(
    results: list[dict[str, Any]],
    run_id: str,
) -> str:
    """Create a readable Markdown report from backend results.

    Args:
        results (list[dict[str, Any]]): Ordered evaluation results.
        run_id (str): Identifier for this evaluation run.

    Returns:
        str: Human-readable Markdown containing agent answers.
    """
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    lines = [
        "# Clinical QA Evaluation Answers",
        "",
        f"- Run ID: `{run_id}`",
        f"- Created at: `{created_at}`",
        f"- Questions: {len(results)}",
        "",
    ]

    for result in results:
        question_number = result["question_number"]
        question = result["question"]

        lines.extend([
            f"## Question {question_number}",
            "",
            f"**Question:** {question}",
            "",
        ])

        if result["status"] != "completed":
            lines.extend([
                "**Status:** Error",
                "",
                (
                    f"**Error:** {result.get('error', 'Unknown error')}"
                ),
                "",
                "---",
                "",
            ])
            continue

        response = result["response"]

        answer = str(
            response.get(
                "answer",
                "",
            )
        ).strip()

        abstained = bool(
            response.get(
                "abstained",
                False,
            )
        )

        confidence = float(
            response.get(
                "confidence",
                0.0,
            )
        )

        confidence_label = str(
            response.get(
                "confidence_label",
                "unknown",
            )
        )

        lines.extend([
            (
                "**Status:** Abstained"
                if abstained
                else "**Status:** Answered"
            ),
            "",
            "**Answer:**",
            "",
            answer or "No answer was returned.",
            "",
            (
                f"**Confidence:** {confidence:.2f} "
                f"({confidence_label})"
            ),
            "",
        ])

        if abstained:
            abstention_reason = response.get(
                "abstention_reason"
            )

            if abstention_reason:
                lines.extend([
                    (
                        "**Abstention reason:** "
                        f"{abstention_reason}"
                    ),
                    "",
                ])

        citations = response.get(
            "citations",
            [],
        )

        lines.extend([
            "**Citations:**",
            "",
        ])

        if citations:
            for citation_number, citation in enumerate(
                citations,
                start=1,
            ):
                lines.extend(
                    _render_citation(
                        citation_number,
                        citation,
                    )
                )
        else:
            lines.extend([
                "No citations were returned.",
                "",
            ])

        limitations = response.get(
            "limitations",
            [],
        )

        if limitations:
            lines.extend([
                "**Limitations:**",
                "",
            ])

            for limitation in limitations:
                lines.append(
                    f"- {limitation}"
                )

            lines.append("")

        lines.extend([
            "---",
            "",
        ])

    return "\n".join(lines)


def _render_citation(
    citation_number: int,
    citation: dict[str, Any],
) -> list[str]:
    """Render one citation for the readable Markdown report.

    Args:
        citation_number (int): Citation number shown to the reader.
        citation (dict[str, Any]): Citation returned by the backend.

    Returns:
        list[str]: Markdown lines representing the citation.
    """
    title = str(
        citation.get(
            "document_title",
            "Unknown document",
        )
    )

    section = str(
        citation.get(
            "section",
            "",
        )
    ).strip()

    page_numbers = citation.get(
        "page_numbers",
        [],
    )

    page_text = _format_page_numbers(
        page_numbers
    )

    location_parts = [
        title,
        page_text,
    ]

    if section:
        location_parts.append(section)

    excerpt = " ".join(
        str(
            citation.get(
                "excerpt",
                "",
            )
        ).split()
    )

    lines = [
        (
            f"{citation_number}. "
            + " — ".join(
                part
                for part in location_parts
                if part
            )
        ),
    ]

    if excerpt:
        lines.extend([
            "",
            f"   > {excerpt}",
        ])

    lines.append("")

    return lines


def _format_page_numbers(
    page_numbers: list[Any],
) -> str:
    """Format citation page numbers for human-readable output.

    Args:
        page_numbers (list[Any]): Page numbers returned by the backend.

    Returns:
        str: Page or pages label.
    """
    normalized_pages = [
        str(page)
        for page in page_numbers
    ]

    if not normalized_pages:
        return "page unknown"

    if len(normalized_pages) == 1:
        return f"page {normalized_pages[0]}"

    return "pages " + ", ".join(
        normalized_pages
    )


def create_run_id() -> str:
    """Create a unique, sortable evaluation-run identifier.

    Returns:
        str: Safe Modal Volume directory name.
    """
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return (
        f"evaluation-{timestamp}-"
        f"{uuid4().hex[:8]}"
    )


def save_results_on_modal(
    run_id: str,
    backend_responses_jsonl: str,
    agent_answers_markdown: str,
) -> dict[str, Any]:
    """Send the two evaluation files to the Modal storage function.

    Args:
        run_id (str): Unique evaluation-run identifier.
        backend_responses_jsonl (str): Complete backend response data.
        agent_answers_markdown (str): Readable answer report.

    Returns:
        dict[str, Any]: Modal Volume paths and saved file sizes.
    """
    save_function = modal.Function.from_name(
        APP_NAME,
        SAVE_FUNCTION_NAME,
    )

    return save_function.remote(
        run_id=run_id,
        backend_responses_jsonl=(
            backend_responses_jsonl
        ),
        agent_answers_markdown=(
            agent_answers_markdown
        ),
    )


def main() -> None:
    """Run the batch and save both output formats on Modal."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all candidate questions and save the results "
            "to a persistent Modal Volume."
        )
    )

    parser.add_argument(
        "question_file",
        type=Path,
        help="Path to candidate_questions.md.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum concurrent Modal requests.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help=(
            "Optional run identifier. A unique ID is generated "
            "when this argument is omitted."
        ),
    )

    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help=(
            "Exclude retrieval diagnostics from backend responses."
        ),
    )

    arguments = parser.parse_args()

    run_id = (
        arguments.run_id.strip()
        or create_run_id()
    )

    results = run_batch(
        question_file=arguments.question_file,
        workers=max(
            1,
            arguments.workers,
        ),
        include_diagnostics=(
            not arguments.no_diagnostics
        ),
    )

    backend_responses_jsonl = (
        serialize_backend_responses(
            results
        )
    )

    agent_answers_markdown = (
        render_agent_answers(
            results=results,
            run_id=run_id,
        )
    )

    storage_result = save_results_on_modal(
        run_id=run_id,
        backend_responses_jsonl=(
            backend_responses_jsonl
        ),
        agent_answers_markdown=(
            agent_answers_markdown
        ),
    )

    completed = sum(
        result["status"] == "completed"
        for result in results
    )

    errors = len(results) - completed

    abstentions = sum(
        bool(
            result.get(
                "response",
                {},
            ).get(
                "abstained",
                False,
            )
        )
        for result in results
        if result["status"] == "completed"
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "questions": len(results),
                "completed": completed,
                "errors": errors,
                "abstentions": abstentions,
                "storage": storage_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()