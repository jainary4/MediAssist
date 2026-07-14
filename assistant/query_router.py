from __future__ import annotations

import re
from typing import Any

from assistant.models import QueryIntent, RoutePlan


ROUTE_CONFIGURATION: dict[
    QueryIntent,
    dict[str, list[str]],
] = {
    "general_text": {
        "required_facts": [
            "clinical_answer",
        ],
        "retrieval_channels": [
            "semantic",
            "keyword",
        ],
        "required_evidence_types": [
            "text_or_table",
        ],
    },
    "classification_code": {
        "required_facts": [
            "classification_code",
        ],
        "retrieval_channels": [
            "metadata",
            "structured_table",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "document_control_record",
        ],
    },
    "review_body": {
        "required_facts": [
            "review_body",
        ],
        "retrieval_channels": [
            "metadata",
            "structured_table",
            "footnote",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "resolved_footnote",
        ],
    },
    "figure_value": {
        "required_facts": [
            "figure_value",
        ],
        "retrieval_channels": [
            "figure",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "figure",
        ],
    },
    "scanned_appendix": {
        "required_facts": [
            "drug",
            "induction_dose",
        ],
        "retrieval_channels": [
            "structured_table",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "ocr_text_or_table",
        ],
    },
    "maintenance_dose": {
        "required_facts": [
            "drug",
            "maintenance_dose",
        ],
        "retrieval_channels": [
            "structured_table",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "structured_table_or_table_window",
        ],
    },
    "induction_dose": {
        "required_facts": [
            "drug",
            "induction_dose",
        ],
        "retrieval_channels": [
            "structured_table",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "structured_table_or_text",
        ],
    },
    "cross_document_dose": {
        "required_facts": [
            "cross_document_link",
            "drug",
            "induction_dose",
        ],
        "retrieval_channels": [
            "document_reference",
            "structured_table",
            "keyword",
            "semantic",
        ],
        "required_evidence_types": [
            "document_reference",
            "target_document_evidence",
        ],
    },
    "corpus_monitoring_tier": {
        "required_facts": [
            "complete_condition_set",
        ],
        "retrieval_channels": [
            "structured_table",
            "corpus_aggregation",
        ],
        "required_evidence_types": [
            "corpus_aggregation",
        ],
    },
    "corpus_formulary_agent": {
        "required_facts": [
            "complete_condition_set",
        ],
        "retrieval_channels": [
            "structured_table",
            "corpus_aggregation",
        ],
        "required_evidence_types": [
            "corpus_aggregation",
        ],
    },
    "reverse_registry_code": {
        "required_facts": [
            "registry_code",
            "document_title",
        ],
        "retrieval_channels": [
            "metadata",
            "structured_table",
            "keyword",
        ],
        "required_evidence_types": [
            "document_metadata_or_control_record",
        ],
    },
}


FIGURE_PATTERNS = (
    r"\bfigure\s+\d+\b",
    r"\baccording to (?:the )?figure\b",
    r"\bshown in (?:the )?figure\b",
    r"\bchart\b",
    r"\bgraph\b",
)


CROSS_DOCUMENT_PATTERNS = (
    r"\banother monograph\b",
    r"\bpoints to\b",
    r"\bdo not use local values\b",
    r"\bconsult formulary\b",
    r"\btarget monograph\b",
)


CORPUS_PATTERNS = (
    r"\bevery condition\b",
    r"\bwhich conditions\b",
    r"\ball conditions\b",
    r"\bacross the corpus\b",
    r"\ball monographs\b",
)


def normalize_text(value: str) -> str:
    """Normalize text for rule matching and title matching.

    Args:
        value (str): Original text.

    Returns:
        str: Lowercase text containing normalized spaces.
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.casefold(),
    ).strip()


def matches_any_pattern(
    value: str,
    patterns: tuple[str, ...],
) -> bool:
    """Check whether text matches at least one regular expression.

    Args:
        value (str): Normalized or original text.
        patterns (tuple[str, ...]): Regular expressions to check.

    Returns:
        bool: True when one pattern matches.
    """
    return any(
        re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def detect_intent(question: str) -> QueryIntent:
    """Determine the principal intent of a user question.

    More-specific intents are checked before general intents.

    Args:
        question (str): User question.

    Returns:
        QueryIntent: Deterministic question category.
    """
    normalized = normalize_text(question)

    is_corpus_question = matches_any_pattern(
        normalized,
        CORPUS_PATTERNS,
    )

    if (
        is_corpus_question
        and "monitoring tier" in normalized
    ):
        return "corpus_monitoring_tier"

    if (
        is_corpus_question
        and "formulary agent" in normalized
    ):
        return "corpus_formulary_agent"

    if matches_any_pattern(
        normalized,
        CROSS_DOCUMENT_PATTERNS,
    ):
        return "cross_document_dose"

    if matches_any_pattern(
        normalized,
        FIGURE_PATTERNS,
    ):
        return "figure_value"

    if (
        "scanned appendix" in normalized
        or "appendix c" in normalized
    ):
        return "scanned_appendix"

    if "classification code" in normalized:
        return "classification_code"

    if (
        "review body" in normalized
        or "approving body" in normalized
        or (
            "approved" in normalized
            and "monograph" in normalized
        )
    ):
        return "review_body"

    registry_code_present = bool(
        re.search(
            r"\bCDR-\d+\b",
            question,
            flags=re.IGNORECASE,
        )
    )

    if (
        registry_code_present
        and any(
            signal in normalized
            for signal in (
                "assigned to",
                "which monograph",
                "what is registry code",
                "belongs to",
            )
        )
    ):
        return "reverse_registry_code"

    if (
        "maintenance dose" in normalized
        or "landscape staging matrix" in normalized
        or "appendix a" in normalized
    ):
        return "maintenance_dose"

    if (
        "induction dose" in normalized
        or "formulary agent" in normalized
    ):
        return "induction_dose"

    return "general_text"


def find_named_documents(
    question: str,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find documents whose titles explicitly occur in a question.

    Args:
        question (str): User question.
        documents (list[dict[str, Any]]): Document metadata records.

    Returns:
        list[dict[str, Any]]: Matching document metadata records.
    """
    normalized_question = normalize_text(question)

    matches: list[dict[str, Any]] = []

    for document in documents:
        title = str(
            document.get("title", "")
        ).strip()

        normalized_title = normalize_text(title)

        if (
            normalized_title
            and normalized_title in normalized_question
        ):
            matches.append(document)

    return matches


def analyse_query(
    question: str,
    documents: list[dict[str, Any]],
) -> RoutePlan:
    """Create a deterministic route plan for one question.

    Args:
        question (str): User question.
        documents (list[dict[str, Any]]): Known document records.

    Returns:
        RoutePlan: Intent, retrieval channels and evidence requirements.
    """
    intent = detect_intent(question)
    configuration = ROUTE_CONFIGURATION[intent]

    named_documents = find_named_documents(
        question=question,
        documents=documents,
    )

    retrieval_channels = list(
        configuration["retrieval_channels"]
    )

    requires_metadata = (
        "metadata" in retrieval_channels
    )
    requires_structured_table = (
        "structured_table" in retrieval_channels
    )
    requires_figure = (
        intent == "figure_value"
    )
    requires_cross_document = (
        intent == "cross_document_dose"
    )
    requires_corpus_aggregation = intent in {
        "corpus_monitoring_tier",
        "corpus_formulary_agent",
    }

    return RoutePlan(
        intent=intent,
        named_document_ids=[
            str(document["document_id"])
            for document in named_documents
        ],
        named_document_titles=[
            str(document["title"])
            for document in named_documents
        ],
        required_facts=list(
            configuration["required_facts"]
        ),
        retrieval_channels=retrieval_channels,
        required_evidence_types=list(
            configuration["required_evidence_types"]
        ),
        requires_metadata=requires_metadata,
        requires_structured_table=(
            requires_structured_table
        ),
        requires_figure=requires_figure,
        requires_cross_document=(
            requires_cross_document
        ),
        requires_corpus_aggregation=(
            requires_corpus_aggregation
        ),
    )