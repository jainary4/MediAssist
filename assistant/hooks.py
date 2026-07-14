from __future__ import annotations

from assistant.models import (
    AgentDraft,
    Citation,
    ClinicalAnswer,
    ConfidenceBreakdown,
    EvidenceItem,
    EvidencePack,
    RoutePlan,
    ValidationReport,
)


ANSWER_THRESHOLD = 0.75
MAXIMUM_PROTOTYPE_CONFIDENCE = 0.95


def pre_answer_check(
    evidence_pack: EvidencePack,
) -> list[str]:
    """Check whether retrieval returned any evidence.

    Args:
        evidence_pack (EvidencePack): Evidence selected for the question.

    Returns:
        list[str]: A blocking error only when retrieval found no evidence.
    """
    if evidence_pack.evidence:
        return []

    return [
        "No evidence was found in the document collection."
    ]


def validate_agent_draft(
    draft: AgentDraft,
    evidence_registry: dict[str, EvidenceItem],
) -> ValidationReport:
    """Validate answer structure and citation references.

    The validator does not inspect whether values are numeric or alphanumeric.
    Retrieval and the answer-writing model are trusted to interpret the contents
    of cited chunks. The deterministic checks only ensure that citations exist
    and that every factual claim has at least one valid citation.

    Args:
        draft (AgentDraft): Structured draft returned by the answer agent.
        evidence_registry (dict[str, EvidenceItem]): Evidence available during
            the current request.

    Returns:
        ValidationReport: Citation and claim validation results.
    """
    errors: list[str] = []

    if draft.abstained:
        if not draft.abstention_reason:
            errors.append(
                "An abstained answer must include a reason."
            )

        return ValidationReport(
            valid=not errors,
            errors=errors,
            citation_validity=1.0,
            claim_citation_coverage=1.0,
            numeric_support=1.0,
        )

    if not draft.answer.strip():
        errors.append(
            "A non-abstained answer cannot be empty."
        )

    if not draft.claims:
        errors.append(
            "A non-abstained answer must contain at least one claim."
        )

    all_cited_ids: set[str] = set(
        draft.cited_evidence_ids
    )

    for claim in draft.claims:
        all_cited_ids.update(
            claim.evidence_ids
        )

    valid_citation_ids = {
        evidence_id
        for evidence_id in all_cited_ids
        if evidence_id in evidence_registry
    }

    invalid_citation_ids = sorted(
        all_cited_ids - valid_citation_ids
    )

    if invalid_citation_ids:
        errors.append(
            "The answer used unknown evidence IDs: "
            + ", ".join(invalid_citation_ids)
        )

    citation_validity = (
        len(valid_citation_ids)
        / len(all_cited_ids)
        if all_cited_ids
        else 0.0
    )

    claims_with_valid_citations = 0

    for claim in draft.claims:
        valid_claim_ids = [
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_registry
        ]

        if valid_claim_ids:
            claims_with_valid_citations += 1
        else:
            errors.append(
                "Claim has no valid citation: "
                f"{claim.text}"
            )

    claim_citation_coverage = (
        claims_with_valid_citations
        / len(draft.claims)
        if draft.claims
        else 0.0
    )

    return ValidationReport(
        valid=not errors,
        errors=errors,
        citation_validity=citation_validity,
        claim_citation_coverage=(
            claim_citation_coverage
        ),
        numeric_support=1.0,
    )


def calculate_confidence(
    route: RoutePlan,
    draft: AgentDraft,
    validation: ValidationReport,
    evidence_registry: dict[str, EvidenceItem],
) -> ConfidenceBreakdown:
    """Calculate confidence from retrieval and citation support.

    No evidence type receives an automatic cap. OCR, figures, scans, tables, and
    visually flagged items contribute their extraction quality as a continuous
    input rather than becoming an automatic abstention. Invalid citations remain
    a hard failure because they break grounding.

    Args:
        route (RoutePlan): Retrieval route selected for the question.
        draft (AgentDraft): Structured answer produced by the model.
        validation (ValidationReport): Deterministic citation validation.
        evidence_registry (dict[str, EvidenceItem]): Request-local evidence.

    Returns:
        ConfidenceBreakdown: Transparent confidence components and final score.
    """
    cited_evidence = _collect_cited_evidence(
        draft=draft,
        evidence_registry=evidence_registry,
    )

    coverage = (
        0.0
        if draft.abstained
        else validation.claim_citation_coverage
    )

    retrieval_support = _calculate_retrieval_support(
        route=route,
        cited_evidence=cited_evidence,
        evidence_registry=evidence_registry,
    )

    evidence_quality = (
        sum(
            item.extraction_quality
            for item in cited_evidence
        )
        / len(cited_evidence)
        if cited_evidence
        else 0.0
    )

    grounding = (
        0.50 * validation.citation_validity
        + 0.50
        * validation.claim_citation_coverage
    )

    consistency = _calculate_consistency(
        route=route,
        cited_evidence=cited_evidence,
    )

    raw_score = (
        0.30 * coverage
        + 0.25 * retrieval_support
        + 0.10 * evidence_quality
        + 0.25 * grounding
        + 0.10 * consistency
    )

    applied_caps: list[str] = []

    if not validation.valid:
        final_score = 0.0
        applied_caps.append(
            "Citation or answer-structure validation failed."
        )
    else:
        final_score = min(
            raw_score,
            MAXIMUM_PROTOTYPE_CONFIDENCE,
        )

        if raw_score > MAXIMUM_PROTOTYPE_CONFIDENCE:
            applied_caps.append(
                "Prototype scores are capped until calibration."
            )

    return ConfidenceBreakdown(
        coverage=round(coverage, 4),
        retrieval_support=round(
            retrieval_support,
            4,
        ),
        evidence_quality=round(
            evidence_quality,
            4,
        ),
        grounding=round(
            grounding,
            4,
        ),
        consistency=round(
            consistency,
            4,
        ),
        raw_score=round(
            raw_score,
            4,
        ),
        final_score=round(
            final_score,
            4,
        ),
        applied_caps=applied_caps,
    )


def build_final_answer(
    session_id: str,
    question: str,
    draft: AgentDraft,
    validation: ValidationReport,
    confidence: ConfidenceBreakdown,
    evidence_registry: dict[str, EvidenceItem],
    diagnostics: dict | None,
) -> ClinicalAnswer:
    """Build the final response after citation and confidence checks.

    Args:
        session_id (str): Conversation session identifier.
        question (str): Original user question.
        draft (AgentDraft): Structured model draft.
        validation (ValidationReport): Citation validation results.
        confidence (ConfidenceBreakdown): Retrieval-focused confidence score.
        evidence_registry (dict[str, EvidenceItem]): Request-local evidence.
        diagnostics (dict | None): Optional retrieval diagnostics.

    Returns:
        ClinicalAnswer: Frontend-ready grounded response.
    """
    should_abstain = (
        draft.abstained
        or not validation.valid
        or confidence.final_score
        < ANSWER_THRESHOLD
    )

    cited_ids = _ordered_cited_ids(
        draft=draft,
        evidence_registry=evidence_registry,
    )

    citations = [
        _build_citation(
            evidence_registry[evidence_id]
        )
        for evidence_id in cited_ids
    ]

    limitations = list(
        draft.limitations
    )

    if any(
        evidence_registry[evidence_id]
        .requires_visual_check
        for evidence_id in cited_ids
    ):
        limitations.append(
            "At least one cited extraction carries a "
            "visual-review warning."
        )

    if any(
        evidence_registry[evidence_id]
        .content_type
        == "figure"
        for evidence_id in cited_ids
    ):
        limitations.append(
            "At least one cited source uses stored figure OCR."
        )

    limitations.append(
        "Confidence is heuristic and is not yet "
        "statistically calibrated."
    )

    limitations = list(
        dict.fromkeys(limitations)
    )

    if should_abstain:
        abstention_reason = (
            draft.abstention_reason
            or (
                "; ".join(validation.errors)
                if validation.errors
                else (
                    "Retrieved evidence and citations did not "
                    "reach the answer threshold."
                )
            )
        )

        answer = (
            "I could not find sufficiently supported evidence "
            "in the provided documents to answer this question."
        )
        claims = []
        confidence_label = "insufficient"
    else:
        abstention_reason = None
        answer = draft.answer
        claims = draft.claims

        confidence_label = (
            "high"
            if confidence.final_score >= 0.85
            else "medium"
        )

    return ClinicalAnswer(
        session_id=session_id,
        question=question,
        answer=answer,
        claims=claims,
        citations=citations,
        abstained=should_abstain,
        abstention_reason=abstention_reason,
        confidence=confidence.final_score,
        confidence_label=confidence_label,
        confidence_breakdown=confidence,
        limitations=limitations,
        diagnostics=diagnostics,
    )


def _collect_cited_evidence(
    draft: AgentDraft,
    evidence_registry: dict[str, EvidenceItem],
) -> list[EvidenceItem]:
    """Collect unique evidence cited by the answer.

    Args:
        draft (AgentDraft): Structured agent draft.
        evidence_registry (dict[str, EvidenceItem]): Request-local evidence.

    Returns:
        list[EvidenceItem]: Unique cited evidence records.
    """
    return [
        evidence_registry[evidence_id]
        for evidence_id in _ordered_cited_ids(
            draft=draft,
            evidence_registry=evidence_registry,
        )
    ]


def _ordered_cited_ids(
    draft: AgentDraft,
    evidence_registry: dict[str, EvidenceItem],
) -> list[str]:
    """Return valid citation IDs in stable first-seen order.

    Args:
        draft (AgentDraft): Structured agent draft.
        evidence_registry (dict[str, EvidenceItem]): Request-local evidence.

    Returns:
        list[str]: Ordered valid evidence IDs.
    """
    ordered_ids: list[str] = []

    candidate_ids = list(
        draft.cited_evidence_ids
    )

    for claim in draft.claims:
        candidate_ids.extend(
            claim.evidence_ids
        )

    for evidence_id in candidate_ids:
        if (
            evidence_id in evidence_registry
            and evidence_id not in ordered_ids
        ):
            ordered_ids.append(
                evidence_id
            )

    return ordered_ids


def _calculate_retrieval_support(
    route: RoutePlan,
    cited_evidence: list[EvidenceItem],
    evidence_registry: dict[str, EvidenceItem],
) -> float:
    """Measure rank strength, retriever agreement and document matching.

    Args:
        route (RoutePlan): Retrieval route.
        cited_evidence (list[EvidenceItem]): Evidence cited by the answer.
        evidence_registry (dict[str, EvidenceItem]): All retrieved evidence.

    Returns:
        float: Retrieval support from zero to one.
    """
    if not cited_evidence:
        return 0.0

    maximum_fusion = max(
        (
            item.fusion_score
            for item in evidence_registry.values()
        ),
        default=1.0,
    )

    if maximum_fusion <= 0.0:
        maximum_fusion = 1.0

    rank_strength = sum(
        min(
            item.fusion_score
            / maximum_fusion,
            1.0,
        )
        for item in cited_evidence
    ) / len(cited_evidence)

    retriever_agreement = sum(
        min(
            max(
                len(item.retrieval_channels),
                1,
            )
            / 2.0,
            1.0,
        )
        for item in cited_evidence
    ) / len(cited_evidence)

    if route.named_document_ids:
        cited_document_ids = {
            item.document_id
            for item in cited_evidence
        }

        matched_documents = sum(
            document_id in cited_document_ids
            for document_id
            in route.named_document_ids
        )

        document_match = (
            matched_documents
            / len(route.named_document_ids)
        )
    else:
        document_match = 1.0

    return (
        0.40 * rank_strength
        + 0.35 * retriever_agreement
        + 0.25 * document_match
    )


def _calculate_consistency(
    route: RoutePlan,
    cited_evidence: list[EvidenceItem],
) -> float:
    """Measure whether the cited evidence forms a coherent chain.

    This is a graded signal rather than a blocking hook. Cross-document answers
    receive full consistency when they cite multiple documents and an explicit
    reference record. Other grounded answers receive full consistency.

    Args:
        route (RoutePlan): Retrieval route.
        cited_evidence (list[EvidenceItem]): Evidence cited by the answer.

    Returns:
        float: Consistency score from zero to one.
    """
    if not cited_evidence:
        return 0.0

    if not route.requires_cross_document:
        return 1.0

    cited_documents = {
        item.document_id
        for item in cited_evidence
    }

    has_reference = any(
        item.content_type
        == "document_reference"
        for item in cited_evidence
    )

    if (
        len(cited_documents) >= 2
        and has_reference
    ):
        return 1.0

    if len(cited_documents) >= 2:
        return 0.75

    return 0.50


def _build_citation(
    evidence: EvidenceItem,
) -> Citation:
    """Convert internal evidence into a frontend citation.

    Args:
        evidence (EvidenceItem): Cited evidence record.

    Returns:
        Citation: Frontend-ready citation.
    """
    excerpt = " ".join(
        evidence.text.split()
    )

    if len(excerpt) > 320:
        excerpt = (
            excerpt[:317]
            + "..."
        )

    return Citation(
        evidence_id=evidence.evidence_id,
        document_title=(
            evidence.document_title
        ),
        page_numbers=(
            evidence.page_numbers
        ),
        content_type=(
            evidence.content_type
        ),
        section=evidence.section,
        source_refs=evidence.source_refs,
        excerpt=excerpt,
        asset_path=evidence.asset_path,
        source_members=(
            evidence.metadata.get(
                "source_members",
                [],
            )
        ),
    )