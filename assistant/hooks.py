from __future__ import annotations

import re

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


def pre_answer_check(
    evidence_pack: EvidencePack,
) -> list[str]:
    """Check whether retrieval found minimally usable evidence.

    Args:
        evidence_pack (EvidencePack): Initial retrieval result.

    Returns:
        list[str]: Blocking errors. An empty list means the LLM may run.
    """
    errors: list[str] = []
    route = evidence_pack.route
    evidence = evidence_pack.evidence

    if not evidence:
        return [
            "No evidence was found in the document collection."
        ]

    if route.requires_cross_document:
        references = [
            item
            for item in evidence
            if item.content_type == "document_reference"
        ]

        target_document_ids = {
            item.metadata.get("target_document_id")
            for item in references
            if item.metadata.get("target_document_id")
        }

        target_evidence_found = any(
            item.document_id in target_document_ids
            for item in evidence
        )

        if not references:
            errors.append(
                "The question requires a document link, but no "
                "explicit reference was retrieved."
            )
        elif not target_evidence_found:
            errors.append(
                "The source reference was found, but no evidence "
                "was retrieved from its target document."
            )

    if route.requires_figure:
        has_figure_evidence = any(
            item.content_type == "figure"
            for item in evidence
        )

        if not has_figure_evidence:
            errors.append(
                "The question requires figure evidence, but no "
                "figure record was retrieved."
            )

    if route.requires_corpus_aggregation:
        has_aggregation = any(
            item.content_type == "corpus_aggregation"
            for item in evidence
        )

        if not has_aggregation:
            errors.append(
                "The question asks for a complete corpus-wide set, "
                "but no structured aggregation was produced."
            )

    return errors


def validate_agent_draft(
    draft: AgentDraft,
    evidence_registry: dict[str, EvidenceItem],
) -> ValidationReport:
    """Validate citations, claim coverage and numeric support.

    Args:
        draft (AgentDraft): Structured LLM output.
        evidence_registry (dict[str, EvidenceItem]): Retrieved evidence.

    Returns:
        ValidationReport: Deterministic validation result.
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
        errors.append("The answer is empty.")

    if not draft.claims:
        errors.append(
            "A non-abstained answer must contain atomic claims."
        )

    all_cited_ids = set(
        draft.cited_evidence_ids
    )

    for claim in draft.claims:
        all_cited_ids.update(claim.evidence_ids)

    valid_citation_ids = {
        evidence_id
        for evidence_id in all_cited_ids
        if evidence_id in evidence_registry
    }

    invalid_ids = sorted(
        all_cited_ids - valid_citation_ids
    )

    if invalid_ids:
        errors.append(
            "The answer invented unknown evidence IDs: "
            + ", ".join(invalid_ids)
        )

    citation_validity = (
        len(valid_citation_ids) / len(all_cited_ids)
        if all_cited_ids
        else 0.0
    )

    claims_with_valid_citations = 0
    supported_numeric_tokens = 0
    total_numeric_tokens = 0

    for claim in draft.claims:
        valid_claim_ids = [
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_registry
        ]

        if not valid_claim_ids:
            errors.append(
                f"Claim has no valid citation: {claim.text}"
            )
            continue

        claims_with_valid_citations += 1

        cited_text = " ".join(
            evidence_registry[evidence_id].text
            for evidence_id in valid_claim_ids
        )

        claim_numbers = _extract_numeric_tokens(
            claim.text
        )
        evidence_numbers = _extract_numeric_tokens(
            cited_text
        )

        for number in claim_numbers:
            total_numeric_tokens += 1

            if number in evidence_numbers:
                supported_numeric_tokens += 1
            else:
                errors.append(
                    "Numeric claim is absent from its cited "
                    f"evidence: {number}"
                )

    claim_citation_coverage = (
        claims_with_valid_citations / len(draft.claims)
        if draft.claims
        else 0.0
    )

    numeric_support = (
        supported_numeric_tokens / total_numeric_tokens
        if total_numeric_tokens
        else 1.0
    )

    return ValidationReport(
        valid=not errors,
        errors=errors,
        citation_validity=citation_validity,
        claim_citation_coverage=(
            claim_citation_coverage
        ),
        numeric_support=numeric_support,
    )


def calculate_confidence(
    route: RoutePlan,
    draft: AgentDraft,
    validation: ValidationReport,
    evidence_registry: dict[str, EvidenceItem],
) -> ConfidenceBreakdown:
    """Calculate a transparent heuristic reliability score.

    Args:
        route (RoutePlan): Deterministic query route.
        draft (AgentDraft): Structured LLM response.
        validation (ValidationReport): Output validation results.
        evidence_registry (dict[str, EvidenceItem]): All retrieved evidence.

    Returns:
        ConfidenceBreakdown: Score components, caps and final score.
    """
    cited_ids = {
        evidence_id
        for evidence_id in draft.cited_evidence_ids
        if evidence_id in evidence_registry
    }

    for claim in draft.claims:
        cited_ids.update(
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_registry
        )

    cited_evidence = [
        evidence_registry[evidence_id]
        for evidence_id in cited_ids
    ]

    coverage = _calculate_coverage(
        route=route,
        draft=draft,
        cited_evidence=cited_evidence,
    )

    retrieval_support = _calculate_retrieval_support(
        route=route,
        cited_evidence=cited_evidence,
        evidence_registry=evidence_registry,
    )

    evidence_quality = (
        min(
            item.extraction_quality
            for item in cited_evidence
        )
        if cited_evidence
        else 0.0
    )

    grounding = (
        0.35 * validation.citation_validity
        + 0.35 * validation.claim_citation_coverage
        + 0.30 * validation.numeric_support
    )

    consistency = _calculate_consistency(
        route=route,
        cited_evidence=cited_evidence,
    )

    raw_score = (
        0.30 * coverage
        + 0.20 * retrieval_support
        + 0.20 * evidence_quality
        + 0.20 * grounding
        + 0.10 * consistency
    )

    applied_caps: list[str] = []
    final_score = raw_score

    if not validation.valid:
        final_score = 0.0
        applied_caps.append(
            "Output validation failed."
        )

    if any(
        item.requires_visual_check
        for item in cited_evidence
    ):
        final_score = min(final_score, 0.60)
        applied_caps.append(
            "Critical evidence requires visual review."
        )

    if (
        cited_evidence
        and all(
            item.content_type == "figure"
            for item in cited_evidence
        )
    ):
        final_score = min(final_score, 0.80)
        applied_caps.append(
            "The answer relies only on figure OCR."
        )

    if route.requires_cross_document:
        cited_documents = {
            item.document_id
            for item in cited_evidence
        }

        has_reference = any(
            item.content_type == "document_reference"
            for item in cited_evidence
        )

        if (
            len(cited_documents) < 2
            or not has_reference
        ):
            final_score = 0.0
            applied_caps.append(
                "Cross-document evidence chain is incomplete."
            )

    if coverage < 1.0:
        final_score = min(final_score, 0.60)
        applied_caps.append(
            "At least one required fact is unsupported."
        )

    final_score = min(final_score, 0.95)

    if final_score == 0.95 and raw_score > 0.95:
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
        grounding=round(grounding, 4),
        consistency=round(consistency, 4),
        raw_score=round(raw_score, 4),
        final_score=round(final_score, 4),
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
    """Build the final API response after confidence gating.

    Args:
        session_id (str): Conversation session identifier.
        question (str): Original user question.
        draft (AgentDraft): LLM-generated structured draft.
        validation (ValidationReport): Grounding validation.
        confidence (ConfidenceBreakdown): Reliability calculation.
        evidence_registry (dict[str, EvidenceItem]): Retrieved evidence.
        diagnostics (dict | None): Optional retrieval diagnostics.

    Returns:
        ClinicalAnswer: Validated frontend-ready answer.
    """
    should_abstain = (
        draft.abstained
        or not validation.valid
        or confidence.final_score < ANSWER_THRESHOLD
    )

    cited_ids: list[str] = []

    for evidence_id in draft.cited_evidence_ids:
        if (
            evidence_id in evidence_registry
            and evidence_id not in cited_ids
        ):
            cited_ids.append(evidence_id)

    for claim in draft.claims:
        for evidence_id in claim.evidence_ids:
            if (
                evidence_id in evidence_registry
                and evidence_id not in cited_ids
            ):
                cited_ids.append(evidence_id)

    citations = [
        _build_citation(
            evidence_registry[evidence_id]
        )
        for evidence_id in cited_ids
    ]

    limitations = list(draft.limitations)

    if any(
        item.content_type == "figure"
        for item in (
            evidence_registry[evidence_id]
            for evidence_id in cited_ids
        )
    ):
        limitations.append(
            "At least one cited source uses figure OCR."
        )

    if any(
        evidence_registry[evidence_id]
        .requires_visual_check
        for evidence_id in cited_ids
    ):
        limitations.append(
            "At least one cited extraction requires visual review."
        )

    limitations.append(
        "Confidence is a heuristic evidence-reliability "
        "score and is not yet statistically calibrated."
    )

    limitations = list(dict.fromkeys(limitations))

    if should_abstain:
        abstention_reason = (
            draft.abstention_reason
            or (
                "; ".join(validation.errors)
                if validation.errors
                else "Evidence reliability was below the answer threshold."
            )
        )

        answer = (
            "I could not find sufficiently reliable evidence "
            "in the provided documents to answer this question."
        )
        claims = []
        label = "insufficient"
    else:
        abstention_reason = None
        answer = draft.answer
        claims = draft.claims
        label = (
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
        confidence_label=label,
        confidence_breakdown=confidence,
        limitations=limitations,
        diagnostics=diagnostics,
    )


def _calculate_coverage(
    route: RoutePlan,
    draft: AgentDraft,
    cited_evidence: list[EvidenceItem],
) -> float:
    """Measure whether every requested fact is supported.

    Args:
        route (RoutePlan): Query route.
        draft (AgentDraft): Agent answer.
        cited_evidence (list[EvidenceItem]): Cited evidence.

    Returns:
        float: Required-fact coverage from zero to one.
    """
    if draft.abstained:
        return 0.0

    answer = draft.answer.casefold()
    evidence_text = " ".join(
        item.text
        for item in cited_evidence
    ).casefold()

    cited_types = {
        item.content_type
        for item in cited_evidence
    }
    cited_documents = {
        item.document_id
        for item in cited_evidence
    }

    support: list[bool] = []

    for fact in route.required_facts:
        if fact == "classification_code":
            support.append(
                bool(
                    re.search(
                        r"\bCMX-[A-Z0-9]+\b",
                        draft.answer,
                        flags=re.IGNORECASE,
                    )
                )
            )
        elif fact == "registry_code":
            support.append(
                bool(
                    re.search(
                        r"\bCDR-\d+\b",
                        draft.answer,
                        flags=re.IGNORECASE,
                    )
                )
            )
        elif fact == "induction_dose":
            support.append(
                "induction" in evidence_text
                and bool(
                    re.search(
                        r"\b\d+(?:\.\d+)?\s*"
                        r"(?:mg|mcg|g)\b",
                        answer,
                    )
                )
            )
        elif fact == "maintenance_dose":
            support.append(
                "maintenance" in evidence_text
                and bool(
                    re.search(
                        r"\b\d+(?:\.\d+)?\s*"
                        r"(?:mg|mcg|g)\b",
                        answer,
                    )
                )
            )
        elif fact == "figure_value":
            support.append(
                "figure" in cited_types
                and bool(re.search(r"\d", answer))
            )
        elif fact == "cross_document_link":
            support.append(
                "document_reference" in cited_types
                and len(cited_documents) >= 2
            )
        elif fact == "complete_condition_set":
            support.append(
                "corpus_aggregation" in cited_types
            )
        else:
            support.append(
                bool(draft.claims)
                and bool(cited_evidence)
            )

    return (
        sum(support) / len(support)
        if support
        else 0.0
    )


def _calculate_retrieval_support(
    route: RoutePlan,
    cited_evidence: list[EvidenceItem],
    evidence_registry: dict[str, EvidenceItem],
) -> float:
    """Measure ranking, retriever agreement and document matching.

    Args:
        route (RoutePlan): Query route.
        cited_evidence (list[EvidenceItem]): Cited items.
        evidence_registry (dict[str, EvidenceItem]): All retrieved items.

    Returns:
        float: Retrieval-support score.
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

    rank_strength = sum(
        min(
            item.fusion_score / maximum_fusion,
            1.0,
        )
        for item in cited_evidence
    ) / len(cited_evidence)

    agreement = sum(
        min(
            len(item.retrieval_channels) / 2.0,
            1.0,
        )
        for item in cited_evidence
    ) / len(cited_evidence)

    if route.named_document_ids:
        retrieved_documents = {
            item.document_id
            for item in cited_evidence
        }

        matched = sum(
            1
            for document_id in route.named_document_ids
            if document_id in retrieved_documents
        )

        entity_match = (
            matched / len(route.named_document_ids)
        )
    else:
        entity_match = 1.0

    return (
        0.40 * rank_strength
        + 0.35 * agreement
        + 0.25 * entity_match
    )


def _calculate_consistency(
    route: RoutePlan,
    cited_evidence: list[EvidenceItem],
) -> float:
    """Calculate a simple evidence-consistency score.

    Args:
        route (RoutePlan): Query route.
        cited_evidence (list[EvidenceItem]): Cited evidence.

    Returns:
        float: Consistency score.
    """
    if not cited_evidence:
        return 0.0

    if route.requires_cross_document:
        has_reference = any(
            item.content_type == "document_reference"
            for item in cited_evidence
        )
        document_count = len({
            item.document_id
            for item in cited_evidence
        })

        if not has_reference or document_count < 2:
            return 0.0

    if any(
        item.requires_visual_check
        for item in cited_evidence
    ):
        return 0.60

    return 1.0


def _extract_numeric_tokens(
    text: str,
) -> set[str]:
    """Extract normalized numbers, codes and dose values.

    Args:
        text (str): Claim or evidence text.

    Returns:
        set[str]: Normalized numeric tokens.
    """
    matches = re.findall(
        r"\b(?:CDR-\d+|CMX-[A-Z0-9]+|"
        r"\d+(?:[.,]\d+)?(?:\s*(?:mg|mcg|g|%))?)\b",
        text,
        flags=re.IGNORECASE,
    )

    return {
        re.sub(r"[\s,]+", "", match.casefold())
        for match in matches
    }


def _build_citation(
    evidence: EvidenceItem,
) -> Citation:
    """Convert internal evidence into a frontend citation.

    Args:
        evidence (EvidenceItem): Cited evidence.

    Returns:
        Citation: Frontend-ready citation.
    """
    excerpt = " ".join(
        evidence.text.split()
    )

    if len(excerpt) > 320:
        excerpt = excerpt[:317] + "..."

    return Citation(
        evidence_id=evidence.evidence_id,
        document_title=evidence.document_title,
        page_numbers=evidence.page_numbers,
        content_type=evidence.content_type,
        section=evidence.section,
        source_refs=evidence.source_refs,
        excerpt=excerpt,
        asset_path=evidence.asset_path,
        source_members=evidence.metadata.get(
            "source_members",
            [],
        ),
    )