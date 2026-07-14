from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

"""Define structured models used by retrieval and the clinical QA agent."""

"""
Contains every structured object passed between retrieval, hooks, the LLM and the API.
The LLM returns AgentDraft. It does not return ClinicalAnswer because it is not allowed to determine confidence."""

class StrictModel(BaseModel):
    """Reject unexpected fields in structured assistant objects."""

    model_config = ConfigDict(extra="forbid")


class RoutePlan(StrictModel):
    """Describe which deterministic retrieval paths a question requires."""

    named_document_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs explicitly identified in the question.",
    )
    named_document_titles: list[str] = Field(
        default_factory=list,
        description="Document titles explicitly identified in the question.",
    )
    required_facts: list[str] = Field(
        default_factory=list,
        description="Facts that must be supported before answering.",
    )
    requires_metadata: bool = False
    requires_structured_table: bool = False
    requires_figure: bool = False
    requires_cross_document: bool = False
    requires_corpus_aggregation: bool = False


class EvidenceItem(StrictModel):
    """Represent one canonical piece of evidence supplied to the agent."""

    evidence_id: str
    document_id: str
    document_title: str
    content_type: str
    section: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    text: str
    parent_id: str | None = None
    asset_path: str | None = None
    ingestion_quality: str = "unknown"
    extraction_quality: float = Field(ge=0.0, le=1.0)
    requires_visual_check: bool = False
    retrieval_channels: list[str] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    fusion_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(StrictModel):
    """Record how the deterministic controller built an evidence pack."""

    route: RoutePlan
    semantic_result_count: int = 0
    keyword_result_count: int = 0
    structured_result_count: int = 0
    figure_result_count: int = 0
    reference_result_count: int = 0
    aggregation_result_count: int = 0
    selected_evidence_ids: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class EvidencePack(StrictModel):
    """Collect selected evidence and its retrieval trace."""

    question: str
    route: RoutePlan
    evidence: list[EvidenceItem]
    trace: RetrievalTrace


class AgentClaim(StrictModel):
    """Represent one independently supported claim in an answer."""

    text: str = Field(
        description="One factual claim written for the user."
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs directly supporting this claim.",
    )


class AgentDraft(StrictModel):
    """Represent the answer produced by the LLM before validation."""

    answer: str = Field(
        description="Concise answer grounded only in supplied evidence."
    )
    claims: list[AgentClaim] = Field(
        default_factory=list,
        description="Atomic factual claims and their supporting evidence IDs.",
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="All evidence IDs cited in the answer.",
    )
    abstained: bool = Field(
        description="Whether the documents are insufficient to answer safely."
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Reason for abstaining when abstained is true.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Relevant extraction or evidence limitations.",
    )


class ValidationReport(StrictModel):
    """Store deterministic post-generation validation results."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    citation_validity: float = Field(ge=0.0, le=1.0)
    claim_citation_coverage: float = Field(ge=0.0, le=1.0)
    numeric_support: float = Field(ge=0.0, le=1.0)


class ConfidenceBreakdown(StrictModel):
    """Explain how the system reliability score was calculated."""

    coverage: float = Field(ge=0.0, le=1.0)
    retrieval_support: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    applied_caps: list[str] = Field(default_factory=list)


class Citation(StrictModel):
    """Represent one source citation returned to the frontend."""

    evidence_id: str
    document_title: str
    page_numbers: list[int] = Field(default_factory=list)
    content_type: str
    section: str = ""
    source_refs: list[str] = Field(default_factory=list)
    excerpt: str
    asset_path: str | None = None
    source_members: list[dict[str, Any]] = Field(default_factory=list)


class ClinicalAnswer(StrictModel):
    """Represent the final validated response returned by Modal."""

    session_id: str
    question: str
    answer: str
    claims: list[AgentClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool
    abstention_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str
    confidence_status: str = "heuristic_not_yet_calibrated"
    confidence_breakdown: ConfidenceBreakdown
    limitations: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None