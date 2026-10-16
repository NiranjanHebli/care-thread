"""Data models for answer provenance, citations, traces, and explainability.

Supports machine-verifiable answer traces with cited memory nodes,
intra-subgraph reasoning edges, rejected candidate tracking,
and counterfactual Why-Not analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProvenanceCitation:
    """Represents a supporting memory node cited by a clinical answer."""

    node_id: str
    label: str
    graph: str
    role: str = "supporting"
    rank: int = 1
    score: float = 1.0
    retrieval_method: str = "graph_traversal"
    summary: str | None = None
    confidence: float | None = None
    effective_confidence: float | None = None
    status: str | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    last_confirmed_at: int | None = None
    age_days: int | None = None
    is_stale: bool = False
    authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts citation to JSON-compatible dictionary."""
        d = asdict(self)
        # Ensure id alias matches node_id for convenience
        d["id"] = self.node_id
        return d


@dataclass
class RejectedCitation:
    """Represents an evaluated candidate node excluded from the final answer."""

    node_id: str
    label: str
    reason_code: str
    reason: str
    score: float = 0.0
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts rejected citation to JSON-compatible dictionary."""
        d = asdict(self)
        d["id"] = self.node_id
        return d


@dataclass
class ReasoningEdge:
    """Represents a graph relationship between cited provenance nodes."""

    type: str
    from_id: str
    to_id: str
    graph: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts reasoning edge to JSON-compatible dictionary."""
        return {
            "type": self.type,
            "from": self.from_id,
            "to": self.to_id,
            "graph": self.graph,
        }


@dataclass
class ProvenanceClaim:
    """Represents an atomic claim asserted within a clinical answer."""

    claim_id: str
    text: str
    supported_by: list[str] = field(default_factory=list)
    support_type: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        """Converts claim to JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class WhyNotExplanation:
    """Represents a counterfactual explanation for why an entity was not chosen."""

    target: dict[str, Any]
    was_candidate: bool
    verdict: str  # BLOCKED, REJECTED, EXCLUDED, NOT_FOUND, CROSS_TENANT_DENIED
    reason_code: str
    reason: str
    blocking_path: list[dict[str, Any]] = field(default_factory=list)
    superseded_by: str | None = None
    score: float | None = None
    clinician_review_required: bool = False
    counterfactual_recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts Why-Not explanation to JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class ProvenanceTrace:
    """Machine-verifiable trace recording how an answer was derived."""

    trace_id: str
    answer_id: str
    patient_id: str
    session_id: str | None
    agent_id: str
    generated_at: int
    question: str
    answer_text: str
    answer_type: str = "informational"
    clinician_review_required: bool = False
    overall_confidence: float = 1.0
    claims: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)
    used: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""
    verified: bool = True
    queries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Converts trace to JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class ExplanationResult:
    """Consolidated explanation containing trace, Why-Not analysis, or ablation."""

    trace: ProvenanceTrace
    why_not: WhyNotExplanation | None = None
    ablated: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts explanation result to JSON-compatible dictionary."""
        res = self.trace.to_dict()
        if self.why_not:
            res["why_not"] = self.why_not.to_dict()
        if self.ablated:
            res["ablated"] = self.ablated
        return res
