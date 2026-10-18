"""Provenance engine facade in memory layer.

Re-exports ProvenanceEngine and models from carethread.provenance.
"""

from carethread.provenance.engine import ProvenanceEngine
from carethread.provenance.models import (
    ExplanationResult,
    ProvenanceCitation,
    ProvenanceClaim,
    ProvenanceTrace,
    ReasoningEdge,
    RejectedCitation,
    WhyNotExplanation,
)

__all__ = [
    "ExplanationResult",
    "ProvenanceCitation",
    "ProvenanceClaim",
    "ProvenanceEngine",
    "ProvenanceTrace",
    "ReasoningEdge",
    "RejectedCitation",
    "WhyNotExplanation",
]
