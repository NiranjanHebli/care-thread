"""CareThread Clinical Agents package.

Contains:
- LLMClient: Provider-neutral inference interface (mock/replay/live).
- BaseClinicalAgent: Abstract agent class with tool dispatch capabilities.
- IntakeAgent: Gathers observations, updates semantic memory, delegates to specialist.
- SpecialistAgent: Diagnostic evaluation and proposed medication delegation.
- PharmacyAgent: Cross-graph allergy screening and provenance commitment.
- ClinicalWorkflowRunner: Full Intake -> Specialist -> Pharmacy multi-agent coordinator.
"""

from carethread.agents.base import BaseClinicalAgent
from carethread.agents.intake import IntakeAgent
from carethread.agents.llm import LLMClient, LLMResponse
from carethread.agents.pharmacy import PharmacyAgent
from carethread.agents.specialist import SpecialistAgent
from carethread.agents.workflow import ClinicalWorkflowRunner, WorkflowResult

__all__ = [
    "BaseClinicalAgent",
    "ClinicalWorkflowRunner",
    "IntakeAgent",
    "LLMClient",
    "LLMResponse",
    "PharmacyAgent",
    "SpecialistAgent",
    "WorkflowResult",
]
