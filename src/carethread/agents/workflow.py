"""Multi-agent clinical workflow orchestrator coordinating Intake, Specialist, and Pharmacy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from carethread.agents.intake import IntakeAgent
from carethread.agents.llm import LLMClient
from carethread.agents.pharmacy import PharmacyAgent
from carethread.agents.specialist import SpecialistAgent
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Consolidated outcome of the multi-agent clinical consultation workflow."""

    patient_id: str
    session_id: str
    intake_handoff_id: str
    specialist_handoff_id: str
    answer_id: str
    diagnosis: str
    is_safe: bool
    recommendation: str
    intake: dict[str, Any] = field(default_factory=dict)
    specialist: dict[str, Any] = field(default_factory=dict)
    pharmacy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "intake_handoff_id": self.intake_handoff_id,
            "specialist_handoff_id": self.specialist_handoff_id,
            "answer_id": self.answer_id,
            "diagnosis": self.diagnosis,
            "is_safe": self.is_safe,
            "recommendation": self.recommendation,
            "intake": self.intake,
            "specialist": self.specialist,
            "pharmacy": self.pharmacy,
        }


class ClinicalWorkflowRunner:
    """Executes the standard multi-agent interaction workflow across clinical roles.

    Sequence:
    1. Intake Agent receives patient complaint, recalls background, extracts facts,
       and delegates case to Specialist Agent via coordination handoff.
    2. Specialist Agent accepts handoff, synthesizes differential diagnosis, records
       assessment and proposed medication, and delegates prescription to Pharmacy Agent.
    3. Pharmacy Agent accepts handoff, executes cross-graph safety recall against
       allergies and shared clinic rules, flags conflicts, and commits verified trace.
    """

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        llm: LLMClient | None = None,
        intake_token: str = "tok_intake_01",
        specialist_token: str = "tok_specialist_01",
        pharmacy_token: str = "tok_pharmacy_01",
    ) -> None:
        shared_llm = llm or LLMClient()
        self.dispatcher = dispatcher
        self.intake_agent = IntakeAgent(dispatcher=dispatcher, token=intake_token, llm=shared_llm)
        self.specialist_agent = SpecialistAgent(
            dispatcher=dispatcher, token=specialist_token, llm=shared_llm
        )
        self.pharmacy_agent = PharmacyAgent(
            dispatcher=dispatcher, token=pharmacy_token, llm=shared_llm
        )

    def run(
        self,
        patient_id: str,
        patient_message: str,
        session_id: str | None = None,
    ) -> WorkflowResult:
        """Executes the full Intake -> Specialist -> Pharmacy clinical consultation workflow.

        Args:
            patient_id: Target patient tenant identifier (e.g. 'p_0123456789ab').
            patient_message: Patient's chief complaint or symptoms.
            session_id: Optional existing session identifier.

        Returns:
            WorkflowResult summarizing transitions, diagnoses, and safety commitments.
        """
        logger.info(f"Starting multi-agent clinical workflow for patient '{patient_id}'")

        # Stage 1: Intake Consultation
        intake_res = self.intake_agent.process_patient_input(
            patient_id=patient_id,
            message=patient_message,
            session_id=session_id,
        )
        sid = intake_res["session_id"]
        ho_intake_id = intake_res["handoff_id"]
        logger.info(f"Stage 1 complete: Intake handoff '{ho_intake_id}' published.")

        # Stage 2: Specialist Evaluation & Diagnosis
        specialist_res = self.specialist_agent.process_handoff(
            patient_id=patient_id,
            handoff_id=ho_intake_id,
            session_id=sid,
        )
        ho_pharmacy_id = specialist_res["pharmacy_handoff_id"]
        diagnosis = specialist_res["diagnosis"]
        logger.info(f"Stage 2 complete: Specialist handoff '{ho_pharmacy_id}' published.")

        # Stage 3: Pharmacy Review & Safety Commitment
        pharmacy_res = self.pharmacy_agent.process_handoff(
            patient_id=patient_id,
            handoff_id=ho_pharmacy_id,
            session_id=sid,
        )
        ans_id = pharmacy_res["answer_id"]
        is_safe = pharmacy_res["is_safe"]
        rec = pharmacy_res["recommendation"]
        logger.info(f"Stage 3 complete: Pharmacy committed answer '{ans_id}' (Safe: {is_safe}).")

        return WorkflowResult(
            patient_id=patient_id,
            session_id=sid,
            intake_handoff_id=ho_intake_id,
            specialist_handoff_id=ho_pharmacy_id,
            answer_id=ans_id,
            diagnosis=diagnosis,
            is_safe=is_safe,
            recommendation=rec,
            intake=intake_res,
            specialist=specialist_res,
            pharmacy=pharmacy_res,
        )
