"""Specialist clinical agent for diagnostic evaluation and prescription delegation."""

from __future__ import annotations

import logging
from typing import Any

from carethread.agents.base import BaseClinicalAgent
from carethread.agents.llm import LLMClient
from carethread.errors import ValidationError
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

DEFAULT_SPECIALIST_AGENT_ID: str = "agt_specialist_01"
DEFAULT_SPECIALIST_TOKEN: str = "tok_specialist_01"


class SpecialistAgent(BaseClinicalAgent):
    """Clinical Specialist Agent evaluating symptoms and delegating prescriptions to Pharmacy."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        agent_id: str = DEFAULT_SPECIALIST_AGENT_ID,
        token: str = DEFAULT_SPECIALIST_TOKEN,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role="specialist",
            token=token,
            dispatcher=dispatcher,
            llm=llm,
        )

    def process_handoff(
        self,
        patient_id: str,
        handoff_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Accepts an intake delegation, records diagnosis, and delegates proposed Rx to pharmacy.

        Steps executed:
        1. Identifies open handoff for specialist role (from argument or query).
        2. Atomically accepts the handoff via CAS query.
        3. Recalls patient history and carried memory facts.
        4. Synthesizes clinical assessment and proposed medication using LLM.
        5. Persists specialist evaluation note and proposed medication fact.
        6. Transitions accepted handoff to completed status.
        7. Dispatches new coordination handoff to 'pharmacy'.

        Returns:
            Dictionary with accepted_handoff_id, pharmacy_handoff_id, and assessment details.
        """
        hid = handoff_id
        if not hid:
            h_list = self.call_tool("list_handoffs", {"session_id": session_id, "status": "open"})
            handoffs = h_list.get("handoffs", [])
            if not handoffs:
                raise ValidationError("No open handoffs found for specialist role.")
            hid = handoffs[0].get("handoff_id") or handoffs[0].get("id")

        # Accept handoff
        accepted_pkt = self.call_tool(
            "accept_handoff",
            {"session_id": session_id, "handoff_id": hid},
        )
        carried_facts = accepted_pkt.get("carried_facts", [])
        goal = accepted_pkt.get("goal", "Evaluate symptoms")

        # Recall background context
        recall_res = self.call_tool(
            "recall",
            {"session_id": session_id, "query": goal, "k": 5},
        )
        items = recall_res.get("items", [])
        facts_summary = "; ".join(f"{it.get('summary')}" for it in items[:4])

        # LLM Clinical Reasoning
        prompt = (
            f"Clinical Consultation Request: {goal}\n"
            f"Carried Facts: {carried_facts}\n"
            f"Recalled History: {facts_summary}\n"
            f"Formulate differential diagnosis, treatment plan, and prescription."
        )
        llm_res = self.llm.generate(prompt=prompt, role="specialist")
        structured = llm_res.structured or {}

        diagnosis = structured.get("diagnosis", "Clinical Assessment Finalized")
        clinical_note = (
            structured.get("clinical_assessment")
            or f"Specialist diagnostic assessment: {diagnosis}."
        )
        prop_med = structured.get("proposed_medication") or {
            "kind": "medication",
            "display": "Amoxicillin 500mg",
            "code": "amox_500",
            "value": "500mg PO TID x 10 days",
            "authority": "agent_inferred",
        }

        # Persist specialist evaluation note
        self.call_tool(
            "remember",
            {
                "session_id": session_id,
                "kind": "note",
                "payload": {"text": clinical_note, "kind": "specialist_assessment"},
            },
        )

        # Persist proposed medication fact
        med_res = self.call_tool(
            "remember",
            {
                "session_id": session_id,
                "kind": "fact",
                "payload": prop_med,
            },
        )
        med_fid = med_res["fact_id"]

        # Complete intake handoff
        self.call_tool(
            "complete_handoff",
            {
                "session_id": session_id,
                "handoff_id": hid,
                "result_summary": f"Diagnosed {diagnosis}. Proposed {prop_med.get('display')}.",
            },
        )

        # Create handoff to Pharmacy
        pharmacy_goal = (
            structured.get("pharmacy_goal")
            or f"Review proposed {prop_med.get('display')} for allergies and interactions"
        )
        state_summary = f"Diagnosis: {diagnosis}. Prescribed: {prop_med.get('display')}."
        questions = structured.get("questions") or [
            f"Screen {prop_med.get('display')} against allergy profile"
        ]

        ho_res = self.call_tool(
            "handoff",
            {
                "session_id": session_id,
                "to_role": "pharmacy",
                "goal": pharmacy_goal,
                "state_summary": state_summary,
                "completed_actions": [
                    {"type": "specialist_assessment", "description": f"Diagnosed {diagnosis}"}
                ],
                "open_questions": [{"text": q} for q in questions],
                "memory_ids": [{"label": "Fact", "id": med_fid, "rel": "proposed_rx"}],
            },
        )

        return {
            "accepted_handoff_id": hid,
            "pharmacy_handoff_id": ho_res["handoff_id"],
            "diagnosis": diagnosis,
            "proposed_medication_id": med_fid,
            "assessment": clinical_note,
        }
