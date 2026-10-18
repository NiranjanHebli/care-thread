"""Intake clinical agent for patient onboarding, observation, and specialist delegation."""

from __future__ import annotations

import logging
from typing import Any

from carethread.agents.base import BaseClinicalAgent
from carethread.agents.llm import LLMClient
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

DEFAULT_INTAKE_AGENT_ID: str = "agt_intake_01"
DEFAULT_INTAKE_TOKEN: str = "tok_intake_01"


class IntakeAgent(BaseClinicalAgent):
    """Clinical Intake Agent handling initial encounter observations and triage handoffs."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        agent_id: str = DEFAULT_INTAKE_AGENT_ID,
        token: str = DEFAULT_INTAKE_TOKEN,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role="intake",
            token=token,
            dispatcher=dispatcher,
            llm=llm,
        )

    def process_patient_input(
        self,
        patient_id: str,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Processes patient complaint, updates semantic memory, and delegatesto specialist.

        Steps executed:
        1. Initiates or binds to active Session.
        2. Appends user message to session transcript.
        3. Recalls relevant active facts and background notes.
        4. Extracts clinical facts using LLM reasoning.
        5. Writes semantic facts and encounter observation note.
        6. Constructs and dispatches coordination handoff to 'specialist'.
        7. Appends agent acknowledgement to session transcript.

        Returns:
            Dictionary containing session_id, handoff_id, remembered fact_ids, and summary.
        """
        sid = session_id
        if not sid:
            s_data = self.call_tool("start_session", {"patient_id": patient_id})
            sid = s_data["session_id"]

        # Append patient message
        self.call_tool(
            "append_message",
            {"session_id": sid, "role": "user", "content": message},
        )

        # Contextual recall
        recall_res = self.call_tool(
            "recall",
            {"session_id": sid, "query": message, "k": 5},
        )
        items = recall_res.get("items", [])
        context_str = ", ".join(f"{it.get('label')}: {it.get('summary')}" for it in items[:3])

        # LLM fact extraction and reasoning
        prompt = (
            f"Patient Input: {message}\n"
            f"Prior Context: {context_str or 'None'}\n"
            f"Extract clinical symptoms, severity, and recommend next actions."
        )
        llm_res = self.llm.generate(prompt=prompt, role="intake")
        structured = llm_res.structured or {}

        # Persist extracted facts
        extracted_facts = structured.get("facts") or [
            {
                "kind": "symptom",
                "display": message,
                "value": "patient_reported",
                "authority": "patient_reported",
            }
        ]
        remembered_fact_ids: list[str] = []
        for fact in extracted_facts:
            f_res = self.call_tool(
                "remember",
                {"session_id": sid, "kind": "fact", "payload": fact},
            )
            remembered_fact_ids.append(f_res["fact_id"])

        # Persist intake note
        note_text = structured.get("note") or f"Patient reported: {message}"
        note_res = self.call_tool(
            "remember",
            {
                "session_id": sid,
                "kind": "note",
                "payload": {"text": note_text, "kind": "intake_note"},
            },
        )

        # Prepare handoff memory citations
        memory_ids = [
            {"label": "Fact", "id": fid, "rel": "supporting"} for fid in remembered_fact_ids
        ]

        # Construct and dispatch handoff to specialist
        goal = structured.get("goal") or "Evaluate clinical presentation and provide workup"
        state_summary = structured.get("summary") or f"Intake completed. Symptoms: {message}"
        questions = structured.get("questions") or ["Evaluate symptom etiology and severity"]

        ho_res = self.call_tool(
            "handoff",
            {
                "session_id": sid,
                "to_role": "specialist",
                "goal": goal,
                "state_summary": state_summary,
                "completed_actions": [
                    {"type": "intake_interview", "description": "Recorded patient complaint"}
                ],
                "open_questions": [{"text": q} for q in questions],
                "memory_ids": memory_ids,
            },
        )

        # Append agent response
        self.call_tool(
            "append_message",
            {"session_id": sid, "role": "agent", "content": llm_res.text},
        )

        return {
            "session_id": sid,
            "handoff_id": ho_res["handoff_id"],
            "fact_ids": remembered_fact_ids,
            "note_id": note_res.get("note_id"),
            "summary": llm_res.text,
        }
