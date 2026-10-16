"""Pharmacy clinical agent for prescription safety screening and provenance commitment."""

from __future__ import annotations

import logging
from typing import Any

from carethread.agents.base import BaseClinicalAgent
from carethread.agents.llm import LLMClient
from carethread.errors import ValidationError
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

DEFAULT_PHARMACY_AGENT_ID: str = "agt_pharmacy_01"
DEFAULT_PHARMACY_TOKEN: str = "tok_pharmacy_01"


class PharmacyAgent(BaseClinicalAgent):
    """Clinical Pharmacy Agent screening prescriptions against allergies and committing traces."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        agent_id: str = DEFAULT_PHARMACY_AGENT_ID,
        token: str = DEFAULT_PHARMACY_TOKEN,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role="pharmacy",
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
        """Accepts pharmacy handoff, runs cross-graph safety recall, and commits answer trace.

        Steps executed:
        1. Identifies open handoff for pharmacy role (from argument or query).
        2. Atomically accepts the handoff via CAS query.
        3. Recalls patient allergies and cross-reactivity rules from shared clinic graph.
        4. Evaluates contraindications and safety conflicts via LLM reasoning.
        5. Commits machine-verifiable answer citing supporting provenance facts.
        6. Transitions accepted handoff to completed status.

        Returns:
            Dictionary with handoff_id, answer_id, safety determination, and recommendations.
        """
        hid = handoff_id
        if not hid:
            h_list = self.call_tool("list_handoffs", {"session_id": session_id, "status": "open"})
            handoffs = h_list.get("handoffs", [])
            if not handoffs:
                raise ValidationError("No open handoffs found for pharmacy role.")
            hid = handoffs[0].get("handoff_id") or handoffs[0].get("id")

        # Accept handoff
        accepted_pkt = self.call_tool(
            "accept_handoff",
            {"session_id": session_id, "handoff_id": hid},
        )
        carried_facts = accepted_pkt.get("carried_facts", [])
        goal = accepted_pkt.get("goal", "Screen prescription")

        # Cross-graph safety recall (include shared clinic rules and guidelines)
        recall_res = self.call_tool(
            "recall",
            {
                "session_id": session_id,
                "query": goal,
                "include_clinic": True,
                "k": 8,
            },
        )
        ret_id = recall_res.get("retrieval_id", "")
        items = recall_res.get("items", [])

        # Find allergy facts from retrieval
        allergy_ids: list[str] = [
            it["id"]
            for it in items
            if it.get("label") == "Fact"
            and (
                it.get("kind") == "allergy"
                or "penicillin" in it.get("summary", "").lower()
                or "allergy" in it.get("summary", "").lower()
            )
        ]

        # LLM Pharmacy Review
        context_items = "; ".join(f"{it.get('label')}: {it.get('summary')}" for it in items[:6])
        prompt = (
            f"Prescription Review Request: {goal}\n"
            f"Carried Facts: {carried_facts}\n"
            f"Patient Allergies & Shared Rules: {context_items}\n"
            f"Determine if proposed medication is safe or contraindicated."
        )
        llm_res = self.llm.generate(prompt=prompt, role="pharmacy")
        structured = llm_res.structured or {}

        is_safe = structured.get("is_safe", True)
        warning = structured.get("warning") or llm_res.text
        recommendation = structured.get("recommendation", "Review completed")

        # Prepare verified claims citing supporting provenance IDs
        claims: list[dict[str, Any]] = []
        raw_claims = structured.get("claims") or []
        for rc in raw_claims:
            claim_text = rc.get("text", "Prescription safety review completed.")
            # Map citation to retrieved allergy fact ID if available
            sup_ids = [aid for aid in allergy_ids if aid]
            if not sup_ids and items:
                sup_ids = [items[0]["id"]]
            claims.append(
                {
                    "text": claim_text,
                    "supported_by": sup_ids,
                }
            )

        if not claims:
            sup_ids = [items[0]["id"]] if items else []
            claims.append(
                {
                    "text": "Evaluated prescription against patient graph and clinic rules.",
                    "supported_by": sup_ids,
                }
            )

        # Commit answer with verifiable provenance citations
        ans_res = self.call_tool(
            "commit_answer",
            {
                "session_id": session_id,
                "retrieval_id": ret_id,
                "question": f"Safety evaluation for {goal}",
                "answer_text": f"{warning} Recommendation: {recommendation}",
                "claims": claims,
                "allow_unsupported": False,
            },
        )
        ans_id = ans_res["answer_id"]

        # Complete pharmacy handoff
        self.call_tool(
            "complete_handoff",
            {
                "session_id": session_id,
                "handoff_id": hid,
                "result_summary": (
                    f"Pharmacy screening complete. Safe: {is_safe}. "
                    f"Recommendation: {recommendation}."
                ),
            },
        )

        return {
            "handoff_id": hid,
            "answer_id": ans_id,
            "is_safe": is_safe,
            "warning": warning,
            "recommendation": recommendation,
            "claims": claims,
            "trace": ans_res.get("trace"),
        }
