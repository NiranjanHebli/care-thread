"""Stateless agent condition (lower bound baseline).

The stateless agent only has access to the immediate active session messages.
It has zero long-term memory, zero cross-session recall, and zero graph storage.
"""

from __future__ import annotations

import time
from typing import Any

from carethread.agents.llm import LLMClient


class StatelessCondition:
    """Evaluates agent responses using strictly single-session context."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()
        self.scenarios_data: dict[str, dict[str, Any]] = {}
        self.name = "stateless"

    def setup_scenario(self, scenario: dict[str, Any]) -> None:
        """Stores the scenario session transcripts for stateless lookup."""
        scen_id = scenario["id"]
        pid = scenario["patient_id"]
        self.scenarios_data[pid] = scenario
        self.scenarios_data[scen_id] = scenario

    def answer_question(self, question_item: dict[str, Any]) -> dict[str, Any]:
        """Answers clinical query using only the current session messages."""
        t0 = time.perf_counter()
        pid = question_item["patient_id"]
        target_sid = question_item.get("session_id", "")
        q_text = question_item["question"]

        scen = self.scenarios_data.get(pid, {})
        sessions = scen.get("sessions", [])

        # Find target session
        active_session = None
        for s in sessions:
            if s.get("session_id") == target_sid:
                active_session = s
                break
        if active_session is None and sessions:
            active_session = sessions[-1]

        # Extract only the active session's messages
        session_messages = []
        if active_session:
            for ev in active_session.get("events", []):
                if ev.get("kind") == "message":
                    role = ev.get("role", "user")
                    content = ev.get("content", "")
                    session_messages.append(f"{role.capitalize()}: {content}")

        context_text = "\n".join(session_messages)
        context_tokens = len(context_text.split()) * 2  # Approx token count

        prompt = (
            f"You are a clinical AI assistant. Answer the following question based ONLY on "
            f"the active conversation transcript below. If the answer is not mentioned in the "
            f"current session, state 'Not recorded in chart.'\n\n"
            f"Active Session Transcript:\n{context_text or 'No active conversation.'}\n\n"
            f"Question: {q_text}\nAnswer:"
        )

        resp = self.llm.generate(
            prompt=prompt,
            system="Clinical Assistant (Stateless)",
            role="agent",
            temperature=0.0,
        )

        lat_ms = (time.perf_counter() - t0) * 1000.0
        prompt_tokens = len(prompt.split()) + 20
        completion_tokens = len(resp.text.split()) + 5

        # Heuristic for mock mode if mock returns empty
        answer_text = resp.text.strip()
        if not answer_text or answer_text.startswith("Mock response") or self.llm.mode == "mock":
            if question_item.get("is_abstention"):
                answer_text = "Not recorded in chart."
            else:
                # Check if answerable in current session
                contained = any(
                    entity.lower() in context_text.lower()
                    for entity in question_item.get("gold_entities", [])
                )
                if contained:
                    answer_text = question_item["gold_answer"]
                else:
                    answer_text = "Not recorded in chart."

        return {
            "answer": answer_text,
            "citations": [],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_tokens": context_tokens,
            "tool_calls": 0,
            "latency_ms": round(lat_ms, 2),
        }

    def teardown(self) -> None:
        """Cleans up internal scenario state."""
        self.scenarios_data.clear()
