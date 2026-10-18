"""FastMCP server definition exposing outpatient memory tools for CareThread.

Exposes tools:
- start_session, end_session, append_message, get_session_messages
- remember, recall, history
- handoff, list_handoffs, accept_handoff, complete_handoff
- consolidate, record_outcome
- commit_answer, explain
- list_flags, resolve_flag, confirm_fact
- forget_request, confirm_forget
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from carethread.mcp.dispatcher import ToolDispatcher

# Alias for compatibility with FastMCP interface
FastMCP = MCPServer
DEFAULT_AGENT_TOKEN_ENV: str = "CT_AGENT_TOKEN"


def _resolve_token(provided_token: str | None = None) -> str | None:
    """Resolves caller token from argument or environment variable."""
    if provided_token and provided_token.strip():
        return provided_token.strip()
    return os.environ.get(DEFAULT_AGENT_TOKEN_ENV)


def create_mcp_server(dispatcher: ToolDispatcher | None = None) -> MCPServer:
    """Constructs and registers all clinical memory tools on an MCPServer instance.

    Args:
        dispatcher: Shared ToolDispatcher instance (constructed if None).

    Returns:
        Configured MCPServer named 'carethread-memory'.
    """
    disp = dispatcher or ToolDispatcher()
    server = MCPServer("carethread-memory")

    # 1. Session Lifecycle
    @server.tool(
        name="start_session",
        description="Initiate a new clinical conversation session bound to a patient tenant.",
    )
    def start_session(
        patient_id: str,
        channel: str = "chat",
        agent_id: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "start_session",
            {"patient_id": patient_id, "channel": channel, "agent_id": agent_id},
            token=_resolve_token(token),
        )

    @server.tool(
        name="end_session",
        description="Close an active clinical conversation session and enqueue consolidation.",
    )
    def end_session(
        session_id: str,
        outcome: str = "unknown",
        summary: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "end_session",
            {"session_id": session_id, "outcome": outcome, "summary": summary},
            token=_resolve_token(token),
        )

    @server.tool(
        name="append_message",
        description="Atomically append a message to an active clinical conversation session.",
    )
    def append_message(
        session_id: str,
        role: str,
        content: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "append_message",
            {"session_id": session_id, "role": role, "content": content},
            token=_resolve_token(token),
        )

    @server.tool(
        name="get_session_messages",
        description="Retrieve message history from a session ordered by sequence number.",
    )
    def get_session_messages(
        session_id: str,
        after_seq: int | None = None,
        limit: int = 50,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "get_session_messages",
            {"session_id": session_id, "after_seq": after_seq, "limit": limit},
            token=_resolve_token(token),
        )

    # 2. Semantic Memory & Recall
    @server.tool(
        name="remember",
        description="Write clinical facts, notes, or candidate observations to memory.",
    )
    def remember(
        session_id: str,
        kind: str,
        payload: dict[str, Any],
        source_message_ids: list[str] | None = None,
        expected_version: int | None = None,
        op_id: str = "",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "remember",
            {
                "session_id": session_id,
                "kind": kind,
                "payload": payload,
                "source_message_ids": source_message_ids,
                "expected_version": expected_version,
                "op_id": op_id,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="recall",
        description="Hybrid recall over patient facts, vector notes, and clinic guidelines.",
    )
    def recall(
        session_id: str,
        query: str,
        kinds: list[str] | None = None,
        as_of: int | None = None,
        k: int = 8,
        include_stale: bool = False,
        include_clinic: bool = True,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "recall",
            {
                "session_id": session_id,
                "query": query,
                "kinds": kinds,
                "as_of": as_of,
                "k": k,
                "include_stale": include_stale,
                "include_clinic": include_clinic,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="history",
        description="Retrieve bitemporal supersession audit history for a fact or slot.",
    )
    def history(
        session_id: str,
        fact_id: str | None = None,
        slot: str | None = None,
        as_of: int | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "history",
            {"session_id": session_id, "fact_id": fact_id, "slot": slot, "as_of": as_of},
            token=_resolve_token(token),
        )

    # 3. Coordination & Handoffs
    @server.tool(
        name="handoff",
        description="Draft and issue an inter-agent clinical handoff packet.",
    )
    def handoff(
        session_id: str,
        to_role: str,
        goal: str,
        state_summary: str,
        completed_actions: list[dict[str, Any]],
        open_questions: list[dict[str, Any]],
        memory_ids: list[dict[str, Any]],
        clinician_review_required: bool = False,
        ttl_minutes: int = 1440,
        op_id: str = "",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "handoff",
            {
                "session_id": session_id,
                "to_role": to_role,
                "goal": goal,
                "state_summary": state_summary,
                "completed_actions": completed_actions,
                "open_questions": open_questions,
                "memory_ids": memory_ids,
                "clinician_review_required": clinician_review_required,
                "ttl_minutes": ttl_minutes,
                "op_id": op_id,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="list_handoffs",
        description="List active or pending handoff packets addressed to caller role.",
    )
    def list_handoffs(
        session_id: str,
        status: str = "open",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "list_handoffs",
            {"session_id": session_id, "status": status},
            token=_resolve_token(token),
        )

    @server.tool(
        name="accept_handoff",
        description="Atomically accept an open handoff via compare-and-set locking.",
    )
    def accept_handoff(
        session_id: str,
        handoff_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "accept_handoff",
            {"session_id": session_id, "handoff_id": handoff_id},
            token=_resolve_token(token),
        )

    @server.tool(
        name="complete_handoff",
        description="Complete or reject an accepted handoff with summary and outcome.",
    )
    def complete_handoff(
        session_id: str,
        handoff_id: str,
        result_summary: str,
        new_memory_ids: list[str] | None = None,
        outcome: str = "success",
        reject: bool = False,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "complete_handoff",
            {
                "session_id": session_id,
                "handoff_id": handoff_id,
                "result_summary": result_summary,
                "new_memory_ids": new_memory_ids,
                "outcome": outcome,
                "reject": reject,
            },
            token=_resolve_token(token),
        )

    # 4. Procedural Memory & Consolidation
    @server.tool(
        name="consolidate",
        description="Trigger memory consolidation runs: fact promotion, decay, and sweeps.",
    )
    def consolidate(
        patient_id: str,
        scope: str = "all",
        dry_run: bool = False,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "consolidate",
            {"patient_id": patient_id, "scope": scope, "dry_run": dry_run},
            token=_resolve_token(token),
        )

    @server.tool(
        name="record_outcome",
        description="Record success or failure outcomes for procedural playbooks.",
    )
    def record_outcome(
        session_id: str,
        playbook_id: str | None = None,
        episode_id: str | None = None,
        outcome: str = "success",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "record_outcome",
            {
                "session_id": session_id,
                "playbook_id": playbook_id,
                "episode_id": episode_id,
                "outcome": outcome,
            },
            token=_resolve_token(token),
        )

    # 5. Provenance & Explanations
    @server.tool(
        name="commit_answer",
        description="Commit a clinical answer citing verifiable provenance memory nodes.",
    )
    def commit_answer(
        session_id: str,
        retrieval_id: str,
        question: str,
        answer_text: str,
        answer_type: str = "informational",
        claims: list[dict[str, Any]] | None = None,
        allow_unsupported: bool = False,
        op_id: str = "",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "commit_answer",
            {
                "session_id": session_id,
                "retrieval_id": retrieval_id,
                "question": question,
                "answer_text": answer_text,
                "answer_type": answer_type,
                "claims": claims or [],
                "allow_unsupported": allow_unsupported,
                "op_id": op_id,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="explain",
        description="Inspect answer provenance graph or query 'Why Not?' exclusions.",
    )
    def explain(
        answer_id: str,
        why_not: str | None = None,
        mode: str = "trace",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "explain",
            {"answer_id": answer_id, "why_not": why_not, "mode": mode},
            token=_resolve_token(token),
        )

    # 6. Safety & Human-in-the-Loop
    @server.tool(
        name="list_flags",
        description="List active clinical safety flags (allergies, drug interactions).",
    )
    def list_flags(
        session_id: str,
        status: str = "open",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "list_flags",
            {"session_id": session_id, "status": status},
            token=_resolve_token(token),
        )

    @server.tool(
        name="resolve_flag",
        description="Clinician resolution or dismissal of an active clinical safety flag.",
    )
    def resolve_flag(
        flag_id: str,
        resolution: str = "resolved",
        note: str = "",
        clinician_id: str = "",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "resolve_flag",
            {
                "flag_id": flag_id,
                "resolution": resolution,
                "note": note,
                "clinician_id": clinician_id,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="confirm_fact",
        description="Clinician review and confirmation or rejection of an unconfirmed fact.",
    )
    def confirm_fact(
        fact_id: str,
        clinician_id: str,
        decision: str = "confirm",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "confirm_fact",
            {"fact_id": fact_id, "clinician_id": clinician_id, "decision": decision},
            token=_resolve_token(token),
        )

    # 7. Forgetting & Deletion
    @server.tool(
        name="forget_request",
        description="Request deletion of a patient graph, fact, or session under retention policy.",
    )
    def forget_request(
        patient_id: str,
        scope: str = "all",
        target_id: str | None = None,
        reason: str = "",
        requested_by: str = "",
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "forget_request",
            {
                "patient_id": patient_id,
                "scope": scope,
                "target_id": target_id,
                "reason": reason,
                "requested_by": requested_by,
            },
            token=_resolve_token(token),
        )

    @server.tool(
        name="confirm_forget",
        description="Clinician-admin execution of an approved forgetting deletion request.",
    )
    def confirm_forget(
        request_id: str,
        confirm_token: str,
        approver_id: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        return disp.dispatch(
            "confirm_forget",
            {
                "request_id": request_id,
                "confirm_token": confirm_token,
                "approver_id": approver_id,
            },
            token=_resolve_token(token),
        )

    return server
