"""Unified tool dispatcher for MCP, REST APIs, and load testing harnesses.

Provides:
- Centralized authentication, rate limiting, and RBAC enforcement
- Automatic tenant resolution from clinical conversation sessions
- Write idempotency tracking via op_id keys
- Synchronous and asynchronous audit event emission
- Consistent response envelopes: {ok: True, data: ..., request_id: ...}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    AnswerNotFoundError,
    CareThreadError,
    HandoffIncompleteError,
    MemoryIdNotFoundError,
    RateLimitError,
    RetrievalExpiredError,
    SessionClosedError,
    SessionNotFoundError,
    TenantMismatchError,
    ValidationError,
)
from carethread.mcp.auth import CallerIdentity, TokenStore
from carethread.mcp.errors import error_envelope, success_envelope
from carethread.memory.audit import AuditLogger
from carethread.memory.embeddings import Embedder, get_embedder
from carethread.memory.facts import FactManager
from carethread.memory.forgetting import ForgettingManager
from carethread.memory.handoff import HandoffManager
from carethread.memory.procedural import PlaybookManager
from carethread.memory.recall import RecallManager
from carethread.memory.safety import SafetyManager
from carethread.memory.sessions import SessionManager
from carethread.memory.worker import ConsolidationWorker
from carethread.provenance.engine import ProvenanceEngine
from carethread.util.id import (
    request_id as gen_request_id,
)
from carethread.util.id import (
    retrieval_id as gen_retrieval_id,
)

logger = logging.getLogger(__name__)

MAX_PAYLOAD_SIZE_BYTES: int = 64 * 1024  # 64 KB
MAX_RECALL_K: int = 25
RETRIEVAL_TTL_SECONDS: int = 3600  # 1 hour in-memory TTL
DEFAULT_RATE_LIMIT: int = 50  # 50 calls/s per token

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "start_session",
        "end_session",
        "append_message",
        "remember",
        "handoff",
        "accept_handoff",
        "complete_handoff",
        "consolidate",
        "record_outcome",
        "commit_answer",
        "resolve_flag",
        "confirm_fact",
        "forget_request",
        "confirm_forget",
    }
)


class ToolDispatcher:
    """Unified dispatcher executing memory tools with security and audit controls."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        token_store: TokenStore | None = None,
        clock: Clock | None = None,
        max_calls_per_second: int = DEFAULT_RATE_LIMIT,
        audit_logger: AuditLogger | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        client = router._client if router is not None else get_db_client()
        self._client = client
        self._router = router or TenantRouter(client)
        self._token_store = token_store or TokenStore()
        self._clock = clock or get_clock()
        self._max_rate = max_calls_per_second
        self._embedder = embedder or get_embedder()

        # Domain memory services
        self._session_mgr = SessionManager(
            router=self._router, clock=self._clock, embedder=self._embedder
        )
        self._fact_mgr = FactManager(router=self._router, clock=self._clock)
        self._recall_mgr = RecallManager(
            router=self._router, clock=self._clock, embedder=self._embedder
        )
        self._handoff_mgr = HandoffManager(router=self._router, clock=self._clock)
        self._safety_mgr = SafetyManager(router=self._router, clock=self._clock)
        self._playbook_mgr = PlaybookManager(router=self._router, clock=self._clock)
        self._worker = ConsolidationWorker(router=self._router, clock=self._clock)
        self._forget_mgr = ForgettingManager(router=self._router, clock=self._clock)
        self._clinic_reader = ClinicReader(self._client)
        self._provenance_engine = ProvenanceEngine(
            router=self._router,
            clinic_reader=self._clinic_reader,
            clock=self._clock,
        )

        self._audit_logger = audit_logger or AuditLogger(router=self._router, clock=self._clock)
        self._audit_mgr = self._audit_logger

        # In-memory tracking structures
        self._session_to_patient: dict[str, str] = {}
        self._retrievals: dict[str, dict[str, Any]] = {}
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._call_history: dict[str, list[float]] = {}

    def bind_session_patient(self, session_id: str, patient_id: str) -> None:
        """Explicitly binds a session to a patient tenant identifier."""
        self._session_to_patient[session_id] = patient_id

    def resolve_patient_from_session(self, session_id: str) -> str:
        """Resolves patient ID bound to session_id from memory or graph search.

        Raises:
            SessionNotFoundError: If session cannot be located across tenants.
        """
        cached = self._session_to_patient.get(session_id)
        if cached:
            return cached

        # Search across registered tenants in ct_control
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (s:Session {id: $sid}) RETURN s.id",
                    {"sid": session_id},
                )
                if res.result_set:
                    self._session_to_patient[session_id] = pid
                    return pid
            except Exception:
                continue

        raise SessionNotFoundError(
            f"Session '{session_id}' not found.",
            details={"session_id": session_id},
        )

    def resolve_patient_from_fact(self, fact_id: str) -> str:
        """Resolves patient ID bound to fact_id from graph search across tenants."""
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (f:Fact {id: $fid}) RETURN f.id",
                    {"fid": fact_id},
                )
                if res.result_set:
                    return pid
            except Exception:
                continue
        raise MemoryIdNotFoundError(
            f"Fact '{fact_id}' not found across tenants.",
            details={"fact_id": fact_id},
        )

    def resolve_patient_from_handoff(self, handoff_id: str) -> str:
        """Resolves patient ID bound to handoff_id from graph search across tenants."""
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (h:Handoff {id: $hid}) RETURN h.id",
                    {"hid": handoff_id},
                )
                if res.result_set:
                    return pid
            except Exception:
                continue
        raise ValidationError(
            f"Handoff '{handoff_id}' not found across tenants.",
            details={"handoff_id": handoff_id},
        )

    def resolve_patient_from_answer(self, answer_id: str) -> str:
        """Resolves patient ID bound to answer_id from graph search across tenants."""
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (a:Answer {id: $aid}) RETURN a.id",
                    {"aid": answer_id},
                )
                if res.result_set:
                    return pid
            except Exception:
                continue
        raise AnswerNotFoundError(
            f"Answer '{answer_id}' not found across tenants.",
            details={"answer_id": answer_id},
        )

    def resolve_patient_from_flag(self, flag_id: str) -> str:
        """Resolves patient ID bound to flag_id from graph search across tenants."""
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (f:SafetyConcern {id: $fid}) RETURN f.id",
                    {"fid": flag_id},
                )
                if res.result_set:
                    return pid
            except Exception:
                continue
        raise ValidationError(
            f"Safety flag '{flag_id}' not found across tenants.",
            details={"flag_id": flag_id},
        )

    def resolve_patient_from_playbook(self, playbook_id: str) -> str:
        """Resolves patient ID bound to playbook_id from graph search across tenants."""
        tenants = self._router.list_tenants()
        for pid in tenants:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    "MATCH (pb:Playbook {id: $pbid}) RETURN pb.id",
                    {"pbid": playbook_id},
                )
                if res.result_set:
                    return pid
            except Exception:
                continue
        raise ValidationError(
            f"Playbook '{playbook_id}' not found across tenants.",
            details={"playbook_id": playbook_id},
        )

    def _check_rate_limit(self, token: str) -> None:
        """Enforces sliding-window rate limit quotas per authentication token."""
        now_perf = time.perf_counter()
        calls = self._call_history.setdefault(token, [])

        # Evict calls older than 1.0 second
        threshold = now_perf - 1.0
        self._call_history[token] = [ts for ts in calls if ts > threshold]

        if len(self._call_history[token]) >= self._max_rate:
            raise RateLimitError(
                f"Rate limit exceeded. Maximum {self._max_rate} calls/s allowed.",
                details={"retry_after_ms": 20},
            )

        self._call_history[token].append(now_perf)

    def dispatch(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Dispatches an MCP tool request through authentication and audit pipelines.

        Args:
            tool_name: Registered name of the tool to invoke.
            args: Parameter dictionary for the tool invocation.
            token: Bearer authentication token identifying the caller.

        Returns:
            Standard envelope: {ok: True, data: ..., request_id: ...}
            or {ok: False, error: ..., request_id: ...}.
        """
        req_id = gen_request_id()
        params = args or {}

        # 1. Payload size enforcement
        try:
            raw_payload = json.dumps(params)
            if len(raw_payload.encode("utf-8")) > MAX_PAYLOAD_SIZE_BYTES:
                return error_envelope(
                    code="VALIDATION_ERROR",
                    message="Payload exceeds maximum allowed size of 64 KB.",
                    request_id=req_id,
                    details={"size_bytes": len(raw_payload)},
                )
        except Exception as e:
            return error_envelope(
                code="VALIDATION_ERROR",
                message=f"Invalid payload format: {e}",
                request_id=req_id,
            )

        # 2. Authenticate caller
        try:
            caller = self._token_store.resolve_token(token)
        except CareThreadError as err:
            return error_envelope(
                code=err.code,
                message=err.message,
                request_id=req_id,
                details=err.details,
            )

        # 3. Rate limiting check
        try:
            self._check_rate_limit(caller.token)
        except RateLimitError as err:
            return error_envelope(
                code=err.code,
                message=err.message,
                request_id=req_id,
                details=err.details,
            )

        # 4. Resolve tenant identifier
        target_pid: str | None = None
        try:
            if "patient_id" in params and params["patient_id"]:
                raw_pid = str(params["patient_id"]).strip()
                target_pid = self._router.validate_patient_id(raw_pid)
            elif "session_id" in params and params["session_id"]:
                target_pid = self.resolve_patient_from_session(str(params["session_id"]).strip())
            elif "fact_id" in params and params["fact_id"]:
                target_pid = self.resolve_patient_from_fact(str(params["fact_id"]).strip())
            elif "handoff_id" in params and params["handoff_id"]:
                target_pid = self.resolve_patient_from_handoff(str(params["handoff_id"]).strip())
            elif "answer_id" in params and params["answer_id"]:
                target_pid = self.resolve_patient_from_answer(str(params["answer_id"]).strip())
            elif "flag_id" in params and params["flag_id"]:
                target_pid = self.resolve_patient_from_flag(str(params["flag_id"]).strip())
            elif "playbook_id" in params and params["playbook_id"]:
                target_pid = self.resolve_patient_from_playbook(str(params["playbook_id"]).strip())
            elif isinstance(caller.patient_scope, list) and len(caller.patient_scope) == 1:
                target_pid = caller.patient_scope[0]
        except CareThreadError as err:
            return error_envelope(
                code=err.code,
                message=err.message,
                request_id=req_id,
                details=err.details,
            )

        # 5. Permission and authorization checks
        authority = None
        if tool_name == "remember" and isinstance(params.get("payload"), dict):
            authority = params["payload"].get("authority")

        try:
            self._token_store.check_permission(
                caller=caller,
                tool_name=tool_name,
                patient_id=target_pid,
                authority=authority,
            )
        except CareThreadError as err:
            # Audit denied call; if tenant mismatch or unverified tenant, log to ct_control
            try:
                self._audit_logger.log_event(
                    patient_id=target_pid if not isinstance(err, TenantMismatchError) else None,
                    agent_id=caller.agent_id,
                    role=caller.role,
                    tool=tool_name,
                    op="denied",
                    outcome="denied",
                    args=params,
                )
            except Exception:
                try:
                    self._audit_logger.log_event(
                        patient_id=None,
                        agent_id=caller.agent_id,
                        role=caller.role,
                        tool=tool_name,
                        op="denied",
                        outcome="denied",
                        args=params,
                    )
                except Exception as log_err:
                    logger.debug(f"Audit log fallback error: {log_err}")
            return error_envelope(
                code=err.code,
                message=err.message,
                request_id=req_id,
                details=err.details,
            )

        # 6. Idempotency replay check
        op_id = str(params.get("op_id", "")).strip() if "op_id" in params else ""
        if op_id and op_id in self._idempotency_cache:
            replayed_result = dict(self._idempotency_cache[op_id])
            replayed_result["replayed"] = True
            return success_envelope(data=replayed_result, request_id=req_id)

        # 7. Audited execution
        op_type = "write" if tool_name in WRITE_TOOLS else "read"
        with self._audit_mgr.audit_scope(
            patient_id=target_pid,
            agent_id=caller.agent_id,
            role=caller.role,
            tool=tool_name,
            op=op_type,
            args=params,
        ) as audit_ctx:
            try:
                data = self._execute_tool(
                    tool_name=tool_name,
                    params=params,
                    caller=caller,
                    patient_id=target_pid,
                    audit_scope=audit_ctx,
                )
                if op_id:
                    self._idempotency_cache[op_id] = data
                return success_envelope(data=data, request_id=req_id)
            except CareThreadError as err:
                audit_ctx.set_outcome("error")
                return error_envelope(
                    code=err.code,
                    message=err.message,
                    request_id=req_id,
                    details=err.details,
                )
            except Exception as exc:
                audit_ctx.set_outcome("error")
                logger.exception(f"Unhandled error during tool '{tool_name}' execution: {exc}")
                return error_envelope(
                    code="INTERNAL_ERROR",
                    message=f"An internal error occurred: {exc}",
                    request_id=req_id,
                )

    def _execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        """Dispatches parameters to corresponding domain handler."""
        method_name = f"_tool_{tool_name}"
        handler = getattr(self, method_name, None)
        if handler is None:
            raise ValidationError(
                f"Unknown tool '{tool_name}'.",
                code="TOOL_NOT_FOUND",
                details={"tool": tool_name},
            )
        return handler(params=params, caller=caller, patient_id=patient_id, audit_scope=audit_scope)

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_start_session(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        if not patient_id:
            raise ValidationError("patient_id is required for start_session.")
        channel = str(params.get("channel", "chat"))
        agent_id = str(params.get("agent_id") or caller.agent_id)

        session = self._session_mgr.create_session(
            patient_id=patient_id,
            channel=channel,
            agent_id=agent_id,
        )
        self.bind_session_patient(session.id, patient_id)

        # Retrieve patient metadata and overview metrics
        target_graph = self._router.graph(patient_id)
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.name, p.birthdate",
            {"pid": patient_id},
        )
        p_name = p_res.result_set[0][0] if p_res.result_set else "Unknown"

        # Count active facts
        f_res = target_graph.query(
            "MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {status:'active'}) RETURN count(f)",
            {"pid": patient_id},
        )
        active_facts = f_res.result_set[0][0] if f_res.result_set else 0

        # Count open flags
        fl_res = target_graph.query("MATCH (fl:Flag {status:'open'}) RETURN count(fl)")
        open_flags = fl_res.result_set[0][0] if fl_res.result_set else 0

        # Count open handoffs for caller's role
        h_res = target_graph.query(
            "MATCH (h:Handoff {status:'open', to_role:$role}) RETURN count(h)",
            {"role": caller.role},
        )
        open_handoffs = h_res.result_set[0][0] if h_res.result_set else 0

        return {
            "session_id": session.id,
            "episode_id": session.episode_id,
            "patient_summary": {
                "name": p_name,
                "active_facts_count": active_facts,
                "open_flags": open_flags,
                "open_handoffs_for_role": open_handoffs,
            },
        }

    def _tool_end_session(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        summary = params.get("summary")

        # Verify session is not already closed
        target_graph = self._router.graph(patient_id or "")
        s_res = target_graph.query(
            "MATCH (s:Session {id:$sid}) RETURN s.status",
            {"sid": session_id},
        )
        if not s_res.result_set:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        if s_res.result_set[0][0] == "closed":
            raise SessionClosedError(f"Session '{session_id}' is already closed.")

        self._session_mgr.close_session(
            patient_id=patient_id or "",
            session_id=session_id,
            summary=summary,
        )
        return {
            "closed_at": self._clock.now_ts(),
            "consolidation_enqueued": True,
        }

    def _tool_append_message(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        role = str(params.get("role", "")).strip().lower()
        content = str(params.get("content", ""))

        if len(content) > 8000:
            raise ValidationError("content exceeds maximum limit of 8000 characters.")

        # Check session status
        target_graph = self._router.graph(patient_id or "")
        s_res = target_graph.query(
            "MATCH (s:Session {id:$sid}) RETURN s.status",
            {"sid": session_id},
        )
        if not s_res.result_set:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        if s_res.result_set[0][0] == "closed":
            raise SessionClosedError(f"Session '{session_id}' is already closed.")

        msg = self._session_mgr.append_message(
            patient_id=patient_id or "",
            session_id=session_id,
            role=role,
            content=content,
            agent_id=caller.agent_id,
        )
        return {"message_id": msg.id, "seq": msg.seq}

    def _tool_get_session_messages(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        after_seq = params.get("after_seq")
        limit = min(int(params.get("limit", 50)), 200)

        target_graph = self._router.graph(patient_id or "")
        cypher = """
        MATCH (s:Session {id:$sid})-[:HAS_MESSAGE]->(m:Message)
        WHERE ($after_seq IS NULL OR m.seq > $after_seq)
        RETURN m
        ORDER BY m.seq ASC
        LIMIT $limit
        """
        res = target_graph.query(
            cypher,
            {"sid": session_id, "after_seq": after_seq, "limit": limit},
        )
        messages: list[dict[str, Any]] = []
        for row in res.result_set:
            props = dict(row[0].properties)
            if "embedding" in props:
                del props["embedding"]
            messages.append(props)

        return {"messages": messages}

    def _tool_remember(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        if session_id and patient_id:
            self.bind_session_patient(session_id, patient_id)
        kind = str(params.get("kind", "")).strip().lower()
        payload = params.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValidationError("remember payload must be a dictionary.")

        expected_version = params.get("expected_version")
        op_id = str(params.get("op_id", "")).strip()

        if kind == "fact":
            fact_kind = payload.get("kind", "condition")
            display = payload.get("display", "")
            code = payload.get("code")
            system = payload.get("system")
            value = payload.get("value")
            authority = payload.get("authority")
            valid_from = payload.get("valid_from")
            valid_to = payload.get("valid_to")

            # Specialist writes become pending_review if not clinician
            if caller.role == "specialist" and authority != "clinician":
                authority = "agent_inferred"

            res = self._fact_mgr.remember_fact(
                patient_id=patient_id or "",
                kind=fact_kind,
                display=display,
                value=value,
                code=code,
                system=system,
                authority=authority,
                caller_role=caller.role,
                valid_from=valid_from,
                valid_to=valid_to,
                op_id=op_id or None,
                expected_version=expected_version,
            )

            # Contradiction and safety scan on medication writes
            flags: list[dict[str, Any]] = []
            if fact_kind == "medication":
                safety_res = self._safety_mgr.run_all_checks(
                    patient_id=patient_id or "",
                    medication_fact_id=res["fact_id"],
                )
                flags = [f.to_dict() for f in safety_res.flags]

            return {
                "action": res["action"],
                "fact_id": res["fact_id"],
                "superseded_id": res.get("superseded_id"),
                "flags": flags,
                "status": "active",
            }

        elif kind == "candidate":
            cand_kind = payload.get("kind", "condition")
            display = payload.get("display", "")
            value = payload.get("value", display)
            code = payload.get("code")
            system = payload.get("system")
            ep_id = payload.get("episode_id") or "epi_default"

            cand = self._worker.create_candidate(
                patient_id=patient_id or "",
                kind=cand_kind,
                display=display,
                episode_id=ep_id,
                value=value,
                code=code,
                system=system,
            )
            return {
                "action": "candidate_created",
                "candidate_id": cand.id,
                "flags": [],
                "status": "pending",
            }

        elif kind == "note":
            text = payload.get("text", "")
            note_kind = payload.get("kind", f"{caller.role}_note")
            about_ids = payload.get("about_ids")

            note = self._recall_mgr.add_note(
                patient_id=patient_id or "",
                text=text,
                kind=note_kind,
                author_agent_id=caller.agent_id,
                about_ids=about_ids,
            )
            return {
                "action": "note_created",
                "note_id": note.id,
                "flags": [],
                "status": "active",
            }

        raise ValidationError(f"Invalid kind '{kind}'. Must be 'fact', 'candidate', or 'note'.")

    def _tool_recall(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        query = str(params.get("query", ""))
        kinds = params.get("kinds")
        as_of = params.get("as_of")
        k = min(int(params.get("k", 8)), MAX_RECALL_K)
        include_clinic = bool(params.get("include_clinic", True))

        recall_res = self._recall_mgr.recall(
            patient_id=patient_id or "",
            query=query,
            kinds=kinds,
            as_of=as_of,
            top_k=k,
            include_clinic=include_clinic,
            include_rejected=True,
        )

        ret_id = gen_retrieval_id()
        items: list[dict[str, Any]] = []

        for f in recall_res.facts:
            items.append(
                {
                    "id": f.id,
                    "label": "Fact",
                    "summary": f"{f.kind}: {f.display}",
                    "score": f.composite_score,
                    "method": "graph_traversal",
                    "confidence": f.confidence,
                    "effective_confidence": f.decayed_confidence,
                    "status": "active",
                    "stale": getattr(f, "is_stale", False),
                    "valid_from": getattr(f, "valid_from", f.last_confirmed_at),
                    "valid_to": getattr(f, "valid_to", None),
                    "graph": f"patient_{patient_id}",
                }
            )

        for n in recall_res.notes:
            items.append(
                {
                    "id": n.id,
                    "label": "Note",
                    "summary": n.text,
                    "score": n.vector_score,
                    "method": "vector_search",
                    "confidence": 1.0,
                    "effective_confidence": 1.0,
                    "status": "active",
                    "stale": False,
                    "valid_from": getattr(n, "ts", 0),
                    "valid_to": None,
                    "graph": f"patient_{patient_id}",
                }
            )

        for g in recall_res.guidelines:
            items.append(
                {
                    "id": g.id,
                    "label": "Guideline",
                    "summary": g.text,
                    "score": g.vector_score,
                    "method": "clinic_guideline",
                    "confidence": 1.0,
                    "effective_confidence": 1.0,
                    "status": "active",
                    "stale": False,
                    "valid_from": None,
                    "valid_to": None,
                    "graph": "clinic_shared",
                }
            )

        rejected = [r.to_dict() for r in recall_res.rejected]

        # Cache retrieval in memory with 1h TTL
        self._retrievals[ret_id] = {
            "retrieval_id": ret_id,
            "patient_id": patient_id,
            "query": query,
            "items": items,
            "rejected": rejected,
            "created_at": self._clock.now_ts(),
        }

        # Track read node IDs in audit scope
        if audit_scope:
            audit_scope.record_read_ids([item["id"] for item in items])

        return {
            "retrieval_id": ret_id,
            "items": items,
            "rejected": rejected,
            "edges": [],
        }

    def _tool_history(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        fact_id = params.get("fact_id")
        as_of = params.get("as_of")

        if fact_id:
            versions = self._fact_mgr.get_fact_history(
                patient_id=patient_id or "",
                fact_id=str(fact_id),
            )
            return {"versions": versions}
        elif as_of:
            facts = self._fact_mgr.get_facts_as_of(
                patient_id=patient_id or "",
                as_of_ts=int(as_of),
            )
            return {"versions": [f.to_dict() for f in facts]}
        return {"versions": []}

    def _tool_handoff(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        to_role = str(params.get("to_role", "")).strip().lower()
        goal = str(params.get("goal", "")).strip()
        state_summary = str(params.get("state_summary", "")).strip()
        completed_actions = params.get("completed_actions") or []
        open_questions = params.get("open_questions") or []
        memory_ids = params.get("memory_ids") or []
        crr = bool(params.get("clinician_review_required", False))
        ttl_minutes = int(params.get("ttl_minutes", 1440))

        if not goal or not state_summary or not memory_ids:
            raise HandoffIncompleteError(
                "Handoff packet is incomplete: goal, state_summary, and memory_ids are required."
            )

        carried_fact_ids: list[str] = []
        for m in memory_ids:
            if isinstance(m, dict) and "id" in m:
                carried_fact_ids.append(str(m["id"]))
            elif isinstance(m, str):
                carried_fact_ids.append(m)

        questions: list[str] = []
        for q in open_questions:
            if isinstance(q, dict) and "text" in q:
                questions.append(str(q["text"]))
            elif isinstance(q, str):
                questions.append(q)

        actions: list[str] = []
        for a in completed_actions:
            if isinstance(a, dict):
                actions.append(str(a.get("description") or a.get("type") or str(a)))
            elif isinstance(a, str):
                actions.append(a)

        pkt = self._handoff_mgr.create_handoff(
            patient_id=patient_id or "",
            from_agent_id=caller.agent_id,
            from_role=caller.role,
            to_role=to_role,
            goal=goal,
            state_summary=state_summary,
            actions=actions,
            open_questions=questions,
            carried_fact_ids=carried_fact_ids,
            clinician_review_required=crr,
            ttl_seconds=ttl_minutes * 60,
            session_id=session_id,
            publish=True,
        )
        return {
            "handoff_id": pkt.handoff_id,
            "status": "open",
            "carried_count": len(carried_fact_ids),
        }

    def _tool_list_handoffs(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        status = params.get("status")
        to_role = params.get("to_role") or params.get("role")
        handoffs = self._handoff_mgr.list_handoffs(
            patient_id=patient_id or "",
            to_role=to_role,
            status=status,
        )
        return {"handoffs": [h.to_dict() for h in handoffs]}

    def _tool_accept_handoff(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip()
        handoff_id = str(params.get("handoff_id", "")).strip()

        pkt = self._handoff_mgr.accept_handoff(
            patient_id=patient_id or "",
            handoff_id=handoff_id,
            agent_id=caller.agent_id,
            role=caller.role,
            session_id=session_id,
        )
        return pkt.to_dict()

    def _tool_complete_handoff(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        handoff_id = str(params.get("handoff_id", "")).strip()
        summary = str(params.get("result_summary", ""))

        pkt = self._handoff_mgr.complete_handoff(
            patient_id=patient_id or "",
            handoff_id=handoff_id,
            result_summary=summary,
            completed_by_agent_id=caller.agent_id,
        )
        return {"status": pkt.status}

    def _tool_consolidate(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        res = self._worker.run_consolidation(
            patient_id=patient_id,
            trigger="api",
        )
        if isinstance(res, list):
            return {
                "run_id": res[0].id if res else "cns_none",
                "promoted_facts": sum(r.stats.get("promoted_candidates", 0) for r in res),
                "decayed_facts": sum(r.stats.get("decayed_facts", 0) for r in res),
                "expired_handoffs": sum(r.stats.get("expired_handoffs", 0) for r in res),
            }
        return {
            "run_id": res.id,
            "promoted_facts": res.stats.get("promoted_candidates", 0),
            "decayed_facts": res.stats.get("decayed_facts", 0),
            "expired_handoffs": res.stats.get("expired_handoffs", 0),
        }

    def _tool_record_outcome(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        playbook_id = params.get("playbook_id")
        outcome = str(params.get("outcome", "success"))

        if not playbook_id:
            raise ValidationError("playbook_id is required for record_outcome.")

        pb = self._playbook_mgr.record_outcome(
            patient_id=patient_id or "",
            playbook_id=str(playbook_id),
            outcome=outcome,
        )
        return pb.to_dict()

    def _tool_commit_answer(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        session_id = str(params.get("session_id", "")).strip() or None
        ret_id = str(params.get("retrieval_id", "")).strip() or None
        question = str(params.get("question", "")).strip()
        answer_text = str(params.get("answer_text") or params.get("text", "")).strip()
        answer_type = str(params.get("answer_type", "informational")).strip()
        claims = params.get("claims") or []
        allow_unsupported = bool(params.get("allow_unsupported", False))
        overall_confidence = params.get("overall_confidence")
        clinician_review_required = params.get("clinician_review_required")

        retrieval = None
        if ret_id:
            retrieval = self._retrievals.get(ret_id)
            if not retrieval:
                raise RetrievalExpiredError("Retrieval context has expired or was not found.")

        trace = self._provenance_engine.commit_answer(
            patient_id=patient_id or "",
            question=question,
            answer_text=answer_text,
            session_id=session_id,
            retrieval=retrieval,
            claims=claims,
            answer_type=answer_type,
            overall_confidence=overall_confidence,
            clinician_review_required=clinician_review_required,
            author_agent_id=caller.agent_id,
            allow_unsupported=allow_unsupported,
        )

        if audit_scope and hasattr(audit_scope, "record_read_ids"):
            audit_scope.record_read_ids([u["node_id"] for u in trace.used])

        return {"answer_id": trace.answer_id, "trace": trace.to_dict()}

    def _tool_explain(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        ans_id = str(params.get("answer_id", "")).strip()
        why_not = params.get("why_not")
        mode = str(params.get("mode", "trace"))

        exp_res = self._provenance_engine.explain_answer(
            patient_id=patient_id or "",
            answer_id=ans_id,
            why_not=why_not,
            mode=mode,
        )

        if audit_scope:
            audit_scope.record_read_ids([u["node_id"] for u in exp_res.trace.used])

        if why_not:
            res = exp_res.why_not.to_dict() if exp_res.why_not else {}
            res["answer_id"] = ans_id
            res["why_not"] = why_not
            return res

        if mode == "ablate" and exp_res.ablated:
            res = dict(exp_res.ablated)
            res["answer_id"] = ans_id
            res["trace_id"] = exp_res.trace.trace_id
            return res

        return exp_res.trace.to_dict()

    def _tool_list_flags(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        status = str(params.get("status", "open"))
        flags = self._safety_mgr.list_flags(
            patient_id=patient_id or "",
            status=status,
        )
        return {"flags": [f.to_dict() for f in flags]}

    def _tool_resolve_flag(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        flag_id = str(params.get("flag_id", "")).strip()
        resolution = str(params.get("resolution", "resolved")).strip().lower()
        note = str(params.get("note", ""))
        clinician_id = str(params.get("clinician_id") or caller.agent_id)

        if resolution not in {"resolved", "dismissed"}:
            raise ValidationError(
                f"Invalid resolution '{resolution}'. Allowed: 'resolved', 'dismissed'."
            )

        flag = self._safety_mgr.resolve_flag(
            patient_id=patient_id or "",
            flag_id=flag_id,
            resolved_by=clinician_id,
            resolution_reason=note or resolution,
        )
        return {"flag": flag.to_dict()}

    def _tool_confirm_fact(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        fact_id = str(params.get("fact_id", "")).strip()
        decision = str(params.get("decision", "confirm")).strip().lower()
        clinician_id = str(params.get("clinician_id") or caller.agent_id)

        if decision not in {"confirm", "reject"}:
            raise ValidationError(f"Invalid decision '{decision}'. Allowed: 'confirm', 'reject'.")

        target_graph = self._router.graph(patient_id or "")
        f_res = target_graph.query(
            "MATCH (f:Fact {id:$fid}) RETURN f",
            {"fid": fact_id},
        )
        if not f_res.result_set:
            raise MemoryIdNotFoundError(f"Fact '{fact_id}' not found.")

        now_ts = self._clock.now_ts()
        if decision == "confirm":
            upd_cypher = """
            MATCH (f:Fact {id:$fid})
            SET f.status = 'active',
                f.authority = 'clinician',
                f.last_confirmed_at = $now
            RETURN f
            """
            target_graph.query(upd_cypher, {"fid": fact_id, "now": now_ts})
        else:
            upd_cypher = """
            MATCH (f:Fact {id:$fid})
            SET f.status = 'retracted',
                f.retracted_at = $now,
                f.retracted_by = $cid
            RETURN f
            """
            target_graph.query(
                upd_cypher,
                {"fid": fact_id, "now": now_ts, "cid": clinician_id},
            )

        f_after = target_graph.query(
            "MATCH (f:Fact {id:$fid}) RETURN f",
            {"fid": fact_id},
        )
        props = dict(f_after.result_set[0][0].properties)
        return {"fact": props}

    def _tool_forget_request(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        if not patient_id:
            raise ValidationError("patient_id is required for forget_request.")
        scope = str(params.get("scope", "all"))
        target_id = params.get("target_id")
        reason = str(params.get("reason", ""))
        requested_by = str(params.get("requested_by") or caller.agent_id)

        return self._forget_mgr.forget_request(
            patient_id=patient_id,
            scope=scope,
            target_id=target_id,
            reason=reason,
            requested_by=requested_by,
        )

    def _tool_confirm_forget(
        self,
        params: dict[str, Any],
        caller: CallerIdentity,
        patient_id: str | None,
        audit_scope: Any,
    ) -> dict[str, Any]:
        req_id = str(params.get("request_id", "")).strip()
        confirm_token = str(params.get("confirm_token", "")).strip()
        approver_id = str(params.get("approver_id") or caller.agent_id)

        return self._forget_mgr.confirm_forget(
            request_id=req_id,
            confirm_token=confirm_token,
            approver_id=approver_id,
            approver_role=caller.role,
        )
