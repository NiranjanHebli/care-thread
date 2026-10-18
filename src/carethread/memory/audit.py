"""Audit logging engine for outpatient clinical memory interactions.

Audit logging guarantees:
- Synchronous write and denied event recording to ensure immediate non-repudiation
- Batched read flushing to preserve low-latency execution during read-heavy interactions
- Cryptographic payload hashing (SHA-256) for argument integrity
- Trace identification linking multi-agent workflow steps
- Bounded-latency automatic buffer flushing and context manager lifecycle
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    AuditError,
    AuditEventNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.util.id import audit_id as gen_audit_id

logger = logging.getLogger(__name__)

VALID_OPS: frozenset[str] = frozenset({"read", "write", "denied"})
VALID_ROLES: frozenset[str] = frozenset({"intake", "specialist", "pharmacy", "clinician", "system"})
VALID_OUTCOMES: frozenset[str] = frozenset({"success", "error", "denied", "partial"})
DEFAULT_BATCH_SIZE: int = 10


def compute_args_hash(args: Any) -> str:
    """Computes a deterministic SHA-256 hex digest for arguments."""
    if args is None:
        return hashlib.sha256(b"").hexdigest()
    try:
        serialized = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        serialized = str(args)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_node_props(node: Any) -> dict[str, Any]:
    """Safely extracts properties dictionary from a graph node or mapping."""
    if hasattr(node, "properties"):
        return dict(node.properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


@dataclass
class AuditEventRecord:
    """Represents an immutable clinical audit event node."""

    id: str
    patient_id: str
    ts: int
    agent_id: str
    role: str
    tool: str
    op: str
    outcome: str
    args_hash: str
    read_ids: list[str] = field(default_factory=list)
    trace_id: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], patient_id: str) -> AuditEventRecord:
        """Create an AuditEventRecord from a raw dictionary or graph properties."""
        raw_reads = data.get("read_ids") or []
        if isinstance(raw_reads, list):
            read_ids = [str(r) for r in raw_reads]
        else:
            read_ids = []

        return cls(
            id=str(data.get("id", "")),
            patient_id=patient_id,
            ts=int(data.get("ts", 0)),
            agent_id=str(data.get("agent_id", "")),
            role=str(data.get("role", "")),
            tool=str(data.get("tool", "")),
            op=str(data.get("op", "")),
            outcome=str(data.get("outcome", "")),
            args_hash=str(data.get("args_hash", "")),
            read_ids=read_ids,
            trace_id=data.get("trace_id") or None,
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


class AuditScope:
    """Context manager for automatically measuring latency and logging audit events."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        patient_id: str,
        agent_id: str,
        role: str,
        tool: str,
        op: str = "read",
        args: Any = None,
        args_hash: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._audit_logger = audit_logger
        self._patient_id = patient_id
        self._agent_id = agent_id
        self._role = role
        self._tool = tool
        self._op = op
        self._args = args
        self._args_hash = args_hash
        self._trace_id = trace_id
        self._read_ids: list[str] = []
        self._outcome: str = "success"
        self._start_perf: float = 0.0

    def record_read_ids(self, ids: list[str]) -> None:
        """Record node IDs accessed during this scope."""
        self._read_ids.extend(ids)

    def set_outcome(self, outcome: str) -> None:
        """Explicitly override the outcome of this audited operation."""
        self._outcome = outcome

    def __enter__(self) -> AuditScope:
        self._start_perf = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start_perf) * 1000.0
        if exc_type is not None and self._outcome == "success":
            self._outcome = "error"

        self._audit_logger.log_event(
            patient_id=self._patient_id,
            agent_id=self._agent_id,
            role=self._role,
            tool=self._tool,
            op=self._op,
            outcome=self._outcome,
            args=self._args,
            args_hash=self._args_hash,
            read_ids=self._read_ids,
            trace_id=self._trace_id,
            latency_ms=elapsed_ms,
        )


class AuditLogger:
    """Audit logging engine for outpatient clinical memory interactions."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        auto_flush: bool = True,
    ) -> None:
        self._client = router._client if router is not None else get_db_client()
        self._router = router or TenantRouter(self._client)
        self._clock = clock or get_clock()
        self._batch_size = max(1, batch_size)
        self._auto_flush = auto_flush
        self._read_buffer: list[AuditEventRecord] = []
        self._lock = threading.Lock()

    @property
    def batch_size(self) -> int:
        """Current buffer batch size for read events."""
        return self._batch_size

    def _validate_common(
        self,
        patient_id: str,
        role: str,
        tool: str,
        op: str,
    ) -> str:
        """Performs input validation for common audit event fields."""
        valid_pid = self._router.validate_patient_id(patient_id)
        if not tool or not tool.strip():
            raise ValidationError("Audit tool name must not be empty.")
        if op not in VALID_OPS:
            raise ValidationError(f"Invalid audit op '{op}'. Must be one of {sorted(VALID_OPS)}.")
        clean_role = role.strip().lower()
        if clean_role not in VALID_ROLES:
            raise ValidationError(
                f"Invalid agent role '{role}'. Must be one of {sorted(VALID_ROLES)}."
            )
        return valid_pid

    def _ensure_patient_exists(self, valid_pid: str) -> None:
        """Verifies that the patient node exists in the tenant graph."""
        target_graph = self._router.graph(valid_pid)
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

    def log_write(
        self,
        patient_id: str,
        agent_id: str,
        role: str,
        tool: str,
        outcome: str = "success",
        args: Any = None,
        args_hash: str | None = None,
        read_ids: list[str] | None = None,
        trace_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> AuditEventRecord:
        """Synchronously records a write audit event to the patient graph."""
        valid_pid = self._validate_common(patient_id, role, tool, "write")
        self._ensure_patient_exists(valid_pid)

        now_ts = int(self._clock.now().timestamp())
        eid = gen_audit_id()
        final_hash = args_hash or compute_args_hash(args)
        final_read_ids = list(read_ids) if read_ids else []

        q_aud_01 = get_query("q_aud_01")
        target_graph = self._router.graph(valid_pid)
        res = target_graph.query(
            q_aud_01,
            {
                "pid": valid_pid,
                "eid": eid,
                "now": now_ts,
                "agent_id": agent_id.strip(),
                "role": role.strip().lower(),
                "tool": tool.strip(),
                "op": "write",
                "outcome": outcome.strip(),
                "args_hash": final_hash,
                "read_ids": final_read_ids,
                "trace_id": (trace_id or "").strip(),
                "latency_ms": float(latency_ms),
            },
        )
        if not res.result_set:
            raise AuditError(f"Failed to record write audit event for patient '{valid_pid}'.")

        return AuditEventRecord(
            id=eid,
            patient_id=valid_pid,
            ts=now_ts,
            agent_id=agent_id.strip(),
            role=role.strip().lower(),
            tool=tool.strip(),
            op="write",
            outcome=outcome.strip(),
            args_hash=final_hash,
            read_ids=final_read_ids,
            trace_id=(trace_id or "").strip() or None,
            latency_ms=float(latency_ms),
        )

    def log_denied(
        self,
        patient_id: str | None,
        agent_id: str,
        role: str,
        tool: str,
        reason: str = "access_denied",
        outcome: str = "denied",
        args: Any = None,
        args_hash: str | None = None,
        read_ids: list[str] | None = None,
        trace_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> AuditEventRecord:
        """Synchronously records an access denied or policy violation event."""
        now_ts = int(self._clock.now().timestamp())
        eid = gen_audit_id()
        final_hash = args_hash or compute_args_hash(args)
        final_read_ids = list(read_ids) if read_ids else []

        if not patient_id:
            # Cross-tenant denied event logged to ct_control to prevent tenant cross-contamination
            ctrl_graph = self._router._control_graph
            cypher = """
            CREATE (a:AuditEvent {
                id: $eid, ts: $now, agent_id: $agent_id, role: $role,
                tool: $tool, op: 'denied', outcome: $outcome,
                args_hash: $args_hash, latency_ms: $latency_ms
            })
            RETURN a.id
            """
            ctrl_graph.query(
                cypher,
                {
                    "eid": eid,
                    "now": now_ts,
                    "agent_id": agent_id.strip(),
                    "role": role.strip().lower(),
                    "tool": tool.strip(),
                    "outcome": outcome.strip(),
                    "args_hash": final_hash,
                    "latency_ms": float(latency_ms),
                },
            )
            return AuditEventRecord(
                id=eid,
                patient_id="ct_control",
                ts=now_ts,
                agent_id=agent_id.strip(),
                role=role.strip().lower(),
                tool=tool.strip(),
                op="denied",
                outcome=outcome.strip(),
                args_hash=final_hash,
                read_ids=final_read_ids,
                trace_id=(trace_id or "").strip() or None,
                latency_ms=float(latency_ms),
            )

        valid_pid = self._validate_common(patient_id, role, tool, "denied")
        self._ensure_patient_exists(valid_pid)

        q_aud_01 = get_query("q_aud_01")
        target_graph = self._router.graph(valid_pid)
        res = target_graph.query(
            q_aud_01,
            {
                "pid": valid_pid,
                "eid": eid,
                "now": now_ts,
                "agent_id": agent_id.strip(),
                "role": role.strip().lower(),
                "tool": tool.strip(),
                "op": "denied",
                "outcome": outcome.strip(),
                "args_hash": final_hash,
                "read_ids": final_read_ids,
                "trace_id": (trace_id or "").strip(),
                "latency_ms": float(latency_ms),
            },
        )
        if not res.result_set:
            raise AuditError(f"Failed to record denied audit event for patient '{valid_pid}'.")

        return AuditEventRecord(
            id=eid,
            patient_id=valid_pid,
            ts=now_ts,
            agent_id=agent_id.strip(),
            role=role.strip().lower(),
            tool=tool.strip(),
            op="denied",
            outcome=outcome.strip(),
            args_hash=final_hash,
            read_ids=final_read_ids,
            trace_id=(trace_id or "").strip() or None,
            latency_ms=float(latency_ms),
        )

    def log_read(
        self,
        patient_id: str,
        agent_id: str,
        role: str,
        tool: str,
        outcome: str = "success",
        args: Any = None,
        args_hash: str | None = None,
        read_ids: list[str] | None = None,
        trace_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> AuditEventRecord:
        """Buffers a read audit event for batched background flushing."""
        valid_pid = self._validate_common(patient_id, role, tool, "read")
        now_ts = int(self._clock.now().timestamp())
        eid = gen_audit_id()
        final_hash = args_hash or compute_args_hash(args)
        final_read_ids = list(read_ids) if read_ids else []

        record = AuditEventRecord(
            id=eid,
            patient_id=valid_pid,
            ts=now_ts,
            agent_id=agent_id.strip(),
            role=role.strip().lower(),
            tool=tool.strip(),
            op="read",
            outcome=outcome.strip(),
            args_hash=final_hash,
            read_ids=final_read_ids,
            trace_id=(trace_id or "").strip() or None,
            latency_ms=float(latency_ms),
        )

        should_flush = False
        with self._lock:
            self._read_buffer.append(record)
            if self._auto_flush and len(self._read_buffer) >= self._batch_size:
                should_flush = True

        if should_flush:
            self.flush()

        return record

    def log_event(
        self,
        patient_id: str | None,
        agent_id: str,
        role: str,
        tool: str,
        op: str,
        outcome: str = "success",
        args: Any = None,
        args_hash: str | None = None,
        read_ids: list[str] | None = None,
        trace_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> AuditEventRecord:
        """Unified logging dispatcher based on operation type."""
        clean_op = op.strip().lower()
        if clean_op == "write":
            return self.log_write(
                patient_id=patient_id,
                agent_id=agent_id,
                role=role,
                tool=tool,
                outcome=outcome,
                args=args,
                args_hash=args_hash,
                read_ids=read_ids,
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
        elif clean_op == "denied":
            return self.log_denied(
                patient_id=patient_id,
                agent_id=agent_id,
                role=role,
                tool=tool,
                outcome=outcome,
                args=args,
                args_hash=args_hash,
                read_ids=read_ids,
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
        elif clean_op == "read":
            return self.log_read(
                patient_id=patient_id,
                agent_id=agent_id,
                role=role,
                tool=tool,
                outcome=outcome,
                args=args,
                args_hash=args_hash,
                read_ids=read_ids,
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
        else:
            raise ValidationError(f"Invalid audit op '{op}'. Must be one of {sorted(VALID_OPS)}.")

    def pending_count(self, patient_id: str | None = None) -> int:
        """Returns the count of un-flushed read audit events in buffer."""
        with self._lock:
            if patient_id is None:
                return len(self._read_buffer)
            valid_pid = self._router.validate_patient_id(patient_id)
            return sum(1 for e in self._read_buffer if e.patient_id == valid_pid)

    def flush(self, patient_id: str | None = None) -> int:
        """Flushes buffered read events to their respective patient graphs."""
        with self._lock:
            if not self._read_buffer:
                return 0

            if patient_id is not None:
                valid_pid = self._router.validate_patient_id(patient_id)
                events_to_flush = [e for e in self._read_buffer if e.patient_id == valid_pid]
                self._read_buffer = [e for e in self._read_buffer if e.patient_id != valid_pid]
            else:
                events_to_flush = list(self._read_buffer)
                self._read_buffer.clear()

        if not events_to_flush:
            return 0

        # Group events by patient_id
        grouped: dict[str, list[AuditEventRecord]] = {}
        for evt in events_to_flush:
            grouped.setdefault(evt.patient_id, []).append(evt)

        total_flushed = 0
        batch_cypher = """
        UNWIND $events AS evt
        MATCH (p:Patient {id:$pid})
        CREATE (e:AuditEvent {
            id: evt.eid,
            ts: evt.ts,
            agent_id: evt.agent_id,
            role: evt.role,
            tool: evt.tool,
            op: evt.op,
            outcome: evt.outcome,
            args_hash: evt.args_hash,
            read_ids: evt.read_ids,
            trace_id: evt.trace_id,
            latency_ms: evt.latency_ms
        })
        RETURN count(e) AS cnt
        """

        for pid, evts in grouped.items():
            try:
                target_graph = self._router.graph(pid)
                payload = [
                    {
                        "eid": e.id,
                        "ts": e.ts,
                        "agent_id": e.agent_id,
                        "role": e.role,
                        "tool": e.tool,
                        "op": e.op,
                        "outcome": e.outcome,
                        "args_hash": e.args_hash,
                        "read_ids": e.read_ids,
                        "trace_id": e.trace_id or "",
                        "latency_ms": e.latency_ms,
                    }
                    for e in evts
                ]
                res = target_graph.query(batch_cypher, {"pid": pid, "events": payload})
                if res.result_set:
                    total_flushed += int(res.result_set[0][0])
                else:
                    total_flushed += len(evts)
            except Exception as ex:
                logger.error(
                    "Error flushing %d audit events for patient %s: %s",
                    len(evts),
                    pid,
                    ex,
                )
                # Re-queue on failure so events are not silently discarded
                with self._lock:
                    self._read_buffer.extend(evts)

        return total_flushed

    def get_audit_events(
        self,
        patient_id: str,
        limit: int = 50,
        op: str | None = None,
        role: str | None = None,
        trace_id: str | None = None,
        flush_first: bool = True,
    ) -> list[AuditEventRecord]:
        """Queries audit events for a patient graph with optional filters."""
        valid_pid = self._router.validate_patient_id(patient_id)
        if flush_first:
            self.flush(valid_pid)

        target_graph = self._router.graph(valid_pid)
        query = """
        MATCH (e:AuditEvent)
        WHERE ($op IS NULL OR e.op = $op)
          AND ($role IS NULL OR e.role = $role)
          AND ($trace_id IS NULL OR e.trace_id = $trace_id)
        RETURN e
        ORDER BY e.ts DESC
        LIMIT $limit
        """
        res = target_graph.query(
            query,
            {
                "op": op.strip().lower() if op else None,
                "role": role.strip().lower() if role else None,
                "trace_id": trace_id.strip() if trace_id else None,
                "limit": max(1, limit),
            },
        )

        records: list[AuditEventRecord] = []
        for row in res.result_set:
            node = row[0]
            props = _extract_node_props(node)
            records.append(AuditEventRecord.from_dict(props, valid_pid))

        return records

    def get_audit_event(self, patient_id: str, event_id: str) -> AuditEventRecord:
        """Retrieves a specific audit event by identifier."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        res = target_graph.query(
            "MATCH (e:AuditEvent {id:$eid}) RETURN e",
            {"eid": event_id},
        )
        if res.result_set:
            node = res.result_set[0][0]
            props = _extract_node_props(node)
            return AuditEventRecord.from_dict(props, valid_pid)

        # Check pending buffer in case un-flushed
        with self._lock:
            for item in self._read_buffer:
                if item.patient_id == valid_pid and item.id == event_id:
                    return item

        raise AuditEventNotFoundError(
            f"Audit event '{event_id}' not found for patient '{valid_pid}'.",
            details={"patient_id": valid_pid, "event_id": event_id},
        )

    def count_audit_events(
        self,
        patient_id: str,
        op: str | None = None,
        flush_first: bool = True,
    ) -> int:
        """Counts persisted audit events in the patient graph."""
        valid_pid = self._router.validate_patient_id(patient_id)
        if flush_first:
            self.flush(valid_pid)

        target_graph = self._router.graph(valid_pid)
        query = """
        MATCH (e:AuditEvent)
        WHERE ($op IS NULL OR e.op = $op)
        RETURN count(e) AS cnt
        """
        res = target_graph.query(
            query,
            {"op": op.strip().lower() if op else None},
        )
        if res.result_set:
            return int(res.result_set[0][0])
        return 0

    def verify_audit_trail(self, patient_id: str, trace_id: str) -> dict[str, Any]:
        """Aggregates and verifies chronological audit events for a workflow trace."""
        valid_pid = self._router.validate_patient_id(patient_id)
        self.flush(valid_pid)

        target_graph = self._router.graph(valid_pid)
        query = """
        MATCH (e:AuditEvent {trace_id:$trace_id})
        RETURN e
        ORDER BY e.ts ASC
        """
        res = target_graph.query(query, {"trace_id": trace_id.strip()})

        events: list[dict[str, Any]] = []
        op_counts: dict[str, int] = {"read": 0, "write": 0, "denied": 0}
        total_latency = 0.0
        tools: set[str] = set()
        roles: set[str] = set()

        for row in res.result_set:
            node = row[0]
            props = _extract_node_props(node)
            rec = AuditEventRecord.from_dict(props, valid_pid)
            events.append(rec.to_dict())

            op_counts[rec.op] = op_counts.get(rec.op, 0) + 1
            total_latency += rec.latency_ms
            tools.add(rec.tool)
            roles.add(rec.role)

        return {
            "patient_id": valid_pid,
            "trace_id": trace_id,
            "total_events": len(events),
            "op_counts": op_counts,
            "total_latency_ms": round(total_latency, 2),
            "tools": sorted(tools),
            "roles": sorted(roles),
            "events": events,
        }

    def audit_scope(
        self,
        patient_id: str,
        agent_id: str,
        role: str,
        tool: str,
        op: str = "read",
        args: Any = None,
        args_hash: str | None = None,
        trace_id: str | None = None,
    ) -> AuditScope:
        """Create an AuditScope context manager to automatically track latency and log."""
        return AuditScope(
            audit_logger=self,
            patient_id=patient_id,
            agent_id=agent_id,
            role=role,
            tool=tool,
            op=op,
            args=args,
            args_hash=args_hash,
            trace_id=trace_id,
        )

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.flush()

    def close(self) -> None:
        """Flushes remaining buffered audit events and terminates logger."""
        self.flush()


# Class alias for nomenclature flexibility
AuditManager = AuditLogger
