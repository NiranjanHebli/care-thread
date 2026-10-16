"""Handoff state machine and coordination engine for inter-agent delegation.

Handoff state machine implementation:
- Handoff lifecycle: draft -> open -> accepted -> completed (or rejected / expired)
- CAS atomic accept: validates status='open', to_role matches, and ttl_at > now
- Carried context payload (Fact, Note, Episode, OpenQuestion, Action)
- TTL enforcement and background expiration
- Inter-agent coordination (Intake -> Specialist -> Pharmacy -> Clinician)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    HandoffNotFoundError,
    HandoffStateError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.util.id import handoff_id as gen_handoff_id
from carethread.util.id import ulid_prefixed

logger = logging.getLogger(__name__)

# Valid handoff statuses and agent roles
VALID_HANDOFF_STATUSES: frozenset[str] = frozenset(
    {"draft", "open", "accepted", "completed", "rejected", "expired"}
)

VALID_ROLES: frozenset[str] = frozenset({"intake", "specialist", "pharmacy", "clinician", "system"})

DEFAULT_HANDOFF_TTL_SECONDS: int = 86400  # 24 hours


@dataclass
class HandoffPacket:
    """Represents a coordination handoff packet between clinical agents."""

    id: str
    patient_id: str
    status: str
    from_role: str
    to_role: str
    goal: str
    state_summary: str = ""
    created_at: int = 0
    ttl_at: int = 0
    accepted_at: int | None = None
    completed_at: int | None = None
    clinician_review_required: bool = False
    result_summary: str | None = None
    version: int = 1
    from_agent_id: str | None = None
    to_agent_id: str | None = None
    session_id: str | None = None
    carried_facts: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HandoffManager:
    """Manages inter-agent handoff packets, CAS delegation, and state transitions."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clock = clock or get_clock()

    def create_handoff(
        self,
        patient_id: str,
        from_role: str,
        to_role: str,
        goal: str,
        state_summary: str = "",
        from_agent_id: str | None = None,
        session_id: str | None = None,
        carried_fact_ids: list[str] | None = None,
        open_questions: list[str] | None = None,
        actions: list[str] | None = None,
        clinician_review_required: bool = False,
        ttl_seconds: int = DEFAULT_HANDOFF_TTL_SECONDS,
        publish: bool = True,
        handoff_id: str | None = None,
    ) -> HandoffPacket:
        """Create a new coordination handoff packet and optionally publish it to open state.

        Args:
            patient_id: Validated patient tenant identifier.
            from_role: Originating agent role ('intake', 'specialist', 'pharmacy', 'clinician').
            to_role: Target agent role.
            goal: Specific clinical objective of the delegation.
            state_summary: Narrative summary of current case findings.
            from_agent_id: Identifier of the initiating agent node.
            session_id: Session identifier where the handoff was created.
            carried_fact_ids: List of Fact node IDs carried as supporting evidence.
            open_questions: List of unresolved questions for the receiving agent.
            actions: List of recommended action item strings.
            clinician_review_required: Whether handoff requires mandatory clinician approval.
            ttl_seconds: Time-to-live expiration window in seconds (default 24h).
            publish: If True, transitions immediately from 'draft' to 'open'.
            handoff_id: Optional explicit ULID identifier.

        Returns:
            HandoffPacket object.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if not goal or not goal.strip():
            raise ValidationError(
                "Handoff goal cannot be empty.",
                details={"patient_id": valid_pid},
            )

        f_role = from_role.strip().lower()
        t_role = to_role.strip().lower()
        if f_role not in VALID_ROLES:
            raise ValidationError(
                f"Invalid from_role '{from_role}'. Allowed: {sorted(VALID_ROLES)}",
                details={"from_role": from_role, "allowed": list(VALID_ROLES)},
            )
        if t_role not in VALID_ROLES:
            raise ValidationError(
                f"Invalid to_role '{to_role}'. Allowed: {sorted(VALID_ROLES)}",
                details={"to_role": to_role, "allowed": list(VALID_ROLES)},
            )

        target_graph = self._router.graph(valid_pid)

        # Check patient exists
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        now_ts = self._clock.now_ts()
        ttl_at = now_ts + ttl_seconds
        hid = handoff_id or gen_handoff_id()
        agent_from = from_agent_id or f"agent_{f_role}"
        ses_id = session_id or "ses_default"

        # Ensure Agent and Session nodes exist in the graph
        target_graph.query(
            """
            MERGE (a:Agent {id:$aid})
            ON CREATE SET a.role = $role, a.created_at = $now
            """,
            {"aid": agent_from, "role": f_role, "now": now_ts},
        )
        target_graph.query(
            """
            MATCH (p:Patient {id:$pid})
            MERGE (s:Session {id:$sid})
            ON CREATE SET s.patient_id = $pid, s.status = 'active', s.created_at = $now
            MERGE (p)-[:HAS_SESSION]->(s)
            """,
            {"pid": valid_pid, "sid": ses_id, "now": now_ts},
        )

        fact_ids = list(carried_fact_ids) if carried_fact_ids else []

        # Execute query q_ho_01a
        q_ho_01a = get_query("q_ho_01a")
        target_graph.query(
            q_ho_01a,
            {
                "pid": valid_pid,
                "from_agent": agent_from,
                "session_id": ses_id,
                "hid": hid,
                "from_role": f_role,
                "to_role": t_role,
                "goal": goal.strip(),
                "state": state_summary.strip(),
                "now": now_ts,
                "ttl_at": ttl_at,
                "cr": clinician_review_required,
                "fact_ids": fact_ids,
            },
        )

        # Create OpenQuestion nodes if provided
        created_questions: list[dict[str, Any]] = []
        if open_questions:
            for q_text in open_questions:
                if q_text and q_text.strip():
                    qid = ulid_prefixed("qst")
                    target_graph.query(
                        """
                        MATCH (h:Handoff {id:$hid})
                        CREATE (q:OpenQuestion {
                            id: $qid, question: $text, asker_agent_id: $aid,
                            status: 'open', created_at: $now
                        })
                        CREATE (h)-[:HAS_QUESTION]->(q)
                        """,
                        {
                            "hid": hid,
                            "qid": qid,
                            "text": q_text.strip(),
                            "aid": agent_from,
                            "now": now_ts,
                        },
                    )
                    created_questions.append({"id": qid, "question": q_text.strip()})

        # Create Action nodes if provided
        created_actions: list[dict[str, Any]] = []
        if actions:
            for act_text in actions:
                if act_text and act_text.strip():
                    act_id = ulid_prefixed("act")
                    target_graph.query(
                        """
                        MATCH (h:Handoff {id:$hid})
                        CREATE (a:Action {
                            id: $act_id, type: 'recommended_action',
                            description: $desc, status: 'pending', assigned_to: $to_role
                        })
                        CREATE (h)-[:HAS_ACTION]->(a)
                        """,
                        {
                            "hid": hid,
                            "act_id": act_id,
                            "desc": act_text.strip(),
                            "to_role": t_role,
                        },
                    )
                    created_actions.append({"id": act_id, "description": act_text.strip()})

        # Step 2: Publish immediately if requested via q_ho_01b
        if publish:
            q_ho_01b = get_query("q_ho_01b")
            target_graph.query(q_ho_01b, {"hid": hid})

        return self.get_handoff(valid_pid, hid)

    def publish_handoff(self, patient_id: str, handoff_id: str) -> HandoffPacket:
        """Publish a draft handoff to open status via query q_ho_01b."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Ensure handoff exists
        cur = self.get_handoff(valid_pid, handoff_id)
        if cur.status != "draft":
            raise HandoffStateError(
                f"Cannot publish handoff '{handoff_id}': status is '{cur.status}', not 'draft'.",
                details={"handoff_id": handoff_id, "current_status": cur.status},
            )

        q_ho_01b = get_query("q_ho_01b")
        res = target_graph.query(q_ho_01b, {"hid": handoff_id})
        if not res.result_set:
            raise HandoffStateError(
                f"Failed to publish handoff '{handoff_id}'.",
                details={"handoff_id": handoff_id},
            )

        return self.get_handoff(valid_pid, handoff_id)

    def accept_handoff(
        self,
        patient_id: str,
        handoff_id: str,
        agent_id: str,
        role: str,
        session_id: str | None = None,
    ) -> HandoffPacket:
        """Atomically accept an open handoff via compare-and-swap (CAS) query q_ho_02a.

        Enforces:
        - status == 'open'
        - to_role == role
        - ttl_at > now
        - Atomic version increment (CAS concurrency guarantee)
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Inspect current state for actionable error messages
        cur = self.get_handoff(valid_pid, handoff_id)
        now_ts = self._clock.now_ts()

        norm_role = role.strip().lower()
        if cur.status != "open":
            raise HandoffStateError(
                f"Cannot accept handoff '{handoff_id}': status is '{cur.status}', not 'open'.",
                details={"handoff_id": handoff_id, "status": cur.status},
            )

        if cur.to_role != norm_role:
            raise HandoffStateError(
                f"Role mismatch for handoff '{handoff_id}': assigned to '{cur.to_role}', "
                f"but role '{norm_role}' attempted acceptance.",
                details={"handoff_id": handoff_id, "expected": cur.to_role, "actual": norm_role},
            )

        if cur.ttl_at <= now_ts:
            # Mark expired
            target_graph.query(
                "MATCH (h:Handoff {id:$hid}) SET h.status = 'expired'",
                {"hid": handoff_id},
            )
            raise HandoffStateError(
                f"Handoff '{handoff_id}' has expired (TTL {cur.ttl_at} <= {now_ts}).",
                details={"handoff_id": handoff_id, "ttl_at": cur.ttl_at, "now": now_ts},
            )

        # Ensure accepting Agent and Session exist
        ses_id = session_id or cur.session_id or "ses_default"
        target_graph.query(
            """
            MERGE (a:Agent {id:$aid})
            ON CREATE SET a.role = $role, a.created_at = $now
            """,
            {"aid": agent_id, "role": norm_role, "now": now_ts},
        )
        target_graph.query(
            """
            MATCH (p:Patient {id:$pid})
            MERGE (s:Session {id:$sid})
            ON CREATE SET s.patient_id = $pid, s.status = 'active', s.created_at = $now
            MERGE (p)-[:HAS_SESSION]->(s)
            """,
            {"pid": valid_pid, "sid": ses_id, "now": now_ts},
        )

        # Execute CAS accept query q_ho_02a
        q_ho_02a = get_query("q_ho_02a")
        res = target_graph.query(
            q_ho_02a,
            {
                "hid": handoff_id,
                "role": norm_role,
                "now": now_ts,
                "agent_id": agent_id,
                "session_id": ses_id,
            },
        )

        if not res.result_set:
            # Concurrency race: another agent accepted simultaneously
            latest = self.get_handoff(valid_pid, handoff_id)
            raise HandoffStateError(
                f"CAS conflict: handoff '{handoff_id}' could not be accepted. "
                f"Current status is '{latest.status}', accepted by '{latest.to_agent_id}'.",
                details={"handoff_id": handoff_id, "current_handoff": latest.to_dict()},
            )

        return self.get_handoff(valid_pid, handoff_id)

    def complete_handoff(
        self,
        patient_id: str,
        handoff_id: str,
        result_summary: str,
        completed_by_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> HandoffPacket:
        """Transition an accepted handoff to completed and record result findings."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_handoff(valid_pid, handoff_id)
        if cur.status != "accepted":
            raise HandoffStateError(
                f"Cannot complete handoff '{handoff_id}': status is '{cur.status}', "
                "must be 'accepted'.",
                details={"handoff_id": handoff_id, "current_status": cur.status},
            )

        now_ts = self._clock.now_ts()
        res_text = result_summary.strip() if result_summary else "Completed"

        cypher = """
        MATCH (h:Handoff {id:$hid})
        WHERE h.status = 'accepted'
        SET h.status = 'completed',
            h.completed_at = $now,
            h.result_summary = $res_summary,
            h.version = h.version + 1
        RETURN h.id
        """
        res = target_graph.query(
            cypher,
            {"hid": handoff_id, "now": now_ts, "res_summary": res_text},
        )
        if not res.result_set:
            raise HandoffStateError(
                f"Failed to complete handoff '{handoff_id}'.",
                details={"handoff_id": handoff_id},
            )

        # Link session if provided
        if session_id:
            target_graph.query(
                """
                MATCH (h:Handoff {id:$hid}), (s:Session {id:$sid})
                MERGE (h)-[:COMPLETED_IN]->(s)
                """,
                {"hid": handoff_id, "sid": session_id},
            )

        return self.get_handoff(valid_pid, handoff_id)

    def reject_handoff(
        self,
        patient_id: str,
        handoff_id: str,
        rejected_by: str,
        rejection_reason: str,
    ) -> HandoffPacket:
        """Reject an open or accepted handoff delegation."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_handoff(valid_pid, handoff_id)
        if cur.status not in ("open", "accepted", "draft"):
            raise HandoffStateError(
                f"Cannot reject handoff '{handoff_id}': status is '{cur.status}'.",
                details={"handoff_id": handoff_id, "status": cur.status},
            )

        now_ts = self._clock.now_ts()
        cypher = """
        MATCH (h:Handoff {id:$hid})
        SET h.status = 'rejected',
            h.rejected_at = $now,
            h.rejected_by = $rejected_by,
            h.rejection_reason = $reason,
            h.version = h.version + 1
        RETURN h.id
        """
        target_graph.query(
            cypher,
            {
                "hid": handoff_id,
                "now": now_ts,
                "rejected_by": rejected_by,
                "reason": rejection_reason,
            },
        )

        return self.get_handoff(valid_pid, handoff_id)

    def expire_stale_handoffs(self, patient_id: str | None = None) -> int:
        """Expire all open handoffs whose TTL timestamp has passed."""
        now_ts = self._clock.now_ts()
        pids = [patient_id] if patient_id else self._router.list_tenants()
        total_expired = 0

        for pid in pids:
            try:
                g = self._router.graph(pid)
                res = g.query(
                    """
                    MATCH (h:Handoff)
                    WHERE h.status = 'open' AND h.ttl_at <= $now
                    SET h.status = 'expired', h.version = h.version + 1
                    RETURN count(h)
                    """,
                    {"now": now_ts},
                )
                if res.result_set:
                    total_expired += res.result_set[0][0]
            except Exception as e:
                logger.warning(f"Failed to check handoff expiry for {pid}: {e}")

        return total_expired

    def get_handoff(self, patient_id: str, handoff_id: str) -> HandoffPacket:
        """Retrieve complete handoff state using query q_ho_02b."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Check patient exists
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        # Query handoff node directly to verify existence
        check_res = target_graph.query(
            """
            MATCH (h:Handoff {id:$hid})
            OPTIONAL MATCH (h)-[:FROM]->(frm:Agent)
            OPTIONAL MATCH (h)-[:TO]->(to_a:Agent)
            OPTIONAL MATCH (h)-[:CREATED_IN]->(s:Session)
            RETURN h, frm.id, to_a.id, s.id
            """,
            {"hid": handoff_id},
        )
        if not check_res.result_set:
            raise HandoffNotFoundError(
                f"Handoff '{handoff_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "handoff_id": handoff_id},
            )

        h_node = check_res.result_set[0][0]
        frm_id = check_res.result_set[0][1]
        to_id = check_res.result_set[0][2]
        s_id = check_res.result_set[0][3]

        # Extract properties
        props = h_node.properties if hasattr(h_node, "properties") else {}

        # Fetch carried items via q_ho_02b
        q_ho_02b = get_query("q_ho_02b")
        res_b = target_graph.query(q_ho_02b, {"hid": handoff_id})

        carried_memory: list[dict[str, Any]] = []
        questions: list[dict[str, Any]] = []
        actions_list: list[dict[str, Any]] = []

        if res_b.result_set:
            row = res_b.result_set[0]
            raw_mem = row[1] or []
            raw_q = row[2] or []
            raw_act = row[3] or []

            for m in raw_mem:
                if isinstance(m, dict) and m.get("id"):
                    carried_memory.append(m)

            for q in raw_q:
                q_props = q.properties if hasattr(q, "properties") else {}
                if q_props.get("id"):
                    questions.append(q_props)

            for a in raw_act:
                a_props = a.properties if hasattr(a, "properties") else {}
                if a_props.get("id"):
                    actions_list.append(a_props)

        return HandoffPacket(
            id=props.get("id", handoff_id),
            patient_id=valid_pid,
            status=props.get("status", "open"),
            from_role=props.get("from_role", ""),
            to_role=props.get("to_role", ""),
            goal=props.get("goal", ""),
            state_summary=props.get("state_summary", ""),
            created_at=props.get("created_at", 0),
            ttl_at=props.get("ttl_at", 0),
            accepted_at=props.get("accepted_at"),
            completed_at=props.get("completed_at"),
            clinician_review_required=bool(props.get("clinician_review_required", False)),
            result_summary=props.get("result_summary"),
            version=props.get("version", 1),
            from_agent_id=frm_id,
            to_agent_id=to_id,
            session_id=s_id,
            carried_facts=carried_memory,
            open_questions=questions,
            actions=actions_list,
        )

    def list_handoffs(
        self,
        patient_id: str,
        to_role: str | None = None,
        status: str | None = None,
    ) -> list[HandoffPacket]:
        """List handoffs for a patient filtered by role or status."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_HANDOFF]->(h:Handoff)
        WHERE ($to_role IS NULL OR h.to_role = $to_role)
          AND ($status IS NULL OR h.status = $status)
        RETURN h.id
        ORDER BY h.created_at DESC
        """
        res = target_graph.query(
            cypher,
            {"pid": valid_pid, "to_role": to_role, "status": status},
        )

        packets: list[HandoffPacket] = []
        for row in res.result_set or []:
            packets.append(self.get_handoff(valid_pid, row[0]))
        return packets
