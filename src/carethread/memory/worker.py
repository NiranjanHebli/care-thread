"""Background worker and procedural memory consolidation engine.

Consolidation capabilities:
- Multi-episode candidate fact durability promotion (query q_cons_01)
- Temporal confidence decay marking for active facts (query q_cons_02)
- Automated handoff expiration sweep
- Playbook promotion tracking
- Immutable ConsolidationRun record logging with telemetry stats
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    ConsolidationError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.facts import FactManager, compute_slot
from carethread.memory.forgetting import ForgettingManager
from carethread.memory.handoff import HandoffManager
from carethread.memory.procedural import PlaybookManager
from carethread.util.id import candidate_id as gen_candidate_id
from carethread.util.id import consolidation_id as gen_consolidation_id

logger = logging.getLogger(__name__)

DEFAULT_STALE_THRESHOLD: float = 0.25


def _extract_props(node: Any) -> dict[str, Any]:
    """Safely extracts properties dictionary from a graph node or mapping."""
    if hasattr(node, "properties"):
        return dict(node.properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


@dataclass
class FactCandidateRecord:
    """Represents an unverified candidate fact awaiting cross-episode durability."""

    id: str
    patient_id: str
    kind: str
    slot: str
    display: str
    value: str = ""
    code: str | None = None
    system: str | None = None
    episode_id: str = ""
    status: str = "pending"
    recorded_at: int = 0
    promoted_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], patient_id: str) -> FactCandidateRecord:
        """Construct FactCandidateRecord from property mapping."""
        return cls(
            id=str(data.get("id", "")),
            patient_id=patient_id,
            kind=str(data.get("kind", "")),
            slot=str(data.get("slot", "")),
            display=str(data.get("display", "")),
            value=str(data.get("value", "")),
            code=data.get("code"),
            system=data.get("system"),
            episode_id=str(data.get("episode_id", "")),
            status=str(data.get("status", "pending")),
            recorded_at=int(data.get("recorded_at", 0)),
            promoted_at=int(data["promoted_at"]) if data.get("promoted_at") else None,
        )


@dataclass
class ConsolidationRunRecord:
    """Represents a logged background consolidation execution cycle."""

    id: str
    patient_id: str
    started_at: int
    finished_at: int
    trigger: str = "manual"
    stats: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        """Convert consolidation record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], patient_id: str) -> ConsolidationRunRecord:
        """Construct ConsolidationRunRecord from property mapping."""
        raw_stats = data.get("stats")
        parsed_stats: dict[str, Any] = {}
        if isinstance(raw_stats, str):
            try:
                parsed_stats = json.loads(raw_stats)
            except Exception:
                parsed_stats = {}
        elif isinstance(raw_stats, dict):
            parsed_stats = dict(raw_stats)

        return cls(
            id=str(data.get("id", "")),
            patient_id=patient_id,
            started_at=int(data.get("started_at", 0)),
            finished_at=int(data.get("finished_at", 0)),
            trigger=str(data.get("trigger", "manual")),
            stats=parsed_stats,
            status=str(data.get("status", "completed")),
        )


class ConsolidationWorker:
    """Consolidation worker executing durability promotions and decay sweeps."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        default_stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ) -> None:
        client = router._client if router is not None else get_db_client()
        self._router = router or TenantRouter(client)
        self._clock = clock or get_clock()
        self._stale_threshold = default_stale_threshold
        self._fact_manager = FactManager(router=self._router, clock=self._clock)
        self._handoff_manager = HandoffManager(router=self._router, clock=self._clock)
        self._playbook_manager = PlaybookManager(router=self._router, clock=self._clock)
        self._forgetting_manager = ForgettingManager(router=self._router, clock=self._clock)

    def resume_incomplete_deletions(self) -> list[str]:
        """Resumes and completes any tenant deletions left in 'deleting' status."""
        return self._forgetting_manager.resume_incomplete_deletions()

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

    def create_candidate(
        self,
        patient_id: str,
        kind: str,
        display: str,
        episode_id: str,
        value: str = "",
        code: str | None = None,
        system: str | None = None,
    ) -> FactCandidateRecord:
        """Creates an unpromoted FactCandidate node pending multi-episode durability."""
        valid_pid = self._router.validate_patient_id(patient_id)
        self._ensure_patient_exists(valid_pid)

        clean_kind = kind.strip().lower()
        clean_display = display.strip()
        if not clean_display:
            raise ValidationError("Candidate display text must not be empty.")

        clean_ep_id = episode_id.strip()
        if not clean_ep_id:
            raise ValidationError("Candidate must be bound to a source episode_id.")

        slot = compute_slot(clean_kind, code=code, display=clean_display, value=value)
        cid = gen_candidate_id()
        now_ts = int(self._clock.now().timestamp())
        target_graph = self._router.graph(valid_pid)

        create_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (c:FactCandidate {
            id: $cid,
            kind: $kind,
            slot: $slot,
            display: $display,
            value: $value,
            code: $code,
            system: $system,
            episode_id: $episode_id,
            status: 'pending',
            recorded_at: $now
        })
        CREATE (p)-[:HAS_CANDIDATE]->(c)
        RETURN c.id
        """
        target_graph.query(
            create_cypher,
            {
                "pid": valid_pid,
                "cid": cid,
                "kind": clean_kind,
                "slot": slot,
                "display": clean_display,
                "value": (value or clean_display).strip(),
                "code": (code or "").strip() or None,
                "system": (system or "").strip() or None,
                "episode_id": clean_ep_id,
                "now": now_ts,
            },
        )

        # Link to Episode if present
        target_graph.query(
            """
            MATCH (c:FactCandidate {id:$cid}), (ep:Episode {id:$eid})
            MERGE (c)-[:ASSERTED_IN]->(ep)
            """,
            {"cid": cid, "eid": clean_ep_id},
        )

        return FactCandidateRecord(
            id=cid,
            patient_id=valid_pid,
            kind=clean_kind,
            slot=slot,
            display=clean_display,
            value=(value or clean_display).strip(),
            code=(code or "").strip() or None,
            system=(system or "").strip() or None,
            episode_id=clean_ep_id,
            status="pending",
            recorded_at=now_ts,
        )

    def run_consolidation(
        self,
        patient_id: str | None = None,
        trigger: str = "manual",
        stale_threshold: float | None = None,
    ) -> ConsolidationRunRecord | list[ConsolidationRunRecord]:
        """Executes full consolidation cycle across one or all registered patients."""
        if patient_id is None:
            tenants = self._router.list_tenants()
            results: list[ConsolidationRunRecord] = []
            for t in tenants:
                try:
                    res = self._run_patient_consolidation(t, trigger, stale_threshold)
                    results.append(res)
                except Exception as ex:
                    logger.error("Consolidation failed for tenant %s: %s", t, ex)
            return results

        valid_pid = self._router.validate_patient_id(patient_id)
        return self._run_patient_consolidation(valid_pid, trigger, stale_threshold)

    def _run_patient_consolidation(
        self,
        valid_pid: str,
        trigger: str,
        stale_threshold: float | None,
    ) -> ConsolidationRunRecord:
        """Internal worker executing consolidation on a single validated patient."""
        self._ensure_patient_exists(valid_pid)
        target_graph = self._router.graph(valid_pid)

        started_at = int(self._clock.now().timestamp())
        perf_start = time.perf_counter()
        thresh = stale_threshold if stale_threshold is not None else self._stale_threshold

        # Step 1: Candidate Promotion via query q_cons_01
        q_cons_01 = get_query("q_cons_01")
        cons_res = target_graph.query(q_cons_01, {"pid": valid_pid})
        promoted_count = 0

        for row in cons_res.result_set:
            kind = row[0]
            slot = row[1]
            cids = row[3] or []

            if not cids:
                continue

            # Fetch first candidate details to construct semantic Fact
            first_cid = cids[0]
            cand_res = target_graph.query(
                "MATCH (c:FactCandidate {id:$cid}) RETURN c",
                {"cid": first_cid},
            )
            if not cand_res.result_set:
                continue

            c_props = _extract_props(cand_res.result_set[0][0])
            c_display = c_props.get("display", slot)
            c_value = c_props.get("value", c_display)
            c_code = c_props.get("code")
            c_system = c_props.get("system")

            # Promote into durable Fact via FactManager
            self._fact_manager.remember_fact(
                patient_id=valid_pid,
                kind=kind,
                display=c_display,
                value=c_value,
                code=c_code,
                system=c_system,
                authority="record",
                caller_role="system",
            )
            promoted_count += 1

            # Mark candidates as promoted
            target_graph.query(
                """
                UNWIND $cids AS cid
                MATCH (c:FactCandidate {id:cid})
                SET c.status = 'promoted', c.promoted_at = $now
                """,
                {"cids": cids, "now": started_at},
            )

        # Step 2: Temporal Confidence Decay Marking via query q_cons_02
        q_cons_02 = get_query("q_cons_02")
        decay_res = target_graph.query(
            q_cons_02,
            {
                "pid": valid_pid,
                "now": started_at,
                "stale_threshold": thresh,
            },
        )
        decayed_count = len(decay_res.result_set) if decay_res.result_set else 0

        # Step 3: Sweep stale open handoffs past TTL
        expired_handoffs = self._handoff_manager.expire_stale_handoffs(valid_pid)

        # Step 4: Count active playbooks
        active_pb_res = target_graph.query(
            "MATCH (pb:Playbook {status:'active'}) RETURN count(pb)",
        )
        active_playbooks = int(active_pb_res.result_set[0][0]) if active_pb_res.result_set else 0

        # Step 5: Record ConsolidationRun node
        finished_at = int(self._clock.now().timestamp())
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0

        stats: dict[str, Any] = {
            "promoted_candidates": promoted_count,
            "decayed_facts": decayed_count,
            "expired_handoffs": expired_handoffs,
            "active_playbooks": active_playbooks,
            "latency_ms": round(elapsed_ms, 2),
        }

        cns_id = gen_consolidation_id()
        create_cns_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (cns:ConsolidationRun {
            id: $cid,
            started_at: $start_ts,
            finished_at: $end_ts,
            trigger: $trigger,
            stats: $stats_json,
            status: 'completed'
        })
        CREATE (p)-[:HAS_CONSOLIDATION]->(cns)
        RETURN cns.id
        """
        cns_res = target_graph.query(
            create_cns_cypher,
            {
                "pid": valid_pid,
                "cid": cns_id,
                "start_ts": started_at,
                "end_ts": finished_at,
                "trigger": trigger.strip().lower(),
                "stats_json": json.dumps(stats),
            },
        )
        if not cns_res.result_set:
            raise ConsolidationError(
                f"Failed to record consolidation run node for patient '{valid_pid}'."
            )

        return ConsolidationRunRecord(
            id=cns_id,
            patient_id=valid_pid,
            started_at=started_at,
            finished_at=finished_at,
            trigger=trigger.strip().lower(),
            stats=stats,
            status="completed",
        )

    def get_consolidation_history(
        self,
        patient_id: str,
        limit: int = 10,
    ) -> list[ConsolidationRunRecord]:
        """Retrieves past consolidation run records ordered by start time descending."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_CONSOLIDATION]->(cns:ConsolidationRun)
        RETURN cns
        ORDER BY cns.started_at DESC
        LIMIT $limit
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "limit": max(1, limit)})

        records: list[ConsolidationRunRecord] = []
        for row in res.result_set:
            props = _extract_props(row[0])
            records.append(ConsolidationRunRecord.from_dict(props, valid_pid))
        return records
