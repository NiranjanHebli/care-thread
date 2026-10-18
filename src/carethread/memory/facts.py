"""Semantic memory management with bitemporal facts and CAS supersession.

Features:
- Bitemporal facts: system belief time (recorded/superseded) and real-world time (validity)
- Deterministic slot assignment for conditions, medications, allergies, and preferences
- Write rules: creation, reinforcement (confidence compounding), and conflict resolution
- Authority hierarchy and base confidence scoring
- Compare-and-swap (CAS) atomic supersession via query q_sup_01
- Version lineage history traversal via query q_sup_02
- Point-in-time as-of queries via query q_recall_02
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    ConflictStaleVersionError,
    FactNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.ingest.synthea_csv import normalize_med
from carethread.util.id import fact_id as gen_fact_id
from carethread.util.id import flag_id as gen_flag_id

logger = logging.getLogger(__name__)

VALID_FACT_KINDS: frozenset[str] = frozenset(
    {"condition", "medication", "allergy", "preference", "symptom"}
)
VALID_FACT_STATUSES: frozenset[str] = frozenset(
    {"active", "superseded", "disputed", "pending_review", "retracted"}
)

# Authority rank hierarchy: higher rank supersedes lower rank
AUTHORITY_RANK: dict[str, int] = {
    "clinician": 5,
    "record": 4,
    "pharmacy": 3,
    "patient_reported": 2,
    "agent_inferred": 1,
}

# Base confidence scores by authority
BASE_CONFIDENCE: dict[str, float] = {
    "clinician": 1.0,
    "record": 0.95,
    "pharmacy": 0.9,
    "patient_reported": 0.7,
    "agent_inferred": 0.5,
}

DEFAULT_HALF_LIFE_DAYS: int = 180


def compute_slot(
    kind: str,
    code: str | None = None,
    display: str = "",
    value: str | None = None,
) -> str:
    """Computes the unique slot identifier for a semantic fact.

    Examples:
    - medication 'Metformin 500 MG' -> 'med:metformin'
    - allergy code '300916003' -> 'allergy:300916003'
    - condition code '44054006' -> 'cond:44054006'
    - preference 'morning visits' -> 'pref:morning_visits'
    """
    clean_kind = kind.strip().lower()
    if clean_kind == "medication":
        med_desc = display or value or ""
        ing = normalize_med(med_desc)
        return f"med:{ing}"
    elif clean_kind == "allergy":
        token = code or display.strip().lower().replace(" ", "_")
        return f"allergy:{token}"
    elif clean_kind == "condition":
        token = code or display.strip().lower().replace(" ", "_")
        return f"cond:{token}"
    elif clean_kind == "preference":
        token = (code or display).strip().lower().replace(" ", "_")
        return f"pref:{token}"
    elif clean_kind == "symptom":
        token = (code or display).strip().lower().replace(" ", "_")
        return f"symp:{token}"
    else:
        token = (code or display).strip().lower().replace(" ", "_")
        return f"{clean_kind}:{token}"


@dataclass
class FactRecord:
    """Represents a bitemporal clinical semantic fact."""

    id: str
    patient_id: str
    kind: str
    slot: str
    display: str
    value: str = ""
    code: str | None = None
    system: str | None = None
    status: str = "active"
    valid_from: int = 0
    valid_to: int | None = None
    recorded_at: int = 0
    superseded_at: int | None = None
    last_confirmed_at: int = 0
    confidence: float = 0.7
    authority: str = "patient_reported"
    evidence_count: int = 1
    stale: bool = False
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS
    version: int = 1
    source_episode_id: str | None = None
    asserted_by: str | None = None
    op_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactManager:
    """Manages bitemporal semantic memory facts, supersession, and belief history."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clock = clock or get_clock()
        self._lock = threading.RLock()

    def get_fact(self, patient_id: str, fact_id: str) -> FactRecord | None:
        """Fetch a specific fact by ID."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {id:$fid})
        RETURN f.id, f.kind, f.slot, f.display, f.value, f.code, f.system,
               f.status, f.valid_from, f.valid_to, f.recorded_at, f.superseded_at,
               f.last_confirmed_at, f.confidence, f.authority, f.evidence_count,
               f.stale, f.half_life_days, f.version, f.source_episode_id,
               f.asserted_by, f.op_id
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "fid": fact_id})
        if not res.result_set:
            return None

        row = res.result_set[0]
        return self._row_to_record(valid_pid, row)

    def get_active_fact_in_slot(self, patient_id: str, slot: str) -> FactRecord | None:
        """Fetch the current active or disputed fact in a given slot."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {slot:$slot})
        WHERE f.status IN ['active', 'disputed', 'pending_review']
        RETURN f.id, f.kind, f.slot, f.display, f.value, f.code, f.system,
               f.status, f.valid_from, f.valid_to, f.recorded_at, f.superseded_at,
               f.last_confirmed_at, f.confidence, f.authority, f.evidence_count,
               f.stale, f.half_life_days, f.version, f.source_episode_id,
               f.asserted_by, f.op_id
        ORDER BY f.version DESC, f.valid_from DESC
        LIMIT 1
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "slot": slot})
        if not res.result_set:
            return None

        return self._row_to_record(valid_pid, res.result_set[0])

    def remember_fact(
        self,
        patient_id: str,
        kind: str,
        display: str,
        value: str = "",
        code: str | None = None,
        system: str | None = None,
        valid_from: int | None = None,
        valid_to: int | None = None,
        authority: str | None = None,
        caller_role: str | None = None,
        episode_id: str | None = None,
        message_id: str | None = None,
        op_id: str | None = None,
        expected_version: int | None = None,
        half_life_days: int | None = None,
    ) -> dict[str, Any]:
        """Store or update a semantic fact applying write rules and conflict policy."""
        with self._lock:
            return self._remember_fact_unlocked(
                patient_id=patient_id,
                kind=kind,
                display=display,
                value=value,
                code=code,
                system=system,
                valid_from=valid_from,
                valid_to=valid_to,
                authority=authority,
                caller_role=caller_role,
                episode_id=episode_id,
                message_id=message_id,
                op_id=op_id,
                expected_version=expected_version,
                half_life_days=half_life_days,
            )

    def _remember_fact_unlocked(
        self,
        patient_id: str,
        kind: str,
        display: str,
        value: str = "",
        code: str | None = None,
        system: str | None = None,
        valid_from: int | None = None,
        valid_to: int | None = None,
        authority: str | None = None,
        caller_role: str | None = None,
        episode_id: str | None = None,
        message_id: str | None = None,
        op_id: str | None = None,
        expected_version: int | None = None,
        half_life_days: int | None = None,
    ) -> dict[str, Any]:
        valid_pid = self._router.validate_patient_id(patient_id)
        if kind not in VALID_FACT_KINDS:
            raise ValidationError(
                f"Invalid fact kind '{kind}'. Allowed: {sorted(VALID_FACT_KINDS)}",
                details={"kind": kind, "allowed": list(VALID_FACT_KINDS)},
            )

        now_ts = self._clock.now_ts()
        v_from = valid_from if valid_from is not None else now_ts
        slot = compute_slot(kind, code=code, display=display, value=value)
        val = (value if value else display).strip()

        # Resolve authority and confidence
        auth = self._resolve_authority(authority, caller_role)
        conf = BASE_CONFIDENCE.get(auth, 0.7)
        h_life = half_life_days or DEFAULT_HALF_LIFE_DAYS

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

        # Replay check for idempotency
        if op_id:
            op_res = target_graph.query(
                "MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {op_id:$op_id}) RETURN f.id",
                {"pid": valid_pid, "op_id": op_id},
            )
            if op_res.result_set:
                existing = self.get_fact(valid_pid, op_res.result_set[0][0])
                return {
                    "action": "replayed",
                    "fact_id": existing.id if existing else None,
                    "status": existing.status if existing else "active",
                    "fact": existing.to_dict() if existing else None,
                    "replayed": True,
                }

        cur = self.get_active_fact_in_slot(valid_pid, slot)

        # Version CAS check
        if expected_version is not None and cur is not None:
            if cur.version != expected_version:
                raise ConflictStaleVersionError(
                    f"Version conflict: expected {expected_version} "
                    f"but slot version is {cur.version}.",
                    details={
                        "slot": slot,
                        "expected_version": expected_version,
                        "current_version": cur.version,
                        "current_fact": cur.to_dict(),
                    },
                )

        # Case 1: No active fact in slot -> Create
        if cur is None:
            # Specialist agent proposing condition starts as pending_review
            is_spec_cond = caller_role == "specialist" and kind == "condition"
            status = "pending_review" if is_spec_cond else "active"
            action = status

            new_fid = gen_fact_id()
            create_cypher = """
            MATCH (p:Patient {id:$pid})
            CREATE (f:Fact {
                id: $fid, kind: $kind, slot: $slot, display: $display, value: $value,
                code: $code, system: $system, status: $status, valid_from: $valid_from,
                valid_to: $valid_to, recorded_at: $now, superseded_at: null,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(f)
            WITH f
            OPTIONAL MATCH (ep:Episode {id:$episode_id})
            FOREACH (_ IN CASE WHEN ep IS NOT NULL THEN [1] ELSE [] END |
                CREATE (f)-[:ASSERTED_IN {message_id:$message_id}]->(ep)
            )
            RETURN f.id
            """
            target_graph.query(
                create_cypher,
                {
                    "pid": valid_pid,
                    "fid": new_fid,
                    "kind": kind,
                    "slot": slot,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "status": status,
                    "valid_from": v_from,
                    "valid_to": valid_to,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                    "message_id": message_id or "",
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": action,
                "fact_id": new_fid,
                "status": status,
                "fact": rec.to_dict() if rec else None,
            }

        # Case 2: Active fact exists with same normalized value -> Reinforce
        if cur.value.strip().lower() == val.lower():
            reinforced = self.reinforce_fact(valid_pid, cur.id, now=now_ts)
            return {
                "action": "reinforced",
                "fact_id": cur.id,
                "status": reinforced.status,
                "fact": reinforced.to_dict(),
            }

        # Case 3: Active fact exists with different value -> Apply conflict policy
        cur_rank = AUTHORITY_RANK.get(cur.authority, 2)
        new_rank = AUTHORITY_RANK.get(auth, 2)

        # Safety asymmetry: allergy removal at equal/lower authority never auto-supersedes
        if kind == "allergy" and self._is_negation_or_removal(val) and new_rank <= cur_rank:
            new_fid = gen_fact_id()
            flg_id = gen_flag_id()
            cypher = """
            MATCH (p:Patient {id:$pid}), (old:Fact {id:$old_id})
            CREATE (new:Fact {
                id: $fid, kind: $kind, slot: $slot, display: $display, value: $value,
                code: $code, system: $system, status: 'pending_review', valid_from: $valid_from,
                valid_to: null, recorded_at: $now, superseded_at: null,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (flg:Flag {
                id: $flg_id, type: 'allergy_med_conflict', severity: 'high',
                status: 'open', rule_id: 'safety_asymmetry_allergy_removal',
                explanation: 'Attempted allergy removal requires clinician confirmation.',
                detected_at: $now
            })
            CREATE (p)-[:HAS_FLAG]->(flg)
            CREATE (flg)-[:ABOUT]->(new)
            CREATE (flg)-[:ABOUT]->(old)
            RETURN new.id, flg.id
            """
            target_graph.query(
                cypher,
                {
                    "pid": valid_pid,
                    "old_id": cur.id,
                    "fid": new_fid,
                    "kind": kind,
                    "slot": slot,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "valid_from": v_from,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                    "flg_id": flg_id,
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": "pending_review",
                "fact_id": new_fid,
                "status": "pending_review",
                "flag_id": flg_id,
                "fact": rec.to_dict() if rec else None,
            }

        # Subcase 3A: Higher authority incoming wins -> Supersedes old fact
        if new_rank > cur_rank:
            new_fid = gen_fact_id()
            cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(old:Fact {id:$old_id})
            SET old.status = 'superseded',
                old.superseded_at = $now,
                old.valid_to = coalesce(old.valid_to, $now),
                old.version = old.version + 1
            WITH p, old
            OPTIONAL MATCH (p)-[:HAS_FACT]->(other:Fact {slot: old.slot, status: 'active'})
            WHERE other.id <> old.id
            WITH p, old, collect(other) AS others
            FOREACH (o IN others |
                SET o.status = 'superseded',
                    o.superseded_at = $now,
                    o.valid_to = coalesce(o.valid_to, $now),
                    o.version = o.version + 1
            )
            CREATE (new:Fact {
                id: $fid, kind: old.kind, slot: old.slot, code: $code, system: $system,
                display: $display, value: $value, status: 'active', valid_from: $valid_from,
                valid_to: null, recorded_at: $now, superseded_at: null,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (new)-[:SUPERSEDES {reason: 'higher_authority', at: $now}]->(old)
            RETURN new.id
            """
            target_graph.query(
                cypher,
                {
                    "pid": valid_pid,
                    "old_id": cur.id,
                    "fid": new_fid,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "valid_from": v_from,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": "superseded",
                "fact_id": new_fid,
                "superseded_id": cur.id,
                "status": "active",
                "fact": rec.to_dict() if rec else None,
            }

        # Subcase 3B: Lower authority incoming loses -> Archived as superseded
        if new_rank < cur_rank:
            new_fid = gen_fact_id()
            cypher = """
            MATCH (p:Patient {id:$pid}), (old:Fact {id:$old_id})
            CREATE (new:Fact {
                id: $fid, kind: old.kind, slot: old.slot, code: $code, system: $system,
                display: $display, value: $value, status: 'superseded', valid_from: $valid_from,
                valid_to: $now, recorded_at: $now, superseded_at: $now,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (old)-[:SUPERSEDES {reason: 'lower_authority', at: $now}]->(new)
            RETURN new.id
            """
            target_graph.query(
                cypher,
                {
                    "pid": valid_pid,
                    "old_id": cur.id,
                    "fid": new_fid,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "valid_from": v_from,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": "superseded",
                "fact_id": new_fid,
                "superseded_id": new_fid,
                "status": "superseded",
                "reason": "lower_authority",
                "fact": rec.to_dict() if rec else None,
            }

        # Subcase 3C: Equal authority -> Compare valid_from
        if v_from > cur.valid_from:
            new_fid = gen_fact_id()
            cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(old:Fact {id:$old_id})
            SET old.status = 'superseded',
                old.superseded_at = $now,
                old.valid_to = coalesce(old.valid_to, $now),
                old.version = old.version + 1
            WITH p, old
            OPTIONAL MATCH (p)-[:HAS_FACT]->(other:Fact {slot: old.slot, status: 'active'})
            WHERE other.id <> old.id
            WITH p, old, collect(other) AS others
            FOREACH (o IN others |
                SET o.status = 'superseded',
                    o.superseded_at = $now,
                    o.valid_to = coalesce(o.valid_to, $now),
                    o.version = o.version + 1
            )
            CREATE (new:Fact {
                id: $fid, kind: old.kind, slot: old.slot, code: $code, system: $system,
                display: $display, value: $value, status: 'active', valid_from: $valid_from,
                valid_to: null, recorded_at: $now, superseded_at: null,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (new)-[:SUPERSEDES {reason: 'later_valid_from', at: $now}]->(old)
            RETURN new.id
            """
            target_graph.query(
                cypher,
                {
                    "pid": valid_pid,
                    "old_id": cur.id,
                    "fid": new_fid,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "valid_from": v_from,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": "superseded",
                "fact_id": new_fid,
                "superseded_id": cur.id,
                "status": "active",
                "fact": rec.to_dict() if rec else None,
            }

        if v_from < cur.valid_from:
            # Older valid_from loses
            new_fid = gen_fact_id()
            cypher = """
            MATCH (p:Patient {id:$pid}), (old:Fact {id:$old_id})
            CREATE (new:Fact {
                id: $fid, kind: old.kind, slot: old.slot, code: $code, system: $system,
                display: $display, value: $value, status: 'superseded', valid_from: $valid_from,
                valid_to: $now, recorded_at: $now, superseded_at: $now,
                last_confirmed_at: $now, confidence: $confidence, authority: $authority,
                evidence_count: 1, stale: false, half_life_days: $half_life,
                version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
                op_id: $op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (old)-[:SUPERSEDES {reason: 'earlier_valid_from', at: $now}]->(new)
            RETURN new.id
            """
            target_graph.query(
                cypher,
                {
                    "pid": valid_pid,
                    "old_id": cur.id,
                    "fid": new_fid,
                    "display": display,
                    "value": val,
                    "code": code,
                    "system": system,
                    "valid_from": v_from,
                    "now": now_ts,
                    "confidence": conf,
                    "authority": auth,
                    "half_life": h_life,
                    "episode_id": episode_id or "",
                    "asserted_by": caller_role or auth,
                    "op_id": op_id or "",
                },
            )
            rec = self.get_fact(valid_pid, new_fid)
            return {
                "action": "superseded",
                "fact_id": new_fid,
                "status": "superseded",
                "fact": rec.to_dict() if rec else None,
            }

        # Identical authority and identical valid_from -> Disputed
        new_fid = gen_fact_id()
        flg_id = gen_flag_id()
        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(old:Fact {id:$old_id})
        SET old.status = 'disputed'
        CREATE (new:Fact {
            id: $fid, kind: old.kind, slot: old.slot, code: $code, system: $system,
            display: $display, value: $value, status: 'disputed', valid_from: $valid_from,
            valid_to: null, recorded_at: $now, superseded_at: null,
            last_confirmed_at: $now, confidence: $confidence, authority: $authority,
            evidence_count: 1, stale: false, half_life_days: $half_life,
            version: 1, source_episode_id: $episode_id, asserted_by: $asserted_by,
            op_id: $op_id
        })
        CREATE (p)-[:HAS_FACT]->(new)
        CREATE (new)-[:CONFLICTS_WITH]->(old)
        CREATE (flg:Flag {
            id: $flg_id, type: 'conflicting_assertion', severity: 'moderate',
            status: 'open', rule_id: 'conflicting_assertion_tie',
            explanation: 'Conflicting values asserted at identical authority and timestamp.',
            detected_at: $now
        })
        CREATE (p)-[:HAS_FLAG]->(flg)
        CREATE (flg)-[:ABOUT]->(new)
        CREATE (flg)-[:ABOUT]->(old)
        RETURN new.id, flg.id
        """
        target_graph.query(
            cypher,
            {
                "pid": valid_pid,
                "old_id": cur.id,
                "fid": new_fid,
                "display": display,
                "value": val,
                "code": code,
                "system": system,
                "valid_from": v_from,
                "now": now_ts,
                "confidence": conf,
                "authority": auth,
                "half_life": h_life,
                "episode_id": episode_id or "",
                "asserted_by": caller_role or auth,
                "op_id": op_id or "",
                "flg_id": flg_id,
            },
        )
        rec = self.get_fact(valid_pid, new_fid)
        return {
            "action": "disputed",
            "fact_id": new_fid,
            "status": "disputed",
            "flag_id": flg_id,
            "fact": rec.to_dict() if rec else None,
        }

    def supersede_fact(
        self,
        patient_id: str,
        old_fact_id: str,
        expected_version: int,
        display: str,
        value: str = "",
        code: str | None = None,
        system: str | None = None,
        valid_from: int | None = None,
        authority: str | None = None,
        agent_id: str | None = None,
        episode_id: str | None = None,
        message_id: str | None = None,
        op_id: str | None = None,
        reason: str = "clinician_update",
    ) -> dict[str, Any]:
        """Atomically supersede an existing fact using Compare-and-Swap (CAS) via query q_sup_01.

        Raises:
            FactNotFoundError: If old fact does not exist.
            ConflictStaleVersionError: If expected_version does not match current version.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Inspect old fact
        old_fact = self.get_fact(valid_pid, old_fact_id)
        if not old_fact:
            raise FactNotFoundError(
                f"Fact '{old_fact_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "fact_id": old_fact_id},
            )

        if old_fact.status != "active":
            raise ConflictStaleVersionError(
                f"Cannot supersede fact '{old_fact_id}': status is '{old_fact.status}', "
                "not 'active'.",
                details={
                    "fact_id": old_fact_id,
                    "status": old_fact.status,
                    "current_fact": old_fact.to_dict(),
                },
            )

        if old_fact.version != expected_version:
            raise ConflictStaleVersionError(
                f"CAS version mismatch for fact '{old_fact_id}': "
                f"expected {expected_version}, got {old_fact.version}.",
                details={
                    "fact_id": old_fact_id,
                    "expected_version": expected_version,
                    "current_version": old_fact.version,
                    "current_fact": old_fact.to_dict(),
                },
            )

        now_ts = self._clock.now_ts()
        v_from = valid_from if valid_from is not None else now_ts
        val = (value if value else display).strip()
        auth = self._resolve_authority(authority, agent_id)
        conf = BASE_CONFIDENCE.get(auth, 0.7)
        new_fid = gen_fact_id()

        # If episode_id was not provided, link to existing source episode if available
        eid = episode_id or old_fact.source_episode_id or ""

        # Check if episode node exists
        ep_exists = False
        if eid:
            ep_res = target_graph.query(
                "MATCH (e:Episode {id:$eid}) RETURN e.id",
                {"eid": eid},
            )
            ep_exists = bool(ep_res.result_set)

        if ep_exists:
            # Execute q_sup_01 directly
            query = get_query("q_sup_01")
            params = {
                "pid": valid_pid,
                "old_id": old_fact_id,
                "expected_version": expected_version,
                "episode_id": eid,
                "now": now_ts,
                "valid_from": v_from,
                "new_id": new_fid,
                "code": code or old_fact.code or "",
                "system": system or old_fact.system or "",
                "display": display,
                "value": val,
                "confidence": conf,
                "authority": auth,
                "half_life": old_fact.half_life_days,
                "agent_id": agent_id or "",
                "op_id": op_id or "",
                "reason": reason,
                "message_id": message_id or "",
            }
            res = target_graph.query(query, params)
        else:
            # Standalone CAS without episode node
            cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(old:Fact {id:$old_id})
            WHERE old.status = 'active' AND old.version = $expected_version
            SET old.status = 'superseded', old.superseded_at = $now,
                old.valid_to = coalesce(old.valid_to, $valid_from), old.version = old.version + 1
            CREATE (new:Fact {
                id:$new_id, kind:old.kind, slot:old.slot, code:$code, system:$system,
                display:$display, value:$value, status:'active', valid_from:$valid_from,
                valid_to:null, recorded_at:$now, superseded_at:null, last_confirmed_at:$now,
                confidence:$confidence, authority:$authority, evidence_count:1, stale:false,
                half_life_days:$half_life, version:1, source_episode_id:$episode_id,
                asserted_by:$agent_id, op_id:$op_id
            })
            CREATE (p)-[:HAS_FACT]->(new)
            CREATE (new)-[:SUPERSEDES {reason:$reason, at:$now}]->(old)
            RETURN new.id AS new_id, old.id AS old_id
            """
            params = {
                "pid": valid_pid,
                "old_id": old_fact_id,
                "expected_version": expected_version,
                "now": now_ts,
                "valid_from": v_from,
                "new_id": new_fid,
                "code": code or old_fact.code or "",
                "system": system or old_fact.system or "",
                "display": display,
                "value": val,
                "confidence": conf,
                "authority": auth,
                "half_life": old_fact.half_life_days,
                "agent_id": agent_id or "",
                "op_id": op_id or "",
                "reason": reason,
                "episode_id": eid,
            }
            res = target_graph.query(cypher, params)

        if not res.result_set:
            refreshed = self.get_fact(valid_pid, old_fact_id)
            raise ConflictStaleVersionError(
                f"CAS update failed for fact '{old_fact_id}'.",
                details={
                    "fact_id": old_fact_id,
                    "expected_version": expected_version,
                    "current_fact": refreshed.to_dict() if refreshed else None,
                },
            )

        new_record = self.get_fact(valid_pid, new_fid)
        return {
            "action": "superseded",
            "new_id": new_fid,
            "old_id": old_fact_id,
            "new_fact": new_record.to_dict() if new_record else None,
        }

    def reinforce_fact(
        self,
        patient_id: str,
        fact_id: str,
        now: int | None = None,
    ) -> FactRecord:
        """Reinforce an active fact: evidence_count+1, last_confirmed_at=now, confidence rises."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_fact(valid_pid, fact_id)
        if not cur:
            raise FactNotFoundError(
                f"Fact '{fact_id}' not found.",
                details={"patient_id": valid_pid, "fact_id": fact_id},
            )

        now_ts = now if now is not None else self._clock.now_ts()
        new_ev = cur.evidence_count + 1
        new_conf = round(min(1.0, cur.confidence + 0.1 * (1.0 - cur.confidence)), 4)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {id:$fid})
        SET f.evidence_count = $ev,
            f.confidence = $conf,
            f.last_confirmed_at = $now
        RETURN f.id
        """
        target_graph.query(
            cypher,
            {"pid": valid_pid, "fid": fact_id, "ev": new_ev, "conf": new_conf, "now": now_ts},
        )
        return self.get_fact(valid_pid, fact_id)  # type: ignore[return-value]

    def get_fact_history(self, patient_id: str, fact_id: str) -> list[dict[str, Any]]:
        """Walk the [:SUPERSEDES*0..10] relationship chain using query q_sup_02."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        query = get_query("q_sup_02")
        res = target_graph.query(query, {"fact_id": fact_id})
        history: list[dict[str, Any]] = []
        for row in res.result_set:
            history.append(
                {
                    "id": row[0],
                    "value": row[1],
                    "status": row[2],
                    "valid_from": row[3],
                    "recorded_at": row[4],
                    "superseded_at": row[5],
                    "asserted_by": row[6],
                }
            )
        return history

    def get_facts_as_of(
        self,
        patient_id: str,
        as_of_ts: int,
        kind: str | None = None,
    ) -> list[FactRecord]:
        """Fetch active facts believed at a point-in-time using query q_recall_02."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        query = get_query("q_recall_02")
        res = target_graph.query(
            query,
            {"pid": valid_pid, "asof": as_of_ts, "kind": kind},
        )
        records: list[FactRecord] = []
        for row in res.result_set:
            records.append(
                FactRecord(
                    id=row[0],
                    patient_id=valid_pid,
                    kind=row[1],
                    slot=row[2],
                    display=row[3],
                    value=row[4] or "",
                    status=row[5] or "active",
                    valid_from=row[6],
                    valid_to=row[7],
                    recorded_at=row[8],
                    superseded_at=row[9],
                    confidence=float(row[10]) if row[10] is not None else 0.7,
                    authority=row[11] or "patient_reported",
                    version=int(row[12]) if row[12] is not None else 1,
                )
            )
        return records

    def get_active_facts(
        self,
        patient_id: str,
        kinds: list[str] | None = None,
        now: int | None = None,
    ) -> list[FactRecord]:
        """Fetch currently active facts using query q_recall_01."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)
        now_ts = now if now is not None else self._clock.now_ts()

        query = get_query("q_recall_01")
        target_kinds = kinds if kinds is not None else list(VALID_FACT_KINDS)
        res = target_graph.query(
            query,
            {"pid": valid_pid, "kinds": target_kinds, "now": now_ts},
        )
        records: list[FactRecord] = []
        for row in res.result_set:
            records.append(
                FactRecord(
                    id=row[0],
                    patient_id=valid_pid,
                    kind=row[1],
                    slot=f"{row[1]}:{row[0]}",
                    display=row[2],
                    value=row[3] or "",
                    status="active",
                    valid_from=now_ts,
                    confidence=float(row[4]) if row[4] is not None else 0.7,
                    last_confirmed_at=row[5] or now_ts,
                    half_life_days=row[6] or DEFAULT_HALF_LIFE_DAYS,
                    authority=row[7] or "patient_reported",
                    version=int(row[8]) if row[8] is not None else 1,
                )
            )
        return records

    def confirm_fact(
        self,
        patient_id: str,
        fact_id: str,
        clinician_id: str,
        decision: str = "confirm",
    ) -> FactRecord:
        """Clinician reviews and confirms or rejects a pending_review fact."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_fact(valid_pid, fact_id)
        if not cur:
            raise FactNotFoundError(
                f"Fact '{fact_id}' not found.",
                details={"patient_id": valid_pid, "fact_id": fact_id},
            )

        now_ts = self._clock.now_ts()
        if decision.lower() == "confirm":
            cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {id:$fid})
            SET f.status = 'active',
                f.authority = 'clinician',
                f.confidence = 1.0,
                f.last_confirmed_at = $now
            RETURN f.id
            """
        else:
            cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {id:$fid})
            SET f.status = 'retracted',
                f.superseded_at = $now
            RETURN f.id
            """
        target_graph.query(cypher, {"pid": valid_pid, "fid": fact_id, "now": now_ts})
        return self.get_fact(valid_pid, fact_id)  # type: ignore[return-value]

    def _resolve_authority(self, authority: str | None, caller_role: str | None) -> str:
        if authority and authority in AUTHORITY_RANK:
            return authority
        if caller_role:
            role_map = {
                "clinician": "clinician",
                "specialist": "clinician",
                "pharmacy": "pharmacy",
                "intake": "patient_reported",
            }
            return role_map.get(caller_role.lower(), "patient_reported")
        return "patient_reported"

    def _is_negation_or_removal(self, value: str) -> bool:
        norm = value.strip().lower()
        negations = {
            "no",
            "none",
            "false",
            "resolved",
            "denies",
            "removed",
            "no known allergies",
            "nka",
        }
        return norm in negations

    def _row_to_record(self, patient_id: str, row: list[Any]) -> FactRecord:
        return FactRecord(
            id=row[0],
            patient_id=patient_id,
            kind=row[1],
            slot=row[2],
            display=row[3],
            value=row[4] or "",
            code=row[5],
            system=row[6],
            status=row[7] or "active",
            valid_from=row[8] or 0,
            valid_to=row[9],
            recorded_at=row[10] or 0,
            superseded_at=row[11],
            last_confirmed_at=row[12] or 0,
            confidence=float(row[13]) if row[13] is not None else 0.7,
            authority=row[14] or "patient_reported",
            evidence_count=int(row[15]) if row[15] is not None else 1,
            stale=bool(row[16]),
            half_life_days=int(row[17]) if row[17] is not None else DEFAULT_HALF_LIFE_DAYS,
            version=int(row[18]) if row[18] is not None else 1,
            source_episode_id=row[19],
            asserted_by=row[20],
            op_id=row[21],
        )
