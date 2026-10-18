"""Procedural memory engine managing patient care playbooks and clinical pathways.

Procedural memory capabilities:
- Reusable care playbooks derived from clinical episodes and care pathways
- Automated promotion from candidate to active status based on clinical successes
- Laplace-smoothed performance scoring ((successes + 1) / (successes + failures + 2))
- Ordered multi-step action execution tracking (PlaybookStep)
- Association with episodes (DERIVED_FROM and FOLLOWED relationships)
- Dense 384-dimensional vector similarity search over clinical playbook text
- Standardized template instantiation from shared clinic pathways (PlaybookTemplate)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    PatientNotFoundError,
    PlaybookNotFoundError,
    PlaybookStateError,
    ValidationError,
)
from carethread.memory.embeddings import Embedder, get_embedder
from carethread.util.id import playbook_id as gen_playbook_id
from carethread.util.id import playbook_step_id as gen_step_id

logger = logging.getLogger(__name__)

VALID_PLAYBOOK_STATUSES: frozenset[str] = frozenset(
    {"candidate", "active", "archived", "deprecated"}
)

VALID_EPISODE_TYPES: frozenset[str] = frozenset(
    {"ambulatory", "emergency", "inpatient", "urgent", "wellness"}
)

VALID_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "assessment",
        "medication",
        "order",
        "instruction",
        "triage",
        "referral",
        "observation",
        "procedure",
    }
)


def _extract_props(node: Any) -> dict[str, Any]:
    """Safely extracts properties dictionary from a graph node or mapping."""
    if hasattr(node, "properties"):
        return dict(node.properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


@dataclass
class PlaybookStepRecord:
    """Represents an ordered step within a clinical care playbook."""

    id: str
    idx: int
    action_type: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert step record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybookStepRecord:
        """Construct PlaybookStepRecord from raw property mapping."""
        return cls(
            id=str(data.get("id", "")),
            idx=int(data.get("idx", 0)),
            action_type=str(data.get("action_type", "instruction")),
            text=str(data.get("text", "")),
        )


@dataclass
class PlaybookRecord:
    """Represents a procedural clinical care playbook in patient memory."""

    id: str
    patient_id: str
    name: str
    episode_type: str
    status: str = "candidate"
    version: int = 1
    success_count: int = 0
    failure_count: int = 0
    score: float = 0.5
    last_outcome_at: int | None = None
    guideline_ref: str | None = None
    steps: list[PlaybookStepRecord] = field(default_factory=list)
    derived_episode_ids: list[str] = field(default_factory=list)
    followed_episode_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert playbook record to dictionary representation."""
        data = asdict(self)
        if data.get("embedding"):
            data["embedding_dim"] = len(data["embedding"])
            del data["embedding"]
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        patient_id: str,
        steps: list[PlaybookStepRecord] | None = None,
        derived_episodes: list[str] | None = None,
        followed_episodes: list[str] | None = None,
    ) -> PlaybookRecord:
        """Construct PlaybookRecord from node properties and related entities."""
        return cls(
            id=str(data.get("id", "")),
            patient_id=patient_id,
            name=str(data.get("name", "")),
            episode_type=str(data.get("episode_type", "ambulatory")),
            status=str(data.get("status", "candidate")),
            version=int(data.get("version", 1)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            score=float(data.get("score", 0.5)),
            last_outcome_at=int(data["last_outcome_at"]) if data.get("last_outcome_at") else None,
            guideline_ref=data.get("guideline_ref"),
            steps=steps or [],
            derived_episode_ids=derived_episodes or [],
            followed_episode_ids=followed_episodes or [],
        )


class PlaybookManager:
    """Procedural memory manager for clinical care playbooks."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        client = router._client if router is not None else get_db_client()
        self._router = router or TenantRouter(client)
        self._clock = clock or get_clock()
        self._embedder = embedder or get_embedder()

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

    def create_playbook(
        self,
        patient_id: str,
        name: str,
        episode_type: str,
        steps: list[dict[str, Any] | str],
        guideline_ref: str | None = None,
        derived_from_episode_ids: list[str] | None = None,
        status: str = "candidate",
    ) -> PlaybookRecord:
        """Creates a new clinical care playbook with ordered execution steps."""
        valid_pid = self._router.validate_patient_id(patient_id)
        self._ensure_patient_exists(valid_pid)

        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("Playbook name must not be empty.")

        clean_ep_type = episode_type.strip().lower()
        if clean_ep_type not in VALID_EPISODE_TYPES:
            allowed = sorted(VALID_EPISODE_TYPES)
            raise ValidationError(
                f"Invalid episode_type '{episode_type}'. Must be one of {allowed}."
            )

        clean_status = status.strip().lower()
        if clean_status not in VALID_PLAYBOOK_STATUSES:
            allowed_st = sorted(VALID_PLAYBOOK_STATUSES)
            raise ValidationError(
                f"Invalid playbook status '{status}'. Must be one of {allowed_st}."
            )

        target_graph = self._router.graph(valid_pid)
        now_ts = int(self._clock.now().timestamp())
        pbid = gen_playbook_id()

        # Parse steps and format textual summary for dense embedding
        step_records: list[PlaybookStepRecord] = []
        step_texts: list[str] = []

        for idx, item in enumerate(steps):
            sid = gen_step_id()
            if isinstance(item, str):
                act_type = "instruction"
                txt = item.strip()
            elif isinstance(item, dict):
                raw_type = item.get("action_type", "instruction").strip().lower()
                act_type = raw_type if raw_type in VALID_ACTION_TYPES else "instruction"
                txt = str(item.get("text", "")).strip()
            else:
                act_type = "instruction"
                txt = str(item).strip()

            step_records.append(
                PlaybookStepRecord(
                    id=sid,
                    idx=idx,
                    action_type=act_type,
                    text=txt,
                )
            )
            step_texts.append(f"{idx + 1}. {txt}")

        full_text = f"{clean_name}. {clean_ep_type}. " + " ".join(step_texts)
        emb = self._embedder.embed(full_text)

        # Create Playbook node in tenant graph
        create_pb_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (pb:Playbook {
            id: $pbid,
            patient_id: $pid,
            name: $name,
            episode_type: $episode_type,
            status: $status,
            version: 1,
            success_count: 0,
            failure_count: 0,
            score: 0.5,
            last_outcome_at: null,
            guideline_ref: $guideline_ref,
            created_at: $now,
            embedding: vecf32($emb)
        })
        CREATE (p)-[:HAS_PLAYBOOK]->(pb)
        RETURN pb.id
        """
        target_graph.query(
            create_pb_cypher,
            {
                "pid": valid_pid,
                "pbid": pbid,
                "name": clean_name,
                "episode_type": clean_ep_type,
                "status": clean_status,
                "guideline_ref": (guideline_ref or "").strip() or None,
                "now": now_ts,
                "emb": emb,
            },
        )

        # Attach PlaybookStep nodes
        for s in step_records:
            create_step_cypher = """
            MATCH (pb:Playbook {id:$pbid})
            CREATE (s:PlaybookStep {
                id: $sid,
                idx: $idx,
                action_type: $action_type,
                text: $text
            })
            CREATE (pb)-[:HAS_STEP]->(s)
            """
            target_graph.query(
                create_step_cypher,
                {
                    "pbid": pbid,
                    "sid": s.id,
                    "idx": s.idx,
                    "action_type": s.action_type,
                    "text": s.text,
                },
            )

        # Connect DERIVED_FROM episodes if provided
        derived_ids = list(derived_from_episode_ids or [])
        for eid in derived_ids:
            target_graph.query(
                """
                MATCH (pb:Playbook {id:$pbid}), (ep:Episode {id:$eid})
                MERGE (pb)-[:DERIVED_FROM]->(ep)
                """,
                {"pbid": pbid, "eid": eid.strip()},
            )

        return PlaybookRecord(
            id=pbid,
            patient_id=valid_pid,
            name=clean_name,
            episode_type=clean_ep_type,
            status=clean_status,
            version=1,
            success_count=0,
            failure_count=0,
            score=0.5,
            last_outcome_at=None,
            guideline_ref=(guideline_ref or "").strip() or None,
            steps=step_records,
            derived_episode_ids=derived_ids,
            embedding=emb,
        )

    def instantiate_from_template(
        self,
        patient_id: str,
        template_id: str,
    ) -> PlaybookRecord:
        """Instantiates a patient-specific playbook from clinic_shared PlaybookTemplate."""
        valid_pid = self._router.validate_patient_id(patient_id)
        self._ensure_patient_exists(valid_pid)

        clinic_reader = self._router.get_clinic_reader()
        t_res = clinic_reader.ro_query(
            "MATCH (pt:PlaybookTemplate {id:$tid}) RETURN pt.name, pt.episode_type, pt.steps",
            {"tid": template_id.strip()},
        )
        if not t_res.result_set:
            raise PlaybookNotFoundError(
                f"Playbook template '{template_id}' not found in clinic_shared.",
                details={"template_id": template_id},
            )

        row = t_res.result_set[0]
        t_name = str(row[0])
        t_ep_type = str(row[1])
        raw_steps = row[2]
        if isinstance(raw_steps, str):
            try:
                t_steps = json.loads(raw_steps)
            except Exception:
                t_steps = [raw_steps]
        elif isinstance(raw_steps, list):
            t_steps = list(raw_steps)
        else:
            t_steps = []

        return self.create_playbook(
            patient_id=valid_pid,
            name=t_name,
            episode_type=t_ep_type,
            steps=t_steps,
            guideline_ref=template_id.strip(),
            status="candidate",
        )

    def record_outcome(
        self,
        patient_id: str,
        playbook_id: str,
        success: bool,
        episode_id: str | None = None,
    ) -> PlaybookRecord:
        """Records outcome and updates performance score via query q_pb_01.

        Candidate playbooks are promoted to 'active' when success_count >= 3.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Verify playbook exists
        self.get_playbook(valid_pid, playbook_id)

        now_ts = int(self._clock.now().timestamp())
        s_inc = 1 if success else 0
        f_inc = 0 if success else 1

        q_pb_01 = get_query("q_pb_01")
        res = target_graph.query(
            q_pb_01,
            {
                "pbid": playbook_id.strip(),
                "s": s_inc,
                "f": f_inc,
                "now": now_ts,
            },
        )
        if not res.result_set:
            raise PlaybookStateError(
                f"Failed to record outcome for playbook '{playbook_id}'.",
                details={"playbook_id": playbook_id},
            )

        # Connect FOLLOWED relationship if episode provided
        if episode_id:
            target_graph.query(
                """
                MATCH (ep:Episode {id:$eid}), (pb:Playbook {id:$pbid})
                MERGE (ep)-[:FOLLOWED]->(pb)
                """,
                {"eid": episode_id.strip(), "pbid": playbook_id.strip()},
            )

        return self.get_playbook(valid_pid, playbook_id)

    def get_playbook(self, patient_id: str, playbook_id: str) -> PlaybookRecord:
        """Retrieves a full playbook record with its steps and related episodes."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        pb_res = target_graph.query(
            "MATCH (pb:Playbook {id:$pbid}) RETURN pb",
            {"pbid": playbook_id.strip()},
        )
        if not pb_res.result_set:
            raise PlaybookNotFoundError(
                f"Playbook '{playbook_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "playbook_id": playbook_id},
            )

        pb_node = pb_res.result_set[0][0]
        pb_props = _extract_props(pb_node)

        # Fetch ordered steps
        step_res = target_graph.query(
            """
            MATCH (pb:Playbook {id:$pbid})-[:HAS_STEP]->(s:PlaybookStep)
            RETURN s
            ORDER BY s.idx ASC
            """,
            {"pbid": playbook_id.strip()},
        )
        steps = [
            PlaybookStepRecord.from_dict(_extract_props(row[0])) for row in step_res.result_set
        ]

        # Fetch DERIVED_FROM episodes
        der_res = target_graph.query(
            "MATCH (pb:Playbook {id:$pbid})-[:DERIVED_FROM]->(ep:Episode) RETURN ep.id",
            {"pbid": playbook_id.strip()},
        )
        derived = [row[0] for row in der_res.result_set]

        # Fetch FOLLOWED episodes
        fol_res = target_graph.query(
            "MATCH (ep:Episode)-[:FOLLOWED]->(pb:Playbook {id:$pbid}) RETURN ep.id",
            {"pbid": playbook_id.strip()},
        )
        followed = [row[0] for row in fol_res.result_set]

        return PlaybookRecord.from_dict(
            data=pb_props,
            patient_id=valid_pid,
            steps=steps,
            derived_episodes=derived,
            followed_episodes=followed,
        )

    def list_playbooks(
        self,
        patient_id: str,
        episode_type: str | None = None,
        status: str | None = None,
    ) -> list[PlaybookRecord]:
        """Lists playbooks for a patient with optional type and status filters."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        query = """
        MATCH (p:Patient {id:$pid})-[:HAS_PLAYBOOK]->(pb:Playbook)
        WHERE ($ep_type IS NULL OR pb.episode_type = $ep_type)
          AND ($status IS NULL OR pb.status = $status)
        RETURN pb.id
        ORDER BY pb.score DESC, pb.name ASC
        """
        res = target_graph.query(
            query,
            {
                "pid": valid_pid,
                "ep_type": episode_type.strip().lower() if episode_type else None,
                "status": status.strip().lower() if status else None,
            },
        )

        return [self.get_playbook(valid_pid, row[0]) for row in res.result_set]

    def search_playbooks(
        self,
        patient_id: str,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[PlaybookRecord, float]]:
        """Dense vector search over playbook embeddings with performance weighting."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        clean_query = query.strip()
        if not clean_query:
            return []

        emb = self._embedder.embed(clean_query)
        v_query = """
        CALL db.idx.vector.queryNodes("Playbook", "embedding", $limit, vecf32($emb))
        YIELD node, score
        RETURN node.id, score
        """
        try:
            res = target_graph.query(v_query, {"limit": max(1, limit * 2), "emb": emb})
        except Exception as ex:
            logger.warning("Vector query on Playbook failed, returning empty: %s", ex)
            return []

        results: list[tuple[PlaybookRecord, float]] = []
        for row in res.result_set:
            pbid = str(row[0])
            raw_dist = float(row[1])
            similarity = max(0.0, min(1.0, 1.0 - raw_dist))
            if similarity >= min_similarity:
                try:
                    pb = self.get_playbook(valid_pid, pbid)
                    results.append((pb, round(similarity, 4)))
                except PlaybookNotFoundError:
                    continue

        results.sort(key=lambda item: item[1] * 0.7 + item[0].score * 0.3, reverse=True)
        return results[:limit]

    def archive_playbook(self, patient_id: str, playbook_id: str) -> PlaybookRecord:
        """Transitions a playbook to archived status."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Verify exists
        self.get_playbook(valid_pid, playbook_id)

        target_graph.query(
            "MATCH (pb:Playbook {id:$pbid}) SET pb.status = 'archived' RETURN pb.id",
            {"pbid": playbook_id.strip()},
        )
        return self.get_playbook(valid_pid, playbook_id)
