"""Episodic memory management with singly-linked chronological chains (NEXT).

Episodic memory management and queries q_ep_01a / q_ep_01b:
- Episode lifecycle: start_episode, close_episode, get_episode
- Singly-linked chronological chains: (e1)-[:NEXT {gap_days, seq}]->(e2)
- Querying past episodes with point-in-time filtering (e.start <= $now)
- Traversal across episodic chains (forward and backward)
- Vector embedding on episode summaries (384-dim vecf32)
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
    EpisodeNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.embeddings import Embedder, get_embedder
from carethread.util.id import episode_id as gen_episode_id

logger = logging.getLogger(__name__)

VALID_EPISODE_KINDS: frozenset[str] = frozenset(
    {
        "synthea_encounter",
        "live_visit",
        "pharmacy_review",
        "specialist_review",
        "wellness_check",
    }
)

VALID_OUTCOMES: frozenset[str] = frozenset(
    {"in_progress", "completed", "transferred", "cancelled", "resolved"}
)


@dataclass
class EpisodeRecord:
    """Represents a clinical encounter or review episode."""

    id: str
    patient_id: str
    kind: str
    start: int
    end: int | None = None
    summary: str = ""
    outcome: str | None = None
    episode_type: str = "ambulatory"
    seq: int = 1
    source: str = "live"
    summary_embedding: list[float] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("summary_embedding"):
            data["summary_embedding_dim"] = len(data["summary_embedding"])
            del data["summary_embedding"]
        return data


class EpisodeManager:
    """Manages clinical episodes and their chronological episodic chains."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clock = clock or get_clock()
        self._embedder = embedder or get_embedder()

    def start_episode(
        self,
        patient_id: str,
        kind: str = "live_visit",
        episode_type: str = "ambulatory",
        summary: str = "",
        start: int | None = None,
        episode_id: str | None = None,
        source: str = "live",
    ) -> EpisodeRecord:
        """Start a new clinical episode and link it into the patient's NEXT chain.

        Args:
            patient_id: Validated patient tenant identifier.
            kind: Episode kind ('live_visit', 'specialist_review', 'pharmacy_review', etc.).
            episode_type: Clinical category ('ambulatory', 'emergency', 'inpatient', 'wellness').
            summary: Short clinical summary or chief complaint.
            start: Optional timestamp in whole seconds (defaults to clock.now_ts()).
            episode_id: Optional explicit episode identifier (defaults to epi_<ulid>).
            source: 'live' or 'synthea'.

        Returns:
            Created EpisodeRecord.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if kind not in VALID_EPISODE_KINDS:
            raise ValidationError(
                f"Invalid episode kind '{kind}'. Allowed: {sorted(VALID_EPISODE_KINDS)}",
                details={"kind": kind, "allowed": list(VALID_EPISODE_KINDS)},
            )

        eid = episode_id or gen_episode_id()
        now_ts = start if start is not None else self._clock.now_ts()
        target_graph = self._router.graph(valid_pid)

        # Check patient exists
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found in patient graph.",
                details={"patient_id": valid_pid},
            )

        # Find current latest episode to determine sequence index and gap_days
        prev_query = """
        MATCH (p:Patient {id:$pid})-[:HAS_EPISODE]->(prev:Episode)
        RETURN prev.id AS id, prev.start AS start, prev.seq AS seq
        ORDER BY prev.seq DESC, prev.start DESC
        LIMIT 1
        """
        prev_res = target_graph.query(prev_query, {"pid": valid_pid})

        prev_id: str | None = None
        new_seq: int = 1
        gap_days: int = 0

        if prev_res.result_set:
            prev_row = prev_res.result_set[0]
            prev_id = prev_row[0]
            prev_start = prev_row[1] or now_ts
            prev_seq = prev_row[2] or 0
            new_seq = prev_seq + 1
            gap_days = max(0, int((now_ts - prev_start) / 86400))

        # Generate 384-dim summary embedding
        emb_text = summary if summary.strip() else f"{kind} {episode_type}"
        emb = self._embedder.embed(emb_text)

        create_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (e:Episode {
            id: $eid,
            kind: $kind,
            start: $start,
            end: null,
            summary: $summary,
            summary_embedding: vecf32($emb),
            outcome: 'in_progress',
            episode_type: $episode_type,
            seq: $seq,
            source: $source
        })
        CREATE (p)-[:HAS_EPISODE]->(e)
        WITH p, e
        OPTIONAL MATCH (p)-[:HAS_EPISODE]->(prev:Episode {id:$prev_id})
        FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (prev)-[:NEXT {gap_days: $gap_days, seq: $seq}]->(e)
        )
        RETURN e.id AS id
        """
        params = {
            "pid": valid_pid,
            "eid": eid,
            "kind": kind,
            "start": now_ts,
            "summary": summary,
            "emb": emb,
            "episode_type": episode_type,
            "seq": new_seq,
            "source": source,
            "prev_id": prev_id or "",
            "gap_days": gap_days,
        }
        target_graph.query(create_cypher, params)

        logger.debug(
            f"Created episode {eid} seq={new_seq} (prev={prev_id}, gap={gap_days}d) for {valid_pid}"
        )

        return EpisodeRecord(
            id=eid,
            patient_id=valid_pid,
            kind=kind,
            start=now_ts,
            end=None,
            summary=summary,
            outcome="in_progress",
            episode_type=episode_type,
            seq=new_seq,
            source=source,
            summary_embedding=emb,
        )

    def get_episode(self, patient_id: str, episode_id: str) -> EpisodeRecord | None:
        """Fetch episode by ID."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_EPISODE]->(e:Episode {id:$eid})
        RETURN e.id AS id, e.kind AS kind, e.start AS start, e.end AS end,
               e.summary AS summary, e.outcome AS outcome, e.episode_type AS episode_type,
               e.seq AS seq, e.source AS source
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "eid": episode_id})
        if not res.result_set:
            return None

        row = res.result_set[0]
        return EpisodeRecord(
            id=row[0],
            patient_id=valid_pid,
            kind=row[1],
            start=row[2],
            end=row[3],
            summary=row[4] or "",
            outcome=row[5],
            episode_type=row[6] or "ambulatory",
            seq=row[7] or 1,
            source=row[8] or "live",
        )

    def close_episode(
        self,
        patient_id: str,
        episode_id: str,
        outcome: str = "completed",
        summary: str | None = None,
        end: int | None = None,
    ) -> EpisodeRecord:
        """Close an episode, setting outcome and end timestamp.

        Raises:
            ValidationError: If outcome is invalid.
            EpisodeNotFoundError: If episode does not exist.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if outcome not in VALID_OUTCOMES:
            raise ValidationError(
                f"Invalid outcome '{outcome}'. Allowed: {sorted(VALID_OUTCOMES)}",
                details={"outcome": outcome, "allowed": list(VALID_OUTCOMES)},
            )

        now_ts = end if end is not None else self._clock.now_ts()
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_EPISODE]->(e:Episode {id:$eid})
        SET e.outcome = $outcome,
            e.end = $end,
            e.summary = CASE
                WHEN $summary IS NOT NULL AND $summary <> '' THEN $summary
                ELSE e.summary
            END
        RETURN e.id AS id, e.kind AS kind, e.start AS start, e.end AS end,
               e.summary AS summary, e.outcome AS outcome, e.episode_type AS episode_type,
               e.seq AS seq, e.source AS source
        """

        res = target_graph.query(
            cypher,
            {
                "pid": valid_pid,
                "eid": episode_id,
                "outcome": outcome,
                "end": now_ts,
                "summary": summary,
            },
        )
        if not res.result_set:
            raise EpisodeNotFoundError(
                f"Episode '{episode_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "episode_id": episode_id},
            )

        row = res.result_set[0]
        # Update vector embedding if summary was updated
        if summary and summary.strip():
            new_emb = self._embedder.embed(summary)
            target_graph.query(
                "MATCH (e:Episode {id:$eid}) SET e.summary_embedding = vecf32($emb)",
                {"eid": episode_id, "emb": new_emb},
            )

        return EpisodeRecord(
            id=row[0],
            patient_id=valid_pid,
            kind=row[1],
            start=row[2],
            end=row[3],
            summary=row[4] or "",
            outcome=row[5],
            episode_type=row[6] or "ambulatory",
            seq=row[7] or 1,
            source=row[8] or "live",
        )

    def get_patient_episodes(
        self,
        patient_id: str,
        limit: int = 20,
        now: int | None = None,
    ) -> list[EpisodeRecord]:
        """Fetch past episodes for a patient using query q_ep_01a with point-in-time filtering."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)
        now_ts = now if now is not None else self._clock.now_ts()

        query = get_query("q_ep_01a")
        res = target_graph.query(
            query,
            {"pid": valid_pid, "now": now_ts, "n": limit},
        )
        records: list[EpisodeRecord] = []
        for row in res.result_set:
            records.append(
                EpisodeRecord(
                    id=row[0],
                    patient_id=valid_pid,
                    kind=row[1],
                    start=row[2],
                    end=row[3],
                    summary=row[4] or "",
                    outcome=row[5],
                    episode_type=row[6] or "ambulatory",
                    seq=row[7] or 1,
                    source=row[8] or "live",
                )
            )
        return records

    def get_episode_chain(
        self,
        patient_id: str,
        episode_id: str,
        hops: int = 3,
        direction: str = "forward",
    ) -> list[dict[str, Any]]:
        """Traverse the singly-linked chronological chain (NEXT) across episodes.

        Supports query q_ep_01b for forward traversal.

        Args:
            patient_id: Validated patient tenant identifier.
            episode_id: Starting episode ID.
            hops: Maximum chain hops to traverse (default 3).
            direction: 'forward' (subsequent episodes) or 'backward' (prior episodes).

        Returns:
            List of connected episode dictionaries ordered by timestamp.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)
        safe_hops = max(1, min(10, int(hops)))

        if direction.lower() == "forward":
            if safe_hops == 3:
                query = get_query("q_ep_01b")
            else:
                query = f"""
                MATCH (e:Episode {{id:$eid}})-[:NEXT*1..{safe_hops}]->(n:Episode)
                RETURN n.id AS id, n.start AS start, n.summary AS summary,
                       n.episode_type AS episode_type, n.seq AS seq
                ORDER BY n.start ASC
                """
        else:
            query = f"""
            MATCH (p:Episode)-[:NEXT*1..{safe_hops}]->(e:Episode {{id:$eid}})
            RETURN p.id AS id, p.start AS start, p.summary AS summary,
                   p.episode_type AS episode_type, p.seq AS seq
            ORDER BY p.start ASC
            """

        res = target_graph.query(query, {"eid": episode_id})
        nodes: list[dict[str, Any]] = []
        for row in res.result_set:
            nodes.append(
                {
                    "id": row[0],
                    "start": row[1],
                    "summary": row[2] or "",
                    "episode_type": row[3] or "ambulatory",
                    "seq": row[4],
                }
            )
        return nodes

    def search_episodes(
        self,
        patient_id: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search over episode summaries using cosine distance."""
        valid_pid = self._router.validate_patient_id(patient_id)
        emb = self._embedder.embed(query_text)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        CALL db.idx.vector.queryNodes('Episode', 'summary_embedding', $top_k, vecf32($emb))
        YIELD node AS e, score
        RETURN e.id AS id, e.kind AS kind, e.summary AS summary,
               e.episode_type AS episode_type, e.start AS start, e.seq AS seq,
               (1.0 - score) AS similarity
        ORDER BY similarity DESC
        """
        res = target_graph.query(cypher, {"top_k": top_k, "emb": emb})
        results: list[dict[str, Any]] = []
        for row in res.result_set:
            results.append(
                {
                    "id": row[0],
                    "kind": row[1],
                    "summary": row[2],
                    "episode_type": row[3],
                    "start": row[4],
                    "seq": row[5],
                    "similarity": round(float(row[6]), 4),
                }
            )
        return results
