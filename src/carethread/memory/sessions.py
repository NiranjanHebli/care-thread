"""Session and message memory management with atomic sequence ordering.

Session and message memory management and queries q_msg_01a / q_msg_01b:
- Session lifecycle: create, get, list, close
- Atomically sequenced messages (s.msg_count + 1)
- Compound uniqueness enforcement on Message(session_id, seq)
- Vector embedding generation (384-dim vecf32) on message content
- History retrieval and vector similarity search across messages
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
    ForgetNotAllowedError,
    PatientNotFoundError,
    SessionNotFoundError,
    ValidationError,
)
from carethread.memory.embeddings import Embedder, get_embedder
from carethread.util.id import message_id as gen_message_id
from carethread.util.id import session_id as gen_session_id

logger = logging.getLogger(__name__)

VALID_ROLES: frozenset[str] = frozenset({"user", "agent", "tool", "system"})
VALID_CHANNELS: frozenset[str] = frozenset({"chat", "api", "console", "portal", "agent"})
VALID_SESSION_STATUSES: frozenset[str] = frozenset({"open", "closed"})


@dataclass
class SessionRecord:
    """Represents a clinical conversation session bound to a patient."""

    id: str
    patient_id: str
    started_at: int
    ended_at: int | None = None
    status: str = "open"
    channel: str = "chat"
    agent_id: str | None = None
    msg_count: int = 0
    summary: str = ""
    episode_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MessageRecord:
    """Represents an atomically sequenced message within a session."""

    id: str
    session_id: str
    seq: int
    role: str
    content: str
    ts: int
    agent_id: str | None = None
    embedding: list[float] | None = field(default=None, repr=False)
    redacted_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("embedding"):
            data["embedding_dim"] = len(data["embedding"])
            del data["embedding"]
        return data


class SessionManager:
    """Manages clinical conversation sessions and atomically sequenced messages."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clock = clock or get_clock()
        self._embedder = embedder or get_embedder()

    def create_session(
        self,
        patient_id: str,
        episode_id: str | None = None,
        channel: str = "chat",
        agent_id: str | None = None,
        session_id: str | None = None,
        started_at: int | None = None,
    ) -> SessionRecord:
        """Create a new session node in the patient graph.

        Args:
            patient_id: Validated patient tenant identifier (e.g. 'p_1a967cbe5b94').
            episode_id: Optional ID of clinical episode to link session to.
            channel: Communication channel ('chat', 'api', 'console', 'portal', 'agent').
            agent_id: Optional identifier of responsible agent.
            session_id: Optional explicit session ID (defaults to generated ses_<ulid>).
            started_at: Optional start timestamp in seconds (defaults to clock.now_ts()).

        Returns:
            Created SessionRecord.

        Raises:
            ValidationError: If channel is invalid.
            PatientNotFoundError: If patient graph or node does not exist.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if self._router.is_deleting(valid_pid):
            raise ForgetNotAllowedError(
                f"Tenant '{valid_pid}' is currently being deleted; new sessions are blocked.",
                details={"patient_id": valid_pid, "status": "deleting"},
            )

        if channel not in VALID_CHANNELS:
            raise ValidationError(
                f"Invalid channel '{channel}'. Allowed: {sorted(VALID_CHANNELS)}",
                details={"channel": channel, "allowed": list(VALID_CHANNELS)},
            )

        sid = session_id or gen_session_id()
        now_ts = started_at if started_at is not None else self._clock.now_ts()
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

        create_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (s:Session {
            id: $sid,
            started_at: $started_at,
            ended_at: null,
            status: 'open',
            channel: $channel,
            agent_id: $agent_id,
            msg_count: 0,
            summary: ''
        })
        CREATE (p)-[:HAS_SESSION]->(s)
        WITH s
        OPTIONAL MATCH (e:Episode {id:$eid})
        FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
            CREATE (s)-[:OF_EPISODE]->(e)
        )
        RETURN s.id AS id
        """
        params = {
            "pid": valid_pid,
            "sid": sid,
            "started_at": now_ts,
            "channel": channel,
            "agent_id": agent_id,
            "eid": episode_id or "",
        }
        target_graph.query(create_cypher, params)

        logger.debug(f"Created session {sid} for patient {valid_pid}")
        return SessionRecord(
            id=sid,
            patient_id=valid_pid,
            started_at=now_ts,
            ended_at=None,
            status="open",
            channel=channel,
            agent_id=agent_id,
            msg_count=0,
            summary="",
            episode_id=episode_id,
        )

    def get_session(self, patient_id: str, session_id: str) -> SessionRecord | None:
        """Fetch session by ID.

        Returns:
            SessionRecord if found, None otherwise.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_SESSION]->(s:Session {id:$sid})
        OPTIONAL MATCH (s)-[:OF_EPISODE]->(e:Episode)
        RETURN s.id AS id, s.started_at AS started_at, s.ended_at AS ended_at,
               s.status AS status, s.channel AS channel, s.agent_id AS agent_id,
               s.msg_count AS msg_count, s.summary AS summary, e.id AS episode_id
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "sid": session_id})
        if not res.result_set:
            return None

        row = res.result_set[0]
        return SessionRecord(
            id=row[0],
            patient_id=valid_pid,
            started_at=row[1],
            ended_at=row[2],
            status=row[3] or "open",
            channel=row[4] or "chat",
            agent_id=row[5],
            msg_count=row[6] or 0,
            summary=row[7] or "",
            episode_id=row[8],
        )

    def close_session(
        self,
        patient_id: str,
        session_id: str,
        summary: str | None = None,
        ended_at: int | None = None,
    ) -> SessionRecord:
        """Close an active session, setting status='closed' and ended_at timestamp.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)
        now_ts = ended_at if ended_at is not None else self._clock.now_ts()

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_SESSION]->(s:Session {id:$sid})
        SET s.status = 'closed',
            s.ended_at = $ended_at,
            s.summary = CASE
                WHEN $summary IS NOT NULL AND $summary <> '' THEN $summary
                ELSE s.summary
            END
        WITH s
        OPTIONAL MATCH (s)-[:OF_EPISODE]->(e:Episode)
        RETURN s.id AS id, s.started_at AS started_at, s.ended_at AS ended_at,
               s.status AS status, s.channel AS channel, s.agent_id AS agent_id,
               s.msg_count AS msg_count, s.summary AS summary, e.id AS episode_id
        """

        res = target_graph.query(
            cypher,
            {
                "pid": valid_pid,
                "sid": session_id,
                "ended_at": now_ts,
                "summary": summary,
            },
        )
        if not res.result_set:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "session_id": session_id},
            )

        row = res.result_set[0]
        return SessionRecord(
            id=row[0],
            patient_id=valid_pid,
            started_at=row[1],
            ended_at=row[2],
            status=row[3],
            channel=row[4],
            agent_id=row[5],
            msg_count=row[6],
            summary=row[7] or "",
            episode_id=row[8],
        )

    def list_sessions(
        self,
        patient_id: str,
        limit: int = 50,
        status: str | None = None,
    ) -> list[SessionRecord]:
        """List sessions for a patient, ordered most recent first."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_SESSION]->(s:Session)
        WHERE ($status IS NULL OR s.status = $status)
        OPTIONAL MATCH (s)-[:OF_EPISODE]->(e:Episode)
        RETURN s.id AS id, s.started_at AS started_at, s.ended_at AS ended_at,
               s.status AS status, s.channel AS channel, s.agent_id AS agent_id,
               s.msg_count AS msg_count, s.summary AS summary, e.id AS episode_id
        ORDER BY s.started_at DESC
        LIMIT $limit
        """
        res = target_graph.query(
            cypher,
            {"pid": valid_pid, "status": status, "limit": limit},
        )
        records: list[SessionRecord] = []
        for row in res.result_set:
            records.append(
                SessionRecord(
                    id=row[0],
                    patient_id=valid_pid,
                    started_at=row[1],
                    ended_at=row[2],
                    status=row[3] or "open",
                    channel=row[4] or "chat",
                    agent_id=row[5],
                    msg_count=row[6] or 0,
                    summary=row[7] or "",
                    episode_id=row[8],
                )
            )
        return records

    def append_message(
        self,
        patient_id: str,
        session_id: str,
        role: str,
        content: str,
        agent_id: str | None = None,
        message_id: str | None = None,
        ts: int | None = None,
        embedding: list[float] | None = None,
    ) -> MessageRecord:
        """Atomically append a message to a session using query q_msg_01a.

        Sequence number is atomically incremented (seq = s.msg_count + 1).
        Vector embedding (384 dimensions) is generated from message content.

        Raises:
            ValidationError: If role is invalid.
            SessionNotFoundError: If session does not exist.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if role not in VALID_ROLES:
            raise ValidationError(
                f"Invalid message role '{role}'. Allowed: {sorted(VALID_ROLES)}",
                details={"role": role, "allowed": list(VALID_ROLES)},
            )

        mid = message_id or gen_message_id()
        now_ts = ts if ts is not None else self._clock.now_ts()

        # Compute dense 384-dim vector embedding
        if embedding is not None:
            emb = embedding
        else:
            emb = self._embedder.embed(content)

        target_graph = self._router.graph(valid_pid)

        # q_msg_01a executes atomic increment and links (s)-[:HAS_MESSAGE]->(m)
        query = get_query("q_msg_01a")
        params = {
            "sid": session_id,
            "mid": mid,
            "role": role,
            "agent_id": agent_id or "",
            "content": content,
            "now": now_ts,
            "emb": emb,
        }
        res = target_graph.query(query, params)

        if not res.result_set:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found in patient graph '{valid_pid}'.",
                details={"patient_id": valid_pid, "session_id": session_id},
            )

        assigned_seq: int = int(res.result_set[0][0])
        logger.debug(
            f"Appended message {mid} seq={assigned_seq} to session {session_id} (pid={valid_pid})"
        )

        return MessageRecord(
            id=mid,
            session_id=session_id,
            seq=assigned_seq,
            role=role,
            content=content,
            ts=now_ts,
            agent_id=agent_id,
            embedding=emb,
        )

    def get_session_history(
        self,
        patient_id: str,
        session_id: str,
        limit: int = 100,
        order: str = "ASC",
    ) -> list[MessageRecord]:
        """Fetch ordered message history for a session using query q_msg_01b.

        Args:
            patient_id: Validated patient tenant identifier.
            session_id: Session identifier.
            limit: Maximum messages to return.
            order: 'ASC' for chronological (default) or 'DESC' for most recent first.

        Returns:
            List of MessageRecord.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        query = get_query("q_msg_01b")
        if order.upper() == "DESC":
            query = query.replace("ORDER BY m.seq ASC", "ORDER BY m.seq DESC")

        if limit > 0:
            query = f"{query} LIMIT {int(limit)}"

        res = target_graph.query(query, {"sid": session_id})
        messages: list[MessageRecord] = []
        for row in res.result_set:
            messages.append(
                MessageRecord(
                    id=row[0],
                    session_id=session_id,
                    seq=row[1],
                    role=row[2],
                    agent_id=row[3] if row[3] else None,
                    ts=row[4],
                    content=row[5] or "",
                )
            )
        return messages

    def search_messages(
        self,
        patient_id: str,
        query_text: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search over message embeddings using cosine distance.

        Args:
            patient_id: Patient tenant ID.
            query_text: Natural language search string.
            session_id: Optional session filter.
            top_k: Number of nearest messages to return.

        Returns:
            List of message dictionaries with similarity scores.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        emb = self._embedder.embed(query_text)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        CALL db.idx.vector.queryNodes('Message', 'embedding', $top_k, vecf32($emb))
        YIELD node AS m, score
        WHERE ($sid IS NULL OR m.session_id = $sid)
        RETURN m.id AS id, m.session_id AS session_id, m.seq AS seq,
               m.role AS role, m.agent_id AS agent_id, m.content AS content,
               m.ts AS ts, (1.0 - score) AS similarity
        ORDER BY similarity DESC
        """
        params = {
            "top_k": top_k,
            "emb": emb,
            "sid": session_id,
        }
        res = target_graph.query(cypher, params)
        results: list[dict[str, Any]] = []
        for row in res.result_set:
            results.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "seq": row[2],
                    "role": row[3],
                    "agent_id": row[4],
                    "content": row[5],
                    "ts": row[6],
                    "similarity": round(float(row[7]), 4),
                }
            )
        return results
