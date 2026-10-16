"""Hybrid recall engine combining fastembed vector search, Cypher traversal, and ranking.

Hybrid recall implementation:
- Deterministic Cypher queries (q_recall_01 active facts, q_recall_02 as-of belief)
- Dense vector similarity search on Note nodes (q_recall_03a, q_recall_03b)
- Multi-factor composite ranking: relevance + decayed confidence + authority weight
- Half-life exponential temporal decay: confidence * pow(0.5, elapsed_days / half_life_days)
- Explainability and Why-Not provenance: classification of rejected candidates with codes
- Optional shared clinic knowledge search on Guideline nodes
- Canary token security filtering preventing prompt injection leakage
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    NoteNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.embeddings import Embedder, get_embedder
from carethread.util.id import note_id as gen_note_id

logger = logging.getLogger(__name__)

# Authority hierarchy weights
AUTHORITY_WEIGHTS: dict[str, float] = {
    "clinician": 1.0,
    "record": 0.95,
    "pharmacy": 0.90,
    "patient_reported": 0.70,
    "agent_inferred": 0.50,
}

# Standard fact kinds
DEFAULT_FACT_KINDS: tuple[str, ...] = (
    "condition",
    "medication",
    "allergy",
    "preference",
    "symptom",
)

# Ranking score weights
ALPHA_RELEVANCE: float = 0.40
BETA_CONFIDENCE: float = 0.40
GAMMA_AUTHORITY: float = 0.20

# Threshold cutoffs
DEFAULT_MIN_CONFIDENCE: float = 0.25
DEFAULT_MIN_SIMILARITY: float = 0.20


@dataclass
class NoteRecord:
    """Represents a clinical observation or note node in the patient graph."""

    id: str
    patient_id: str
    text: str
    kind: str = "intake_note"
    author_agent_id: str = "agent_intake"
    status: str = "active"
    clinician_review_required: bool = False
    ts: int = 0
    episode_id: str | None = None
    about_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("embedding"):
            data["embedding_dim"] = len(data["embedding"])
            del data["embedding"]
        return data


@dataclass
class RecalledFact:
    """Represents a semantic fact recalled from graph memory."""

    id: str
    kind: str
    display: str
    value: str
    confidence: float
    decayed_confidence: float
    authority: str
    authority_weight: float
    last_confirmed_at: int
    half_life_days: int | None
    relevance_score: float
    composite_score: float
    rank: int = 1
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecalledNote:
    """Represents a clinical note recalled via dense vector similarity."""

    id: str
    text: str
    kind: str
    author_agent_id: str
    episode_id: str | None
    about_ids: list[str]
    vector_score: float
    composite_score: float
    rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecalledGuideline:
    """Represents a clinical guideline recalled from the shared clinic graph."""

    id: str
    title: str
    version: str
    text: str
    condition_code: str | None
    vector_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedCandidate:
    """Represents a memory node evaluated during recall but rejected with reason."""

    id: str
    label: str
    reason_code: str
    reason: str
    score: float = 0.0
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecallResult:
    """Consolidated recall output combining facts, notes, guidelines, and rejected nodes."""

    patient_id: str
    query: str
    facts: list[RecalledFact] = field(default_factory=list)
    notes: list[RecalledNote] = field(default_factory=list)
    guidelines: list[RecalledGuideline] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    as_of: int = 0
    total_active_facts: int = 0
    total_vector_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "query": self.query,
            "as_of": self.as_of,
            "total_active_facts": self.total_active_facts,
            "total_vector_candidates": self.total_vector_candidates,
            "facts": [f.to_dict() for f in self.facts],
            "notes": [n.to_dict() for n in self.notes],
            "guidelines": [g.to_dict() for g in self.guidelines],
            "rejected": [r.to_dict() for r in self.rejected],
        }

    def to_context_str(self) -> str:
        """Formats the recalled context into a structured markdown block for LLM prompts."""
        as_of_dt = datetime.fromtimestamp(self.as_of, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        lines: list[str] = [f"### Patient Context (As Of: {as_of_dt})"]

        if self.facts:
            lines.append("\n#### Active Facts:")
            for f in self.facts:
                lines.append(
                    f"- [{f.kind.upper()}] {f.display}: {f.value} "
                    f"(score: {f.composite_score:.2f}, conf: {f.decayed_confidence:.2f}, "
                    f"auth: {f.authority})"
                )
        else:
            lines.append("\n#### Active Facts:\n- No active clinical facts found.")

        if self.notes:
            lines.append("\n#### Relevant Clinical Notes:")
            for n in self.notes:
                preview = n.text[:120] + "..." if len(n.text) > 120 else n.text
                lines.append(f"- [{n.kind}] {preview} (similarity: {n.vector_score:.2f})")

        if self.guidelines:
            lines.append("\n#### Applicable Clinic Guidelines:")
            for g in self.guidelines:
                lines.append(f"- {g.title} (v{g.version}, match: {g.vector_score:.2f})")

        return "\n".join(lines)


class RecallManager:
    """Manages hybrid vector and graph recall with scoring and provenance tracking."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clinic_reader: ClinicReader | None = None,
        clock: Clock | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clinic_reader = clinic_reader
        self._clock = clock or get_clock()
        self._embedder = embedder or get_embedder()

    # -------------------------------------------------------------------------
    # Note Management Helpers
    # -------------------------------------------------------------------------

    def add_note(
        self,
        patient_id: str,
        text: str,
        kind: str = "intake_note",
        author_agent_id: str = "agent_intake",
        episode_id: str | None = None,
        about_ids: list[str] | None = None,
        clinician_review_required: bool = False,
        ts: int | None = None,
        note_id: str | None = None,
    ) -> NoteRecord:
        """Create and embed a clinical note in the patient graph.

        Args:
            patient_id: Validated patient tenant identifier.
            text: Text content of the clinical observation.
            kind: Note category ('intake_note', 'clinical_draft', 'pharmacy_note', 'canary').
            author_agent_id: Identifier of the creating agent.
            episode_id: Optional encounter episode identifier to link.
            about_ids: Optional list of target node IDs this note is about.
            clinician_review_required: Whether note flags mandatory human signoff.
            ts: Creation timestamp in whole seconds.
            note_id: Optional explicit ULID note identifier.

        Returns:
            Created NoteRecord.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        if not text or not text.strip():
            raise ValidationError(
                "Note text cannot be empty.",
                details={"patient_id": valid_pid},
            )

        now_ts = ts if ts is not None else self._clock.now_ts()
        nid = note_id or gen_note_id()
        emb = self._embedder.embed(text.strip())

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

        # Insert Note node
        create_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (n:Note {
            id: $nid,
            text: $text,
            embedding: vecf32($emb),
            kind: $kind,
            author_agent_id: $author,
            status: 'active',
            clinician_review_required: $crr,
            ts: $ts
        })
        CREATE (p)-[:HAS_NOTE]->(n)
        RETURN n.id
        """
        target_graph.query(
            create_cypher,
            {
                "pid": valid_pid,
                "nid": nid,
                "text": text.strip(),
                "emb": emb,
                "kind": kind,
                "author": author_agent_id,
                "crr": clinician_review_required,
                "ts": now_ts,
            },
        )

        # Link Episode if provided
        if episode_id:
            link_ep_cypher = """
            MATCH (e:Episode {id:$eid}), (n:Note {id:$nid})
            MERGE (e)-[:HAS_NOTE]->(n)
            MERGE (n)-[:ASSERTED_IN]->(e)
            """
            target_graph.query(link_ep_cypher, {"eid": episode_id, "nid": nid})

        # Link ABOUT targets if provided
        linked_abouts: list[str] = []
        if about_ids:
            for tid in about_ids:
                if tid:
                    target_graph.query(
                        """
                        MATCH (n:Note {id:$nid}), (t {id:$tid})
                        MERGE (n)-[:ABOUT]->(t)
                        """,
                        {"nid": nid, "tid": tid},
                    )
                    linked_abouts.append(tid)

        return NoteRecord(
            id=nid,
            patient_id=valid_pid,
            text=text.strip(),
            kind=kind,
            author_agent_id=author_agent_id,
            status="active",
            clinician_review_required=clinician_review_required,
            ts=now_ts,
            episode_id=episode_id,
            about_ids=linked_abouts,
            embedding=emb,
        )

    def get_note(self, patient_id: str, note_id: str) -> NoteRecord:
        """Retrieve a specific note by ID."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cypher = """
        MATCH (n:Note {id:$nid})
        OPTIONAL MATCH (ep:Episode)-[:HAS_NOTE]->(n)
        OPTIONAL MATCH (n)-[:ABOUT]->(t)
        RETURN n.id, n.text, n.kind, n.author_agent_id, n.status,
               n.clinician_review_required, n.ts, ep.id, collect(DISTINCT t.id)
        """
        res = target_graph.query(cypher, {"nid": note_id})
        if not res.result_set:
            raise NoteNotFoundError(
                f"Note '{note_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "note_id": note_id},
            )

        row = res.result_set[0]
        return NoteRecord(
            id=row[0],
            patient_id=valid_pid,
            text=row[1],
            kind=row[2] or "intake_note",
            author_agent_id=row[3] or "",
            status=row[4] or "active",
            clinician_review_required=bool(row[5]),
            ts=row[6] or 0,
            episode_id=row[7],
            about_ids=row[8] or [],
        )

    # -------------------------------------------------------------------------
    # Hybrid Recall Pipeline
    # -------------------------------------------------------------------------

    def recall(
        self,
        patient_id: str,
        query: str = "",
        session_id: str | None = None,
        episode_id: str | None = None,
        kinds: list[str] | tuple[str, ...] | None = None,
        top_k: int = 10,
        vector_k: int = 5,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        include_notes: bool = True,
        include_facts: bool = True,
        include_clinic: bool = False,
        include_rejected: bool = True,
        as_of: int | None = None,
    ) -> RecallResult:
        """Execute hybrid recall across patient facts, notes, and clinic guidelines.

        Args:
            patient_id: Validated patient tenant identifier.
            query: Clinical query or question (e.g. "active medications and allergies").
            session_id: Optional active session identifier.
            episode_id: Optional active encounter episode identifier.
            kinds: Filter list of fact kinds (defaults to all standard kinds).
            top_k: Maximum number of top facts and notes to return.
            vector_k: Candidate limit for dense vector search.
            min_similarity: Cutoff threshold for note vector similarity.
            min_confidence: Cutoff threshold for decayed fact confidence.
            include_notes: Whether to run dense vector search on notes.
            include_facts: Whether to query semantic facts from the graph.
            include_clinic: Whether to search shared clinical guidelines.
            include_rejected: Whether to classify and return rejected candidates.
            as_of: Optional point-in-time timestamp (defaults to current clock time).

        Returns:
            RecallResult containing ranked facts, notes, guidelines, and rejected nodes.
        """
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

        now_ts = as_of if as_of is not None else self._clock.now_ts()
        fact_kinds = list(kinds) if kinds else list(DEFAULT_FACT_KINDS)

        recalled_facts: list[RecalledFact] = []
        recalled_notes: list[RecalledNote] = []
        recalled_guidelines: list[RecalledGuideline] = []
        rejected_candidates: list[RejectedCandidate] = []

        total_active_facts = 0
        total_vector_candidates = 0

        # 1. Semantic Facts Retrieval (Deterministic Cypher)
        if include_facts:
            facts, fact_rejections, total_active = self._retrieve_and_rank_facts(
                valid_pid=valid_pid,
                target_graph=target_graph,
                query=query,
                kinds=fact_kinds,
                now_ts=now_ts,
                is_historical=as_of is not None and as_of < self._clock.now_ts(),
                min_confidence=min_confidence,
                include_rejected=include_rejected,
            )
            recalled_facts.extend(facts)
            if include_rejected:
                rejected_candidates.extend(fact_rejections)
            total_active_facts = total_active

        # 2. Dense Vector Retrieval on Notes
        if include_notes and query.strip():
            notes, note_rejections, total_vec = self._retrieve_and_rank_notes(
                target_graph=target_graph,
                query=query.strip(),
                k=vector_k,
                min_similarity=min_similarity,
                include_rejected=include_rejected,
            )
            recalled_notes.extend(notes)
            if include_rejected:
                rejected_candidates.extend(note_rejections)
            total_vector_candidates = total_vec

        # 3. Clinic Shared Graph Guidelines Retrieval
        if include_clinic and query.strip():
            guidelines, guide_rejections = self._retrieve_clinic_guidelines(
                query=query.strip(),
                k=vector_k,
                min_similarity=min_similarity,
                include_rejected=include_rejected,
            )
            recalled_guidelines.extend(guidelines)
            if include_rejected:
                rejected_candidates.extend(guide_rejections)

        # Sort and assign ranks
        recalled_facts.sort(key=lambda f: f.composite_score, reverse=True)
        for idx, f in enumerate(recalled_facts, start=1):
            f.rank = idx
        recalled_facts = recalled_facts[:top_k]

        recalled_notes.sort(key=lambda n: n.composite_score, reverse=True)
        for idx, n in enumerate(recalled_notes, start=1):
            n.rank = idx
        recalled_notes = recalled_notes[:top_k]

        recalled_guidelines.sort(key=lambda g: g.vector_score, reverse=True)
        rejected_candidates.sort(key=lambda r: r.score, reverse=True)

        return RecallResult(
            patient_id=valid_pid,
            query=query,
            facts=recalled_facts,
            notes=recalled_notes,
            guidelines=recalled_guidelines,
            rejected=rejected_candidates,
            as_of=now_ts,
            total_active_facts=total_active_facts,
            total_vector_candidates=total_vector_candidates,
        )

    # -------------------------------------------------------------------------
    # Internal Fact Retrieval & Scoring
    # -------------------------------------------------------------------------

    def _retrieve_and_rank_facts(
        self,
        valid_pid: str,
        target_graph: Any,
        query: str,
        kinds: list[str],
        now_ts: int,
        is_historical: bool,
        min_confidence: float,
        include_rejected: bool,
    ) -> tuple[list[RecalledFact], list[RejectedCandidate], int]:
        """Fetch and score semantic facts using Cypher traversal and decay math."""
        active_facts: list[RecalledFact] = []
        rejected: list[RejectedCandidate] = []

        if not is_historical:
            # Current active belief using q_recall_01
            q_recall_01 = get_query("q_recall_01")
            res = target_graph.query(
                q_recall_01,
                {"pid": valid_pid, "kinds": kinds, "now": now_ts},
            )
            rows = res.result_set or []
        else:
            # Historical as-of point-in-time belief using q_recall_02
            q_recall_02 = get_query("q_recall_02")
            res = target_graph.query(
                q_recall_02,
                {"pid": valid_pid, "kind": None, "asof": now_ts},
            )
            raw_rows = res.result_set or []
            # Transform q_recall_02 row format to match q_recall_01 format
            rows = []
            for r in raw_rows:
                r_kind = r[1]
                if r_kind in kinds:
                    rows.append(
                        [
                            r[0],  # id
                            r[1],  # kind
                            r[3],  # display
                            r[4],  # value
                            r[10],  # confidence
                            r[6],  # last_confirmed_at (approx valid_from)
                            180,  # half_life_days
                            r[11],  # authority
                            r[12],  # version
                        ]
                    )

        total_active = len(rows)

        # Process each active candidate fact
        for row in rows:
            fid = row[0]
            fkind = row[1]
            fdisp = row[2] or ""
            fval = row[3] or ""
            base_conf = float(row[4]) if row[4] is not None else 0.7
            l_conf = int(row[5]) if row[5] is not None else now_ts
            h_life = int(row[6]) if row[6] is not None else None
            auth = str(row[7]) if row[7] is not None else "patient_reported"
            version = int(row[8]) if row[8] is not None else 1

            # 1. Half-life exponential temporal decay
            decayed_conf = self._compute_decayed_confidence(
                base_confidence=base_conf,
                last_confirmed_at=l_conf,
                half_life_days=h_life,
                now_ts=now_ts,
            )

            # Filter low confidence into rejected
            if decayed_conf < min_confidence:
                if include_rejected:
                    rejected.append(
                        RejectedCandidate(
                            id=fid,
                            label="Fact",
                            reason_code="LOW_CONFIDENCE",
                            reason=(
                                f"Decayed confidence {decayed_conf:.2f} is below "
                                f"threshold {min_confidence:.2f}."
                            ),
                            score=decayed_conf,
                        )
                    )
                continue

            # 2. Text relevance score against query
            rel_score = self._compute_relevance(
                query=query,
                display=fdisp,
                value=fval,
                kind=fkind,
            )

            # 3. Authority weight
            auth_wt = AUTHORITY_WEIGHTS.get(auth.lower(), 0.70)

            # 4. Composite ranking score
            comp_score = round(
                ALPHA_RELEVANCE * rel_score
                + BETA_CONFIDENCE * decayed_conf
                + GAMMA_AUTHORITY * auth_wt,
                4,
            )

            active_facts.append(
                RecalledFact(
                    id=fid,
                    kind=fkind,
                    display=fdisp,
                    value=fval,
                    confidence=base_conf,
                    decayed_confidence=decayed_conf,
                    authority=auth,
                    authority_weight=auth_wt,
                    last_confirmed_at=l_conf,
                    half_life_days=h_life,
                    relevance_score=rel_score,
                    composite_score=comp_score,
                    version=version,
                )
            )

        # Classify non-active rejected facts (superseded, disputed, pending_review, expired)
        if include_rejected and not is_historical:
            rejected.extend(
                self._classify_non_active_facts(
                    valid_pid=valid_pid,
                    target_graph=target_graph,
                    now_ts=now_ts,
                )
            )

        return active_facts, rejected, total_active

    def _compute_decayed_confidence(
        self,
        base_confidence: float,
        last_confirmed_at: int,
        half_life_days: int | None,
        now_ts: int,
    ) -> float:
        """Calculates exponential confidence decay based on elapsed days and half life."""
        if half_life_days is None or half_life_days <= 0:
            return round(base_confidence, 4)

        elapsed_sec = max(0, now_ts - last_confirmed_at)
        elapsed_days = elapsed_sec / 86400.0
        decay = math.pow(0.5, elapsed_days / float(half_life_days))
        return round(base_confidence * decay, 4)

    def _compute_relevance(
        self,
        query: str,
        display: str,
        value: str,
        kind: str,
    ) -> float:
        """Computes lexical relevance score between search query and fact properties."""
        if not query.strip():
            return 1.0

        q_lower = query.lower().strip()
        full_text = f"{kind} {display} {value}".lower()

        # Exact substring match
        if q_lower in full_text:
            return 1.0

        q_tokens = set(re.findall(r"\w+", q_lower))
        if not q_tokens:
            return 1.0

        doc_tokens = set(re.findall(r"\w+", full_text))
        overlap = q_tokens.intersection(doc_tokens)

        if not overlap:
            return 0.1

        overlap_ratio = len(overlap) / len(q_tokens)
        return round(0.5 + 0.5 * overlap_ratio, 4)

    def _classify_non_active_facts(
        self,
        valid_pid: str,
        target_graph: Any,
        now_ts: int,
    ) -> list[RejectedCandidate]:
        """Queries superseded, disputed, pending, and expired facts for Why-Not explainability."""
        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact)
        WHERE f.status <> 'active'
           OR (f.valid_to IS NOT NULL AND f.valid_to <= $now)
           OR (f.valid_from > $now)
        OPTIONAL MATCH (newer:Fact)-[:SUPERSEDES]->(f)
        RETURN f.id, f.status, f.valid_from, f.valid_to, f.confidence,
               newer.id, f.display, f.value
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "now": now_ts})
        rejected: list[RejectedCandidate] = []

        for row in res.result_set or []:
            fid = row[0]
            status = row[1] or "active"
            v_from = row[2] or 0
            v_to = row[3]
            conf = float(row[4]) if row[4] is not None else 0.5
            newer_id = row[5]

            if status == "superseded":
                reason = f"Superseded by newer assertion '{newer_id}'" if newer_id else "Superseded"
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="SUPERSEDED",
                        reason=reason,
                        score=conf,
                        superseded_by=newer_id,
                    )
                )
            elif status == "disputed":
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="DISPUTED",
                        reason="Disputed due to conflicting assertions from multiple sources.",
                        score=conf,
                    )
                )
            elif status == "pending_review":
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="PENDING_REVIEW",
                        reason="Requires clinician review before active clinical use.",
                        score=conf,
                    )
                )
            elif status == "retracted":
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="RETRACTED",
                        reason="Fact was explicitly retracted or tombstoned.",
                        score=conf,
                    )
                )
            elif v_to is not None and v_to <= now_ts:
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="EXPIRED_VALIDITY",
                        reason=f"Real-world validity period expired at epoch {v_to}.",
                        score=conf,
                    )
                )
            elif v_from > now_ts:
                rejected.append(
                    RejectedCandidate(
                        id=fid,
                        label="Fact",
                        reason_code="FUTURE_VALIDITY",
                        reason=f"Real-world validity starts in the future at epoch {v_from}.",
                        score=conf,
                    )
                )

        return rejected

    # -------------------------------------------------------------------------
    # Internal Note Vector Retrieval
    # -------------------------------------------------------------------------

    def _retrieve_and_rank_notes(
        self,
        target_graph: Any,
        query: str,
        k: int,
        min_similarity: float,
        include_rejected: bool,
    ) -> tuple[list[RecalledNote], list[RejectedCandidate], int]:
        """Perform dense vector search against Note embeddings and filter candidates."""
        qvec = self._embedder.embed(query)
        q_recall_03a = get_query("q_recall_03a")

        # FalkorDB vector index search
        res_a = target_graph.query(q_recall_03a, {"k": k, "qvec": qvec})
        candidates = res_a.result_set or []
        total_candidates = len(candidates)

        if not candidates:
            return [], [], 0

        # Convert FalkorDB cosine distance (0.0=exact match) to cosine similarity
        score_map = {row[0]: max(0.0, min(1.0, 1.0 - float(row[1]))) for row in candidates}
        candidate_ids = list(score_map.keys())

        # Inspect all candidate nodes to classify status, canary, and similarity
        inspect_cypher = """
        MATCH (n:Note)
        WHERE n.id IN $ids
        RETURN n.id, n.kind, n.status
        """
        insp_res = target_graph.query(inspect_cypher, {"ids": candidate_ids})

        accepted_ids: list[str] = []
        rejected: list[RejectedCandidate] = []

        for row in insp_res.result_set or []:
            nid = row[0]
            nkind = row[1] or "intake_note"
            nstatus = row[2] or "active"
            sim_score = score_map.get(nid, 0.0)

            # Security check: canary tokens must never leak into recall
            if nkind == "canary":
                if include_rejected:
                    rejected.append(
                        RejectedCandidate(
                            id=nid,
                            label="Note",
                            reason_code="CANARY_SECURITY",
                            reason="Canary security token filtered from recall context.",
                            score=sim_score,
                        )
                    )
                continue

            # Status check
            if nstatus != "active":
                if include_rejected:
                    rejected.append(
                        RejectedCandidate(
                            id=nid,
                            label="Note",
                            reason_code="RETRACTED" if nstatus == "retracted" else "SUPERSEDED",
                            reason=f"Note status is '{nstatus}'.",
                            score=sim_score,
                        )
                    )
                continue

            # Similarity threshold check
            if sim_score < min_similarity:
                if include_rejected:
                    rejected.append(
                        RejectedCandidate(
                            id=nid,
                            label="Note",
                            reason_code="LOW_SIMILARITY",
                            reason=(
                                f"Vector similarity {sim_score:.3f} is below "
                                f"threshold {min_similarity:.2f}."
                            ),
                            score=sim_score,
                        )
                    )
                continue

            accepted_ids.append(nid)

        if not accepted_ids:
            return [], rejected, total_candidates

        # Fetch accepted notes details using q_recall_03b
        q_recall_03b = get_query("q_recall_03b")
        res_b = target_graph.query(q_recall_03b, {"ids": accepted_ids})

        notes: list[RecalledNote] = []
        for row in res_b.result_set or []:
            nid = row[0]
            ntext = row[1] or ""
            author = row[2] or ""
            epid = row[3]
            abouts = row[4] or []
            v_score = score_map.get(nid, 0.0)

            notes.append(
                RecalledNote(
                    id=nid,
                    text=ntext,
                    kind="clinical_note",
                    author_agent_id=author,
                    episode_id=epid,
                    about_ids=abouts,
                    vector_score=round(v_score, 4),
                    composite_score=round(v_score, 4),
                )
            )

        return notes, rejected, total_candidates

    # -------------------------------------------------------------------------
    # Internal Clinic Knowledge Search
    # -------------------------------------------------------------------------

    def _retrieve_clinic_guidelines(
        self,
        query: str,
        k: int,
        min_similarity: float,
        include_rejected: bool,
    ) -> tuple[list[RecalledGuideline], list[RejectedCandidate]]:
        """Search shared clinical guidelines using vector similarity."""
        if not self._clinic_reader:
            return [], []

        qvec = self._embedder.embed(query)
        cypher = """
        CALL db.idx.vector.queryNodes('Guideline', 'embedding', $k, vecf32($qvec))
        YIELD node, score
        RETURN node.id, node.title, node.version, node.text, node.condition_code, score
        """
        try:
            res = self._clinic_reader.ro_query(cypher, {"k": k, "qvec": qvec})
        except Exception as e:
            logger.warning(f"Clinic guideline search failed: {e}")
            return [], []

        guidelines: list[RecalledGuideline] = []
        rejected: list[RejectedCandidate] = []

        for row in res.result_set or []:
            gid = row[0]
            title = row[1] or ""
            ver = row[2] or "1.0"
            text = row[3] or ""
            ccode = row[4]
            raw_dist = float(row[5]) if row[5] is not None else 1.0
            score = max(0.0, min(1.0, 1.0 - raw_dist))

            if score < min_similarity:
                if include_rejected:
                    rejected.append(
                        RejectedCandidate(
                            id=gid,
                            label="Guideline",
                            reason_code="LOW_SIMILARITY",
                            reason=(
                                f"Guideline vector similarity {score:.3f} below "
                                f"threshold {min_similarity:.2f}."
                            ),
                            score=score,
                        )
                    )
                continue

            guidelines.append(
                RecalledGuideline(
                    id=gid,
                    title=title,
                    version=str(ver),
                    text=text,
                    condition_code=ccode,
                    vector_score=round(score, 4),
                )
            )

        return guidelines, rejected
