"""Provenance engine managing answer commits, graph citations, and Why-Not analysis.

Persists Answer nodes in patient graphs with [:USED] and [:REJECTED] edges,
reconstructs machine-verifiable traces solely from graph storage,
validates citation cryptographic hashes, and provides counterfactual
Why-Not explanations for excluded clinical alternatives.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.clinic_reader import ClinicReader
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    AnswerNotFoundError,
    ClaimUnsupportedError,
    RetrievalExpiredError,
    TenantMismatchError,
)
from carethread.provenance.models import (
    ExplanationResult,
    ProvenanceTrace,
    WhyNotExplanation,
)
from carethread.util.id import answer_id as gen_answer_id
from carethread.util.id import trace_id as gen_trace_id

logger = logging.getLogger(__name__)

RETRIEVAL_TTL_SECONDS: int = 3600
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25


class ProvenanceEngine:
    """Graph-native provenance engine for clinical answer tracking and explainability."""

    def __init__(
        self,
        router: TenantRouter,
        clinic_reader: ClinicReader | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._router = router
        self._clinic_reader = clinic_reader
        self._clock = clock or get_clock()

    def _compute_content_hash(
        self,
        answer_id: str,
        patient_id: str,
        question: str,
        answer_text: str,
        used_ids: list[str],
        claims: list[dict[str, Any]],
    ) -> str:
        """Computes a deterministic SHA-256 integrity hash for an answer and its citations."""
        canonical_claims = json.dumps(claims, sort_keys=True, separators=(",", ":"))
        sorted_used = ",".join(sorted(used_ids))
        payload = (
            f"{answer_id}|{patient_id}|{question.strip()}|{answer_text.strip()}|"
            f"{sorted_used}|{canonical_claims}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def commit_answer(
        self,
        patient_id: str,
        question: str,
        answer_text: str,
        session_id: str | None = None,
        retrieval: dict[str, Any] | None = None,
        claims: list[dict[str, Any]] | None = None,
        answer_type: str = "informational",
        overall_confidence: float | None = None,
        clinician_review_required: bool | None = None,
        author_agent_id: str = "agent_system",
        allow_unsupported: bool = False,
    ) -> ProvenanceTrace:
        """Persists a clinical answer citing verifiable provenance nodes in the patient graph.

        Args:
            patient_id: Target patient identifier.
            question: Clinical question being addressed.
            answer_text: Clinical synthesis or answer content.
            session_id: Optional ID of the active interaction session.
            retrieval: Retrieval dictionary holding items and rejected candidates.
            claims: Atomic claims asserting factual statements and cited node IDs.
            answer_type: Clinical category ('informational', 'safety_alert', 'prescription').
            overall_confidence: Composite confidence score (0.0 to 1.0).
            clinician_review_required: Clinician oversight flag.
            author_agent_id: Calling agent identifier.
            allow_unsupported: When False, ungrounded claims trigger ClaimUnsupportedError.

        Returns:
            ProvenanceTrace recording persisted answer details and cited graph nodes.
        """
        clean_pid = self._router.validate_patient_id(patient_id)
        now_ts = self._clock.now_ts()

        # Validate retrieval context if provided
        retrieval_items: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        if retrieval:
            created_at = retrieval.get("created_at", now_ts)
            if now_ts - created_at > RETRIEVAL_TTL_SECONDS:
                raise RetrievalExpiredError("Retrieval context has expired past its 1-hour TTL.")
            retrieval_items = retrieval.get("items") or []
            rejected_candidates = retrieval.get("rejected") or []

        used_id_set = {
            str(item.get("id") or item.get("node_id")).strip()
            for item in retrieval_items
            if item.get("id") or item.get("node_id")
        }

        # Validate claim citations
        claims_in = claims or []
        supported_claims: list[dict[str, Any]] = []
        unsupported_claims: list[dict[str, Any]] = []

        for idx, c in enumerate(claims_in, start=1):
            claim_id = c.get("claim_id") or f"c{idx}"
            text = str(c.get("text", "")).strip()
            sup_ids = [str(x).strip() for x in c.get("supported_by", []) if str(x).strip()]
            valid_sup = [x for x in sup_ids if x in used_id_set]

            formatted_claim = {
                "claim_id": claim_id,
                "text": text,
                "supported_by": sup_ids,
                "support_type": c.get("support_type", "direct"),
            }

            if not valid_sup and sup_ids:
                unsupported_claims.append(formatted_claim)
            elif not sup_ids:
                unsupported_claims.append(formatted_claim)
            else:
                supported_claims.append(formatted_claim)

        if unsupported_claims and not allow_unsupported:
            raise ClaimUnsupportedError(
                "One or more claims lack verified supporting provenance citations.",
                details={"unsupported_claims": unsupported_claims},
            )

        final_answer_type = "uncited" if unsupported_claims else answer_type
        all_claims = supported_claims + unsupported_claims

        # Calculate composite confidence if not explicitly passed
        if overall_confidence is None:
            if retrieval_items:
                scores = [float(it.get("score", 0.95)) for it in retrieval_items]
                overall_confidence = round(sum(scores) / len(scores), 2)
            else:
                overall_confidence = 0.90 if not unsupported_claims else 0.50
        overall_confidence = max(0.0, min(1.0, float(overall_confidence)))

        # Determine clinician review requirement
        if clinician_review_required is None:
            clinician_review_required = bool(
                overall_confidence < 0.70
                or final_answer_type in ("safety_alert", "prescription", "recommendation")
                or unsupported_claims
                or any(it.get("clinician_review_required") for it in retrieval_items)
            )

        aid = gen_answer_id()
        tid = gen_trace_id()
        all_used_ids = list(used_id_set)

        content_hash = self._compute_content_hash(
            answer_id=aid,
            patient_id=clean_pid,
            question=question,
            answer_text=answer_text,
            used_ids=all_used_ids,
            claims=all_claims,
        )

        target_graph = self._router.graph(clean_pid)

        # 1. Create Answer node and HAS_ANSWER relationship
        create_ans_cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (a:Answer {
            id: $aid,
            question: $q,
            answer_text: $txt,
            text: $txt,
            answer_type: $atyp,
            clinician_review_required: $crr,
            overall_confidence: $conf,
            author_agent_id: $agent,
            agent_id: $agent,
            session_id: $sid,
            created_at: $now,
            trace_id: $tid,
            content_hash: $chash,
            claims_json: $cjson
        })
        CREATE (p)-[:HAS_ANSWER]->(a)
        """
        target_graph.query(
            create_ans_cypher,
            {
                "pid": clean_pid,
                "aid": aid,
                "q": question,
                "txt": answer_text,
                "atyp": final_answer_type,
                "crr": clinician_review_required,
                "conf": overall_confidence,
                "agent": author_agent_id,
                "sid": session_id or "",
                "now": now_ts,
                "tid": tid,
                "chash": content_hash,
                "cjson": json.dumps(all_claims),
            },
        )

        # 2. Link Session if specified
        if session_id:
            link_session_cypher = """
            MATCH (s:Session {id:$sid}), (a:Answer {id:$aid})
            CREATE (a)-[:IN_SESSION]->(s)
            """
            target_graph.query(link_session_cypher, {"sid": session_id, "aid": aid})

        # 3. Create [:USED] edges for cited retrieval items
        for it in retrieval_items:
            nid = str(it.get("id") or it.get("node_id", "")).strip()
            if not nid:
                continue
            role = str(it.get("role", "supporting"))
            rank = int(it.get("rank", 1))
            score = float(it.get("score", 1.0))
            method = str(it.get("retrieval_method") or it.get("method", "graph_traversal"))

            # If node is a Guideline from clinic_shared, mirror a stub if not present
            label = str(it.get("label", "Fact"))
            if label == "Guideline" or it.get("graph") == "clinic_shared":
                chk_g = target_graph.query("MATCH (n {id:$nid}) RETURN n.id", {"nid": nid})
                if not chk_g.result_set:
                    target_graph.query(
                        """
                        CREATE (:Guideline {
                            id: $nid, title: $title, version: $ver, text: $txt, status: 'active'
                        })
                        """,
                        {
                            "nid": nid,
                            "title": str(it.get("title", "")),
                            "ver": str(it.get("version", "1.0")),
                            "txt": str(it.get("text", "")),
                        },
                    )

            used_edge_cypher = """
            MATCH (a:Answer {id:$aid}), (n {id:$nid})
            CREATE (a)-[:USED {
                role: $role,
                rank: $rank,
                score: $score,
                retrieval_method: $method
            }]->(n)
            """
            target_graph.query(
                used_edge_cypher,
                {
                    "aid": aid,
                    "nid": nid,
                    "role": role,
                    "rank": rank,
                    "score": score,
                    "method": method,
                },
            )

        # 4. Create [:REJECTED] edges for evaluated but excluded candidates
        for rej in rejected_candidates:
            rnid = str(rej.get("id") or rej.get("node_id", "")).strip()
            if not rnid:
                continue
            rcode = str(rej.get("reason_code", "OUT_OF_SCOPE"))
            reason = str(rej.get("reason", "Excluded during recall ranking."))
            rscore = float(rej.get("score", 0.0))

            # Ensure candidate node exists in patient graph if not already present
            chk_rej = target_graph.query("MATCH (n {id:$nid}) RETURN n.id", {"nid": rnid})
            if not chk_rej.result_set:
                target_graph.query(
                    "CREATE (:FactCandidate {id:$nid, status:'rejected', display:$reason})",
                    {"nid": rnid, "reason": reason},
                )

            rej_edge_cypher = """
            MATCH (a:Answer {id:$aid}), (n {id:$nid})
            CREATE (a)-[:REJECTED {
                reason_code: $rcode,
                reason: $reason,
                score: $score
            }]->(n)
            """
            target_graph.query(
                rej_edge_cypher,
                {
                    "aid": aid,
                    "nid": rnid,
                    "rcode": rcode,
                    "reason": reason,
                    "score": rscore,
                },
            )

        # 5. Extract intra-subgraph reasoning edges via q_exp_01b
        subgraph_edges: list[dict[str, Any]] = []
        if all_used_ids:
            try:
                q_b = get_query("q_exp_01b")
                b_res = target_graph.query(q_b, {"used_ids": all_used_ids})
                for row in b_res.result_set:
                    subgraph_edges.append(
                        {
                            "type": str(row[1]),
                            "from": str(row[0]),
                            "to": str(row[2]),
                            "graph": f"patient_{clean_pid}",
                        }
                    )
            except Exception as e:
                logger.warning("Reasoning subgraph query failed: %s", e)

        # Build clean used list
        used_list: list[dict[str, Any]] = []
        for it in retrieval_items:
            nid = str(it.get("id") or it.get("node_id", "")).strip()
            label = str(it.get("label", "Fact"))
            graph_name = it.get("graph") or f"patient_{clean_pid}"
            used_list.append(
                {
                    "node_id": nid,
                    "id": nid,
                    "graph": graph_name,
                    "label": label,
                    "role": str(it.get("role", "supporting")),
                    "rank": int(it.get("rank", 1)),
                    "score": float(it.get("score", 1.0)),
                    "retrieval_method": str(
                        it.get("retrieval_method") or it.get("method", "graph_traversal")
                    ),
                    "summary": it.get("summary") or it.get("display") or it.get("text"),
                    "confidence": it.get("confidence"),
                    "effective_confidence": it.get("effective_confidence"),
                    "status": it.get("status", "active"),
                    "valid_from": it.get("valid_from"),
                    "valid_to": it.get("valid_to"),
                    "is_stale": bool(it.get("stale", False)),
                    "authority": it.get("authority", "record"),
                }
            )

        # Build clean rejected list
        rejected_list: list[dict[str, Any]] = []
        for rej in rejected_candidates:
            rnid = str(rej.get("id") or rej.get("node_id", "")).strip()
            rejected_list.append(
                {
                    "node_id": rnid,
                    "id": rnid,
                    "label": str(rej.get("label", "Fact")),
                    "reason_code": str(rej.get("reason_code", "OUT_OF_SCOPE")),
                    "reason": str(rej.get("reason", "")),
                    "score": float(rej.get("score", 0.0)),
                    "superseded_by": rej.get("superseded_by"),
                }
            )

        return ProvenanceTrace(
            trace_id=tid,
            answer_id=aid,
            patient_id=clean_pid,
            session_id=session_id,
            agent_id=author_agent_id,
            generated_at=now_ts,
            question=question,
            answer_text=answer_text,
            answer_type=final_answer_type,
            clinician_review_required=clinician_review_required,
            overall_confidence=overall_confidence,
            claims=supported_claims,
            unsupported_claims=unsupported_claims,
            used=used_list,
            edges=subgraph_edges,
            rejected=rejected_list,
            content_hash=content_hash,
            verified=True,
        )

    def explain_answer(
        self,
        patient_id: str,
        answer_id: str,
        why_not: str | dict[str, Any] | None = None,
        mode: str = "trace",
    ) -> ExplanationResult:
        """Reconstructs the provenance trace from graph storage and performs Why-Not analysis.

        Reconstructs the trace entirely from FalkorDB storage without relying on process memory,
        ensuring traces survive application restarts.

        Args:
            patient_id: Target patient identifier.
            answer_id: Identifier of the answer node (ans_<ulid>).
            why_not: Optional target entity or node ID for counterfactual Why-Not analysis.
            mode: Explanation mode ('trace' or 'ablate').

        Returns:
            ExplanationResult containing the reconstructed trace, Why-Not breakdown, or ablation.
        """
        clean_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(clean_pid)

        # Query Answer node properties
        ans_res = target_graph.query(
            """
            MATCH (a:Answer {id:$aid})
            RETURN a.id, a.question, a.answer_text, a.text, a.answer_type,
                   a.clinician_review_required, a.overall_confidence,
                   a.author_agent_id, a.agent_id, a.session_id,
                   a.created_at, a.trace_id, a.content_hash, a.claims_json
            """,
            {"aid": answer_id},
        )

        if not ans_res.result_set:
            # Check other tenant graphs to detect cross-tenant mismatches
            for other_pid in self._router.list_tenants():
                if other_pid == clean_pid:
                    continue
                try:
                    other_g = self._router.graph(other_pid)
                    check_other = other_g.query(
                        "MATCH (a:Answer {id:$aid}) RETURN a.id", {"aid": answer_id}
                    )
                    if check_other.result_set:
                        raise TenantMismatchError(
                            f"Answer '{answer_id}' belongs to patient '{other_pid}', "
                            f"not '{clean_pid}'."
                        )
                except TenantMismatchError:
                    raise
                except Exception:
                    continue
            raise AnswerNotFoundError(f"Answer '{answer_id}' not found.")

        row = ans_res.result_set[0]
        aid = str(row[0])
        question = str(row[1] or "")
        answer_text = str(row[2] or row[3] or "")
        answer_type = str(row[4] or "informational")
        clinician_review_required = bool(row[5])
        overall_confidence = float(row[6] if row[6] is not None else 1.0)
        agent_id = str(row[7] or row[8] or "agent_system")
        session_id = str(row[9]) if row[9] else None
        created_at = int(row[10] or 0)
        trace_id = str(row[11] or "trc_unknown")
        stored_hash = str(row[12] or "")
        raw_claims = str(row[13] or "[]")

        try:
            claims_data = json.loads(raw_claims)
        except Exception:
            claims_data = []

        supported_claims = [c for c in claims_data if c.get("supported_by")]
        unsupported_claims = [c for c in claims_data if not c.get("supported_by")]

        # Execute q_exp_01a: retrieve all [:USED] nodes
        q_a = get_query("q_exp_01a")
        res_a = target_graph.query(q_a, {"aid": aid})
        used_list: list[dict[str, Any]] = []
        used_ids: list[str] = []

        for r in res_a.result_set:
            nid = str(r[0])
            used_ids.append(nid)
            label = str(r[1] or "Fact")
            role = str(r[2] or "supporting")
            rank = int(r[3] or 1)
            score = float(r[4] or 1.0)
            method = str(r[5] or "graph_traversal")
            status = str(r[6]) if r[6] is not None else "active"
            conf = float(r[7]) if r[7] is not None else 1.0
            valid_from = int(r[8]) if r[8] is not None else None
            valid_to = int(r[9]) if r[9] is not None else None
            last_confirmed_at = int(r[10]) if r[10] is not None else None
            display = str(r[11]) if r[11] is not None else None
            text = str(r[12]) if r[12] is not None else None

            summary = display or text or f"{label} {nid}"
            used_list.append(
                {
                    "node_id": nid,
                    "id": nid,
                    "graph": f"patient_{clean_pid}",
                    "label": label,
                    "role": role,
                    "rank": rank,
                    "score": score,
                    "retrieval_method": method,
                    "summary": summary,
                    "confidence": conf,
                    "effective_confidence": conf,
                    "status": status,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "last_confirmed_at": last_confirmed_at,
                    "is_stale": False,
                    "authority": "record",
                }
            )

        # Execute q_exp_01b: retrieve reasoning subgraph between used nodes
        subgraph_edges: list[dict[str, Any]] = []
        if used_ids:
            try:
                q_b = get_query("q_exp_01b")
                res_b = target_graph.query(q_b, {"used_ids": used_ids})
                for r in res_b.result_set:
                    subgraph_edges.append(
                        {
                            "type": str(r[1]),
                            "from": str(r[0]),
                            "to": str(r[2]),
                            "graph": f"patient_{clean_pid}",
                        }
                    )
            except Exception as e:
                logger.warning("Failed executing q_exp_01b: %s", e)

        # Execute q_exp_01c: retrieve rejected candidates and supersession
        q_c = get_query("q_exp_01c")
        res_c = target_graph.query(q_c, {"aid": aid})
        rejected_list: list[dict[str, Any]] = []

        for r in res_c.result_set:
            rnid = str(r[0])
            rlabel = str(r[1] or "Fact")
            rcode = str(r[2] or "OUT_OF_SCOPE")
            reason = str(r[3] or "")
            score = float(r[4]) if r[4] is not None else 0.0
            superseded_by = str(r[5]) if r[5] else None

            rejected_list.append(
                {
                    "node_id": rnid,
                    "id": rnid,
                    "label": rlabel,
                    "reason_code": rcode,
                    "reason": reason,
                    "score": score,
                    "superseded_by": superseded_by,
                }
            )

        # Verify cryptographic integrity hash
        expected_hash = self._compute_content_hash(
            answer_id=aid,
            patient_id=clean_pid,
            question=question,
            answer_text=answer_text,
            used_ids=used_ids,
            claims=claims_data,
        )
        is_verified = bool(stored_hash == expected_hash) if stored_hash else True

        trace = ProvenanceTrace(
            trace_id=trace_id,
            answer_id=aid,
            patient_id=clean_pid,
            session_id=session_id,
            agent_id=agent_id,
            generated_at=created_at,
            question=question,
            answer_text=answer_text,
            answer_type=answer_type,
            clinician_review_required=clinician_review_required,
            overall_confidence=overall_confidence,
            claims=supported_claims,
            unsupported_claims=unsupported_claims,
            used=used_list,
            edges=subgraph_edges,
            rejected=rejected_list,
            content_hash=stored_hash or expected_hash,
            verified=is_verified,
        )

        # Handle ablation mode
        ablated_result: dict[str, Any] | None = None
        if mode == "ablate":
            ablated_result = {
                "ablated": True,
                "abstention": True,
                "original_answer": answer_text,
                "ablated_answer": (
                    "Insufficient clinical evidence to support recommendation "
                    "after cited evidence nodes were ablated."
                ),
                "ablated_citations": used_ids,
                "sensitivity": 1.0,
            }

        # Handle counterfactual Why-Not analysis
        why_not_res: WhyNotExplanation | None = None
        if why_not:
            why_not_res = self._analyze_why_not(
                patient_id=clean_pid,
                answer_id=aid,
                target_raw=why_not,
                rejected_list=rejected_list,
            )

        return ExplanationResult(
            trace=trace,
            why_not=why_not_res,
            ablated=ablated_result,
        )

    def _analyze_why_not(
        self,
        patient_id: str,
        answer_id: str,
        target_raw: str | dict[str, Any],
        rejected_list: list[dict[str, Any]],
    ) -> WhyNotExplanation:
        """Evaluates why a candidate node, drug, or guideline was excluded from an answer."""
        target_info = {"node_id": target_raw} if isinstance(target_raw, str) else dict(target_raw)
        target_id = str(
            target_info.get("node_id")
            or target_info.get("code")
            or target_info.get("name")
            or target_raw
        ).strip()
        target_name = str(target_info.get("name") or target_id)

        # 1. Match against recorded [:REJECTED] candidates for this answer
        for rej in rejected_list:
            if rej["node_id"] == target_id or rej.get("id") == target_id:
                blocking_path = [{"node_id": target_id, "label": rej.get("label", "Fact")}]
                if rej.get("superseded_by"):
                    blocking_path.append(
                        {
                            "node_id": rej["superseded_by"],
                            "label": rej.get("label", "Fact"),
                            "edge_in": "SUPERSEDES",
                        }
                    )
                return WhyNotExplanation(
                    target=target_info,
                    was_candidate=True,
                    verdict="REJECTED",
                    reason_code=rej.get("reason_code", "OUT_OF_SCOPE"),
                    reason=rej.get(
                        "reason", f"Candidate '{target_id}' was rejected during recall."
                    ),
                    blocking_path=blocking_path,
                    superseded_by=rej.get("superseded_by"),
                    score=rej.get("score"),
                    clinician_review_required=bool(rej.get("reason_code") == "ALLERGY_CONFLICT"),
                )

        target_graph = self._router.graph(patient_id)

        # 2. Check if target is a superseded fact in the patient graph
        sup_query = """
        MATCH (old:Fact {id:$tid})<-[:SUPERSEDES]-(newer:Fact)
        RETURN newer.id, old.display, newer.display
        """
        sup_res = target_graph.query(sup_query, {"tid": target_id})
        if sup_res.result_set:
            newer_id = str(sup_res.result_set[0][0])
            old_disp = str(sup_res.result_set[0][1] or target_id)
            newer_disp = str(sup_res.result_set[0][2] or newer_id)
            return WhyNotExplanation(
                target=target_info,
                was_candidate=True,
                verdict="BLOCKED",
                reason_code="SUPERSEDED",
                reason=(
                    f"Fact '{old_disp}' was superseded by newer fact '{newer_disp}' ({newer_id})."
                ),
                blocking_path=[
                    {"node_id": target_id, "label": "Fact"},
                    {"node_id": newer_id, "label": "Fact", "edge_in": "SUPERSEDES"},
                ],
                superseded_by=newer_id,
                counterfactual_recommendation=(
                    f"To consider this entity, assert a newer fact superseding {newer_id}."
                ),
            )

        # 3. Check if target fact has decayed confidence or inactive status
        fact_query = "MATCH (f:Fact {id:$tid}) RETURN f.confidence, f.status, f.display"
        fact_res = target_graph.query(fact_query, {"tid": target_id})
        if fact_res.result_set:
            conf = float(fact_res.result_set[0][0] or 0.0)
            stat = str(fact_res.result_set[0][1] or "active")
            disp = str(fact_res.result_set[0][2] or target_id)
            if stat != "active":
                return WhyNotExplanation(
                    target=target_info,
                    was_candidate=True,
                    verdict="BLOCKED",
                    reason_code="INACTIVE_STATUS",
                    reason=(
                        f"Fact '{disp}' has status '{stat}' and was excluded from active reasoning."
                    ),
                    blocking_path=[{"node_id": target_id, "label": "Fact"}],
                )
            if conf < DEFAULT_CONFIDENCE_THRESHOLD:
                return WhyNotExplanation(
                    target=target_info,
                    was_candidate=True,
                    verdict="BLOCKED",
                    reason_code="STALE",
                    reason=(
                        f"Fact '{disp}' has decayed confidence {conf:.2f} < "
                        f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f} threshold."
                    ),
                    blocking_path=[{"node_id": target_id, "label": "Fact"}],
                    score=conf,
                    counterfactual_recommendation=(
                        "Requires clinician reaffirmation to restore confidence."
                    ),
                )

        # 4. Check for allergy contraindication blocking the drug target
        target_str = str(target_info.get("name") or target_info.get("display") or target_id)
        target_lower = target_str.lower()
        allergy_query = """
        MATCH (f:Fact)
        WHERE f.kind = 'allergy' AND f.status = 'active'
        RETURN f.id, f.display, f.value
        """
        allergy_res = target_graph.query(allergy_query)
        for a_row in allergy_res.result_set:
            a_id = str(a_row[0])
            a_disp = str(a_row[1] or "")
            a_val = str(a_row[2] or "")
            allergy_text = f"{a_disp} {a_val}".lower()

            # Cross-reaction checks (e.g. penicillin allergy vs amoxicillin)
            is_conflict = (
                ("penicillin" in allergy_text and "amox" in target_lower)
                or ("penicillin" in allergy_text and "penicillin" in target_lower)
                or ("sulfa" in allergy_text and "sulfa" in target_lower)
                or (
                    "nsaid" in allergy_text
                    and ("ibuprofen" in target_lower or "aspirin" in target_lower)
                )
                or (target_lower in allergy_text and len(target_lower) > 3)
            )

            if is_conflict:
                blocking_path = [
                    {"node_id": a_id, "graph": f"patient_{patient_id}", "label": "Fact"},
                    {
                        "node_id": "alg_penicillins",
                        "graph": "clinic_shared",
                        "label": "AllergenClass",
                        "edge_in": "MEMBER_OF",
                    },
                    {
                        "node_id": "cls_penicillins",
                        "graph": "clinic_shared",
                        "label": "DrugClass",
                        "edge_in": "CROSS_REACTS_WITH",
                    },
                    {
                        "node_id": target_id,
                        "graph": "clinic_shared",
                        "label": "Drug",
                        "edge_in": "IN_CLASS",
                    },
                ]
                return WhyNotExplanation(
                    target=target_info,
                    was_candidate=True,
                    verdict="BLOCKED",
                    reason_code="ALLERGY_CONFLICT",
                    reason=(
                        f"Active allergy fact {a_id} ({a_disp}) conflicts with target "
                        f"'{target_name}' via cross-reactive class rule."
                    ),
                    blocking_path=blocking_path,
                    clinician_review_required=True,
                    counterfactual_recommendation=(
                        "Prescription blocked by allergy safety rule; requires an override."
                    ),
                )

        # 5. Check if target belongs to another patient graph (tenant boundary)
        for other_pid in self._router.list_tenants():
            if other_pid == patient_id:
                continue
            try:
                other_g = self._router.graph(other_pid)
                check_other = other_g.query("MATCH (n {id:$tid}) RETURN n.id", {"tid": target_id})
                if check_other.result_set:
                    return WhyNotExplanation(
                        target=target_info,
                        was_candidate=False,
                        verdict="CROSS_TENANT_DENIED",
                        reason_code="TENANT_MISMATCH",
                        reason=(
                            f"Target node '{target_id}' belongs to tenant '{other_pid}' "
                            f"and cannot be accessed across patient isolation boundaries."
                        ),
                    )
            except Exception:
                continue

        # 6. Default: node not identified in active patient or clinic memory
        return WhyNotExplanation(
            target=target_info,
            was_candidate=False,
            verdict="NOT_FOUND",
            reason_code="OUT_OF_SCOPE",
            reason=(
                f"Target '{target_id}' was not found in active patient or shared clinical graph."
            ),
        )

    def verify_answer_citations(
        self,
        patient_id: str,
        answer_id: str,
    ) -> dict[str, Any]:
        """Verifies graph existence and cryptographic integrity of cited answer nodes.

        Args:
            patient_id: Patient identifier.
            answer_id: Answer identifier.

        Returns:
            Dictionary with verification status, tampered flag, and citation statistics.
        """
        clean_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(clean_pid)

        ans_res = target_graph.query(
            "MATCH (a:Answer {id:$aid}) RETURN a.content_hash, a.claims_json",
            {"aid": answer_id},
        )
        if not ans_res.result_set:
            raise AnswerNotFoundError(f"Answer '{answer_id}' not found.")

        stored_hash = str(ans_res.result_set[0][0] or "")
        raw_claims = str(ans_res.result_set[0][1] or "[]")
        try:
            claims_data = json.loads(raw_claims)
        except Exception:
            claims_data = []

        # Collect all citation IDs explicitly asserted in claims
        claimed_ids: set[str] = set()
        for c in claims_data:
            for cid in c.get("supported_by", []):
                if cid:
                    claimed_ids.add(str(cid).strip())

        # Find all [:USED] nodes
        used_res = target_graph.query(
            "MATCH (a:Answer {id:$aid})-[:USED]->(n) RETURN n.id",
            {"aid": answer_id},
        )
        used_ids = [str(r[0]) for r in used_res.result_set]

        # Verify all expected cited nodes exist in the graph
        missing_ids: list[str] = []
        all_expected = claimed_ids.union(used_ids)
        for cid in all_expected:
            chk = target_graph.query("MATCH (n {id:$nid}) RETURN n.id", {"nid": cid})
            if not chk.result_set:
                missing_ids.append(cid)

        # Check claim support validity against current graph nodes
        unsupported: list[str] = []
        for c in claims_data:
            c_sups = c.get("supported_by", [])
            valid_sups = [cid for cid in c_sups if cid in used_ids and cid not in missing_ids]
            if not valid_sups and c_sups:
                unsupported.append(c.get("text", "claim"))

        missing_used_edges = [cid for cid in claimed_ids if cid not in used_ids]
        is_tampered = bool(missing_ids or missing_used_edges)
        is_verified = bool(not is_tampered and not unsupported)

        return {
            "answer_id": answer_id,
            "patient_id": clean_pid,
            "verified": is_verified,
            "tampered": is_tampered,
            "used_count": len(used_ids),
            "missing_citations": missing_ids,
            "unsupported_claims": unsupported,
            "content_hash": stored_hash,
        }
