"""Unit tests for the graph-native provenance engine, traces, and Why-Not explainability.

Validates:
- commit_answer persisting Answer nodes with [:USED] and [:REJECTED] graph edges
- Claim grounding enforcement and allow_unsupported override
- Retrieval TTL expiration handling
- Trace reconstruction directly from FalkorDB queries (q_exp_01a, q_exp_01b, q_exp_01c)
- Reasoning subgraph extraction connecting cited clinical facts and notes
- Counterfactual Why-Not analysis:
  - Excluded recall candidates
  - Superseded historical facts
  - Decayed confidence facts
  - Allergy contraindications with 4-hop blocking paths
  - Cross-tenant boundary isolation
  - Non-existent target handling
- Ablation mode sensitivity and scripted abstention
- Citation verification and cryptographic tamper detection
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    AnswerNotFoundError,
    ClaimUnsupportedError,
    RetrievalExpiredError,
    TenantMismatchError,
)
from carethread.provenance.engine import ProvenanceEngine
from carethread.util.id import (
    fact_id as gen_fact_id,
)
from carethread.util.id import (
    note_id as gen_note_id,
)
from carethread.util.id import (
    session_id as gen_session_id,
)

TEST_PID_PROV = "p_d17000000001"
OTHER_PID_PROV = "p_d17000000002"


@pytest.fixture
def prov_fixture() -> dict[str, Any]:
    """Sets up isolated patient graphs and a ProvenanceEngine instance."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (TEST_PID_PROV, OTHER_PID_PROV):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g1 = router.graph(TEST_PID_PROV)
    g1.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Provenance Test Patient'",
        {"pid": TEST_PID_PROV},
    )

    g2 = router.graph(OTHER_PID_PROV)
    g2.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Other Patient'",
        {"pid": OTHER_PID_PROV},
    )

    clock = FixedClock(1792170000)
    clinic_reader = ClinicReader(client)
    engine = ProvenanceEngine(router=router, clinic_reader=clinic_reader, clock=clock)

    return {
        "client": client,
        "router": router,
        "clinic_reader": clinic_reader,
        "clock": clock,
        "engine": engine,
        "g1": g1,
        "g2": g2,
    }


def test_commit_answer_with_graph_citations(prov_fixture: dict[str, Any]) -> None:
    """Verifies commit_answer creates Answer, HAS_ANSWER, IN_SESSION, USED, and REJECTED edges."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    sid = gen_session_id()
    fid1 = gen_fact_id()
    fid2 = gen_fact_id()
    rej_id = gen_fact_id()

    # Create Session, Facts, and Rejected candidate in patient graph
    g1.query("CREATE (:Session {id:$sid})", {"sid": sid})
    g1.query(
        """
        CREATE (:Fact {id:$f1, display:'Hypertension', status:'active', confidence:0.95})
        CREATE (:Fact {id:$f2, display:'Lisinopril 10mg', status:'active', confidence:0.90})
        CREATE (:Fact {id:$rej, display:'Historical Lisinopril 5mg', status:'superseded'})
        """,
        {"f1": fid1, "f2": fid2, "rej": rej_id},
    )

    retrieval = {
        "created_at": prov_fixture["clock"].now_ts(),
        "items": [
            {
                "id": fid1,
                "label": "Fact",
                "role": "supporting",
                "rank": 1,
                "score": 0.95,
                "method": "graph_traversal",
                "display": "Hypertension",
            },
            {
                "id": fid2,
                "label": "Fact",
                "role": "treatment",
                "rank": 2,
                "score": 0.90,
                "method": "graph_traversal",
                "display": "Lisinopril 10mg",
            },
        ],
        "rejected": [
            {
                "id": rej_id,
                "label": "Fact",
                "reason_code": "SUPERSEDED",
                "reason": "Replaced by Lisinopril 10mg",
                "score": 0.60,
                "superseded_by": fid2,
            }
        ],
    }

    claims = [
        {"text": "Patient has hypertension", "supported_by": [fid1]},
        {"text": "Patient is on Lisinopril 10mg", "supported_by": [fid2]},
    ]

    trace = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        session_id=sid,
        retrieval=retrieval,
        question="What is the current treatment for hypertension?",
        answer_text="Patient is managed for hypertension on Lisinopril 10mg daily.",
        claims=claims,
        author_agent_id="agt_specialist_01",
    )

    assert trace.answer_id.startswith("ans_")
    assert trace.trace_id.startswith("trc_")
    assert trace.patient_id == TEST_PID_PROV
    assert trace.session_id == sid
    assert trace.verified is True
    assert len(trace.content_hash) == 64
    assert len(trace.used) == 2
    assert len(trace.rejected) == 1

    # Verify graph edges in FalkorDB
    has_ans = g1.query(
        "MATCH (:Patient {id:$pid})-[:HAS_ANSWER]->(a:Answer {id:$aid}) RETURN a.id",
        {"pid": TEST_PID_PROV, "aid": trace.answer_id},
    )
    assert len(has_ans.result_set) == 1

    in_ses = g1.query(
        "MATCH (a:Answer {id:$aid})-[:IN_SESSION]->(:Session {id:$sid}) RETURN a.id",
        {"aid": trace.answer_id, "sid": sid},
    )
    assert len(in_ses.result_set) == 1

    used_edges = g1.query(
        "MATCH (a:Answer {id:$aid})-[u:USED]->(n) RETURN n.id, u.role, u.rank",
        {"aid": trace.answer_id},
    )
    assert len(used_edges.result_set) == 2

    rej_edges = g1.query(
        "MATCH (a:Answer {id:$aid})-[r:REJECTED]->(n) RETURN n.id, r.reason_code",
        {"aid": trace.answer_id},
    )
    assert len(rej_edges.result_set) == 1
    assert rej_edges.result_set[0][1] == "SUPERSEDED"


def test_commit_answer_claim_grounding_validation(prov_fixture: dict[str, Any]) -> None:
    """Verifies ungrounded claims trigger ClaimUnsupportedError unless allow_unsupported=True."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    fid = gen_fact_id()
    now_ts = prov_fixture["clock"].now_ts()

    retrieval = {
        "created_at": now_ts,
        "items": [{"id": fid, "label": "Fact", "score": 0.9}],
        "rejected": [],
    }

    # Claim citing unknown ID
    bad_claims = [{"text": "Ungrounded statement", "supported_by": ["fct_nonexistent_999"]}]

    with pytest.raises(ClaimUnsupportedError) as exc_info:
        engine.commit_answer(
            patient_id=TEST_PID_PROV,
            retrieval=retrieval,
            question="Question?",
            answer_text="Answer with bad citation.",
            claims=bad_claims,
            allow_unsupported=False,
        )
    assert "unsupported_claims" in exc_info.value.details

    # When allow_unsupported=True, should succeed with answer_type='uncited'
    trace_uncited = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        retrieval=retrieval,
        question="Question?",
        answer_text="Answer allowing uncited claims.",
        claims=bad_claims,
        allow_unsupported=True,
    )
    assert trace_uncited.answer_type == "uncited"
    assert len(trace_uncited.unsupported_claims) == 1


def test_commit_answer_expired_retrieval(prov_fixture: dict[str, Any]) -> None:
    """Verifies that expired retrieval contexts (> 1 hour) are rejected."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    now_ts = prov_fixture["clock"].now_ts()

    expired_retrieval = {
        "created_at": now_ts - 3601,  # 1 hour and 1 second ago
        "items": [],
        "rejected": [],
    }

    with pytest.raises(RetrievalExpiredError):
        engine.commit_answer(
            patient_id=TEST_PID_PROV,
            retrieval=expired_retrieval,
            question="Question?",
            answer_text="Answer with expired context.",
        )


def test_explain_trace_reconstructed_from_graph(prov_fixture: dict[str, Any]) -> None:
    """Verifies explain executes q_exp_01a, q_exp_01b, q_exp_01c to rebuild trace from graph."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    fid = gen_fact_id()
    nid = gen_note_id()
    rej_id = gen_fact_id()

    # Create Fact and Note connected via ABOUT relationship
    g1.query(
        """
        CREATE (f:Fact {
            id: $fid, display: 'Type 2 Diabetes', status: 'active',
            confidence: 0.92, valid_from: 1700000000, last_confirmed_at: 1790000000
        })
        CREATE (n:Note {
            id: $nid, text: 'HbA1c elevated at 7.8% on recent screening.', status: 'active'
        })
        CREATE (n)-[:ABOUT]->(f)
        """,
        {"fid": fid, "nid": nid},
    )

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [
            {
                "id": fid,
                "label": "Fact",
                "role": "condition",
                "rank": 1,
                "score": 0.92,
                "display": "Type 2 Diabetes",
            },
            {
                "id": nid,
                "label": "Note",
                "role": "evidence",
                "rank": 2,
                "score": 0.88,
                "text": "HbA1c elevated at 7.8%",
            },
        ],
        "rejected": [
            {
                "id": rej_id,
                "label": "Fact",
                "reason_code": "STALE",
                "reason": "Outdated reading from 3 years ago",
                "score": 0.40,
            }
        ],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="What is the glycemic status?",
        answer_text="Patient has Type 2 Diabetes with recent HbA1c 7.8%.",
        claims=[
            {"text": "Diabetes diagnosis active", "supported_by": [fid]},
            {"text": "HbA1c 7.8%", "supported_by": [nid]},
        ],
        retrieval=retrieval,
    )
    ans_id = commit_res.answer_id

    # Reconstruct trace solely via explain_answer from FalkorDB graph
    exp = engine.explain_answer(patient_id=TEST_PID_PROV, answer_id=ans_id)

    assert exp.trace.answer_id == ans_id
    assert exp.trace.verified is True
    assert len(exp.trace.used) == 2

    # Verify used nodes retrieved via q_exp_01a
    used_ids = {u["node_id"] for u in exp.trace.used}
    assert fid in used_ids
    assert nid in used_ids

    # Verify intra-subgraph reasoning edge retrieved via q_exp_01b
    assert len(exp.trace.edges) >= 1
    edge = exp.trace.edges[0]
    assert edge["type"] == "ABOUT"
    assert edge["from"] == nid
    assert edge["to"] == fid

    # Verify rejected candidate retrieved via q_exp_01c
    assert len(exp.trace.rejected) == 1
    assert exp.trace.rejected[0]["node_id"] == rej_id
    assert exp.trace.rejected[0]["reason_code"] == "STALE"


def test_explain_why_not_superseded_fact(prov_fixture: dict[str, Any]) -> None:
    """Verifies Why-Not analysis identifies superseded facts and names newer replacement."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    old_fid = gen_fact_id()
    new_fid = gen_fact_id()

    # Create supersession relationship between facts
    g1.query(
        """
        CREATE (old:Fact {
            id: $old_id, display: 'Metformin 500mg', status: 'superseded', confidence: 0.8
        })
        CREATE (new:Fact {
            id: $new_id, display: 'Metformin 1000mg', status: 'active', confidence: 0.95
        })
        CREATE (new)-[:SUPERSEDES]->(old)
        """,
        {"old_id": old_fid, "new_id": new_fid},
    )

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": new_fid, "label": "Fact", "display": "Metformin 1000mg"}],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="What is the current Metformin dose?",
        answer_text="Current dose is Metformin 1000mg daily.",
        claims=[{"text": "Metformin 1000mg", "supported_by": [new_fid]}],
        retrieval=retrieval,
    )

    # Ask Why-Not for the old dose
    exp = engine.explain_answer(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
        why_not=old_fid,
    )

    assert exp.why_not is not None
    assert exp.why_not.verdict == "BLOCKED"
    assert exp.why_not.reason_code == "SUPERSEDED"
    assert exp.why_not.superseded_by == new_fid
    assert len(exp.why_not.blocking_path) == 2


def test_explain_why_not_decayed_confidence(prov_fixture: dict[str, Any]) -> None:
    """Verifies Why-Not analysis detects facts with confidence decayed below cutoff."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    active_fid = gen_fact_id()
    decayed_fid = gen_fact_id()

    g1.query(
        """
        CREATE (:Fact {
            id: $act_id, display: 'Active Diagnosis', confidence: 0.9, status: 'active'
        })
        CREATE (:Fact {
            id: $dec_id, display: 'Decayed Assertion', confidence: 0.15, status: 'active'
        })
        """,
        {"act_id": active_fid, "dec_id": decayed_fid},
    )

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": active_fid, "label": "Fact"}],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="Clinical status?",
        answer_text="Active diagnosis confirmed.",
        claims=[{"text": "Confirmed", "supported_by": [active_fid]}],
        retrieval=retrieval,
    )

    exp = engine.explain_answer(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
        why_not=decayed_fid,
    )

    assert exp.why_not is not None
    assert exp.why_not.verdict == "BLOCKED"
    assert exp.why_not.reason_code == "STALE"
    assert "decayed confidence" in exp.why_not.reason


def test_explain_why_not_allergy_contraindication(prov_fixture: dict[str, Any]) -> None:
    """Verifies Why-Not returns ALLERGY_CONFLICT with multi-hop blocking path."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    allergy_fid = gen_fact_id()
    g1.query(
        """
        CREATE (:Fact {
            id: $aid, kind: 'allergy', display: 'Penicillin allergy',
            value: 'penicillin', status: 'active', confidence: 1.0
        })
        """,
        {"aid": allergy_fid},
    )

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [
            {"id": allergy_fid, "label": "Fact", "role": "safety", "display": "Penicillin allergy"}
        ],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="Can we prescribe antibiotics?",
        answer_text="Patient has confirmed penicillin allergy.",
        claims=[{"text": "Allergic to penicillin", "supported_by": [allergy_fid]}],
        retrieval=retrieval,
    )

    # Query Why-Not for Amoxicillin
    exp = engine.explain_answer(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
        why_not={"label": "Drug", "code": "308192", "name": "Amoxicillin"},
    )

    assert exp.why_not is not None
    assert exp.why_not.verdict == "BLOCKED"
    assert exp.why_not.reason_code == "ALLERGY_CONFLICT"
    assert exp.why_not.clinician_review_required is True
    assert len(exp.why_not.blocking_path) == 4
    assert exp.why_not.blocking_path[0]["node_id"] == allergy_fid


def test_explain_why_not_cross_tenant_isolation(prov_fixture: dict[str, Any]) -> None:
    """Verifies Why-Not denies cross-tenant access and reports TENANT_MISMATCH."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    g2 = prov_fixture["g2"]
    clock = prov_fixture["clock"]

    p1_fid = gen_fact_id()
    p2_fid = gen_fact_id()

    g1.query("CREATE (:Fact {id: $fid, display: 'Patient 1 Fact'})", {"fid": p1_fid})
    g2.query("CREATE (:Fact {id: $fid, display: 'Patient 2 Fact'})", {"fid": p2_fid})

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": p1_fid, "label": "Fact"}],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="Summary?",
        answer_text="Patient 1 summary.",
        claims=[{"text": "Fact", "supported_by": [p1_fid]}],
        retrieval=retrieval,
    )

    # Query Why-Not for entity in patient 2's graph
    exp = engine.explain_answer(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
        why_not=p2_fid,
    )

    assert exp.why_not is not None
    assert exp.why_not.verdict == "CROSS_TENANT_DENIED"
    assert exp.why_not.reason_code == "TENANT_MISMATCH"


def test_explain_ablation_mode(prov_fixture: dict[str, Any]) -> None:
    """Verifies explain with mode='ablate' simulates citation removal and abstention."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    fid = gen_fact_id()
    g1.query("CREATE (:Fact {id: $fid, display: 'Essential Fact'})", {"fid": fid})

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": fid, "label": "Fact"}],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="Can we recommend therapy?",
        answer_text="Recommend therapy based on essential fact.",
        claims=[{"text": "Therapy indication", "supported_by": [fid]}],
        retrieval=retrieval,
    )

    exp = engine.explain_answer(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
        mode="ablate",
    )

    assert exp.ablated is not None
    assert exp.ablated["ablated"] is True
    assert exp.ablated["abstention"] is True
    assert exp.ablated["sensitivity"] == 1.0
    assert fid in exp.ablated["ablated_citations"]


def test_verify_answer_citations_and_tamper_detection(prov_fixture: dict[str, Any]) -> None:
    """Verifies verify_answer_citations validates citations and detects missing nodes."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g1 = prov_fixture["g1"]
    clock = prov_fixture["clock"]

    fid = gen_fact_id()
    g1.query("CREATE (:Fact {id: $fid, display: 'Verified Fact'})", {"fid": fid})

    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": fid, "label": "Fact"}],
        "rejected": [],
    }

    commit_res = engine.commit_answer(
        patient_id=TEST_PID_PROV,
        question="Status?",
        answer_text="Verified response.",
        claims=[{"text": "Grounding", "supported_by": [fid]}],
        retrieval=retrieval,
    )

    # Initial check: verified and untampered
    check = engine.verify_answer_citations(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
    )
    assert check["verified"] is True
    assert check["tampered"] is False
    assert check["used_count"] == 1

    # Tamper test: delete the cited fact from the graph
    g1.query("MATCH (f:Fact {id: $fid}) DETACH DELETE f", {"fid": fid})

    check_tampered = engine.verify_answer_citations(
        patient_id=TEST_PID_PROV,
        answer_id=commit_res.answer_id,
    )
    assert check_tampered["verified"] is False
    assert check_tampered["tampered"] is True
    assert fid in check_tampered["missing_citations"]


def test_explain_errors_nonexistent_and_tenant_mismatch(prov_fixture: dict[str, Any]) -> None:
    """Verifies AnswerNotFoundError and TenantMismatchError during explain."""
    engine: ProvenanceEngine = prov_fixture["engine"]
    g2 = prov_fixture["g2"]
    clock = prov_fixture["clock"]

    with pytest.raises(AnswerNotFoundError):
        engine.explain_answer(patient_id=TEST_PID_PROV, answer_id="ans_nonexistent_01")

    # Commit an answer in Patient 2
    fid = gen_fact_id()
    g2.query("CREATE (:Fact {id: $fid, display: 'P2 Fact'})", {"fid": fid})
    retrieval = {
        "created_at": clock.now_ts(),
        "items": [{"id": fid, "label": "Fact"}],
        "rejected": [],
    }
    commit_p2 = engine.commit_answer(
        patient_id=OTHER_PID_PROV,
        question="P2 Question?",
        answer_text="P2 Answer.",
        retrieval=retrieval,
    )

    # Attempting to explain Patient 2's answer under Patient 1 raises TenantMismatchError
    with pytest.raises(TenantMismatchError):
        engine.explain_answer(patient_id=TEST_PID_PROV, answer_id=commit_p2.answer_id)
