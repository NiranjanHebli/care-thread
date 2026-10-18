"""Unit tests for the memory consolidation background worker.

Validates:
- FactCandidate creation with slot derivation and episode link
- Multi-episode durability threshold promotion via query q_cons_01 (n_eps >= 2)
- Temporal confidence decay marking for active facts via query q_cons_02
- Stale open handoff expiration sweep
- ConsolidationRun record persistence and stats tracking
- Multi-tenant consolidation across registered patients
- Validation and error handling (ValidationError, PatientNotFoundError)
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.facts import FactManager
from carethread.memory.handoff import HandoffManager
from carethread.memory.worker import (
    ConsolidationRunRecord,
    ConsolidationWorker,
)

TEST_PID_1 = "p_101010101010"
TEST_PID_2 = "p_202020202020"


@pytest.fixture
def worker_fixture():
    """Sets up a clean test fixture with two tenants and a ConsolidationWorker."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (TEST_PID_1, TEST_PID_2):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g1 = router.graph(TEST_PID_1)
    g1.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Worker Patient 1'",
        {"pid": TEST_PID_1},
    )

    g2 = router.graph(TEST_PID_2)
    g2.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Worker Patient 2'",
        {"pid": TEST_PID_2},
    )

    clock = FixedClock(1792140000)
    worker = ConsolidationWorker(router=router, clock=clock, default_stale_threshold=0.25)
    fact_mgr = FactManager(router=router, clock=clock)
    handoff_mgr = HandoffManager(router=router, clock=clock)

    yield {
        "router": router,
        "clock": clock,
        "worker": worker,
        "fact_manager": fact_mgr,
        "handoff_manager": handoff_mgr,
        "graph1": g1,
        "graph2": g2,
    }

    for pid in (TEST_PID_1, TEST_PID_2):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass


def test_create_candidate(worker_fixture: dict[str, Any]):
    """Verify create_candidate creates FactCandidate node linked to Patient and Episode."""
    worker: ConsolidationWorker = worker_fixture["worker"]
    g = worker_fixture["graph1"]

    # Pre-create episode node
    g.query("MERGE (ep:Episode {id:'epi_cand_01'})")

    cand = worker.create_candidate(
        patient_id=TEST_PID_1,
        kind="condition",
        display="Type 2 Diabetes Mellitus",
        episode_id="epi_cand_01",
        code="44054006",
        value="HbA1c 7.4%",
    )

    assert cand.id.startswith("can_")
    assert cand.patient_id == TEST_PID_1
    assert cand.kind == "condition"
    assert cand.slot == "cond:44054006"
    assert cand.status == "pending"
    assert cand.episode_id == "epi_cand_01"


def test_candidate_promotion_at_two_episodes(worker_fixture: dict[str, Any]):
    """Verify candidates are promoted to durable Fact only when observed in >= 2 episodes."""
    worker: ConsolidationWorker = worker_fixture["worker"]
    fact_mgr: FactManager = worker_fixture["fact_manager"]
    g = worker_fixture["graph1"]

    g.query("MERGE (e1:Episode {id:'epi_visit_1'})")
    g.query("MERGE (e2:Episode {id:'epi_visit_2'})")

    # Observation 1 in visit 1
    worker.create_candidate(
        patient_id=TEST_PID_1,
        kind="condition",
        display="Essential hypertension",
        code="59621000",
        episode_id="epi_visit_1",
    )

    # 1st consolidation run: only 1 episode -> should NOT promote
    run1 = worker.run_consolidation(TEST_PID_1)
    assert isinstance(run1, ConsolidationRunRecord)
    assert run1.stats["promoted_candidates"] == 0

    # Verify fact does not exist in patient facts yet
    facts1 = fact_mgr.get_active_facts(TEST_PID_1)
    assert not any(f.slot == "cond:59621000" for f in facts1)

    # Observation 2 in visit 2 (different episode)
    worker.create_candidate(
        patient_id=TEST_PID_1,
        kind="condition",
        display="Essential hypertension",
        code="59621000",
        episode_id="epi_visit_2",
    )

    # 2nd consolidation run: 2 distinct episodes -> promotes to Fact via q_cons_01!
    run2 = worker.run_consolidation(TEST_PID_1)
    assert run2.stats["promoted_candidates"] == 1

    # Verify Fact now exists as durable semantic fact!
    promoted_fact = fact_mgr.get_active_fact_in_slot(TEST_PID_1, "cond:59621000")
    assert promoted_fact is not None
    assert promoted_fact.authority == "record"

    # Verify candidates are now marked promoted
    cand_check = g.query(
        "MATCH (c:FactCandidate {slot:'cond:59621000'}) RETURN c.status",
    )
    assert all(row[0] == "promoted" for row in cand_check.result_set)


def test_temporal_decay_marking(worker_fixture: dict[str, Any]):
    """Verify q_cons_02 marks active facts stale when confidence decays below threshold."""
    worker: ConsolidationWorker = worker_fixture["worker"]
    fact_mgr: FactManager = worker_fixture["fact_manager"]
    g = worker_fixture["graph1"]

    # Insert an active fact confirmed 100 days ago with 30-day half life
    # eff = 0.7 * (0.5 ** (100 / 30)) = 0.7 * (0.5 ** 3.33) ~ 0.069 < 0.25
    hundred_days_ago = 1792140000 - (100 * 86400)
    g.query(
        """
        MATCH (p:Patient {id:$pid})
        CREATE (f:Fact {
            id: 'fct_decay_test_01',
            kind: 'symptom',
            slot: 'symp:cough',
            display: 'Dry cough',
            value: 'mild',
            status: 'active',
            confidence: 0.7,
            half_life_days: 30,
            last_confirmed_at: $last_conf,
            stale: false,
            version: 1
        })
        CREATE (p)-[:HAS_FACT]->(f)
        """,
        {"pid": TEST_PID_1, "last_conf": hundred_days_ago},
    )

    # Before consolidation: not stale
    f_before = fact_mgr.get_fact(TEST_PID_1, "fct_decay_test_01")
    assert f_before is not None
    assert f_before.stale is False

    # Run consolidation with default stale_threshold=0.25
    run = worker.run_consolidation(TEST_PID_1)
    assert run.stats["decayed_facts"] >= 1

    # After consolidation: fact marked stale
    f_after = fact_mgr.get_fact(TEST_PID_1, "fct_decay_test_01")
    assert f_after is not None
    assert f_after.stale is True


def test_handoff_expiration_sweep(worker_fixture: dict[str, Any]):
    """Verify consolidation sweeps open handoffs past TTL."""
    worker: ConsolidationWorker = worker_fixture["worker"]
    handoff_mgr: HandoffManager = worker_fixture["handoff_manager"]
    clock: FixedClock = worker_fixture["clock"]

    # Create a handoff with 10-second TTL
    ho = handoff_mgr.create_handoff(
        patient_id=TEST_PID_1,
        from_role="intake",
        to_role="specialist",
        goal="Urgent review",
        ttl_seconds=10,
        publish=True,
    )
    assert ho.status == "open"

    # Advance clock past TTL
    clock.advance(100)

    # Run consolidation
    run = worker.run_consolidation(TEST_PID_1)
    assert run.stats["expired_handoffs"] >= 1

    # Verify handoff status updated to expired
    ho_after = handoff_mgr.get_handoff(TEST_PID_1, ho.id)
    assert ho_after.status == "expired"


def test_consolidation_run_record_and_history(worker_fixture: dict[str, Any]):
    """Verify ConsolidationRun nodes are logged with stats and queryable in history."""
    worker: ConsolidationWorker = worker_fixture["worker"]

    run = worker.run_consolidation(TEST_PID_1, trigger="scheduled")
    assert run.id.startswith("cns_")
    assert run.trigger == "scheduled"
    assert run.status == "completed"
    assert "latency_ms" in run.stats

    history = worker.get_consolidation_history(TEST_PID_1, limit=5)
    assert len(history) >= 1
    assert history[0].id == run.id
    assert history[0].trigger == "scheduled"


def test_multi_tenant_consolidation(worker_fixture: dict[str, Any]):
    """Verify running consolidation with patient_id=None iterates across all tenants."""
    worker: ConsolidationWorker = worker_fixture["worker"]

    runs = worker.run_consolidation(patient_id=None, trigger="scheduled")
    assert isinstance(runs, list)
    pids = [r.patient_id for r in runs]
    assert TEST_PID_1 in pids
    assert TEST_PID_2 in pids


def test_worker_validation_errors(worker_fixture: dict[str, Any]):
    """Verify validation errors for bad input."""
    worker: ConsolidationWorker = worker_fixture["worker"]

    # Invalid patient
    with pytest.raises(ValidationError):
        worker.create_candidate("bad_pid", "condition", "display", "epi_1")

    # Non-existent patient
    with pytest.raises(PatientNotFoundError):
        worker.create_candidate("p_999988887777", "condition", "display", "epi_1")

    # Empty display
    with pytest.raises(ValidationError):
        worker.create_candidate(TEST_PID_1, "condition", "", "epi_1")

    # Empty episode
    with pytest.raises(ValidationError):
        worker.create_candidate(TEST_PID_1, "condition", "display", "")
