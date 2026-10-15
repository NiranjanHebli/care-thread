"""Unit tests for FactManager, bitemporal facts, CAS supersession, and as-of queries.

Validates:
- Fact creation with authority and base confidence scoring
- Reinforcement compounding on re-assertion of identical normalized values
- Authority conflict hierarchy (higher authority wins, lower authority archived)
- Equal authority tie-breaking and conflicting_assertion flags
- Safety asymmetry: allergy negation at equal/lower authority routes to pending_review
- Atomic compare-and-swap (CAS) supersession and CONFLICT_STALE_VERSION error handling
- Lineage traversal via query q_sup_02
- Point-in-time bitemporal queries via query q_recall_02
"""

from datetime import datetime, timezone

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    ConflictStaleVersionError,
    FactNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.facts import FactManager, compute_slot

TEST_PID = "p_777788889999"


@pytest.fixture
def fact_fixture():
    """Sets up a clean bootstrapped patient graph for fact tests."""
    client = get_db_client()
    router = TenantRouter(client)
    router.register_patient(TEST_PID, bootstrap_schema=True)

    target_graph = router.graph(TEST_PID)
    target_graph.query(
        "MERGE (p:Patient {id:$pid}) "
        "ON CREATE SET p.name = 'Fact Test Patient', p.created_at = 123456",
        {"pid": TEST_PID},
    )

    clock = FixedClock(datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc))
    manager = FactManager(router=router, clock=clock)

    yield manager, router, clock

    # Teardown
    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    router.deregister_patient(TEST_PID)


def test_slot_computation() -> None:
    assert compute_slot("medication", display="Metformin 500 MG Oral Tablet") == "med:metformin"
    assert compute_slot("medication", display="Lisinopril 10 MG") == "med:lisinopril"
    assert compute_slot("allergy", code="300916003", display="Penicillin") == "allergy:300916003"
    assert (
        compute_slot("condition", code="44054006", display="Diabetes mellitus type 2")
        == "cond:44054006"
    )
    assert compute_slot("preference", display="Morning Visits") == "pref:morning_visits"


def test_remember_fact_creation(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # 1. Patient reported medication
    res_med = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Metformin hydrochloride 1000 mg",
        value="1000 mg daily",
        caller_role="intake",
    )
    assert res_med["action"] == "active" or res_med["action"] == "created"
    assert res_med["status"] == "active"

    fact_med = manager.get_fact(TEST_PID, res_med["fact_id"])
    assert fact_med is not None
    assert fact_med.slot == "med:metformin"
    assert fact_med.authority == "patient_reported"
    assert fact_med.confidence == 0.7
    assert fact_med.version == 1
    assert fact_med.evidence_count == 1
    assert fact_med.recorded_at == clock.now_ts()

    # 2. Patient preference
    res_pref = manager.remember_fact(
        patient_id=TEST_PID,
        kind="preference",
        display="morning visits preferred",
        value="morning",
        caller_role="intake",
    )
    assert res_pref["status"] == "active"
    fact_pref = manager.get_fact(TEST_PID, res_pref["fact_id"])
    assert fact_pref is not None
    assert fact_pref.slot == "pref:morning_visits_preferred"


def test_remember_fact_reinforcement(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Initial assertion
    res1 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Aspirin 81 mg",
        value="81 mg daily",
        caller_role="intake",
    )
    fid = res1["fact_id"]
    initial_fact = manager.get_fact(TEST_PID, fid)
    assert initial_fact is not None
    assert initial_fact.confidence == 0.7
    assert initial_fact.evidence_count == 1

    # Advance clock by 3 days
    clock.advance(3 * 86400)
    now_reassert = clock.now_ts()

    # Re-assert exact same value
    res2 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Aspirin 81 mg",
        value="81 mg daily",
        caller_role="intake",
    )
    assert res2["action"] == "reinforced"
    assert res2["fact_id"] == fid

    reinforced = manager.get_fact(TEST_PID, fid)
    assert reinforced is not None
    assert reinforced.evidence_count == 2
    # Formula: min(1.0, 0.7 + 0.1 * (1.0 - 0.7)) == 0.73
    assert reinforced.confidence == 0.73
    assert reinforced.last_confirmed_at == now_reassert


def test_authority_supersession_hierarchy(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # 1. Intake agent (patient_reported: rank 2) asserts 500 mg
    res1 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Metformin 500 mg",
        value="500 mg daily",
        caller_role="intake",
    )
    f1_id = res1["fact_id"]

    clock.advance(3600)
    # 2. Pharmacy agent (pharmacy: rank 3) asserts 1000 mg
    res2 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Metformin 1000 mg",
        value="1000 mg daily",
        caller_role="pharmacy",
    )
    f2_id = res2["fact_id"]

    assert res2["action"] == "superseded"
    assert res2["status"] == "active"
    assert res2["superseded_id"] == f1_id

    # Check that old fact f1 is now superseded
    f1_updated = manager.get_fact(TEST_PID, f1_id)
    assert f1_updated is not None
    assert f1_updated.status == "superseded"
    assert f1_updated.superseded_at == clock.now_ts()

    # Check that new fact f2 is active with higher confidence
    f2_updated = manager.get_fact(TEST_PID, f2_id)
    assert f2_updated is not None
    assert f2_updated.status == "active"
    assert f2_updated.authority == "pharmacy"
    assert f2_updated.confidence == 0.9

    # Check SUPERSEDES edge in graph
    target_graph = router.graph(TEST_PID)
    edge_res = target_graph.query(
        "MATCH (f2:Fact {id:$f2})-[:SUPERSEDES]->(f1:Fact {id:$f1}) RETURN f2.id, f1.id",
        {"f2": f2_id, "f1": f1_id},
    )
    assert len(edge_res.result_set) == 1


def test_lower_authority_archived_as_superseded(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Clinician asserts active dose
    res_clinician = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Lisinopril 20 mg",
        value="20 mg oral",
        caller_role="clinician",
    )
    f_clinician_id = res_clinician["fact_id"]

    clock.advance(1800)
    # Patient reports conflicting lower dose
    res_patient = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Lisinopril 10 mg",
        value="10 mg oral",
        caller_role="intake",
    )
    f_patient_id = res_patient["fact_id"]

    assert res_patient["status"] == "superseded"

    # Clinician fact remains active
    f_clinician = manager.get_fact(TEST_PID, f_clinician_id)
    assert f_clinician is not None
    assert f_clinician.status == "active"

    # Patient fact is archived as superseded
    f_patient = manager.get_fact(TEST_PID, f_patient_id)
    assert f_patient is not None
    assert f_patient.status == "superseded"


def test_safety_asymmetry_allergy_removal(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Existing confirmed allergy
    res_allergy = manager.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Penicillin allergy",
        value="anaphylaxis",
        authority="record",
    )
    f_allergy_id = res_allergy["fact_id"]

    clock.advance(86400)
    # Intake agent reports patient denies allergy ("no")
    res_denial = manager.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Penicillin allergy",
        value="no",
        caller_role="intake",
    )
    assert res_denial["action"] == "pending_review"
    assert res_denial["status"] == "pending_review"
    assert "flag_id" in res_denial

    # Original allergy must remain active
    orig = manager.get_fact(TEST_PID, f_allergy_id)
    assert orig is not None
    assert orig.status == "active"


def test_equal_authority_conflict_flagging(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    t_same = clock.now_ts()
    # First assertion
    res1 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Atorvastatin 20 mg",
        value="20 mg bedtime",
        caller_role="pharmacy",
        valid_from=t_same,
    )
    f1_id = res1["fact_id"]

    # Second assertion at identical authority and identical valid_from
    res2 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Atorvastatin 40 mg",
        value="40 mg bedtime",
        caller_role="pharmacy",
        valid_from=t_same,
    )
    assert res2["action"] == "disputed"
    assert res2["status"] == "disputed"

    # Both facts should now be disputed
    f1 = manager.get_fact(TEST_PID, f1_id)
    f2 = manager.get_fact(TEST_PID, res2["fact_id"])
    assert f1 is not None and f1.status == "disputed"
    assert f2 is not None and f2.status == "disputed"


def test_atomic_cas_supersession_and_stale_version_error(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Create initial fact
    res = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Warfarin 2 mg",
        value="2 mg daily",
        caller_role="clinician",
    )
    f_id = res["fact_id"]
    fact = manager.get_fact(TEST_PID, f_id)
    assert fact is not None
    assert fact.version == 1

    # Atomic CAS supersede with matching expected_version=1
    clock.advance(3600)
    cas_res = manager.supersede_fact(
        patient_id=TEST_PID,
        old_fact_id=f_id,
        expected_version=1,
        display="Warfarin 5 mg",
        value="5 mg daily",
        authority="clinician",
        reason="INR below target",
    )
    assert cas_res["action"] == "superseded"
    new_fid = cas_res["new_id"]

    # Old fact version incremented, status superseded
    old_fact = manager.get_fact(TEST_PID, f_id)
    assert old_fact is not None
    assert old_fact.status == "superseded"
    assert old_fact.version == 2

    # New fact active at version 1
    new_fact = manager.get_fact(TEST_PID, new_fid)
    assert new_fact is not None
    assert new_fact.status == "active"
    assert new_fact.version == 1

    # Attempting CAS on old fact again fails with ConflictStaleVersionError
    with pytest.raises(ConflictStaleVersionError) as exc_info:
        manager.supersede_fact(
            patient_id=TEST_PID,
            old_fact_id=f_id,
            expected_version=1,
            display="Warfarin 7.5 mg",
            value="7.5 mg daily",
        )
    assert exc_info.value.code == "CONFLICT_STALE_VERSION"


def test_fact_history_lineage(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Version 1
    t1 = clock.now_ts()
    res1 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Metoprolol 25 mg",
        value="25 mg daily",
        caller_role="clinician",
        valid_from=t1,
    )
    f1 = res1["fact_id"]

    # Version 2
    clock.advance(7 * 86400)
    t2 = clock.now_ts()
    res2 = manager.supersede_fact(
        patient_id=TEST_PID,
        old_fact_id=f1,
        expected_version=1,
        display="Metoprolol 50 mg",
        value="50 mg daily",
        valid_from=t2,
        reason="Dose escalation",
    )
    f2 = res2["new_id"]

    # Version 3
    clock.advance(7 * 86400)
    t3 = clock.now_ts()
    res3 = manager.supersede_fact(
        patient_id=TEST_PID,
        old_fact_id=f2,
        expected_version=1,
        display="Metoprolol 100 mg",
        value="100 mg daily",
        valid_from=t3,
        reason="Target heart rate",
    )
    f3 = res3["new_id"]

    # History from current head (f3)
    history = manager.get_fact_history(TEST_PID, f3)
    assert len(history) == 3
    assert [h["id"] for h in history] == [f3, f2, f1]
    assert history[0]["value"] == "100 mg daily"
    assert history[1]["value"] == "50 mg daily"
    assert history[2]["value"] == "25 mg daily"


def test_bitemporal_as_of_queries(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    t0 = clock.now_ts()
    # At t0, patient takes Lisinopril 10 mg
    res1 = manager.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        display="Lisinopril 10 mg",
        value="10 mg daily",
        caller_role="clinician",
        valid_from=t0,
    )
    f1 = res1["fact_id"]

    # Advance 14 days
    clock.advance(14 * 86400)
    t1 = clock.now_ts()
    # At t1, medication is increased to Lisinopril 20 mg
    res2 = manager.supersede_fact(
        patient_id=TEST_PID,
        old_fact_id=f1,
        expected_version=1,
        display="Lisinopril 20 mg",
        value="20 mg daily",
        valid_from=t1,
    )
    f2 = res2["new_id"]

    # Advance another 7 days to t2
    clock.advance(7 * 86400)
    t2 = clock.now_ts()

    # Query as of t0 + 1 day: system believed Lisinopril 10 mg
    as_of_t0 = manager.get_facts_as_of(TEST_PID, as_of_ts=t0 + 86400, kind="medication")
    assert len(as_of_t0) == 1
    assert as_of_t0[0].id == f1
    assert as_of_t0[0].value == "10 mg daily"

    # Query as of t2: system believes Lisinopril 20 mg
    as_of_t2 = manager.get_facts_as_of(TEST_PID, as_of_ts=t2, kind="medication")
    assert len(as_of_t2) == 1
    assert as_of_t2[0].id == f2
    assert as_of_t2[0].value == "20 mg daily"


def test_specialist_condition_pending_review_and_clinician_confirm(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    # Specialist writes a condition fact -> must start as pending_review
    res = manager.remember_fact(
        patient_id=TEST_PID,
        kind="condition",
        code="44054006",
        display="Diabetes mellitus type 2",
        value="active",
        caller_role="specialist",
    )
    assert res["action"] == "pending_review"
    assert res["status"] == "pending_review"
    fid = res["fact_id"]

    fact = manager.get_fact(TEST_PID, fid)
    assert fact is not None
    assert fact.status == "pending_review"

    # Clinician confirms the fact
    clock.advance(3600)
    confirmed = manager.confirm_fact(TEST_PID, fid, clinician_id="dr_smith", decision="confirm")
    assert confirmed.status == "active"
    assert confirmed.authority == "clinician"
    assert confirmed.confidence == 1.0


def test_fact_not_found_and_validation_errors(fact_fixture) -> None:
    manager, router, clock = fact_fixture

    with pytest.raises(PatientNotFoundError):
        manager.remember_fact(patient_id="p_000000000000", kind="medication", display="Aspirin")

    with pytest.raises(ValidationError):
        manager.remember_fact(patient_id=TEST_PID, kind="unsupported_kind", display="Aspirin")

    assert manager.get_fact(patient_id=TEST_PID, fact_id="fct_nonexistent0000000000") is None

    with pytest.raises(FactNotFoundError):
        manager.reinforce_fact(patient_id=TEST_PID, fact_id="fct_nonexistent0000000000")
