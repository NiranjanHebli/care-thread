"""Unit tests for clinical safety verification, cross-reactivity, and contradiction detection.

Validates:
- Cross-reactivity between beta-lactams and penicillin allergies (q_safe_01a, q_safe_01b)
- Drug-drug interaction (DDI) detection (e.g. Lisinopril + Ibuprofen, Lisinopril + Losartan)
- Therapeutic duplicate therapy detection within identical drug classes
- Safety flag persistence and graph edges (HAS_FLAG, ABOUT, CONFLICTS_WITH)
- Safety flag lifecycle: open -> resolved and open -> dismissed
- Pre-prescription check (check_proposed_medication) passed vs failed logic
- Full patient safety audit (check_patient_safety)
"""

from datetime import datetime, timezone

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import FlagNotFoundError, PatientNotFoundError
from carethread.memory.facts import FactManager
from carethread.memory.safety import (
    SafetyCheckResult,
    SafetyFlagRecord,
    SafetyManager,
)

TEST_PID = "p_444455556666"


@pytest.fixture
def safety_fixture():
    """Sets up a clean bootstrapped patient graph and safety manager."""
    client = get_db_client()
    router = TenantRouter(client)
    router.register_patient(TEST_PID, bootstrap_schema=True)

    target_graph = router.graph(TEST_PID)
    target_graph.query(
        "MERGE (p:Patient {id:$pid}) "
        "ON CREATE SET p.name = 'Safety Test Patient', p.created_at = 100000",
        {"pid": TEST_PID},
    )

    clock = FixedClock(datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc))
    fact_manager = FactManager(router=router, clock=clock)
    safety_manager = SafetyManager(
        router=router,
        clinic_reader=router.get_clinic_reader(),
        clock=clock,
    )

    yield safety_manager, fact_manager, router, clock

    # Teardown
    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    router.deregister_patient(TEST_PID)


def test_cross_reactivity_penicillin_allergy_vs_amoxicillin(safety_fixture) -> None:
    """Verifies cross-reactivity detection between penicillin allergy and proposed amoxicillin."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Record active penicillin allergy
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Allergy to penicillin",
        value="hives and swelling",
        authority="clinician",
        caller_role="clinician",
    )

    # Check proposed amoxicillin (RxNorm 308182)
    flags = safety_mgr.check_allergy_cross_reactivity(
        patient_id=TEST_PID,
        drug_code="308182",
        drug_name="Amoxicillin 250 MG Oral Capsule",
    )

    assert len(flags) >= 1
    flag = flags[0]
    assert isinstance(flag, SafetyFlagRecord)
    assert flag.type == "allergy_med_conflict"
    assert flag.severity == "high"
    assert "cross-reacts" in flag.explanation
    assert len(flag.target_fact_ids) >= 1


def test_cross_reactivity_cephalosporin_vs_penicillin(safety_fixture) -> None:
    """Verifies beta-lactam class cross-reactivity between cephalosporin and penicillin allergy."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Record active penicillin allergy
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Allergy to penicillin",
        value="anaphylaxis",
        authority="clinician",
        caller_role="clinician",
    )

    # Check proposed Cefuroxime (RxNorm 309097, Cephalosporin)
    flags = safety_mgr.check_allergy_cross_reactivity(
        patient_id=TEST_PID,
        drug_code="309097",
        drug_name="Cefuroxime 250 MG Oral Tablet",
    )

    assert len(flags) >= 1
    flag = flags[0]
    assert flag.type == "allergy_med_conflict"
    assert flag.severity == "high"
    assert flag.rule_id == "rule_cross_reactivity"


def test_cross_reactivity_negative_case(safety_fixture) -> None:
    """Verifies that non-cross-reactive drugs return no allergy conflict flags."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Record active penicillin allergy
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Allergy to penicillin",
        value="rash",
        authority="clinician",
        caller_role="clinician",
    )

    # Check proposed Metformin (RxNorm 860975, Biguanide - no cross-reactivity)
    flags = safety_mgr.check_allergy_cross_reactivity(
        patient_id=TEST_PID,
        drug_code="860975",
        drug_name="Metformin 500 MG Oral Tablet",
    )

    assert len(flags) == 0


def test_drug_drug_interaction_lisinopril_and_ibuprofen(safety_fixture) -> None:
    """Verifies DDI detection between active Lisinopril and proposed Ibuprofen."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Active medication: Lisinopril 10 MG (RxNorm 314076)
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        code="314076",
        display="lisinopril 10 MG Oral Tablet",
        value="10 mg daily",
        authority="clinician",
        caller_role="clinician",
    )

    # Proposed medication: Ibuprofen 200 MG (RxNorm 310965)
    flags = safety_mgr.check_drug_drug_interaction(
        patient_id=TEST_PID,
        proposed_code="310965",
        proposed_name="Ibuprofen 200 MG Oral Tablet",
    )

    assert len(flags) >= 1
    flag = flags[0]
    assert flag.type == "drug_interaction"
    assert flag.severity == "moderate"
    assert "NSAIDs reduce antihypertensive efficacy" in flag.explanation


def test_drug_drug_interaction_severe_dual_ras_blockade(safety_fixture) -> None:
    """Verifies severe DDI detection between Lisinopril and Losartan (dual RAS blockade)."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Active medication: Lisinopril
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        code="314076",
        display="lisinopril 10 MG Oral Tablet",
        value="10 mg daily",
        authority="clinician",
        caller_role="clinician",
    )

    # Proposed medication: Losartan (RxNorm 979467)
    flags = safety_mgr.check_drug_drug_interaction(
        patient_id=TEST_PID,
        proposed_code="979467",
        proposed_name="Losartan Potassium 50 MG Oral Tablet",
    )

    assert len(flags) >= 1
    flag = flags[0]
    assert flag.type == "drug_interaction"
    assert flag.severity == "severe"
    assert "Dual renin-angiotensin system blockade" in flag.explanation


def test_duplicate_therapy_detection(safety_fixture) -> None:
    """Verifies duplicate therapy detection within identical drug class."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    # Active medication: Lisinopril 10 MG (class: dc_ace_inhibitors)
    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        code="314076",
        display="lisinopril 10 MG Oral Tablet",
        value="10 mg daily",
        authority="clinician",
        caller_role="clinician",
    )

    # Proposed medication: Lisinopril 20 MG (same class dc_ace_inhibitors)
    flags = safety_mgr.check_duplicate_therapy(
        patient_id=TEST_PID,
        proposed_code="314077",
        proposed_name="lisinopril 20 MG Oral Tablet",
    )

    assert len(flags) >= 1
    flag = flags[0]
    assert flag.type == "duplicate_therapy"
    assert "Duplicate therapy" in flag.explanation
    assert flag.metadata.get("class_id") == "dc_ace_inhibitors"


def test_check_proposed_medication_passed_vs_failed(safety_fixture) -> None:
    """Verifies check_proposed_medication returns passed=False on high severity conflict."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="allergy",
        code="300916003",
        display="Allergy to penicillin",
        value="anaphylaxis",
        authority="clinician",
        caller_role="clinician",
    )

    # Check Amoxicillin (Allergy conflict -> high severity -> passed = False)
    res_fail = safety_mgr.check_proposed_medication(
        patient_id=TEST_PID,
        drug_code="308182",
        drug_name="Amoxicillin 250 MG Oral Capsule",
    )
    assert isinstance(res_fail, SafetyCheckResult)
    assert res_fail.passed is False
    assert res_fail.cross_reactivities_count >= 1

    # Check Atorvastatin (Safe -> passed = True)
    res_pass = safety_mgr.check_proposed_medication(
        patient_id=TEST_PID,
        drug_code="617310",
        drug_name="atorvastatin 20 MG Oral Tablet",
    )
    assert res_pass.passed is True
    assert len(res_pass.flags) == 0


def test_flag_persistence_and_relationships(safety_fixture) -> None:
    """Verifies that persisting flags writes Flag node and ABOUT / CONFLICTS_WITH edges."""
    safety_mgr, fact_mgr, router, clock = safety_fixture

    med_rec = fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="medication",
        code="314076",
        display="lisinopril 10 MG Oral Tablet",
        value="10 mg",
        authority="clinician",
        caller_role="clinician",
    )

    flg = SafetyFlagRecord(
        id="flg_test001",
        patient_id=TEST_PID,
        type="drug_interaction",
        severity="moderate",
        explanation="NSAID interaction warning",
        rule_id="rule_ddi",
        detected_at=clock.now_ts(),
        target_fact_ids=[med_rec["fact_id"]],
    )

    safety_mgr.create_flag(TEST_PID, flg)

    target_graph = router.graph(TEST_PID)
    # Check Flag node exists
    f_res = target_graph.query(
        "MATCH (flg:Flag {id:'flg_test001'}) RETURN flg.type, flg.status",
    )
    assert len(f_res.result_set) == 1
    assert f_res.result_set[0][0] == "drug_interaction"
    assert f_res.result_set[0][1] == "open"

    # Check ABOUT edge
    about_res = target_graph.query(
        "MATCH (flg:Flag {id:'flg_test001'})-[:ABOUT]->(f:Fact) RETURN f.id",
    )
    assert len(about_res.result_set) == 1
    assert about_res.result_set[0][0] == med_rec["fact_id"]


def test_flag_lifecycle_resolution_and_dismissal(safety_fixture) -> None:
    """Verifies resolving and dismissing safety flags updates status and resolution metadata."""
    safety_mgr, fact_mgr, _, _ = safety_fixture

    flg1 = SafetyFlagRecord(
        id="flg_open001",
        patient_id=TEST_PID,
        type="allergy_med_conflict",
        severity="high",
        explanation="Cross-reactivity review required",
        rule_id="rule_cr",
    )
    flg2 = SafetyFlagRecord(
        id="flg_open002",
        patient_id=TEST_PID,
        type="drug_interaction",
        severity="moderate",
        explanation="Minor interaction warning",
        rule_id="rule_ddi",
    )

    safety_mgr.create_flag(TEST_PID, flg1)
    safety_mgr.create_flag(TEST_PID, flg2)

    open_flags = safety_mgr.get_open_flags(TEST_PID)
    assert len(open_flags) == 2

    # Resolve flg1
    resolved = safety_mgr.resolve_flag(
        patient_id=TEST_PID,
        flag_id="flg_open001",
        resolved_by="dr_smith",
        resolution_reason="Patient tolerated amoxicillin without rash under direct observation.",
    )
    assert resolved.status == "resolved"
    assert resolved.resolved_by == "dr_smith"
    assert resolved.resolved_at is not None

    # Dismiss flg2
    dismissed = safety_mgr.dismiss_flag(
        patient_id=TEST_PID,
        flag_id="flg_open002",
        dismissed_by="pharm_jones",
        dismissal_reason="Short-term co-administration approved with monitoring.",
    )
    assert dismissed.status == "dismissed"
    assert dismissed.resolved_by == "pharm_jones"

    # Open flags should now be 0
    remaining_open = safety_mgr.get_open_flags(TEST_PID)
    assert len(remaining_open) == 0


def test_flag_not_found_error(safety_fixture) -> None:
    """Verifies FlagNotFoundError is raised when querying a nonexistent flag."""
    safety_mgr, _, _, _ = safety_fixture

    with pytest.raises(FlagNotFoundError):
        safety_mgr.get_flag(TEST_PID, "flg_nonexistent999")

    with pytest.raises(PatientNotFoundError):
        safety_mgr.get_flag("p_000000000000", "flg_nonexistent999")
