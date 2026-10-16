"""Unit tests for procedural memory and clinical playbook management.

Validates:
- Playbook creation with ordered PlaybookStep nodes
- Dense 384-dimensional vector embedding generation
- Instantiation of standardized templates from clinic_shared (PlaybookTemplate)
- Outcome tracking with Laplace performance scoring ((s + 1) / (s + f + 2))
- Automatic promotion from candidate to active status at threshold (3 successes)
- Association with clinical episodes (DERIVED_FROM and FOLLOWED relationships)
- Dense vector similarity search over playbooks
- Listing and filtering by episode type and status
- Archive lifecycle transitions
- Validation and error handling (PlaybookNotFoundError, ValidationError, PatientNotFoundError)
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    PatientNotFoundError,
    PlaybookNotFoundError,
    ValidationError,
)
from carethread.memory.procedural import PlaybookManager

TEST_PID = "p_777788889999"


@pytest.fixture
def procedural_fixture():
    """Sets up a clean test tenant and PlaybookManager."""
    client = get_db_client()
    router = TenantRouter(client)

    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    try:
        router.deregister_patient(TEST_PID)
    except Exception:
        pass

    router.register_patient(TEST_PID, bootstrap_schema=True)
    g = router.graph(TEST_PID)
    g.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Procedural Test Patient'",
        {"pid": TEST_PID},
    )

    clock = FixedClock(1792139000)
    manager = PlaybookManager(router=router, clock=clock)

    yield {
        "router": router,
        "clock": clock,
        "manager": manager,
        "graph": g,
    }

    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    try:
        router.deregister_patient(TEST_PID)
    except Exception:
        pass


def test_create_playbook_with_steps(procedural_fixture: dict[str, Any]):
    """Verify creating a playbook stores Playbook node, steps, and generates embedding."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    steps = [
        {"action_type": "assessment", "text": "Check seated blood pressure and heart rate."},
        {"action_type": "order", "text": "Order standard 12-lead ECG."},
        {"action_type": "instruction", "text": "Advise patient to sit upright."},
    ]

    pb = mgr.create_playbook(
        patient_id=TEST_PID,
        name="Hypertension Crisis Protocol",
        episode_type="ambulatory",
        steps=steps,
        status="candidate",
    )

    assert pb.id.startswith("pb_")
    assert pb.patient_id == TEST_PID
    assert pb.name == "Hypertension Crisis Protocol"
    assert pb.episode_type == "ambulatory"
    assert pb.status == "candidate"
    assert pb.score == 0.5
    assert pb.success_count == 0
    assert pb.failure_count == 0
    assert len(pb.steps) == 3
    assert pb.steps[0].idx == 0
    assert pb.steps[0].action_type == "assessment"
    assert pb.steps[1].idx == 1
    assert pb.steps[1].action_type == "order"
    assert pb.steps[2].idx == 2
    assert pb.steps[2].action_type == "instruction"

    # Fetch back
    fetched = mgr.get_playbook(TEST_PID, pb.id)
    assert fetched.id == pb.id
    assert len(fetched.steps) == 3


def test_instantiate_from_template(procedural_fixture: dict[str, Any]):
    """Verify instantiating a playbook from clinic_shared PlaybookTemplate."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    # Use seeded template 'pb_ed_chest_pain'
    pb = mgr.instantiate_from_template(
        patient_id=TEST_PID,
        template_id="pb_ed_chest_pain",
    )

    assert pb.id.startswith("pb_")
    assert pb.guideline_ref == "pb_ed_chest_pain"
    assert pb.episode_type == "emergency"
    assert "Chest Pain" in pb.name
    assert len(pb.steps) >= 4
    assert pb.status == "candidate"


def test_outcome_tracking_and_auto_promotion(procedural_fixture: dict[str, Any]):
    """Verify outcome updates, Laplace scoring, and promotion at 3 successes (q_pb_01)."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    pb = mgr.create_playbook(
        patient_id=TEST_PID,
        name="Diabetes Insulin Titration",
        episode_type="ambulatory",
        steps=["Review fasting blood glucose.", "Adjust long-acting insulin dose."],
        status="candidate",
    )
    assert pb.status == "candidate"
    assert pb.score == 0.5

    # 1st success: ns=1, nf=0 -> score = (1 + 1)/(1 + 0 + 2) = 2/3 ~ 0.6667
    pb1 = mgr.record_outcome(TEST_PID, pb.id, success=True)
    assert pb1.success_count == 1
    assert pb1.failure_count == 0
    assert abs(pb1.score - (2.0 / 3.0)) < 0.001
    assert pb1.status == "candidate"

    # 2nd success: ns=2, nf=0 -> score = (2 + 1)/(2 + 0 + 2) = 3/4 = 0.75
    pb2 = mgr.record_outcome(TEST_PID, pb.id, success=True)
    assert pb2.success_count == 2
    assert pb2.status == "candidate"
    assert abs(pb2.score - 0.75) < 0.001

    # 3rd success: reaches promotion threshold -> status promoted to 'active'!
    pb3 = mgr.record_outcome(TEST_PID, pb.id, success=True)
    assert pb3.success_count == 3
    assert pb3.status == "active"
    assert abs(pb3.score - 0.8) < 0.001

    # 4th outcome: failure -> ns=3, nf=1 -> score = (3 + 1)/(3 + 1 + 2) = 4/6 ~ 0.6667
    pb4 = mgr.record_outcome(TEST_PID, pb.id, success=False)
    assert pb4.success_count == 3
    assert pb4.failure_count == 1
    assert pb4.status == "active"
    assert abs(pb4.score - (4.0 / 6.0)) < 0.001


def test_episode_associations(procedural_fixture: dict[str, Any]):
    """Verify DERIVED_FROM and FOLLOWED relationships with Episode nodes."""
    mgr: PlaybookManager = procedural_fixture["manager"]
    g = procedural_fixture["graph"]

    # Create dummy episodes
    g.query("MERGE (e1:Episode {id:'epi_init_01'}) ON CREATE SET e1.type='ambulatory'")
    g.query("MERGE (e2:Episode {id:'epi_followup_02'}) ON CREATE SET e2.type='ambulatory'")

    pb = mgr.create_playbook(
        patient_id=TEST_PID,
        name="Hypertension Care Protocol",
        episode_type="ambulatory",
        steps=["Review blood pressure.", "Check adherence."],
        derived_from_episode_ids=["epi_init_01"],
    )

    assert "epi_init_01" in pb.derived_episode_ids

    # Record outcome following episode epi_followup_02
    pb_updated = mgr.record_outcome(
        patient_id=TEST_PID,
        playbook_id=pb.id,
        success=True,
        episode_id="epi_followup_02",
    )
    assert "epi_followup_02" in pb_updated.followed_episode_ids


def test_search_playbooks_vector_similarity(procedural_fixture: dict[str, Any]):
    """Verify vector search on Playbook.embedding retrieves relevant playbooks."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    mgr.create_playbook(
        patient_id=TEST_PID,
        name="Asthma Exacerbation Triage",
        episode_type="emergency",
        steps=["Measure peak expiratory flow.", "Administer albuterol nebulizer."],
        status="active",
    )

    mgr.create_playbook(
        patient_id=TEST_PID,
        name="Routine Annual Wellness Exam",
        episode_type="wellness",
        steps=["Screen vaccination status.", "Counsel on healthy lifestyle."],
        status="active",
    )

    results = mgr.search_playbooks(TEST_PID, query="acute wheezing respiratory distress", limit=2)
    assert len(results) >= 1
    top_pb, similarity = results[0]
    assert "Asthma" in top_pb.name
    assert similarity > 0.0


def test_list_and_archive_playbooks(procedural_fixture: dict[str, Any]):
    """Verify listing with filters and archiving a playbook."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    p1 = mgr.create_playbook(
        patient_id=TEST_PID,
        name="Protocol 1",
        episode_type="ambulatory",
        steps=["Step 1"],
        status="active",
    )
    p2 = mgr.create_playbook(
        patient_id=TEST_PID,
        name="Protocol 2",
        episode_type="emergency",
        steps=["Step 2"],
        status="candidate",
    )

    # Filter by episode_type
    amb = mgr.list_playbooks(TEST_PID, episode_type="ambulatory")
    assert any(p.id == p1.id for p in amb)
    assert not any(p.id == p2.id for p in amb)

    # Archive p1
    archived = mgr.archive_playbook(TEST_PID, p1.id)
    assert archived.status == "archived"

    # Filter by status
    arch = mgr.list_playbooks(TEST_PID, status="archived")
    assert any(p.id == p1.id for p in arch)


def test_validation_and_errors(procedural_fixture: dict[str, Any]):
    """Verify input validation and error raising."""
    mgr: PlaybookManager = procedural_fixture["manager"]

    # Invalid patient
    with pytest.raises(ValidationError):
        mgr.create_playbook("bad_pid", "Name", "ambulatory", ["Step"])

    # Non-existent patient
    with pytest.raises(PatientNotFoundError):
        mgr.create_playbook("p_000011112222", "Name", "ambulatory", ["Step"])

    # Empty name
    with pytest.raises(ValidationError):
        mgr.create_playbook(TEST_PID, "", "ambulatory", ["Step"])

    # Invalid episode_type
    with pytest.raises(ValidationError):
        mgr.create_playbook(TEST_PID, "Name", "invalid_type", ["Step"])

    # Non-existent playbook
    with pytest.raises(PlaybookNotFoundError):
        mgr.get_playbook(TEST_PID, "pb_nonexistent999")
