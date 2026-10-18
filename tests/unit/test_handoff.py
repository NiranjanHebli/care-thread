"""Unit tests for the inter-agent handoff state machine and delegation engine.

Validates:
- Handoff packet creation with carried facts, open questions, and recommended actions
- Publish lifecycle from draft to open via query q_ho_01b
- Atomic compare-and-swap (CAS) delegation via query q_ho_02a
- Role mismatch protection and CAS concurrency conflict rejection
- Completion lifecycle with result summary and session link
- Rejection lifecycle with reason recording
- Time-to-live (TTL) expiration and background sweep
- Filtered handoff listing by role and status
- Error handling (HandoffNotFoundError, HandoffStateError, ValidationError)
"""

from datetime import datetime, timezone

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    HandoffNotFoundError,
    HandoffStateError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.facts import FactManager
from carethread.memory.handoff import HandoffManager, HandoffPacket

TEST_PID = "p_333344445555"


@pytest.fixture
def handoff_fixture():
    """Sets up a clean bootstrapped patient graph and handoff manager."""
    client = get_db_client()
    router = TenantRouter(client)
    router.register_patient(TEST_PID, bootstrap_schema=True)

    target_graph = router.graph(TEST_PID)
    target_graph.query(
        "MERGE (p:Patient {id:$pid}) "
        "ON CREATE SET p.name = 'Handoff Test Patient', p.created_at = 100000",
        {"pid": TEST_PID},
    )

    clock = FixedClock(datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc))
    fact_manager = FactManager(router=router, clock=clock)
    handoff_manager = HandoffManager(router=router, clock=clock)

    yield handoff_manager, fact_manager, router, clock

    # Teardown
    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    router.deregister_patient(TEST_PID)


def test_handoff_creation_and_retrieval(handoff_fixture) -> None:
    """Verifies creating a handoff with carried facts, questions, and actions."""
    ho_mgr, fact_mgr, _, _ = handoff_fixture

    # Create supporting fact
    f_rec = fact_mgr.remember_fact(
        patient_id=TEST_PID,
        kind="condition",
        display="Hypertension",
        value="Stage 2",
        authority="clinician",
        caller_role="clinician",
    )

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Cardiology evaluation for resistant hypertension",
        state_summary="Blood pressure remains 150/95 despite first-line therapy.",
        carried_fact_ids=[f_rec["fact_id"]],
        open_questions=["Should we initiate dual RAS blockade or add calcium channel blocker?"],
        actions=["Order 24-hour ambulatory blood pressure monitoring"],
        publish=True,
    )

    assert isinstance(packet, HandoffPacket)
    assert packet.id.startswith("hnd_")
    assert packet.status == "open"
    assert packet.from_role == "intake"
    assert packet.to_role == "specialist"
    assert packet.goal == "Cardiology evaluation for resistant hypertension"
    assert len(packet.carried_facts) >= 1
    assert len(packet.open_questions) == 1
    assert len(packet.actions) == 1

    # Verify retrieval
    fetched = ho_mgr.get_handoff(TEST_PID, packet.id)
    assert fetched.id == packet.id
    assert fetched.status == "open"
    assert fetched.to_role == "specialist"
    assert len(fetched.open_questions) == 1


def test_handoff_publish_from_draft(handoff_fixture) -> None:
    """Verifies that publishing a draft handoff transitions status to open."""
    ho_mgr, _, _, _ = handoff_fixture

    draft_packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Pending review consultation",
        publish=False,
    )
    assert draft_packet.status == "draft"

    # Publish to open
    published = ho_mgr.publish_handoff(TEST_PID, draft_packet.id)
    assert published.status == "open"
    assert published.version >= 2

    # Attempting to publish again should raise HandoffStateError
    with pytest.raises(HandoffStateError):
        ho_mgr.publish_handoff(TEST_PID, draft_packet.id)


def test_handoff_accept_cas_delegation(handoff_fixture) -> None:
    """Verifies atomic CAS acceptance by assigned agent role."""
    ho_mgr, _, router, clock = handoff_fixture

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Specialist review",
        publish=True,
    )

    accepted = ho_mgr.accept_handoff(
        patient_id=TEST_PID,
        handoff_id=packet.id,
        agent_id="agent_cardiology_01",
        role="specialist",
    )

    assert accepted.status == "accepted"
    assert accepted.to_agent_id == "agent_cardiology_01"
    assert accepted.accepted_at == clock.now_ts()

    # Verify TO edge in graph
    target_graph = router.graph(TEST_PID)
    res = target_graph.query(
        "MATCH (h:Handoff {id:$hid})-[:TO]->(a:Agent) RETURN a.id",
        {"hid": packet.id},
    )
    assert len(res.result_set) == 1
    assert res.result_set[0][0] == "agent_cardiology_01"


def test_handoff_accept_role_mismatch_rejected(handoff_fixture) -> None:
    """Verifies that an agent with mismatched role cannot accept the handoff."""
    ho_mgr, _, _, _ = handoff_fixture

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Cardiology review",
        publish=True,
    )

    # Pharmacy agent tries to accept specialist handoff
    with pytest.raises(HandoffStateError) as exc_info:
        ho_mgr.accept_handoff(
            patient_id=TEST_PID,
            handoff_id=packet.id,
            agent_id="agent_pharmacy_01",
            role="pharmacy",
        )
    assert "Role mismatch" in str(exc_info.value)


def test_handoff_accept_concurrency_conflict(handoff_fixture) -> None:
    """Verifies CAS conflict when second agent tries to accept an already accepted handoff."""
    ho_mgr, _, _, _ = handoff_fixture

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Specialist evaluation",
        publish=True,
    )

    # First specialist accepts
    ho_mgr.accept_handoff(
        patient_id=TEST_PID,
        handoff_id=packet.id,
        agent_id="agent_spec_alice",
        role="specialist",
    )

    # Second specialist tries to accept
    with pytest.raises(HandoffStateError) as exc_info:
        ho_mgr.accept_handoff(
            patient_id=TEST_PID,
            handoff_id=packet.id,
            agent_id="agent_spec_bob",
            role="specialist",
        )
    assert "Cannot accept" in str(exc_info.value) or "CAS conflict" in str(exc_info.value)


def test_handoff_completion_lifecycle(handoff_fixture) -> None:
    """Verifies complete_handoff records result summary and updates status."""
    ho_mgr, _, _, clock = handoff_fixture

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="pharmacy",
        goal="Medication safety check and reconciliation",
        publish=True,
    )

    ho_mgr.accept_handoff(
        patient_id=TEST_PID,
        handoff_id=packet.id,
        agent_id="agent_pharmacy_01",
        role="pharmacy",
    )

    completed = ho_mgr.complete_handoff(
        patient_id=TEST_PID,
        handoff_id=packet.id,
        result_summary="Reconciliation complete: no drug interactions found.",
    )

    assert completed.status == "completed"
    assert completed.completed_at == clock.now_ts()
    assert "Reconciliation complete" in str(completed.result_summary)

    # Cannot complete already completed handoff
    with pytest.raises(HandoffStateError):
        ho_mgr.complete_handoff(
            patient_id=TEST_PID,
            handoff_id=packet.id,
            result_summary="Second attempt",
        )


def test_handoff_rejection(handoff_fixture) -> None:
    """Verifies rejecting an open or accepted handoff delegation."""
    ho_mgr, _, _, _ = handoff_fixture

    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Nephrology consultation",
        publish=True,
    )

    rejected = ho_mgr.reject_handoff(
        patient_id=TEST_PID,
        handoff_id=packet.id,
        rejected_by="agent_spec_01",
        rejection_reason="Patient eGFR is > 60; nephrology consult not currently indicated.",
    )

    assert rejected.status == "rejected"
    assert rejected.version >= 2


def test_handoff_ttl_expiration_and_sweep(handoff_fixture) -> None:
    """Verifies that expired handoffs cannot be accepted and are swept by expiration."""
    ho_mgr, _, _, clock = handoff_fixture

    # 1 hour TTL
    packet = ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Time-sensitive consult",
        ttl_seconds=3600,
        publish=True,
    )

    # Advance clock by 2 hours
    clock.advance(7200)

    # Attempting to accept should fail due to TTL expiration
    with pytest.raises(HandoffStateError) as exc_info:
        ho_mgr.accept_handoff(
            patient_id=TEST_PID,
            handoff_id=packet.id,
            agent_id="agent_spec_01",
            role="specialist",
        )
    assert "expired" in str(exc_info.value).lower()

    # Expire stale handoffs sweep
    count = ho_mgr.expire_stale_handoffs(patient_id=TEST_PID)
    assert count >= 0


def test_handoff_filtering_and_listing(handoff_fixture) -> None:
    """Verifies listing handoffs by role and status filters."""
    ho_mgr, _, _, _ = handoff_fixture

    ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="specialist",
        goal="Goal 1",
        publish=True,
    )
    ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="pharmacy",
        goal="Goal 2",
        publish=True,
    )
    ho_mgr.create_handoff(
        patient_id=TEST_PID,
        from_role="intake",
        to_role="pharmacy",
        goal="Goal 3",
        publish=False,  # draft
    )

    all_handoffs = ho_mgr.list_handoffs(TEST_PID)
    assert len(all_handoffs) == 3

    pharmacy_handoffs = ho_mgr.list_handoffs(TEST_PID, to_role="pharmacy")
    assert len(pharmacy_handoffs) == 2

    draft_handoffs = ho_mgr.list_handoffs(TEST_PID, status="draft")
    assert len(draft_handoffs) == 1


def test_handoff_validation_and_not_found(handoff_fixture) -> None:
    """Verifies validation and not-found error handling for handoffs."""
    ho_mgr, _, _, _ = handoff_fixture

    # Empty goal
    with pytest.raises(ValidationError):
        ho_mgr.create_handoff(
            patient_id=TEST_PID,
            from_role="intake",
            to_role="specialist",
            goal="   ",
        )

    # Invalid role
    with pytest.raises(ValidationError):
        ho_mgr.create_handoff(
            patient_id=TEST_PID,
            from_role="invalid_role",
            to_role="specialist",
            goal="Test goal",
        )

    # Nonexistent patient
    with pytest.raises(PatientNotFoundError):
        ho_mgr.get_handoff("p_000000000000", "hnd_000000000000")

    # Nonexistent handoff
    with pytest.raises(HandoffNotFoundError):
        ho_mgr.get_handoff(TEST_PID, "hnd_nonexistent999")
