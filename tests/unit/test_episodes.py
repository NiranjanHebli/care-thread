"""Unit tests for EpisodeManager, NEXT chains, and queries q_ep_01a / q_ep_01b.

Validates episode lifecycle, NEXT chains, and queries:
- Episode creation and lifecycle
- Singly-linked chronological chain (NEXT) with gap_days calculation
- Chain traversal (forward and backward) via query q_ep_01b
- Point-in-time episode history via query q_ep_01a
- Vector similarity search over episode summaries
"""

from datetime import datetime, timezone

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    EpisodeNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.embeddings import HashEmbedder
from carethread.memory.episodes import EpisodeManager

TEST_PID = "p_666677778888"


@pytest.fixture
def episode_fixture():
    """Sets up a clean bootstrapped patient graph for episode tests."""
    client = get_db_client()
    router = TenantRouter(client)
    router.register_patient(TEST_PID, bootstrap_schema=True)

    target_graph = router.graph(TEST_PID)
    target_graph.query(
        "MERGE (p:Patient {id:$pid}) "
        "ON CREATE SET p.name = 'Episode Test Patient', p.created_at = 123456",
        {"pid": TEST_PID},
    )

    clock = FixedClock(datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc))
    embedder = HashEmbedder()
    manager = EpisodeManager(router=router, clock=clock, embedder=embedder)

    yield manager, router, clock

    # Teardown
    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    router.deregister_patient(TEST_PID)


def test_episode_lifecycle(episode_fixture) -> None:
    manager, router, clock = episode_fixture

    # 1. Start episode
    ep = manager.start_episode(
        patient_id=TEST_PID,
        kind="live_visit",
        episode_type="ambulatory",
        summary="Annual routine physical examination.",
    )
    assert ep.id.startswith("epi_")
    assert ep.seq == 1
    assert ep.kind == "live_visit"
    assert ep.outcome == "in_progress"
    assert ep.start == clock.now_ts()
    assert ep.end is None

    # 2. Get episode
    fetched = manager.get_episode(TEST_PID, ep.id)
    assert fetched is not None
    assert fetched.id == ep.id
    assert fetched.summary == "Annual routine physical examination."

    # 3. Close episode
    clock.advance(3600)  # 1 hour encounter
    closed = manager.close_episode(
        TEST_PID,
        ep.id,
        outcome="completed",
        summary="Annual routine physical examination. Normal vitals.",
    )
    assert closed.outcome == "completed"
    assert closed.end == clock.now_ts()
    assert "Normal vitals" in closed.summary


def test_episodic_chronological_chain_and_traversal(episode_fixture) -> None:
    """Verifies that sequential episodes form a singly-linked NEXT chain with gap_days."""
    manager, router, clock = episode_fixture

    # Day 1: Episode 1 (Intake)
    ep1 = manager.start_episode(
        patient_id=TEST_PID,
        kind="live_visit",
        episode_type="wellness",
        summary="Initial clinic intake visit.",
    )
    assert ep1.seq == 1

    # Advance 10 days
    clock.advance(10 * 86400)

    # Day 11: Episode 2 (Specialist consultation)
    ep2 = manager.start_episode(
        patient_id=TEST_PID,
        kind="specialist_review",
        episode_type="ambulatory",
        summary="Specialist cardiology evaluation for arrhythmia.",
    )
    assert ep2.seq == 2

    # Advance 5 days
    clock.advance(5 * 86400)

    # Day 16: Episode 3 (Pharmacy reconciliation)
    ep3 = manager.start_episode(
        patient_id=TEST_PID,
        kind="pharmacy_review",
        episode_type="ambulatory",
        summary="Clinical pharmacy review of antiarrhythmic medications.",
    )
    assert ep3.seq == 3

    # Direct Cypher verification of NEXT edges and gap_days
    target_graph = router.graph(TEST_PID)
    edge_res = target_graph.query(
        """
        MATCH (e1:Episode)-[r:NEXT]->(e2:Episode)
        RETURN e1.id, e2.id, r.gap_days, r.seq
        ORDER BY r.seq ASC
        """
    )
    assert len(edge_res.result_set) == 2

    # e1 -> e2: gap_days should be 10
    row1 = edge_res.result_set[0]
    assert row1[0] == ep1.id
    assert row1[1] == ep2.id
    assert row1[2] == 10
    assert row1[3] == 2

    # e2 -> e3: gap_days should be 5
    row2 = edge_res.result_set[1]
    assert row2[0] == ep2.id
    assert row2[1] == ep3.id
    assert row2[2] == 5
    assert row2[3] == 3

    # Forward chain traversal from ep1 (q_ep_01b)
    forward_chain = manager.get_episode_chain(TEST_PID, ep1.id, hops=3, direction="forward")
    assert len(forward_chain) == 2
    assert [n["id"] for n in forward_chain] == [ep2.id, ep3.id]

    # Backward chain traversal from ep3
    backward_chain = manager.get_episode_chain(TEST_PID, ep3.id, hops=3, direction="backward")
    assert len(backward_chain) == 2
    assert [n["id"] for n in backward_chain] == [ep1.id, ep2.id]


def test_point_in_time_episode_history(episode_fixture) -> None:
    """Verifies q_ep_01a point-in-time filtering (WHERE e.start <= $now)."""
    manager, router, clock = episode_fixture

    ep1 = manager.start_episode(TEST_PID, kind="live_visit", summary="Visit 1")

    clock.advance(7 * 86400)
    t1 = clock.now_ts()
    ep2 = manager.start_episode(TEST_PID, kind="specialist_review", summary="Visit 2")

    clock.advance(7 * 86400)
    ep3 = manager.start_episode(TEST_PID, kind="pharmacy_review", summary="Visit 3")

    # At time t1, only ep1 and ep2 should be visible
    past_eps = manager.get_patient_episodes(TEST_PID, limit=10, now=t1)
    assert len(past_eps) == 2
    assert [e.id for e in past_eps] == [ep2.id, ep1.id]  # Ordered DESC by start

    # At current time, all 3 are visible
    all_eps = manager.get_patient_episodes(TEST_PID, limit=10)
    assert len(all_eps) == 3
    assert [e.id for e in all_eps] == [ep3.id, ep2.id, ep1.id]


def test_episode_vector_search(episode_fixture) -> None:
    manager, router, clock = episode_fixture

    manager.start_episode(
        TEST_PID,
        kind="live_visit",
        summary="Dermatology examination for eczema flare on bilateral arms.",
    )
    ep2 = manager.start_episode(
        TEST_PID,
        kind="specialist_review",
        summary="Comprehensive cardiology assessment for congestive heart failure.",
    )

    results = manager.search_episodes(
        TEST_PID,
        query_text="Comprehensive cardiology assessment for congestive heart failure.",
        top_k=2,
    )
    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["id"] == ep2.id
    assert "cardiology" in top_hit["summary"]
    assert top_hit["similarity"] > 0.95


def test_episode_validation_and_errors(episode_fixture) -> None:
    manager, router, clock = episode_fixture

    # Non-existent patient
    with pytest.raises(PatientNotFoundError):
        manager.start_episode(patient_id="p_000000000000")

    # Invalid kind
    with pytest.raises(ValidationError):
        manager.start_episode(patient_id=TEST_PID, kind="unsupported_kind")

    ep = manager.start_episode(patient_id=TEST_PID, kind="live_visit")

    # Invalid outcome
    with pytest.raises(ValidationError):
        manager.close_episode(patient_id=TEST_PID, episode_id=ep.id, outcome="bogus_outcome")

    # Non-existent episode
    with pytest.raises(EpisodeNotFoundError):
        manager.close_episode(patient_id=TEST_PID, episode_id="epi_nonexistent0000000000")
