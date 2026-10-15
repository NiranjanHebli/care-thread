"""Unit and concurrency tests for SessionManager and atomic message sequencing.

Validates session lifecycle, atomic messages, and q_msg_01a / q_msg_01b:
- Session lifecycle (create, get, list, close)
- Atomic message sequence ordering (seq = s.msg_count + 1)
- Concurrent message appends without collisions or race conditions
- Compound uniqueness constraint on Message(session_id, seq)
- 384-dimensional vector embedding and similarity search on messages
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from redis.exceptions import ResponseError

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    PatientNotFoundError,
    SessionNotFoundError,
    ValidationError,
)
from carethread.memory.embeddings import HashEmbedder
from carethread.memory.sessions import SessionManager

TEST_PID = "p_555544443333"


@pytest.fixture
def session_fixture():
    """Sets up a clean bootstrapped patient graph for session tests."""
    client = get_db_client()
    router = TenantRouter(client)
    router.register_patient(TEST_PID, bootstrap_schema=True)

    # Create anchor Patient node in patient graph
    target_graph = router.graph(TEST_PID)
    target_graph.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Test Patient', p.created_at = 123456",
        {"pid": TEST_PID},
    )

    clock = FixedClock(datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc))
    embedder = HashEmbedder()
    manager = SessionManager(router=router, clock=clock, embedder=embedder)

    yield manager, router, clock

    # Teardown: delete patient graph and deregister
    try:
        router.graph(TEST_PID).delete()
    except Exception:
        pass
    router.deregister_patient(TEST_PID)


def test_session_lifecycle(session_fixture) -> None:
    manager, router, clock = session_fixture

    # 1. Create session
    session = manager.create_session(
        patient_id=TEST_PID,
        channel="chat",
        agent_id="intake_agent",
    )
    assert session.id.startswith("ses_")
    assert session.patient_id == TEST_PID
    assert session.status == "open"
    assert session.msg_count == 0
    assert session.channel == "chat"
    assert session.agent_id == "intake_agent"
    assert session.started_at == clock.now_ts()

    # 2. Get session
    fetched = manager.get_session(TEST_PID, session.id)
    assert fetched is not None
    assert fetched.id == session.id
    assert fetched.status == "open"

    # 3. List sessions
    sessions = manager.list_sessions(TEST_PID)
    assert len(sessions) >= 1
    assert any(s.id == session.id for s in sessions)

    # 4. Close session
    clock.advance(1800)  # 30 minutes later
    closed = manager.close_session(
        TEST_PID,
        session.id,
        summary="Intake assessment completed successfully.",
    )
    assert closed.status == "closed"
    assert closed.ended_at == clock.now_ts()
    assert closed.summary == "Intake assessment completed successfully."

    # Verify state persisted
    persisted = manager.get_session(TEST_PID, session.id)
    assert persisted is not None
    assert persisted.status == "closed"
    assert persisted.ended_at == clock.now_ts()


def test_atomic_message_sequencing(session_fixture) -> None:
    manager, router, clock = session_fixture

    session = manager.create_session(patient_id=TEST_PID)

    # Append 4 sequential messages
    m1 = manager.append_message(
        patient_id=TEST_PID,
        session_id=session.id,
        role="user",
        content="Good morning, I feel dizzy.",
    )
    assert m1.seq == 1
    assert m1.role == "user"
    assert m1.id.startswith("msg_")
    assert m1.embedding is not None
    assert len(m1.embedding) == 384

    clock.advance(5)
    m2 = manager.append_message(
        patient_id=TEST_PID,
        session_id=session.id,
        role="agent",
        content="I am sorry to hear that. Did the dizziness start suddenly?",
        agent_id="agent_intake",
    )
    assert m2.seq == 2
    assert m2.role == "agent"

    clock.advance(10)
    m3 = manager.append_message(
        patient_id=TEST_PID,
        session_id=session.id,
        role="user",
        content="Yes, after taking my morning medication.",
    )
    assert m3.seq == 3

    clock.advance(5)
    m4 = manager.append_message(
        patient_id=TEST_PID,
        session_id=session.id,
        role="tool",
        content="Observation recorded: dizziness post-medication.",
    )
    assert m4.seq == 4

    # Verify session msg_count was updated
    refreshed_session = manager.get_session(TEST_PID, session.id)
    assert refreshed_session is not None
    assert refreshed_session.msg_count == 4

    # Verify ascending history retrieval (q_msg_01b)
    history_asc = manager.get_session_history(TEST_PID, session.id, order="ASC")
    assert len(history_asc) == 4
    assert [m.seq for m in history_asc] == [1, 2, 3, 4]
    assert history_asc[0].content == "Good morning, I feel dizzy."
    assert history_asc[3].role == "tool"

    # Verify descending history retrieval
    history_desc = manager.get_session_history(TEST_PID, session.id, order="DESC")
    assert [m.seq for m in history_desc] == [4, 3, 2, 1]


def test_concurrent_message_sequencing(session_fixture) -> None:
    """Verifies that concurrent appends produce strictly contiguous sequences without collisions."""
    manager, router, clock = session_fixture
    session = manager.create_session(patient_id=TEST_PID)

    num_messages = 20

    def _worker_append(idx: int):
        # Create thread-isolated router and manager for socket thread safety
        thread_router = TenantRouter(get_db_client())
        thread_mgr = SessionManager(router=thread_router, clock=clock, embedder=HashEmbedder())
        return thread_mgr.append_message(
            patient_id=TEST_PID,
            session_id=session.id,
            role="user" if idx % 2 == 0 else "agent",
            content=f"Concurrent message payload number {idx}",
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_worker_append, range(num_messages)))

    # Verify exactly 20 distinct sequences from 1 to 20
    seqs = sorted(m.seq for m in results)
    assert seqs == list(range(1, num_messages + 1)), f"Sequences violated: {seqs}"

    # Verify session counter in graph
    final_session = manager.get_session(TEST_PID, session.id)
    assert final_session is not None
    assert final_session.msg_count == num_messages


def test_compound_uniqueness_constraint_violation(session_fixture) -> None:
    """Verifies that FalkorDB strictly enforces Message(session_id, seq) compound uniqueness."""
    manager, router, clock = session_fixture
    session = manager.create_session(patient_id=TEST_PID)

    # First message seq=1
    manager.append_message(TEST_PID, session.id, role="user", content="First message")

    target_graph = router.graph(TEST_PID)
    # Attempt duplicate insert with seq=1
    with pytest.raises(ResponseError) as exc_info:
        target_graph.query(
            """
            CREATE (m:Message {
                id: 'msg_violator',
                session_id: $sid,
                seq: 1,
                role: 'user',
                content: 'Duplicate seq 1',
                ts: 12345
            })
            """,
            {"sid": session.id},
        )

    err_text = str(exc_info.value).lower()
    assert "already exists" in err_text or "violates" in err_text or "unique" in err_text


def test_message_vector_search(session_fixture) -> None:
    manager, router, clock = session_fixture
    session = manager.create_session(patient_id=TEST_PID)

    manager.append_message(
        TEST_PID,
        session.id,
        role="user",
        content="Patient presents with acute chest pain and shortness of breath.",
    )
    manager.append_message(
        TEST_PID,
        session.id,
        role="agent",
        content="Ordering an urgent 12-lead electrocardiogram.",
    )
    manager.append_message(
        TEST_PID,
        session.id,
        role="user",
        content="Patient inquired about parking validation voucher.",
    )

    # Search for cardiac symptoms
    results = manager.search_messages(
        TEST_PID,
        query_text="Patient presents with acute chest pain and shortness of breath.",
        session_id=session.id,
        top_k=3,
    )
    assert len(results) >= 1
    # Top result should match exact phrase with high similarity
    top_hit = results[0]
    assert "chest pain" in top_hit["content"]
    assert top_hit["similarity"] > 0.95


def test_validation_and_not_found_errors(session_fixture) -> None:
    manager, router, clock = session_fixture

    # Non-existent patient
    with pytest.raises(PatientNotFoundError):
        manager.create_session(patient_id="p_000000000000")

    # Invalid channel
    with pytest.raises(ValidationError):
        manager.create_session(patient_id=TEST_PID, channel="unsupported_channel")

    session = manager.create_session(patient_id=TEST_PID)

    # Invalid role
    with pytest.raises(ValidationError):
        manager.append_message(
            TEST_PID,
            session.id,
            role="unauthorized_role",
            content="Hello",
        )

    # Non-existent session
    with pytest.raises(SessionNotFoundError):
        manager.append_message(
            TEST_PID,
            session_id="ses_nonexistent00000000000",
            role="user",
            content="Hello",
        )

    with pytest.raises(SessionNotFoundError):
        manager.close_session(TEST_PID, session_id="ses_nonexistent00000000000")
