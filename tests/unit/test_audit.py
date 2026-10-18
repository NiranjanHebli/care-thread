"""Unit tests for the outpatient clinical audit logging engine.

Validates:
- Synchronous write audit event recording via query q_aud_01
- Synchronous access-denied and policy violation event recording
- In-memory buffering and batched flushing of read audit events
- Automatic threshold-based flush when buffer reaches batch_size
- Targeted patient flush and universal flush
- Context manager auto-flushing on exit
- AuditScope context manager measuring latency and handling errors
- Filtered audit event queries and pagination
- Trace verification aggregating multi-agent interaction trails
- Argument cryptographic hashing (SHA-256)
- Input validation and exception handling
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    AuditEventNotFoundError,
    PatientNotFoundError,
    ValidationError,
)
from carethread.memory.audit import (
    AuditLogger,
    AuditManager,
    compute_args_hash,
)

TEST_PID_A = "p_111122223333"
TEST_PID_B = "p_444455556666"


@pytest.fixture
def audit_fixture():
    """Sets up a clean test fixture with two bootstrapped patient tenants."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (TEST_PID_A, TEST_PID_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g_a = router.graph(TEST_PID_A)
    g_a.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient A'",
        {"pid": TEST_PID_A},
    )

    g_b = router.graph(TEST_PID_B)
    g_b.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient B'",
        {"pid": TEST_PID_B},
    )

    clock = FixedClock(1792137000)
    logger = AuditLogger(router=router, clock=clock, batch_size=3)

    yield {
        "router": router,
        "clock": clock,
        "logger": logger,
        "graph_a": g_a,
        "graph_b": g_b,
    }

    for pid in (TEST_PID_A, TEST_PID_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass


def test_compute_args_hash_deterministic():
    """Verify compute_args_hash produces deterministic SHA-256 digests."""
    h1 = compute_args_hash({"slot": "med:metformin", "dose": 500})
    h2 = compute_args_hash({"dose": 500, "slot": "med:metformin"})
    assert h1 == h2
    assert len(h1) == 64

    # None input
    h_none = compute_args_hash(None)
    assert len(h_none) == 64
    assert h_none != h1


def test_synchronous_write(audit_fixture: dict[str, Any]):
    """Verify log_write persists immediately to graph via query q_aud_01."""
    audit_log: AuditLogger = audit_fixture["logger"]

    rec = audit_log.log_write(
        patient_id=TEST_PID_A,
        agent_id="agent_intake_01",
        role="intake",
        tool="write_fact",
        outcome="success",
        args={"slot": "med:metformin", "status": "active"},
        read_ids=["fct_old_1"],
        trace_id="trc_session_01",
        latency_ms=15.4,
    )

    assert rec.id.startswith("aud_")
    assert rec.patient_id == TEST_PID_A
    assert rec.op == "write"
    assert rec.ts == 1792137000
    assert rec.agent_id == "agent_intake_01"
    assert rec.role == "intake"
    assert rec.tool == "write_fact"
    assert rec.outcome == "success"
    assert rec.read_ids == ["fct_old_1"]
    assert rec.trace_id == "trc_session_01"
    assert rec.latency_ms == 15.4

    # Verify directly in FalkorDB patient graph without any flush
    graph_a = audit_fixture["graph_a"]
    check = graph_a.query(
        "MATCH (e:AuditEvent {id:$eid}) RETURN e.op, e.tool, e.args_hash",
        {"eid": rec.id},
    )
    assert len(check.result_set) == 1
    assert check.result_set[0][0] == "write"
    assert check.result_set[0][1] == "write_fact"


def test_synchronous_denied(audit_fixture: dict[str, Any]):
    """Verify log_denied persists access violations synchronously."""
    audit_log: AuditLogger = audit_fixture["logger"]

    rec = audit_log.log_denied(
        patient_id=TEST_PID_A,
        agent_id="agent_pharmacy_01",
        role="pharmacy",
        tool="retract_note",
        reason="insufficient_permissions",
        outcome="denied",
        args={"note_id": "not_123"},
        trace_id="trc_sec_01",
        latency_ms=3.2,
    )

    assert rec.op == "denied"
    assert rec.outcome == "denied"

    # Query back
    fetched = audit_log.get_audit_event(TEST_PID_A, rec.id)
    assert fetched.id == rec.id
    assert fetched.op == "denied"
    assert fetched.role == "pharmacy"


def test_batched_read_buffering_and_auto_flush(audit_fixture: dict[str, Any]):
    """Verify read events buffer in memory and flush when batch_size is reached."""
    audit_log: AuditLogger = audit_fixture["logger"]
    assert audit_log.batch_size == 3

    # First read: buffered, not in graph
    audit_log.log_read(
        patient_id=TEST_PID_A,
        agent_id="agent_specialist_01",
        role="specialist",
        tool="recall",
        read_ids=["fct_1"],
    )
    assert audit_log.pending_count(TEST_PID_A) == 1
    assert audit_log.count_audit_events(TEST_PID_A, flush_first=False) == 0

    # Second read: still buffered
    audit_log.log_read(
        patient_id=TEST_PID_A,
        agent_id="agent_specialist_01",
        role="specialist",
        tool="get_patient",
        read_ids=[],
    )
    assert audit_log.pending_count(TEST_PID_A) == 2
    assert audit_log.count_audit_events(TEST_PID_A, flush_first=False) == 0

    # Third read: reaches batch_size=3 -> auto flushes!
    audit_log.log_read(
        patient_id=TEST_PID_A,
        agent_id="agent_specialist_01",
        role="specialist",
        tool="list_flags",
        read_ids=["flg_1"],
    )
    assert audit_log.pending_count(TEST_PID_A) == 0
    assert audit_log.count_audit_events(TEST_PID_A, flush_first=False) == 3


def test_explicit_patient_flush(audit_fixture: dict[str, Any]):
    """Verify flush(patient_id) flushes only the targeted patient."""
    audit_log: AuditLogger = audit_fixture["logger"]

    # Log reads for Patient A and Patient B
    audit_log.log_read(
        patient_id=TEST_PID_A,
        agent_id="agent_intake_01",
        role="intake",
        tool="search",
    )
    audit_log.log_read(
        patient_id=TEST_PID_B,
        agent_id="agent_specialist_01",
        role="specialist",
        tool="search",
    )

    assert audit_log.pending_count(TEST_PID_A) == 1
    assert audit_log.pending_count(TEST_PID_B) == 1

    # Flush only Patient A
    flushed_a = audit_log.flush(TEST_PID_A)
    assert flushed_a == 1
    assert audit_log.pending_count(TEST_PID_A) == 0
    assert audit_log.pending_count(TEST_PID_B) == 1

    # Now flush remaining
    flushed_all = audit_log.flush()
    assert flushed_all == 1
    assert audit_log.pending_count() == 0


def test_context_manager_lifecycle(audit_fixture: dict[str, Any]):
    """Verify AuditLogger auto-flushes all pending reads upon context exit."""
    router: TenantRouter = audit_fixture["router"]
    clock: FixedClock = audit_fixture["clock"]

    # Use a large batch size so auto-flush does not trigger inside the block
    with AuditLogger(router=router, clock=clock, batch_size=50) as log:
        log.log_read(
            patient_id=TEST_PID_A,
            agent_id="agent_intake_01",
            role="intake",
            tool="check_vitals",
        )
        assert log.pending_count(TEST_PID_A) == 1

    # Outside context block: buffer was flushed
    events = log.get_audit_events(TEST_PID_A, flush_first=False)
    assert len(events) >= 1
    assert events[0].tool == "check_vitals"


def test_audit_scope_success_and_error(audit_fixture: dict[str, Any]):
    """Verify AuditScope measures latency and records outcomes."""
    audit_log: AuditLogger = audit_fixture["logger"]

    # Successful scope
    with audit_log.audit_scope(
        patient_id=TEST_PID_A,
        agent_id="agent_clinician_01",
        role="clinician",
        tool="complex_review",
        op="write",
        trace_id="trc_rev_01",
    ) as scope:
        scope.record_read_ids(["fct_10", "fct_11"])
        time.sleep(0.01)

    evts = audit_log.get_audit_events(TEST_PID_A, trace_id="trc_rev_01")
    assert len(evts) == 1
    assert evts[0].outcome == "success"
    assert evts[0].read_ids == ["fct_10", "fct_11"]
    assert evts[0].latency_ms >= 5.0

    # Scope capturing an unhandled exception
    with pytest.raises(RuntimeError):
        with audit_log.audit_scope(
            patient_id=TEST_PID_A,
            agent_id="agent_clinician_01",
            role="clinician",
            tool="failing_action",
            op="write",
            trace_id="trc_rev_02",
        ):
            raise RuntimeError("Database connection glitch")

    fail_evts = audit_log.get_audit_events(TEST_PID_A, trace_id="trc_rev_02")
    assert len(fail_evts) == 1
    assert fail_evts[0].outcome == "error"


def test_get_audit_events_filters(audit_fixture: dict[str, Any]):
    """Verify get_audit_events filters by op, role, trace_id, and limits."""
    audit_log: AuditLogger = audit_fixture["logger"]

    audit_log.log_write(
        patient_id=TEST_PID_A,
        agent_id="a1",
        role="intake",
        tool="t1",
        trace_id="t_flow",
    )
    audit_log.log_write(
        patient_id=TEST_PID_A,
        agent_id="a2",
        role="specialist",
        tool="t2",
        trace_id="t_flow",
    )
    audit_log.log_denied(
        patient_id=TEST_PID_A,
        agent_id="a3",
        role="pharmacy",
        tool="t3",
        trace_id="t_other",
    )

    # Filter by op
    writes = audit_log.get_audit_events(TEST_PID_A, op="write")
    assert all(e.op == "write" for e in writes)

    # Filter by role
    pharm = audit_log.get_audit_events(TEST_PID_A, role="pharmacy")
    assert len(pharm) == 1
    assert pharm[0].role == "pharmacy"

    # Filter by trace_id
    flow = audit_log.get_audit_events(TEST_PID_A, trace_id="t_flow")
    assert len(flow) == 2


def test_verify_audit_trail(audit_fixture: dict[str, Any]):
    """Verify verify_audit_trail aggregates cross-agent execution telemetry."""
    audit_log: AuditLogger = audit_fixture["logger"]
    trace = "trc_multistep_consult"

    audit_log.log_write(
        patient_id=TEST_PID_A,
        agent_id="agent_intake_01",
        role="intake",
        tool="t_intake",
        trace_id=trace,
        latency_ms=10.0,
    )
    audit_log.log_read(
        patient_id=TEST_PID_A,
        agent_id="agent_specialist_01",
        role="specialist",
        tool="t_specialist",
        trace_id=trace,
        latency_ms=25.0,
    )
    audit_log.log_write(
        patient_id=TEST_PID_A,
        agent_id="agent_pharmacy_01",
        role="pharmacy",
        tool="t_pharmacy",
        trace_id=trace,
        latency_ms=15.0,
    )

    report = audit_log.verify_audit_trail(TEST_PID_A, trace)
    assert report["trace_id"] == trace
    assert report["total_events"] == 3
    assert report["op_counts"]["write"] == 2
    assert report["op_counts"]["read"] == 1
    assert report["total_latency_ms"] == 50.0
    assert report["roles"] == ["intake", "pharmacy", "specialist"]
    assert report["tools"] == ["t_intake", "t_pharmacy", "t_specialist"]


def test_validation_and_not_found(audit_fixture: dict[str, Any]):
    """Verify error checking for invalid roles, ops, and missing events/patients."""
    audit_log: AuditLogger = audit_fixture["logger"]

    # Invalid patient format
    with pytest.raises(ValidationError):
        audit_log.log_write(
            patient_id="invalid_pid",
            agent_id="a1",
            role="intake",
            tool="tool1",
        )

    # Invalid role
    with pytest.raises(ValidationError):
        audit_log.log_write(
            patient_id=TEST_PID_A,
            agent_id="a1",
            role="hacker_role",
            tool="tool1",
        )

    # Empty tool
    with pytest.raises(ValidationError):
        audit_log.log_write(
            patient_id=TEST_PID_A,
            agent_id="a1",
            role="intake",
            tool="",
        )

    # Non-existent patient for write
    with pytest.raises(PatientNotFoundError):
        audit_log.log_write(
            patient_id="p_999988887777",
            agent_id="a1",
            role="intake",
            tool="tool1",
        )

    # Audit event not found
    with pytest.raises(AuditEventNotFoundError):
        audit_log.get_audit_event(TEST_PID_A, "aud_nonexistent123")


def test_audit_manager_alias(audit_fixture: dict[str, Any]):
    """Verify AuditManager is a functional alias for AuditLogger."""
    router: TenantRouter = audit_fixture["router"]
    mgr = AuditManager(router=router)
    assert isinstance(mgr, AuditLogger)
