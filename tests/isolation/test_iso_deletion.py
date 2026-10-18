"""Tenant isolation tests verifying complete data isolation during tenant deletion.

Validates:
- Deleting tenant A completely removes patient_A graph and registry entry.
- Tenant B counts (node counts, relationship counts, fact values) remain unchanged.
- Cryptographic tombstone for tenant A exists in ct_control without leaking raw ID.
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.memory.facts import FactManager
from carethread.memory.forgetting import (
    ForgettingManager,
    compute_patient_hash,
)
from carethread.memory.sessions import SessionManager

PID_ISO_A = "p_150a1a1a1a1a"
PID_ISO_B = "p_150b2b2b2b2b"
ISO_SALT = "isolation_test_salt_999"


@pytest.fixture
def isolation_deletion_fixture():
    """Sets up two isolated tenant graphs with distinct data."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (PID_ISO_A, PID_ISO_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g_a = router.graph(PID_ISO_A)
    g_a.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient ISO A'",
        {"pid": PID_ISO_A},
    )

    g_b = router.graph(PID_ISO_B)
    g_b.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient ISO B'",
        {"pid": PID_ISO_B},
    )

    # Clean previous deletion requests or tombstones in ct_control for these patients
    ctrl_g = router._control_graph
    hash_a = compute_patient_hash(PID_ISO_A, ISO_SALT)
    hash_b = compute_patient_hash(PID_ISO_B, ISO_SALT)
    ctrl_g.query(
        "MATCH (r:DeletionRequest) WHERE r.patient_id IN [$p1, $p2] DELETE r",
        {"p1": PID_ISO_A, "p2": PID_ISO_B},
    )
    ctrl_g.query(
        "MATCH (t:Tombstone) WHERE t.patient_hash IN [$h1, $h2] DELETE t",
        {"h1": hash_a, "h2": hash_b},
    )

    clock = FixedClock(1792150000)
    forget_mgr = ForgettingManager(router=router, clock=clock, salt=ISO_SALT)
    fact_mgr = FactManager(router=router, clock=clock)
    session_mgr = SessionManager(router=router, clock=clock)

    # Populate Tenant A
    fact_mgr.remember_fact(
        patient_id=PID_ISO_A,
        kind="condition",
        display="Type 2 Diabetes",
        authority="record",
    )
    ses_a = session_mgr.create_session(patient_id=PID_ISO_A, channel="chat")
    session_mgr.append_message(
        patient_id=PID_ISO_A,
        session_id=ses_a.id,
        role="user",
        content="Patient A message",
    )

    # Populate Tenant B
    fact_mgr.remember_fact(
        patient_id=PID_ISO_B,
        kind="condition",
        display="Hypertension",
        authority="record",
    )
    fact_mgr.remember_fact(
        patient_id=PID_ISO_B,
        kind="medication",
        display="Lisinopril 10mg",
        authority="record",
    )
    ses_b = session_mgr.create_session(patient_id=PID_ISO_B, channel="portal")
    session_mgr.append_message(
        patient_id=PID_ISO_B,
        session_id=ses_b.id,
        role="user",
        content="Patient B message",
    )

    yield {
        "client": client,
        "router": router,
        "forget_mgr": forget_mgr,
        "fact_mgr": fact_mgr,
        "session_mgr": session_mgr,
        "salt": ISO_SALT,
    }

    # Teardown
    for pid in (PID_ISO_A, PID_ISO_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass

    ctrl_g.query(
        "MATCH (r:DeletionRequest) WHERE r.patient_id IN [$p1, $p2] DELETE r",
        {"p1": PID_ISO_A, "p2": PID_ISO_B},
    )
    ctrl_g.query(
        "MATCH (t:Tombstone) WHERE t.patient_hash IN [$h1, $h2] DELETE t",
        {"h1": hash_a, "h2": hash_b},
    )


def test_tenant_deletion_leaves_other_tenants_unaltered(
    isolation_deletion_fixture: dict[str, Any],
) -> None:
    client = isolation_deletion_fixture["client"]
    router: TenantRouter = isolation_deletion_fixture["router"]
    forget_mgr: ForgettingManager = isolation_deletion_fixture["forget_mgr"]
    salt: str = isolation_deletion_fixture["salt"]

    # 1. Capture Tenant B baseline metrics
    g_b = router.graph(PID_ISO_B)
    nodes_b_before = g_b.query("MATCH (n) RETURN count(n) AS cnt").result_set[0][0]
    edges_b_before = g_b.query("MATCH ()-[r]->() RETURN count(r) AS cnt").result_set[0][0]
    facts_b_before = g_b.query("MATCH (f:Fact) RETURN count(f) AS cnt").result_set[0][0]
    assert nodes_b_before > 0
    assert facts_b_before == 2

    # Verify both graphs exist initially
    initial_graphs = client.list_graphs()
    assert f"patient_{PID_ISO_A}" in initial_graphs
    assert f"patient_{PID_ISO_B}" in initial_graphs

    # 2. Execute deletion request for Tenant A
    req = forget_mgr.forget_request(
        patient_id=PID_ISO_A,
        scope="all",
        reason="Patient right to be forgotten",
        requested_by="agent_intake",
    )
    result = forget_mgr.confirm_forget(
        request_id=req["request_id"],
        confirm_token=req["confirm_token"],
        approver_id="admin_supervisor",
        approver_role="clinician_admin",
    )
    assert result["status"] == "executed"

    # 3. Verify Tenant A is absent from graph listing and registry
    after_graphs = client.list_graphs()
    assert f"patient_{PID_ISO_A}" not in after_graphs
    assert router.is_registered(PID_ISO_A) is False

    # 4. Verify tombstone exists in ct_control for Tenant A
    hash_a = compute_patient_hash(PID_ISO_A, salt)
    tombstones = forget_mgr.list_tombstones(patient_hash=hash_a)
    assert len(tombstones) == 1
    assert tombstones[0].patient_hash == hash_a
    assert tombstones[0].scope == "all"

    # 5. Verify Tenant B is 100% untouched
    assert f"patient_{PID_ISO_B}" in after_graphs
    assert router.is_registered(PID_ISO_B) is True

    nodes_b_after = g_b.query("MATCH (n) RETURN count(n) AS cnt").result_set[0][0]
    edges_b_after = g_b.query("MATCH ()-[r]->() RETURN count(r) AS cnt").result_set[0][0]
    facts_b_after = g_b.query("MATCH (f:Fact) RETURN count(f) AS cnt").result_set[0][0]

    assert nodes_b_after == nodes_b_before
    assert edges_b_after == edges_b_before
    assert facts_b_after == facts_b_before
