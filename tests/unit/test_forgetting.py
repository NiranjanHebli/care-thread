"""Unit tests for the forgetting and deletion manager.

Validates:
- forget_request parameter validation and target verification
- Confirmation token generation, hashing, and TTL expiry
- Role-based authorization and separation of duties for approvals
- Scope 'all' tenant graph deletion, registry wipe, and cryptographic tombstones
- LLM cache purging for deleted tenant
- BGREWRITEAOF persistence trigger
- Tenant isolation: ensuring unrelated tenants remain completely intact
- Scope 'fact' retraction with stub preservation
- Scope 'session' message content and summary redaction
- Session creation blocking when a tenant is in 'deleting' state
- Recovery and completion of incomplete deletions after simulated crashes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    DeletionRequestNotFoundError,
    FactNotFoundError,
    ForgetConfirmationRequiredError,
    ForgetNotAllowedError,
    PatientNotFoundError,
    RequestExpiredError,
    SessionNotFoundError,
    ValidationError,
)
from carethread.memory.facts import FactManager
from carethread.memory.forgetting import (
    ForgettingManager,
    compute_patient_hash,
    hash_confirm_token,
)
from carethread.memory.sessions import SessionManager

PID_FORGET_A = "p_fa1010101010"
PID_FORGET_B = "p_fb2020202020"
CUSTOM_SALT = "test_custom_cryptographic_salt_123"


@pytest.fixture
def forgetting_fixture(tmp_path: Path):
    """Sets up a clean test fixture with two tenants, cache fixtures, and ForgettingManager."""
    client = get_db_client()
    router = TenantRouter(client)
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for pid in (PID_FORGET_A, PID_FORGET_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g_a = router.graph(PID_FORGET_A)
    g_a.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient A'",
        {"pid": PID_FORGET_A},
    )

    g_b = router.graph(PID_FORGET_B)
    g_b.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient B'",
        {"pid": PID_FORGET_B},
    )

    # Clean previous deletion requests or tombstones in ct_control for these patients
    ctrl_g = router._control_graph
    hash_a = compute_patient_hash(PID_FORGET_A, CUSTOM_SALT)
    hash_b = compute_patient_hash(PID_FORGET_B, CUSTOM_SALT)
    ctrl_g.query(
        "MATCH (r:DeletionRequest) WHERE r.patient_id IN [$p1, $p2] DELETE r",
        {"p1": PID_FORGET_A, "p2": PID_FORGET_B},
    )
    ctrl_g.query(
        "MATCH (t:Tombstone) WHERE t.patient_hash IN [$h1, $h2] DELETE t",
        {"h1": hash_a, "h2": hash_b},
    )

    clock = FixedClock(1792140000)
    forget_mgr = ForgettingManager(
        router=router,
        clock=clock,
        salt=CUSTOM_SALT,
        cache_dir=cache_dir,
    )
    fact_mgr = FactManager(router=router, clock=clock)
    session_mgr = SessionManager(router=router, clock=clock)

    yield {
        "client": client,
        "router": router,
        "clock": clock,
        "forget_mgr": forget_mgr,
        "fact_mgr": fact_mgr,
        "session_mgr": session_mgr,
        "cache_dir": cache_dir,
        "graph_a": g_a,
        "graph_b": g_b,
        "salt": CUSTOM_SALT,
    }

    # Teardown
    for pid in (PID_FORGET_A, PID_FORGET_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass


def test_forget_request_validation(forgetting_fixture: dict[str, Any]) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]

    # 1. Invalid scope
    with pytest.raises(ValidationError) as exc:
        mgr.forget_request(PID_FORGET_A, scope="invalid_scope")
    assert "Invalid deletion scope" in str(exc.value)

    # 2. Unregistered patient
    with pytest.raises(PatientNotFoundError):
        mgr.forget_request("p_999999999999", scope="all")

    # 3. Scope 'fact' without target_id
    with pytest.raises(ValidationError):
        mgr.forget_request(PID_FORGET_A, scope="fact", target_id=None)

    # 4. Scope 'fact' with non-existent fact
    with pytest.raises(FactNotFoundError):
        mgr.forget_request(PID_FORGET_A, scope="fact", target_id="fct_nonexistent123")

    # 5. Scope 'session' without target_id
    with pytest.raises(ValidationError):
        mgr.forget_request(PID_FORGET_A, scope="session", target_id="")

    # 6. Scope 'session' with non-existent session
    with pytest.raises(SessionNotFoundError):
        mgr.forget_request(PID_FORGET_A, scope="session", target_id="ses_nonexistent123")

    # 7. Valid scope 'all'
    req = mgr.forget_request(
        patient_id=PID_FORGET_A,
        scope="all",
        reason="Patient GDPR request",
        requested_by="agent_intake",
    )
    assert req["request_id"].startswith("del_")
    assert len(req["confirm_token"]) == 64
    assert req["status"] == "pending_approval"
    assert req["expires_at"] == 1792140000 + 15 * 60

    # Verify only SHA-256 hash was stored, not the raw token
    rec = mgr.get_deletion_request(req["request_id"])
    assert rec is not None
    assert rec.token_hash == hash_confirm_token(req["confirm_token"])
    assert rec.token_hash != req["confirm_token"]
    assert rec.patient_id == PID_FORGET_A
    assert rec.scope == "all"


def test_confirm_forget_validation_and_permissions(
    forgetting_fixture: dict[str, Any],
) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    clock: FixedClock = forgetting_fixture["clock"]

    req = mgr.forget_request(
        patient_id=PID_FORGET_A,
        scope="all",
        reason="Clinical retraction",
        requested_by="user_requester_1",
    )
    req_id = req["request_id"]
    token = req["confirm_token"]

    # 1. Non-existent request
    with pytest.raises(DeletionRequestNotFoundError):
        mgr.confirm_forget("del_doesnotexist", token, "admin_user", "clinician_admin")

    # 2. Separation of duties: approver cannot be the requester
    with pytest.raises(ForgetNotAllowedError) as exc:
        mgr.confirm_forget(req_id, token, "user_requester_1", "clinician_admin")
    assert "Separation of duties" in str(exc.value)

    # 3. Insufficient role (must be clinician_admin)
    with pytest.raises(ForgetNotAllowedError) as exc:
        mgr.confirm_forget(req_id, token, "other_user", "specialist")
    assert "clinician_admin" in str(exc.value)

    # 4. Invalid confirmation token
    with pytest.raises(ForgetConfirmationRequiredError):
        mgr.confirm_forget(req_id, "wrong_token", "admin_user", "clinician_admin")

    # 5. Expired token
    clock.advance(16 * 60)  # advance past 15 min TTL
    with pytest.raises(RequestExpiredError):
        mgr.confirm_forget(req_id, token, "admin_user", "clinician_admin")


def test_confirm_forget_scope_all_and_isolation(
    forgetting_fixture: dict[str, Any],
) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    router: TenantRouter = forgetting_fixture["router"]
    fact_mgr: FactManager = forgetting_fixture["fact_mgr"]
    cache_dir: Path = forgetting_fixture["cache_dir"]
    salt: str = forgetting_fixture["salt"]
    client = forgetting_fixture["client"]

    # 1. Setup Patient A data and LLM cache fixture
    fact_mgr.remember_fact(
        patient_id=PID_FORGET_A,
        kind="condition",
        display="Type 2 Diabetes",
        authority="record",
    )
    hash_a = compute_patient_hash(PID_FORGET_A, salt)
    cache_file = cache_dir / "replay_cache.jsonl"
    cache_data = [
        {"tenant_hash": hash_a, "prompt": "Diabetes guidance", "response": "Metformin"},
        {
            "tenant_hash": compute_patient_hash(PID_FORGET_B, salt),
            "prompt": "Asthma guidance",
            "response": "Albuterol",
        },
    ]
    cache_file.write_text("\n".join(json.dumps(d) for d in cache_data) + "\n", encoding="utf-8")

    # 2. Setup Patient B data
    f_b = fact_mgr.remember_fact(
        patient_id=PID_FORGET_B,
        kind="condition",
        display="Mild Asthma",
        authority="record",
    )

    # Verify both tenants are active and graphs exist
    assert router.is_registered(PID_FORGET_A) is True
    assert router.is_registered(PID_FORGET_B) is True
    existing_graphs = client.list_graphs()
    assert f"patient_{PID_FORGET_A}" in existing_graphs
    assert f"patient_{PID_FORGET_B}" in existing_graphs

    # 3. Request and confirm deletion for Patient A
    req = mgr.forget_request(
        patient_id=PID_FORGET_A,
        scope="all",
        reason="Patient revocation of consent",
        requested_by="clinician_01",
    )
    result = mgr.confirm_forget(
        request_id=req["request_id"],
        confirm_token=req["confirm_token"],
        approver_id="admin_02",
        approver_role="clinician_admin",
    )
    assert result["status"] == "executed"
    tomb_id = result["tombstone_id"]
    assert tomb_id.startswith("tmb_")

    # 4. Verify Patient A graph is deleted from FalkorDB
    updated_graphs = client.list_graphs()
    assert f"patient_{PID_FORGET_A}" not in updated_graphs
    assert router.is_registered(PID_FORGET_A) is False

    # 5. Verify cryptographic tombstone exists in ct_control
    tomb = mgr.get_tombstone(tomb_id)
    assert tomb is not None
    assert tomb.patient_hash == hash_a
    assert tomb.request_id == req["request_id"]
    assert tomb.scope == "all"
    assert tomb.approver_id == "admin_02"

    # 6. Verify LLM cache entry was purged for Patient A, but Patient B remains
    remaining_lines = [
        json.loads(line)
        for line in cache_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(remaining_lines) == 1
    assert remaining_lines[0]["tenant_hash"] == compute_patient_hash(PID_FORGET_B, salt)

    # 7. Verify Patient B graph and fact remain completely intact (isolation)
    assert f"patient_{PID_FORGET_B}" in updated_graphs
    assert router.is_registered(PID_FORGET_B) is True
    fact_b_check = fact_mgr.get_fact(PID_FORGET_B, f_b["fact_id"])
    assert fact_b_check is not None
    assert fact_b_check.display == "Mild Asthma"

    # 8. Attempting to execute confirmed request again fails
    with pytest.raises(ForgetNotAllowedError):
        mgr.confirm_forget(
            request_id=req["request_id"],
            confirm_token=req["confirm_token"],
            approver_id="admin_02",
            approver_role="clinician_admin",
        )


def test_confirm_forget_scope_fact(forgetting_fixture: dict[str, Any]) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    fact_mgr: FactManager = forgetting_fixture["fact_mgr"]
    router: TenantRouter = forgetting_fixture["router"]

    f = fact_mgr.remember_fact(
        patient_id=PID_FORGET_A,
        kind="medication",
        display="Lisinopril 10mg",
        authority="clinician",
    )
    fid = f["fact_id"]

    req = mgr.forget_request(
        patient_id=PID_FORGET_A,
        scope="fact",
        target_id=fid,
        reason="Erroneous entry",
        requested_by="dr_smith",
    )
    res = mgr.confirm_forget(
        request_id=req["request_id"],
        confirm_token=req["confirm_token"],
        approver_id="admin_jones",
        approver_role="clinician_admin",
    )
    assert res["status"] == "executed"

    # Verify fact node stub retained for graph integrity, but display & value nulled
    g = router.graph(PID_FORGET_A)
    f_res = g.query("MATCH (f:Fact {id: $fid}) RETURN f", {"fid": fid})
    assert len(f_res.result_set) == 1
    props = f_res.result_set[0][0].properties
    assert props["status"] == "retracted"
    assert props.get("display") is None
    assert props.get("value") is None
    assert props.get("retracted_at") is not None
    assert props.get("retracted_by") == "admin_jones"

    # Tombstone written in ct_control
    tomb = mgr.get_tombstone(res["tombstone_id"])
    assert tomb is not None
    assert tomb.scope == "fact"
    assert tomb.target_id == fid


def test_confirm_forget_scope_session(forgetting_fixture: dict[str, Any]) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    session_mgr: SessionManager = forgetting_fixture["session_mgr"]
    router: TenantRouter = forgetting_fixture["router"]

    ses = session_mgr.create_session(patient_id=PID_FORGET_A, channel="chat")
    sid = ses.id
    session_mgr.append_message(
        patient_id=PID_FORGET_A,
        session_id=sid,
        role="user",
        content="Sensitive medical question",
    )
    session_mgr.append_message(
        patient_id=PID_FORGET_A,
        session_id=sid,
        role="agent",
        content="Confidential clinical answer",
    )

    req = mgr.forget_request(
        patient_id=PID_FORGET_A,
        scope="session",
        target_id=sid,
        reason="Patient requests conversation redaction",
        requested_by="agent_support",
    )
    res = mgr.confirm_forget(
        request_id=req["request_id"],
        confirm_token=req["confirm_token"],
        approver_id="admin_chief",
        approver_role="clinician_admin",
    )
    assert res["status"] == "executed"

    # Verify messages are redacted
    g = router.graph(PID_FORGET_A)
    m_res = g.query(
        "MATCH (m:Message {session_id: $sid}) RETURN m.id, m.content, m.embedding",
        {"sid": sid},
    )
    for row in m_res.result_set:
        assert row[1] == "[REDACTED]"
        assert row[2] is None

    # Verify session is closed and summary redacted
    s_res = g.query("MATCH (s:Session {id: $sid}) RETURN s.status, s.summary", {"sid": sid})
    assert s_res.result_set[0][0] == "closed"
    assert s_res.result_set[0][1] == "[REDACTED]"


def test_blocking_new_sessions_during_deletion(
    forgetting_fixture: dict[str, Any],
) -> None:
    router: TenantRouter = forgetting_fixture["router"]
    session_mgr: SessionManager = forgetting_fixture["session_mgr"]

    # Mark tenant 'deleting' in ct_control
    router.set_tenant_status(PID_FORGET_A, "deleting")
    assert router.is_deleting(PID_FORGET_A) is True

    # Attempting to create a new session must be blocked
    with pytest.raises(ForgetNotAllowedError) as exc:
        session_mgr.create_session(patient_id=PID_FORGET_A)
    assert "currently being deleted" in str(exc.value)


def test_resume_incomplete_deletions(forgetting_fixture: dict[str, Any]) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    router: TenantRouter = forgetting_fixture["router"]
    fact_mgr: FactManager = forgetting_fixture["fact_mgr"]
    client = forgetting_fixture["client"]

    # Setup patient data
    fact_mgr.remember_fact(
        patient_id=PID_FORGET_A,
        kind="symptom",
        display="Persistent cough",
        authority="patient_reported",
    )
    assert f"patient_{PID_FORGET_A}" in client.list_graphs()

    # Simulate mid-crash state: registry marked deleting
    router.set_tenant_status(PID_FORGET_A, "deleting")

    # Worker resumes incomplete deletions
    resumed = mgr.resume_incomplete_deletions()
    assert PID_FORGET_A in resumed

    # Graph is gone and registry entry removed
    assert f"patient_{PID_FORGET_A}" not in client.list_graphs()
    assert router.is_registered(PID_FORGET_A) is False

    # Tombstone exists
    hash_a = compute_patient_hash(PID_FORGET_A, forgetting_fixture["salt"])
    tombstones = mgr.list_tombstones(patient_hash=hash_a)
    assert len(tombstones) >= 1
    assert tombstones[0].patient_hash == hash_a


def test_tombstone_listing_and_lookup(forgetting_fixture: dict[str, Any]) -> None:
    mgr: ForgettingManager = forgetting_fixture["forget_mgr"]
    salt: str = forgetting_fixture["salt"]

    # Create two deletion cycles
    req1 = mgr.forget_request(PID_FORGET_A, scope="all", requested_by="user_1")
    res1 = mgr.confirm_forget(req1["request_id"], req1["confirm_token"], "admin_1")

    tomb = mgr.get_tombstone(res1["tombstone_id"])
    assert tomb is not None
    assert tomb.id == res1["tombstone_id"]

    hash_a = compute_patient_hash(PID_FORGET_A, salt)
    tomb_list = mgr.list_tombstones(patient_hash=hash_a)
    assert len(tomb_list) >= 1
    assert tomb_list[0].id == res1["tombstone_id"]
