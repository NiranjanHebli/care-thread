"""Tenant isolation tests verifying cross-tenant mismatch detection (T-ISO-01 and T-ISO-02).

Validates:
- T-ISO-01: Session bound to patient A; calling recall, remember, history, explain
  with patient B's ID or B's node IDs returns TENANT_MISMATCH or MEMORY_ID_NOT_FOUND,
  returns zero patient B data in outputs, and logs a denied audit event for each attempt.
- T-ISO-02: Token scoped to patient A; calling start_session for patient B returns
  TENANT_MISMATCH and records a denied audit event.
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher

PID_A = "p_150a00000001"
PID_B = "p_150b00000002"
SECRET_CANARY_B = "canary_b_secret_allergy_999"


@pytest.fixture
def mismatch_fixture():
    """Sets up two isolated tenant graphs and dispatcher with scoped tokens."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (PID_A, PID_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g_a = router.graph(PID_A)
    g_a.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient A'",
        {"pid": PID_A},
    )

    g_b = router.graph(PID_B)
    g_b.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient B'",
        {"pid": PID_B},
    )

    # Clean ct_control audit events
    ctrl_g = router._control_graph
    try:
        ctrl_g.query("MATCH (a:AuditEvent) DELETE a")
    except Exception:
        pass

    token_store = TokenStore()
    token_store.register_token("tok_global_intake", "agt_intake_01", "intake", "*")
    token_store.register_token("tok_global_clinician", "agt_clnc_01", "clinician", "*")
    token_store.register_token("tok_scoped_a", "agt_scoped_a", "intake", [PID_A])

    clock = FixedClock(1792180000)
    dispatcher = ToolDispatcher(
        router=router,
        token_store=token_store,
        clock=clock,
    )

    # 1. Start Session for Patient A
    s_a = dispatcher.dispatch(
        "start_session",
        {"patient_id": PID_A, "channel": "chat"},
        token="tok_global_intake",
    )
    assert s_a["ok"] is True
    sid_a = s_a["data"]["session_id"]

    # 2. Start Session for Patient B and seed fact B
    s_b = dispatcher.dispatch(
        "start_session",
        {"patient_id": PID_B, "channel": "chat"},
        token="tok_global_intake",
    )
    assert s_b["ok"] is True
    sid_b = s_b["data"]["session_id"]

    rem_b = dispatcher.dispatch(
        "remember",
        {
            "session_id": sid_b,
            "kind": "fact",
            "payload": {
                "kind": "allergy",
                "display": "Penicillin",
                "value": SECRET_CANARY_B,
                "authority": "patient_reported",
            },
        },
        token="tok_global_intake",
    )
    assert rem_b["ok"] is True
    fact_b_id = rem_b["data"]["fact_id"]

    # 3. Commit Answer for Patient B
    ans_b = dispatcher.dispatch(
        "commit_answer",
        {
            "session_id": sid_b,
            "question": "What allergies does patient B have?",
            "answer_text": f"Patient has allergy: {SECRET_CANARY_B}",
            "claims": [
                {
                    "text": f"Allergy {SECRET_CANARY_B}",
                    "citations": [fact_b_id],
                }
            ],
            "allow_unsupported": True,
        },
        token="tok_global_intake",
    )
    assert ans_b["ok"] is True
    answer_b_id = ans_b["data"]["answer_id"]

    yield {
        "client": client,
        "router": router,
        "dispatcher": dispatcher,
        "clock": clock,
        "sid_a": sid_a,
        "sid_b": sid_b,
        "fact_b_id": fact_b_id,
        "answer_b_id": answer_b_id,
        "graph_a": g_a,
        "graph_b": g_b,
    }

    # Teardown
    for pid in (PID_A, PID_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
    try:
        ctrl_g.query("MATCH (a:AuditEvent) DELETE a")
    except Exception:
        pass


def test_t_iso_01_cross_tenant_mismatch_denied(mismatch_fixture: dict[str, Any]) -> None:
    """T-ISO-01: Session bound to patient A rejected when calling with patient B identifiers."""
    disp: ToolDispatcher = mismatch_fixture["dispatcher"]
    sid_a: str = mismatch_fixture["sid_a"]
    fact_b_id: str = mismatch_fixture["fact_b_id"]
    answer_b_id: str = mismatch_fixture["answer_b_id"]
    router: TenantRouter = mismatch_fixture["router"]

    # 1. Call recall with session A but specifying patient B ID
    res_recall = disp.dispatch(
        "recall",
        {"session_id": sid_a, "patient_id": PID_B, "query": "allergy"},
        token="tok_global_intake",
    )
    assert res_recall["ok"] is False
    assert res_recall["error"]["code"] == "TENANT_MISMATCH"
    assert SECRET_CANARY_B not in str(res_recall)

    # 2. Call remember with session A but specifying patient B ID
    res_rem = disp.dispatch(
        "remember",
        {
            "session_id": sid_a,
            "patient_id": PID_B,
            "kind": "fact",
            "payload": {"kind": "condition", "display": "Asthma"},
        },
        token="tok_global_intake",
    )
    assert res_rem["ok"] is False
    assert res_rem["error"]["code"] == "TENANT_MISMATCH"

    # 3. Call history with session A specifying patient B's fact ID
    res_hist = disp.dispatch(
        "history",
        {"session_id": sid_a, "fact_id": fact_b_id},
        token="tok_global_intake",
    )
    assert res_hist["ok"] is False
    assert res_hist["error"]["code"] in ("TENANT_MISMATCH", "MEMORY_ID_NOT_FOUND")
    assert SECRET_CANARY_B not in str(res_hist)

    # 4. Call explain with session A specifying patient B's answer ID
    res_exp = disp.dispatch(
        "explain",
        {"session_id": sid_a, "answer_id": answer_b_id},
        token="tok_global_intake",
    )
    assert res_exp["ok"] is False
    assert res_exp["error"]["code"] in ("TENANT_MISMATCH", "ANSWER_NOT_FOUND")
    assert SECRET_CANARY_B not in str(res_exp)

    # 5. Verify denied audit events are logged
    # Query ct_control for denied events recorded for these failed attempts
    ctrl_g = router._control_graph
    denied_ctrl = ctrl_g.query(
        "MATCH (a:AuditEvent {op: 'denied'}) RETURN count(a) AS cnt"
    ).result_set[0][0]

    # Query patient A graph for denied events
    g_a = router.graph(PID_A)
    denied_a = g_a.query("MATCH (a:AuditEvent {op: 'denied'}) RETURN count(a) AS cnt").result_set[
        0
    ][0]

    total_denied = denied_ctrl + denied_a
    assert total_denied >= 4, f"Expected at least 4 denied audit events, found {total_denied}"


def test_t_iso_02_scoped_token_tenant_mismatch(mismatch_fixture: dict[str, Any]) -> None:
    """T-ISO-02: Token scoped to patient A rejected when starting session for patient B."""
    disp: ToolDispatcher = mismatch_fixture["dispatcher"]
    router: TenantRouter = mismatch_fixture["router"]

    ctrl_g = router._control_graph
    before_denied = ctrl_g.query(
        "MATCH (a:AuditEvent {op: 'denied'}) RETURN count(a) AS cnt"
    ).result_set[0][0]

    res = disp.dispatch(
        "start_session",
        {"patient_id": PID_B, "channel": "chat"},
        token="tok_scoped_a",
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "TENANT_MISMATCH"

    after_denied = ctrl_g.query(
        "MATCH (a:AuditEvent {op: 'denied'}) RETURN count(a) AS cnt"
    ).result_set[0][0]

    assert after_denied > before_denied, (
        "Expected denied audit event to be logged for scoped token mismatch"
    )
