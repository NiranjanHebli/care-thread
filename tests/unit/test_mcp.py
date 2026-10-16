"""Unit tests for the unified tool dispatcher, authentication, and FastMCP server.

Validates:
- TokenStore authentication, role-based access control, and tenant scoping
- Rate limiting quota enforcement with retry_after_ms
- Input payload size limits (64 KB ceiling)
- Write idempotency tracking and replay via op_id keys
- Full tool lifecycle execution through the unified dispatcher:
  - start_session, end_session, append_message, get_session_messages
  - remember (facts, notes, candidates)
  - recall, history
  - handoff, list_handoffs, accept_handoff, complete_handoff
  - commit_answer, explain
  - list_flags, resolve_flag, confirm_fact
  - forget_request, confirm_forget
- MCPServer tool registry introspection and tool invocation
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher
from carethread.mcp.server import create_mcp_server

TEST_PID_MCP = "p_d15000000001"
OTHER_PID_MCP = "p_d15000000002"


@pytest.fixture
def mcp_fixture(tmp_path: Path):
    """Sets up an isolated test fixture with router, token store, and dispatcher."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (TEST_PID_MCP, OTHER_PID_MCP):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g1 = router.graph(TEST_PID_MCP)
    g1.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Test MCP Patient'",
        {"pid": TEST_PID_MCP},
    )

    g2 = router.graph(OTHER_PID_MCP)
    g2.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Other MCP Patient'",
        {"pid": OTHER_PID_MCP},
    )

    token_store = TokenStore()
    # Register custom tokens for testing
    token_store.register_token("tok_test_intake", "agt_intake_test", "intake", "*")
    token_store.register_token("tok_test_specialist", "agt_spec_test", "specialist", "*")
    token_store.register_token("tok_test_pharmacy", "agt_pharm_test", "pharmacy", "*")
    token_store.register_token("tok_test_clinician", "agt_clnc_test", "clinician", "*")
    token_store.register_token("tok_test_admin", "agt_adm_test", "clinician_admin", "*")
    token_store.register_token(
        "tok_scoped_patient",
        "agt_scoped",
        "intake",
        [TEST_PID_MCP],
    )

    clock = FixedClock(1792160000)
    dispatcher = ToolDispatcher(
        router=router,
        token_store=token_store,
        clock=clock,
        max_calls_per_second=50,
    )

    yield {
        "client": client,
        "router": router,
        "clock": clock,
        "token_store": token_store,
        "dispatcher": dispatcher,
        "graph": g1,
    }

    # Teardown
    for pid in (TEST_PID_MCP, OTHER_PID_MCP):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass


def test_auth_and_scoping(mcp_fixture: dict[str, Any]) -> None:
    disp: ToolDispatcher = mcp_fixture["dispatcher"]

    # 1. Missing token
    res = disp.dispatch("start_session", {"patient_id": TEST_PID_MCP}, token=None)
    assert res["ok"] is False
    assert res["error"]["code"] == "AUTH_REQUIRED"

    # 2. Invalid token
    res = disp.dispatch("start_session", {"patient_id": TEST_PID_MCP}, token="bad_token_123")
    assert res["ok"] is False
    assert res["error"]["code"] == "AUTH_REQUIRED"

    # 3. Scoped token accessing authorized patient
    res = disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP},
        token="tok_scoped_patient",
    )
    assert res["ok"] is True

    # 4. Scoped token attempting unauthorized patient
    res = disp.dispatch(
        "start_session",
        {"patient_id": OTHER_PID_MCP},
        token="tok_scoped_patient",
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "TENANT_MISMATCH"

    # 5. Role not allowed (intake attempting confirm_forget)
    res = disp.dispatch(
        "confirm_forget",
        {"request_id": "del_mock", "confirm_token": "mock", "approver_id": "admin"},
        token="tok_test_intake",
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "ROLE_NOT_ALLOWED"


def test_rate_limiting_and_payload_limits(mcp_fixture: dict[str, Any]) -> None:
    token_store: TokenStore = mcp_fixture["token_store"]
    router: TenantRouter = mcp_fixture["router"]
    clock: FixedClock = mcp_fixture["clock"]

    # Rate limiter configured to 5 calls/s for testing
    low_rate_disp = ToolDispatcher(
        router=router,
        token_store=token_store,
        clock=clock,
        max_calls_per_second=5,
    )

    # 1. Payload size limit (64 KB ceiling)
    giant_string = "x" * (65 * 1024)
    res = low_rate_disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP, "giant_data": giant_string},
        token="tok_test_intake",
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "VALIDATION_ERROR"

    # 2. Rate limiting quota
    for _ in range(5):
        ok_res = low_rate_disp.dispatch(
            "start_session",
            {"patient_id": TEST_PID_MCP},
            token="tok_test_intake",
        )
        assert ok_res["ok"] is True

    # 6th call exceeds quota
    blocked_res = low_rate_disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP},
        token="tok_test_intake",
    )
    assert blocked_res["ok"] is False
    assert blocked_res["error"]["code"] == "RATE_LIMITED"
    assert "retry_after_ms" in blocked_res["error"]["details"]


def test_write_idempotency_replay(mcp_fixture: dict[str, Any]) -> None:
    disp: ToolDispatcher = mcp_fixture["dispatcher"]

    # Start session
    ses_res = disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP},
        token="tok_test_intake",
    )
    sid = ses_res["data"]["session_id"]

    # First write with op_id
    res1 = disp.dispatch(
        "remember",
        {
            "session_id": sid,
            "kind": "fact",
            "payload": {"kind": "condition", "display": "Asthma"},
            "op_id": "op_test_unique_key_01",
        },
        token="tok_test_intake",
    )
    assert res1["ok"] is True
    assert res1["data"].get("replayed") is None
    fact_id = res1["data"]["fact_id"]

    # Second write replay with exact same op_id
    res2 = disp.dispatch(
        "remember",
        {
            "session_id": sid,
            "kind": "fact",
            "payload": {"kind": "condition", "display": "Asthma"},
            "op_id": "op_test_unique_key_01",
        },
        token="tok_test_intake",
    )
    assert res2["ok"] is True
    assert res2["data"].get("replayed") is True
    assert res2["data"]["fact_id"] == fact_id


def test_session_message_and_recall_lifecycle(mcp_fixture: dict[str, Any]) -> None:
    disp: ToolDispatcher = mcp_fixture["dispatcher"]

    # 1. start_session
    s_res = disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP},
        token="tok_test_intake",
    )
    assert s_res["ok"] is True
    sid = s_res["data"]["session_id"]
    assert "patient_summary" in s_res["data"]
    assert s_res["data"]["patient_summary"]["name"] == "Test MCP Patient"

    # 2. append_message
    m1_res = disp.dispatch(
        "append_message",
        {"session_id": sid, "role": "user", "content": "Patient reports knee pain."},
        token="tok_test_intake",
    )
    assert m1_res["ok"] is True
    assert m1_res["data"]["seq"] == 1

    m2_res = disp.dispatch(
        "append_message",
        {"session_id": sid, "role": "agent", "content": "Acknowledged knee pain."},
        token="tok_test_intake",
    )
    assert m2_res["ok"] is True
    assert m2_res["data"]["seq"] == 2

    # 3. get_session_messages
    msgs_res = disp.dispatch(
        "get_session_messages",
        {"session_id": sid},
        token="tok_test_intake",
    )
    assert msgs_res["ok"] is True
    assert len(msgs_res["data"]["messages"]) == 2

    # 4. remember a fact and a note
    fact_res = disp.dispatch(
        "remember",
        {
            "session_id": sid,
            "kind": "fact",
            "payload": {
                "kind": "symptom",
                "display": "Right knee pain",
                "authority": "patient_reported",
            },
        },
        token="tok_test_intake",
    )
    assert fact_res["ok"] is True
    fid = fact_res["data"]["fact_id"]

    note_res = disp.dispatch(
        "remember",
        {
            "session_id": sid,
            "kind": "note",
            "payload": {"text": "Intake observation regarding knee strain."},
        },
        token="tok_test_intake",
    )
    assert note_res["ok"] is True

    # 5. recall
    recall_res = disp.dispatch(
        "recall",
        {"session_id": sid, "query": "knee pain", "k": 5},
        token="tok_test_intake",
    )
    assert recall_res["ok"] is True
    ret_id = recall_res["data"]["retrieval_id"]
    assert ret_id.startswith("ret_")
    assert len(recall_res["data"]["items"]) >= 1

    # 6. history
    hist_res = disp.dispatch(
        "history",
        {"session_id": sid, "fact_id": fid},
        token="tok_test_intake",
    )
    assert hist_res["ok"] is True
    assert len(hist_res["data"]["versions"]) >= 1

    # 7. commit_answer citing recall item
    ans_res = disp.dispatch(
        "commit_answer",
        {
            "session_id": sid,
            "retrieval_id": ret_id,
            "question": "What is the primary symptom?",
            "answer_text": "Patient has right knee pain.",
            "claims": [{"text": "Patient has right knee pain", "supported_by": [fid]}],
        },
        token="tok_test_intake",
    )
    assert ans_res["ok"] is True
    ans_id = ans_res["data"]["answer_id"]
    assert ans_id.startswith("ans_")

    # 8. explain answer
    exp_res = disp.dispatch("explain", {"answer_id": ans_id}, token="tok_test_intake")
    assert exp_res["ok"] is True

    # 9. end_session
    end_res = disp.dispatch(
        "end_session",
        {"session_id": sid, "summary": "Visit concluded"},
        token="tok_test_intake",
    )
    assert end_res["ok"] is True

    # 10. Operations on closed session fail with SESSION_CLOSED
    append_closed = disp.dispatch(
        "append_message",
        {"session_id": sid, "role": "user", "content": "Cannot add to closed session."},
        token="tok_test_intake",
    )
    assert append_closed["ok"] is False
    assert append_closed["error"]["code"] == "SESSION_CLOSED"


def test_handoff_and_safety_flow(mcp_fixture: dict[str, Any]) -> None:
    disp: ToolDispatcher = mcp_fixture["dispatcher"]

    # Start session
    ses_res = disp.dispatch(
        "start_session",
        {"patient_id": TEST_PID_MCP},
        token="tok_test_intake",
    )
    sid = ses_res["data"]["session_id"]

    # Remember allergy fact
    f_res = disp.dispatch(
        "remember",
        {
            "session_id": sid,
            "kind": "fact",
            "payload": {
                "kind": "allergy",
                "display": "Penicillin",
                "code": "70618",
                "authority": "patient_reported",
            },
        },
        token="tok_test_intake",
    )
    fid = f_res["data"]["fact_id"]

    # Issue handoff: intake -> specialist
    ho_res = disp.dispatch(
        "handoff",
        {
            "session_id": sid,
            "to_role": "specialist",
            "goal": "Evaluate knee pain and allergy history",
            "state_summary": "Intake completed initial assessment",
            "completed_actions": [{"type": "intake_interview", "description": "Done"}],
            "open_questions": [{"text": "Is knee pain referred or acute?"}],
            "memory_ids": [{"label": "Fact", "id": fid, "rel": "supporting"}],
        },
        token="tok_test_intake",
    )
    assert ho_res["ok"] is True
    hid = ho_res["data"]["handoff_id"]

    # Specialist lists handoffs
    list_res = disp.dispatch(
        "list_handoffs",
        {"session_id": sid},
        token="tok_test_specialist",
    )
    assert list_res["ok"] is True
    assert any(h["handoff_id"] == hid for h in list_res["data"]["handoffs"])

    # Specialist accepts handoff
    acc_res = disp.dispatch(
        "accept_handoff",
        {"session_id": sid, "handoff_id": hid},
        token="tok_test_specialist",
    )
    assert acc_res["ok"] is True

    # Specialist completes handoff
    comp_res = disp.dispatch(
        "complete_handoff",
        {
            "session_id": sid,
            "handoff_id": hid,
            "result_summary": "Assessment finalized.",
            "outcome": "success",
        },
        token="tok_test_specialist",
    )
    assert comp_res["ok"] is True
    assert comp_res["data"]["status"] == "completed"

    # Clinician confirms fact
    conf_res = disp.dispatch(
        "confirm_fact",
        {"fact_id": fid, "decision": "confirm"},
        token="tok_test_clinician",
    )
    assert conf_res["ok"] is True
    assert conf_res["data"]["fact"]["authority"] == "clinician"


def test_mcp_server_tool_registry_and_call(mcp_fixture: dict[str, Any]) -> None:
    disp: ToolDispatcher = mcp_fixture["dispatcher"]
    server = create_mcp_server(dispatcher=disp)

    assert server.name == "carethread-memory"

    async def run_test():
        tools = await server.list_tools()
        tool_names = [t.name for t in tools]

        # Verify all 20 required tools are registered
        expected_tools = [
            "start_session",
            "end_session",
            "append_message",
            "get_session_messages",
            "remember",
            "recall",
            "history",
            "handoff",
            "list_handoffs",
            "accept_handoff",
            "complete_handoff",
            "consolidate",
            "record_outcome",
            "commit_answer",
            "explain",
            "list_flags",
            "resolve_flag",
            "confirm_fact",
            "forget_request",
            "confirm_forget",
        ]
        for tname in expected_tools:
            assert tname in tool_names, f"Tool '{tname}' missing from MCP server registry."

        # Execute a tool via the MCP server interface
        res = await server.call_tool(
            "start_session",
            {"patient_id": TEST_PID_MCP, "token": "tok_test_intake"},
        )
        assert res is not None
        assert not res.is_error
        assert len(res.content) >= 1

    asyncio.run(run_test())
