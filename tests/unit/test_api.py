"""Unit tests for the FastAPI REST and Server-Sent Events API layer.

Validates:
- Health and readiness probes (/healthz, /readyz, /metrics)
- Patient selector and search filtering (/api/patients)
- Cytoscape.js graph payload serialization and node collapsing (/api/patients/{id}/graph)
- SSE streaming agent chat (/api/agents/{role}/chat) emitting tool_call,
  tool_result, token, final, and trace events
- Provenance trace retrieval and Why-Not explainability (/api/answers)
- Handoff board listing and safety flag lifecycle (/api/handoffs, /api/flags)
- Audit log inspection (/api/audit)
- Gated internal harness endpoints (/internal/tools, /internal/clock)
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from carethread.agents.llm import LLMClient
from carethread.api.app import create_app
from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher
from carethread.util.id import (
    fact_id as gen_fact_id,
)

TEST_PID_API = "p_d18000000001"
OTHER_PID_API = "p_d18000000002"


@pytest.fixture
def api_fixture() -> dict[str, Any]:
    """Sets up an isolated FastAPI TestClient with database fixtures and tokens."""
    client = get_db_client()
    router = TenantRouter(client)

    for pid in (TEST_PID_API, OTHER_PID_API):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g1 = router.graph(TEST_PID_API)
    g1.query(
        """
        MERGE (p:Patient {id:$pid})
        ON CREATE SET p.name = 'API Test Patient', p.gender = 'F', p.birth_date = '1985-04-12'
        """,
        {"pid": TEST_PID_API},
    )

    token_store = TokenStore()
    token_store.register_token("tok_api_admin", "agt_api_admin", "clinician", "*")
    token_store.register_token("tok_api_intake", "agt_intake_01", "intake", "*")
    token_store.register_token("tok_api_spec", "agt_specialist_01", "specialist", "*")
    token_store.register_token("tok_api_pharm", "agt_pharmacy_01", "pharmacy", "*")

    clock = FixedClock(1792180000)
    dispatcher = ToolDispatcher(
        router=router,
        token_store=token_store,
        clock=clock,
    )
    llm = LLMClient(mode="mock")

    app = create_app(
        dispatcher=dispatcher,
        router=router,
        clock=clock,
        llm=llm,
        enable_internal=True,
    )
    test_client = TestClient(app)

    return {
        "client": test_client,
        "dispatcher": dispatcher,
        "router": router,
        "clock": clock,
        "g1": g1,
    }


def test_health_and_readiness_endpoints(api_fixture: dict[str, Any]) -> None:
    """Verifies /healthz, /readyz, and /metrics responses."""
    client: TestClient = api_fixture["client"]

    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

    res_ready = client.get("/readyz")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["ready"] is True
    assert data_ready["clinic_graph_present"] is True

    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "carethread_up 1" in res_metrics.text
    assert "carethread_tenants_total" in res_metrics.text


def test_patient_picker_and_search(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/patients lists registered patients with clinical summary counts."""
    client: TestClient = api_fixture["client"]
    g1 = api_fixture["g1"]

    # Add active fact to test patient
    fid = gen_fact_id()
    g1.query(
        "CREATE (:Fact {id:$fid, kind:'condition', display:'Asthma', status:'active'})",
        {"fid": fid},
    )

    res = client.get("/api/patients")
    assert res.status_code == 200
    patients = res.json()["patients"]
    assert len(patients) >= 1

    p = next((x for x in patients if x["id"] == TEST_PID_API), None)
    assert p is not None
    assert p["name"] == "API Test Patient"
    assert p["gender"] == "F"
    assert p["active_facts"] >= 1

    # Search filter query
    filtered = client.get(f"/api/patients?q={TEST_PID_API}")
    assert filtered.status_code == 200
    assert len(filtered.json()["patients"]) == 1


def test_patient_cytoscape_graph_serialization(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/patients/{id}/graph returns Cytoscape elements with collapsed observations."""
    client: TestClient = api_fixture["client"]
    g1 = api_fixture["g1"]

    fid = gen_fact_id()
    g1.query(
        """
        CREATE (e:Episode {id:'epi_test_01', start:1700000000, episode_type:'encounter'})
        CREATE (f:Fact {id:$fid, display:'Diabetes', status:'active'})
        CREATE (e)-[:ASSERTED_IN]->(f)
        CREATE (o:Observation {id:'obs_test_01', code:'8480-6'})
        CREATE (e)-[:PRODUCED]->(o)
        """,
        {"fid": fid},
    )

    # With observation collapsing (default)
    res = client.get(f"/api/patients/{TEST_PID_API}/graph?collapse_observations=true")
    assert res.status_code == 200
    elements = res.json()["elements"]
    assert "nodes" in elements
    assert "edges" in elements

    node_ids = {n["data"]["id"] for n in elements["nodes"]}
    assert TEST_PID_API in node_ids
    assert "epi_test_01" in node_ids
    assert fid in node_ids
    # Observations should be collapsed into counter attribute, not standalone nodes
    assert "obs_test_01" not in node_ids

    # Find episode node and verify observation_count counter
    epi_node = next(n for n in elements["nodes"] if n["data"]["id"] == "epi_test_01")
    assert epi_node["data"].get("observation_count") == 1


def test_agent_chat_sse_workflow_streaming(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/agents/workflow/chat streams tool_call, tool_result, token,
    final, and trace.
    """
    client: TestClient = api_fixture["client"]

    payload = {
        "patient_id": TEST_PID_API,
        "message": "Patient reports sore throat and fever for 2 days.",
    }

    response = client.post("/api/agents/workflow/chat", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events: list[tuple[str, dict[str, Any]]] = []
    for line in response.text.split("\n\n"):
        line = line.strip()
        if not line:
            continue
        ev_name = None
        ev_data = None
        for part in line.split("\n"):
            if part.startswith("event:"):
                ev_name = part[len("event:") :].strip()
            elif part.startswith("data:"):
                ev_data = json.loads(part[len("data:") :].strip())
        if ev_name and ev_data is not None:
            events.append((ev_name, ev_data))

    event_types = [ev[0] for ev in events]
    assert "token" in event_types
    assert "tool_call" in event_types
    assert "final" in event_types
    assert "trace" in event_types

    # Verify final event payload structure
    final_ev = next(ev[1] for ev in events if ev[0] == "final")
    assert "session_id" in final_ev
    assert "answer_id" in final_ev
    assert "recommendation" in final_ev

    # Verify trace event payload structure
    trace_ev = next(ev[1] for ev in events if ev[0] == "trace")
    assert trace_ev["answer_id"] == final_ev["answer_id"]
    assert "used" in trace_ev


def test_agent_chat_sse_intake_streaming(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/agents/intake/chat streams intake triage events."""
    client: TestClient = api_fixture["client"]

    payload = {
        "patient_id": TEST_PID_API,
        "message": "Patient reports persistent dry cough and fatigue.",
    }

    response = client.post("/api/agents/intake/chat", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events: list[str] = []
    for line in response.text.split("\n\n"):
        if "event:" in line:
            ev_name = line.split("event:")[1].split("\n")[0].strip()
            events.append(ev_name)

    assert "token" in events
    assert "tool_call" in events
    assert "final" in events


def test_provenance_trace_and_why_not_endpoints(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/answers/{id}/trace and /api/answers/{id}/why-not."""
    client: TestClient = api_fixture["client"]
    disp: ToolDispatcher = api_fixture["dispatcher"]
    g1 = api_fixture["g1"]

    # Commit an answer to query
    fid = gen_fact_id()
    g1.query(
        "CREATE (:Fact {id:$fid, display:'Penicillin allergy', kind:'allergy', status:'active'})",
        {"fid": fid},
    )

    retrieval = {
        "created_at": api_fixture["clock"].now_ts(),
        "items": [{"id": fid, "label": "Fact", "display": "Penicillin allergy"}],
        "rejected": [],
    }

    trace = disp._provenance_engine.commit_answer(
        patient_id=TEST_PID_API,
        question="Allergy status?",
        answer_text="Active penicillin allergy.",
        claims=[{"text": "Allergy", "supported_by": [fid]}],
        retrieval=retrieval,
    )
    aid = trace.answer_id

    # 1. GET /api/answers/{id}/trace
    res_trace = client.get(f"/api/answers/{aid}/trace")
    assert res_trace.status_code == 200
    trace_data = res_trace.json()
    assert trace_data["answer_id"] == aid
    assert len(trace_data["used"]) == 1

    # 2. POST /api/answers/{id}/why-not for contraindicated Amoxicillin
    res_why_not = client.post(
        f"/api/answers/{aid}/why-not",
        json={"name": "Amoxicillin", "code": "308192"},
    )
    assert res_why_not.status_code == 200
    why_data = res_why_not.json()
    assert why_data["verdict"] == "BLOCKED"
    assert why_data["reason_code"] == "ALLERGY_CONFLICT"
    assert len(why_data["blocking_path"]) >= 1

    # 3. GET /api/answers/nonexistent/trace returns 404
    res_404 = client.get("/api/answers/ans_nonexistent_99/trace")
    assert res_404.status_code == 404


def test_handoffs_and_safety_flags_endpoints(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/handoffs, /api/flags, and /api/flags/{id}/resolve."""
    client: TestClient = api_fixture["client"]
    disp: ToolDispatcher = api_fixture["dispatcher"]

    # Start session and record a safety flag
    s_res = disp.dispatch("start_session", {"patient_id": TEST_PID_API}, token="tok_api_admin")
    sid = s_res["data"]["session_id"]

    # Create a handoff
    h_res = disp.dispatch(
        "handoff",
        {
            "session_id": sid,
            "to_role": "specialist",
            "goal": "Evaluate severe chest pain",
            "state_summary": "Intake completed",
            "memory_ids": ["fct_test_01"],
        },
        token="tok_api_admin",
    )
    assert h_res["ok"] is True
    hid = h_res["data"]["handoff_id"]

    # 1. GET /api/handoffs
    res_h = client.get(f"/api/handoffs?session_id={sid}")
    assert res_h.status_code == 200
    assert any(h["handoff_id"] == hid for h in res_h.json()["handoffs"])

    # Create a safety flag
    from carethread.memory.safety import SafetyFlagRecord
    from carethread.util.id import flag_id as gen_flag_id

    flag_rec = SafetyFlagRecord(
        id=gen_flag_id(),
        patient_id=TEST_PID_API,
        type="allergy_med_conflict",
        severity="high",
        explanation="Conflict detected between penicillin and amoxicillin.",
    )
    fl = disp._safety_mgr.create_flag(patient_id=TEST_PID_API, flag=flag_rec)

    # 2. GET /api/flags
    res_fl = client.get(f"/api/flags?patient_id={TEST_PID_API}&status=open")
    assert res_fl.status_code == 200
    flags = res_fl.json()["flags"]
    assert any(f["id"] == fl.id for f in flags)

    # 3. POST /api/flags/{id}/resolve
    res_res = client.post(
        f"/api/flags/{fl.id}/resolve",
        json={"resolved_by": "dr_smith", "patient_id": TEST_PID_API},
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "resolved"


def test_audit_log_endpoint(api_fixture: dict[str, Any]) -> None:
    """Verifies /api/audit retrieves audit records."""
    client: TestClient = api_fixture["client"]
    disp: ToolDispatcher = api_fixture["dispatcher"]

    # Perform an audited operation
    disp.dispatch("start_session", {"patient_id": TEST_PID_API}, token="tok_api_admin")

    res = client.get(f"/api/audit?patient_id={TEST_PID_API}&limit=50")
    assert res.status_code == 200
    events = res.json()["events"]
    assert len(events) >= 1
    assert any(ev["tool"] == "start_session" for ev in events)


def test_internal_harness_endpoints(api_fixture: dict[str, Any]) -> None:
    """Verifies /internal/tools and /internal/clock endpoints."""
    client: TestClient = api_fixture["client"]

    # 1. Advance clock
    adv_res = client.post("/internal/clock/advance", json={"seconds": 3600})
    assert adv_res.status_code == 200
    assert adv_res.json()["advanced_seconds"] == 3600

    # 2. Invoke tool directly through internal harness
    tool_res = client.post(
        "/internal/tools/start_session",
        json={"params": {"patient_id": TEST_PID_API}, "token": "tok_api_admin"},
    )
    assert tool_res.status_code == 200
    assert tool_res.json()["ok"] is True
    assert tool_res.json()["data"]["session_id"].startswith("ses_")
