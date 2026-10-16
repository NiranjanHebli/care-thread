"""Unit and integration tests for CareThread clinical agents and multi-agent workflow."""

from __future__ import annotations

from typing import Any

import pytest

from carethread.agents.intake import IntakeAgent
from carethread.agents.llm import LLMClient
from carethread.agents.pharmacy import PharmacyAgent
from carethread.agents.specialist import SpecialistAgent
from carethread.agents.workflow import ClinicalWorkflowRunner
from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher

TEST_PID_AGENT: str = "p_a99887766554"


@pytest.fixture
def agent_fixture(tmp_path: Any):
    """Sets up a clean test patient tenant and configured ToolDispatcher for clinical agents."""
    client = get_db_client()
    router = TenantRouter(client)

    try:
        router.graph(TEST_PID_AGENT).delete()
    except Exception:
        pass
    try:
        router.deregister_patient(TEST_PID_AGENT)
    except Exception:
        pass
    router.register_patient(TEST_PID_AGENT, bootstrap_schema=True)

    g = router.graph(TEST_PID_AGENT)
    g.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Agent Workflow Patient'",
        {"pid": TEST_PID_AGENT},
    )

    # Seed an allergy fact into patient graph: Penicillin allergy
    clock = FixedClock(1792200000)
    now_ts = clock.now_ts()
    g.query(
        """
        MATCH (p:Patient {id:$pid})
        CREATE (f:Fact {
            id: 'fct_allergy_pnc',
            kind: 'allergy',
            slot: 'allergy_penicillin',
            code: '70618',
            system: 'RxNorm',
            display: 'Penicillin allergy',
            value: 'anaphylaxis_history',
            status: 'active',
            valid_from: $now,
            recorded_at: $now,
            confidence: 1.0,
            authority: 'patient_reported',
            stale: false,
            version: 1
        })
        CREATE (p)-[:HAS_FACT]->(f)
        """,
        {"pid": TEST_PID_AGENT, "now": now_ts},
    )

    token_store = TokenStore()
    token_store.register_token("tok_intake_01", "agt_intake_01", "intake", "*")
    token_store.register_token("tok_specialist_01", "agt_specialist_01", "specialist", "*")
    token_store.register_token("tok_pharmacy_01", "agt_pharmacy_01", "pharmacy", "*")
    token_store.register_token("tok_clinician_01", "agt_clinician_01", "clinician", "*")

    dispatcher = ToolDispatcher(
        router=router,
        token_store=token_store,
        clock=clock,
        max_calls_per_second=50,
    )

    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(mode="mock", cache_dir=cache_dir)

    yield {
        "client": client,
        "router": router,
        "clock": clock,
        "dispatcher": dispatcher,
        "llm": llm,
        "graph": g,
        "cache_dir": cache_dir,
    }

    try:
        router.graph(TEST_PID_AGENT).delete()
    except Exception:
        pass
    try:
        router.deregister_patient(TEST_PID_AGENT)
    except Exception:
        pass


def test_llm_client_mock_and_replay(tmp_path: Any) -> None:
    """Verifies LLMClient behavior in mock and replay cache modes."""
    cache_dir = tmp_path / "cache_test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client_mock = LLMClient(mode="mock", cache_dir=cache_dir)
    res_intake = client_mock.generate("Patient reports knee pain and stiffness", role="intake")
    assert res_intake.structured is not None
    assert "facts" in res_intake.structured
    assert len(res_intake.structured["facts"]) > 0

    res_specialist = client_mock.generate("Consult for sore throat and fever", role="specialist")
    assert "diagnosis" in res_specialist.structured
    assert "proposed_medication" in res_specialist.structured

    res_pharmacy = client_mock.generate(
        "Evaluate amoxicillin given penicillin allergy", role="pharmacy"
    )
    assert res_pharmacy.structured["is_safe"] is False
    assert "Azithromycin" in res_pharmacy.structured["recommendation"]

    # Test replay mode with disk cache
    client_replay = LLMClient(mode="replay", cache_dir=cache_dir)
    # Write to cache
    client_replay._write_cache("test_prompt", None, "agent", res_intake)
    replayed = client_replay.generate("test_prompt", role="agent")
    assert replayed.cached is True
    assert replayed.structured["summary"] == res_intake.structured["summary"]


def test_intake_agent(agent_fixture: dict[str, Any]) -> None:
    """Verifies IntakeAgent session creation, fact recording, and specialist handoff."""
    disp: ToolDispatcher = agent_fixture["dispatcher"]
    llm: LLMClient = agent_fixture["llm"]

    agent = IntakeAgent(dispatcher=disp, llm=llm)
    res = agent.process_patient_input(
        patient_id=TEST_PID_AGENT,
        message="Patient has severe sore throat, fever, and difficulty swallowing.",
    )

    assert res["session_id"].startswith("ses_")
    assert res["handoff_id"].startswith("hnd_")
    assert len(res["fact_ids"]) >= 1
    assert res["note_id"].startswith("not_")

    # Verify handoff exists in open status for specialist
    ho_list = disp.dispatch(
        "list_handoffs",
        {"session_id": res["session_id"], "status": "open"},
        token="tok_specialist_01",
    )
    assert ho_list["ok"] is True
    assert any(h["handoff_id"] == res["handoff_id"] for h in ho_list["data"]["handoffs"])


def test_specialist_agent(agent_fixture: dict[str, Any]) -> None:
    """Verifies SpecialistAgent accepts handoff, assesses patient, and delegates to pharmacy."""
    disp: ToolDispatcher = agent_fixture["dispatcher"]
    llm: LLMClient = agent_fixture["llm"]

    # Step 1: Intake creates handoff
    intake = IntakeAgent(dispatcher=disp, llm=llm)
    in_res = intake.process_patient_input(
        patient_id=TEST_PID_AGENT,
        message="Patient reports high fever and severe throat inflammation.",
    )
    sid = in_res["session_id"]
    in_hid = in_res["handoff_id"]

    # Step 2: Specialist processes handoff
    specialist = SpecialistAgent(dispatcher=disp, llm=llm)
    spec_res = specialist.process_handoff(
        patient_id=TEST_PID_AGENT,
        handoff_id=in_hid,
        session_id=sid,
    )

    assert spec_res["accepted_handoff_id"] == in_hid
    assert spec_res["pharmacy_handoff_id"].startswith("hnd_")
    assert "Pharyngitis" in spec_res["diagnosis"] or "Assessment" in spec_res["diagnosis"]
    assert spec_res["proposed_medication_id"].startswith("fct_")

    # Verify intake handoff transitioned to completed
    intake_ho = disp.dispatch(
        "list_handoffs",
        {"session_id": sid, "status": "completed"},
        token="tok_intake_01",
    )
    assert intake_ho["ok"] is True
    assert any(h["handoff_id"] == in_hid for h in intake_ho["data"]["handoffs"])


def test_pharmacy_agent_contraindication_and_provenance(agent_fixture: dict[str, Any]) -> None:
    """Verifies PharmacyAgent detects allergy contraindication and commits cited answer."""
    disp: ToolDispatcher = agent_fixture["dispatcher"]
    llm: LLMClient = agent_fixture["llm"]

    # Setup intake -> specialist to produce a pharmacy handoff for Amoxicillin
    intake = IntakeAgent(dispatcher=disp, llm=llm)
    in_res = intake.process_patient_input(
        patient_id=TEST_PID_AGENT,
        message="Patient has bacterial tonsillitis symptoms.",
    )
    sid = in_res["session_id"]
    in_hid = in_res["handoff_id"]

    specialist = SpecialistAgent(dispatcher=disp, llm=llm)
    spec_res = specialist.process_handoff(
        patient_id=TEST_PID_AGENT,
        handoff_id=in_hid,
        session_id=sid,
    )
    pharm_hid = spec_res["pharmacy_handoff_id"]

    # Pharmacy agent processes prescription
    pharmacy = PharmacyAgent(dispatcher=disp, llm=llm)
    pharm_res = pharmacy.process_handoff(
        patient_id=TEST_PID_AGENT,
        handoff_id=pharm_hid,
        session_id=sid,
    )

    assert pharm_res["handoff_id"] == pharm_hid
    assert pharm_res["answer_id"].startswith("ans_")
    assert pharm_res["is_safe"] is False
    assert "Azithromycin" in pharm_res["recommendation"]
    assert len(pharm_res["claims"]) >= 1

    # Verify answer can be explained via explain tool
    exp_res = disp.dispatch(
        "explain",
        {"answer_id": pharm_res["answer_id"]},
        token="tok_intake_01",
    )
    assert exp_res["ok"] is True
    assert exp_res["data"]["answer_id"] == pharm_res["answer_id"]


def test_end_to_end_clinical_workflow(agent_fixture: dict[str, Any]) -> None:
    """Verifies end-to-end execution across Intake -> Specialist -> Pharmacy workflow."""
    disp: ToolDispatcher = agent_fixture["dispatcher"]
    llm: LLMClient = agent_fixture["llm"]

    runner = ClinicalWorkflowRunner(
        dispatcher=disp,
        llm=llm,
    )

    result = runner.run(
        patient_id=TEST_PID_AGENT,
        patient_message="Patient reports painful throat, fever, and difficulty swallowing.",
    )

    assert result.patient_id == TEST_PID_AGENT
    assert result.session_id.startswith("ses_")
    assert result.intake_handoff_id.startswith("hnd_")
    assert result.specialist_handoff_id.startswith("hnd_")
    assert result.answer_id.startswith("ans_")
    assert result.is_safe is False
    assert "Azithromycin" in result.recommendation

    # Verify both handoffs are in completed status
    h_list = disp.dispatch(
        "list_handoffs",
        {"session_id": result.session_id, "status": "completed"},
        token="tok_clinician_01",
    )
    assert h_list["ok"] is True
    completed_ids = [h["handoff_id"] for h in h_list["data"]["handoffs"]]
    assert result.intake_handoff_id in completed_ids
    assert result.specialist_handoff_id in completed_ids

    # Verify session transcript contains both patient and agent utterances
    msgs_res = disp.dispatch(
        "get_session_messages",
        {"session_id": result.session_id},
        token="tok_intake_01",
    )
    assert msgs_res["ok"] is True
    messages = msgs_res["data"]["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "agent"
