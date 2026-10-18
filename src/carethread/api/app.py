"""FastAPI application providing REST endpoints, Cytoscape graph serialization, and SSE.

Exposes:
- Root healthcheck endpoints (/healthz, /readyz, /metrics)
- Patient selector and Cytoscape.js graph serialization (/api/patients)
- SSE streaming agent chat (/api/agents/{role}/chat) emitting tool_call,
  tool_result, token, final, and trace events
- Provenance trace retrieval and Why-Not explainability (/api/answers)
- Handoff board and safety flag management (/api/handoffs, /api/flags)
- Audit log inspection (/api/audit)
- Test and load harness internal tools (/internal/tools, /internal/clock)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from carethread.agents.intake import IntakeAgent
from carethread.agents.llm import LLMClient
from carethread.agents.pharmacy import PharmacyAgent
from carethread.agents.specialist import SpecialistAgent
from carethread.agents.workflow import ClinicalWorkflowRunner
from carethread.api.models import (
    AdvanceClockRequest,
    ChatRequest,
    ResolveFlagRequest,
    ToolInvokeRequest,
    WhyNotRequest,
)
from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


def sse_message(event: str, data: dict[str, Any]) -> str:
    """Formats an event name and data dictionary into a Server-Sent Event block."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(
    dispatcher: ToolDispatcher | None = None,
    router: TenantRouter | None = None,
    clock: Clock | None = None,
    llm: LLMClient | None = None,
    enable_internal: bool = False,
) -> FastAPI:
    """Creates and configures the FastAPI CareThread application."""
    client = get_db_client()
    effective_router = router or TenantRouter(client)
    effective_clock = clock or get_clock()
    effective_clinic = ClinicReader(client)

    token_store = TokenStore()
    token_store.register_token("tok_api_admin", "agt_api_admin", "clinician", "*")
    token_store.register_token("tok_api_intake", "agt_intake_01", "intake", "*")
    token_store.register_token("tok_api_spec", "agt_specialist_01", "specialist", "*")
    token_store.register_token("tok_api_pharm", "agt_pharmacy_01", "pharmacy", "*")

    effective_disp = dispatcher or ToolDispatcher(
        router=effective_router,
        token_store=token_store,
        clock=effective_clock,
    )
    effective_llm = llm or LLMClient(mode="mock")

    app = FastAPI(
        title="CareThread Clinical Graph Memory API",
        version="0.1.0",
        description="Graph-native agentic memory service for clinical coordination.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Health and Metrics Endpoints
    # -------------------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness check returning service status and current clock timestamp."""
        return {
            "status": "ok",
            "timestamp": effective_clock.now_ts(),
        }

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        """Readiness check verifying FalkorDB connectivity and shared clinic graph."""
        try:
            res = effective_clinic.ro_query("RETURN 1")
            clinic_ready = bool(res.result_set)
        except Exception as e:
            logger.warning("Clinic graph readiness check failed: %s", e)
            clinic_ready = False

        if not clinic_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Shared clinic graph is unavailable or not seeded.",
            )

        return {
            "ready": True,
            "clinic_graph_present": True,
            "timestamp": effective_clock.now_ts(),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus-compatible plaintext metrics endpoint."""
        tenants = effective_router.list_tenants()
        lines = [
            "# HELP carethread_up Service availability flag",
            "# TYPE carethread_up gauge",
            "carethread_up 1",
            "# HELP carethread_tenants_total Total registered patient tenants",
            "# TYPE carethread_tenants_total gauge",
            f"carethread_tenants_total {len(tenants)}",
            "# HELP carethread_clock_seconds Current synthetic clock timestamp",
            "# TYPE carethread_clock_seconds gauge",
            f"carethread_clock_seconds {effective_clock.now_ts()}",
        ]
        return Response(content="\n".join(lines) + "\n", media_type="text/plain")

    # -------------------------------------------------------------------------
    # Patient Picker and Graph Serialization
    # -------------------------------------------------------------------------

    @app.get("/api/patients")
    def list_patients(
        q: str = Query(default="", description="Search query filter"),
    ) -> dict[str, Any]:
        """Lists registered patient tenants with clinical overview counts."""
        tenants = effective_router.list_tenants()
        results: list[dict[str, Any]] = []

        q_lower = q.strip().lower()
        for pid in tenants:
            try:
                g = effective_router.graph(pid)
                p_res = g.query(
                    "MATCH (p:Patient {id:$pid}) RETURN p.name, p.gender, p.birth_date",
                    {"pid": pid},
                )
                name = "Unknown"
                gender = None
                birth_date = None
                if p_res.result_set:
                    row = p_res.result_set[0]
                    name = str(row[0] or "Unknown")
                    gender = str(row[1]) if row[1] else None
                    birth_date = str(row[2]) if row[2] else None

                if q_lower and q_lower not in pid.lower() and q_lower not in name.lower():
                    continue

                f_res = g.query("MATCH (f:Fact {status:'active'}) RETURN count(f)")
                active_facts = f_res.result_set[0][0] if f_res.result_set else 0

                fl_res = g.query("MATCH (fl:Flag {status:'open'}) RETURN count(fl)")
                open_flags = fl_res.result_set[0][0] if fl_res.result_set else 0

                results.append(
                    {
                        "id": pid,
                        "patient_id": pid,
                        "name": name,
                        "gender": gender,
                        "birth_date": birth_date,
                        "active_facts": active_facts,
                        "open_flags": open_flags,
                    }
                )
            except Exception as e:
                logger.warning("Error reading patient %s: %s", pid, e)
                continue

        return {"patients": results}

    @app.get("/api/patients/{patient_id}/graph")
    def get_patient_graph(
        patient_id: str,
        collapse_observations: bool = Query(default=True),
    ) -> dict[str, Any]:
        """Serializes patient graph into Cytoscape.js JSON payload with node collapsing."""
        clean_pid = effective_router.validate_patient_id(patient_id)
        if not effective_router.is_registered(clean_pid):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{clean_pid}' not found.",
            )

        g = effective_router.graph(clean_pid)
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()

        # Query main nodes excluding observations if collapsed and audit events
        label_filter = (
            "WHERE NOT 'Observation' IN labels(n) AND NOT 'ClinicalProcedure' IN labels(n) "
            "AND NOT 'AuditEvent' IN labels(n)"
            if collapse_observations
            else "WHERE NOT 'AuditEvent' IN labels(n)"
        )
        node_query = f"MATCH (n) {label_filter} RETURN n.id, labels(n)[0], properties(n)"
        n_res = g.query(node_query)

        for row in n_res.result_set:
            nid = str(row[0]) if row[0] else None
            if not nid or nid in seen_node_ids:
                continue
            seen_node_ids.add(nid)
            label = str(row[1] or "Node")
            props = dict(row[2] or {})
            clean_props = {k: v for k, v in props.items() if not k.endswith("embedding")}

            display = (
                clean_props.get("display")
                or clean_props.get("name")
                or clean_props.get("text")
                or clean_props.get("description")
                or clean_props.get("question")
            )
            if not display or display == nid:
                if label == "Agent" or nid.startswith("agt_"):
                    if "intake" in nid:
                        display = "Intake Agent"
                    elif "specialist" in nid:
                        display = "Specialist Agent"
                    elif "pharmacy" in nid:
                        display = "Pharmacy Agent"
                    else:
                        display = "Clinical Agent"
                elif label == "Episode" or nid.startswith("epi_"):
                    seq = clean_props.get("seq")
                    ep_type = (clean_props.get("episode_type") or "visit").capitalize()
                    display = f"Visit {seq} ({ep_type})" if seq else f"{ep_type} Visit"
                elif label == "Handoff" or nid.startswith("hnd_"):
                    to_role = clean_props.get("to_role") or "agent"
                    display = f"Handoff to {to_role.replace('_', ' ').capitalize()}"
                elif label == "Session" or nid.startswith("ses_"):
                    display = "Clinical Session"
                elif label == "Action" or nid.startswith("act_"):
                    desc = clean_props.get("description") or "Clinical Action"
                    display = desc[:35] + "..." if len(desc) > 38 else desc
                elif label == "OpenQuestion" or nid.startswith("qst_"):
                    q = clean_props.get("question") or "Open Question"
                    display = q[:35] + "..." if len(q) > 38 else q
                elif label == "Message" or nid.startswith("msg_"):
                    role = clean_props.get("role") or "User"
                    display = f"{role.capitalize()} Message"
                elif label == "Flag" or nid.startswith("flg_"):
                    display = clean_props.get("category") or "Safety Flag"
                else:
                    display = nid

            if isinstance(display, str) and len(display) > 80:
                display = display[:77] + "..."

            node_data: dict[str, Any] = {
                **clean_props,
                "id": nid,
                "label": label,
                "display": display,
                "status": clean_props.get("status", "active"),
                "confidence": clean_props.get("confidence"),
                "role": clean_props.get("role"),
                "created_at": clean_props.get("created_at"),
            }
            nodes.append({"data": node_data})

        # Attach observation count to episodes when collapsed
        if collapse_observations:
            obs_counts = g.query(
                "MATCH (e:Episode)-[:PRODUCED]->(o:Observation) RETURN e.id, count(o)"
            )
            obs_map = {str(r[0]): int(r[1]) for r in obs_counts.result_set}
            for n in nodes:
                if n["data"]["label"] == "Episode" and n["data"]["id"] in obs_map:
                    n["data"]["observation_count"] = obs_map[n["data"]["id"]]

        # Query relationships
        edge_query = """
        MATCH (a)-[r]->(b)
        RETURN a.id, type(r), b.id, properties(r)
        """
        e_res = g.query(edge_query)
        seen_edge_ids: set[str] = set()

        for row in e_res.result_set:
            src = str(row[0]) if row[0] else None
            rel_type = str(row[1]) if row[1] else "RELATED"
            tgt = str(row[2]) if row[2] else None

            if not src or not tgt or src not in seen_node_ids or tgt not in seen_node_ids:
                continue

            edge_id = f"{src}-{rel_type}-{tgt}"
            if edge_id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge_id)

            edges.append(
                {
                    "data": {
                        "id": edge_id,
                        "source": src,
                        "target": tgt,
                        "label": rel_type,
                    }
                }
            )

        return {"elements": {"nodes": nodes, "edges": edges}}

    # -------------------------------------------------------------------------
    # Agent Chat Execution (Server-Sent Events)
    # -------------------------------------------------------------------------

    @app.post("/api/agents/{role}/chat")
    async def chat_agent(role: str, req: ChatRequest) -> StreamingResponse:
        """Executes a clinical agent reasoning run, streaming events via SSE.

        Emits event types: tool_call, tool_result, token, final, trace.
        """
        clean_role = role.strip().lower()
        if clean_role not in ("intake", "specialist", "pharmacy", "workflow"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown agent role '{role}'. Allowed: intake, specialist, pharmacy, workflow."
                ),
            )

        clean_pid = effective_router.validate_patient_id(req.patient_id)
        if not effective_router.is_registered(clean_pid):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{clean_pid}' not found.",
            )

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # 1. Starting token notification
                yield sse_message("token", {"token": f"Initializing {clean_role} agent run..."})

                if clean_role == "workflow":
                    runner = ClinicalWorkflowRunner(
                        dispatcher=effective_disp,
                        llm=effective_llm,
                    )
                    yield sse_message(
                        "tool_call",
                        {"tool": "start_session", "args": {"patient_id": clean_pid}},
                    )
                    wf_res = runner.run(
                        patient_id=clean_pid,
                        patient_message=req.message,
                    )
                    yield sse_message(
                        "tool_result",
                        {
                            "tool": "start_session",
                            "result": {"session_id": wf_res.session_id},
                        },
                    )

                    yield sse_message(
                        "token",
                        {"token": "Completed clinical triage and specialist review."},
                    )
                    yield sse_message(
                        "final",
                        wf_res.to_dict(),
                    )

                    # Stream trace event
                    exp = effective_disp.dispatch(
                        "explain", {"answer_id": wf_res.answer_id}, token="tok_api_admin"
                    )
                    if exp.get("ok"):
                        yield sse_message("trace", exp["data"])

                elif clean_role == "intake":
                    agent = IntakeAgent(dispatcher=effective_disp, llm=effective_llm)
                    yield sse_message(
                        "tool_call",
                        {"tool": "start_session", "args": {"patient_id": clean_pid}},
                    )
                    res = agent.process_patient_input(
                        patient_id=clean_pid,
                        message=req.message,
                    )
                    yield sse_message(
                        "tool_result",
                        {"tool": "start_session", "result": {"session_id": res["session_id"]}},
                    )
                    yield sse_message(
                        "token",
                        {"token": "Patient intake recorded and handoff prepared."},
                    )
                    yield sse_message("final", res)

                elif clean_role == "specialist":
                    spec_agent = SpecialistAgent(dispatcher=effective_disp, llm=effective_llm)
                    if not req.handoff_id or not req.session_id:
                        yield sse_message(
                            "token",
                            {"token": "Specialist requires active session_id and handoff_id."},
                        )
                        yield sse_message("final", {"error": "Missing handoff_id or session_id"})
                        return

                    yield sse_message(
                        "tool_call",
                        {"tool": "accept_handoff", "args": {"handoff_id": req.handoff_id}},
                    )
                    res = spec_agent.process_handoff(
                        patient_id=clean_pid,
                        handoff_id=req.handoff_id,
                        session_id=req.session_id,
                    )
                    yield sse_message(
                        "tool_result",
                        {"tool": "accept_handoff", "result": {"status": "accepted"}},
                    )
                    yield sse_message("token", {"token": "Specialist assessment completed."})
                    yield sse_message("final", res)

                elif clean_role == "pharmacy":
                    pharm_agent = PharmacyAgent(dispatcher=effective_disp, llm=effective_llm)
                    if not req.handoff_id or not req.session_id:
                        yield sse_message(
                            "token",
                            {"token": "Pharmacy requires active session_id and handoff_id."},
                        )
                        yield sse_message("final", {"error": "Missing handoff_id or session_id"})
                        return

                    yield sse_message(
                        "tool_call",
                        {"tool": "accept_handoff", "args": {"handoff_id": req.handoff_id}},
                    )
                    res = pharm_agent.process_handoff(
                        patient_id=clean_pid,
                        handoff_id=req.handoff_id,
                        session_id=req.session_id,
                    )
                    yield sse_message(
                        "tool_result",
                        {"tool": "accept_handoff", "result": {"status": "accepted"}},
                    )
                    yield sse_message("token", {"token": "Pharmacy safety screening complete."})
                    yield sse_message("final", res)

                    # Emit trace
                    if res.get("answer_id"):
                        exp = effective_disp.dispatch(
                            "explain", {"answer_id": res["answer_id"]}, token="tok_api_admin"
                        )
                        if exp.get("ok"):
                            yield sse_message("trace", exp["data"])

            except Exception as err:
                logger.error("SSE stream error: %s", err, exc_info=True)
                yield sse_message("token", {"token": f"Error occurred: {err}"})
                yield sse_message("final", {"error": str(err)})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # -------------------------------------------------------------------------
    # Provenance and Explainability
    # -------------------------------------------------------------------------

    @app.get("/api/answers/{answer_id}/trace")
    def get_answer_trace(answer_id: str) -> dict[str, Any]:
        """Retrieves reconstructed machine-verifiable provenance trace for an answer."""
        res = effective_disp.dispatch("explain", {"answer_id": answer_id}, token="tok_api_admin")
        if not res.get("ok"):
            err_code = res.get("error", {}).get("code", "NOT_FOUND")
            if err_code == "ANSWER_NOT_FOUND":
                raise HTTPException(status_code=404, detail=f"Answer '{answer_id}' not found.")
            raise HTTPException(status_code=400, detail=res.get("error", {}).get("message"))
        return res["data"]

    @app.post("/api/answers/{answer_id}/why-not")
    def explain_why_not(answer_id: str, req: WhyNotRequest) -> dict[str, Any]:
        """Evaluates counterfactual Why-Not analysis for an answer candidate."""
        target: Any = req.target or req.model_dump(exclude_none=True)
        if isinstance(target, dict) and len(target) == 1 and "node_id" in target:
            target = target["node_id"]

        res = effective_disp.dispatch(
            "explain",
            {"answer_id": answer_id, "why_not": target},
            token="tok_api_admin",
        )
        if not res.get("ok"):
            err_code = res.get("error", {}).get("code", "NOT_FOUND")
            if err_code == "ANSWER_NOT_FOUND":
                raise HTTPException(status_code=404, detail=f"Answer '{answer_id}' not found.")
            raise HTTPException(status_code=400, detail=res.get("error", {}).get("message"))
        return res["data"]

    # -------------------------------------------------------------------------
    # Handoffs and Safety Flags
    # -------------------------------------------------------------------------

    @app.get("/api/handoffs")
    def get_handoffs(
        patient_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Lists coordination handoffs across agents."""
        params: dict[str, Any] = {}
        if status and status.lower() != "all":
            params["status"] = status
        if session_id:
            params["session_id"] = session_id
        if patient_id:
            params["patient_id"] = patient_id

        res = effective_disp.dispatch("list_handoffs", params, token="tok_api_admin")
        if not res.get("ok"):
            raise HTTPException(status_code=400, detail=res.get("error", {}).get("message"))
        return res["data"]

    @app.get("/api/flags")
    def get_flags(
        patient_id: str = Query(..., description="Target patient identifier"),
        status: str = Query(default="open"),
    ) -> dict[str, Any]:
        """Lists active safety flags for a patient."""
        res = effective_disp.dispatch(
            "list_flags",
            {"patient_id": patient_id, "status": status},
            token="tok_api_admin",
        )
        if not res.get("ok"):
            raise HTTPException(status_code=400, detail=res.get("error", {}).get("message"))
        return res["data"]

    @app.post("/api/flags/{flag_id}/resolve")
    def resolve_flag(flag_id: str, req: ResolveFlagRequest) -> dict[str, Any]:
        """Resolves an active clinical safety flag."""
        res = effective_disp.dispatch(
            "resolve_flag",
            {
                "flag_id": flag_id,
                "resolved_by": req.resolved_by,
                "patient_id": req.patient_id,
            },
            token="tok_api_admin",
        )
        if not res.get("ok"):
            raise HTTPException(status_code=400, detail=res.get("error", {}).get("message"))
        return res["data"].get("flag", res["data"])

    # -------------------------------------------------------------------------
    # Audit Log Inspection
    # -------------------------------------------------------------------------

    @app.get("/api/audit")
    def get_audit(
        patient_id: str = Query(..., description="Target patient identifier"),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        """Retrieves recent audit events for clinical inspection."""
        clean_pid = effective_router.validate_patient_id(patient_id)
        events = effective_disp._audit_logger.get_audit_events(patient_id=clean_pid, limit=limit)
        return {"events": [ev.to_dict() for ev in events]}

    # -------------------------------------------------------------------------
    # Internal Test and Load Harness Endpoints (Gated)
    # -------------------------------------------------------------------------

    is_internal_enabled = enable_internal or os.environ.get("CT_INTERNAL") == "1"

    @app.post("/internal/tools/{name}")
    def invoke_internal_tool(name: str, req: ToolInvokeRequest) -> dict[str, Any]:
        """Directly dispatches tool calls for test harness (gated by CT_INTERNAL)."""
        if not is_internal_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Internal tool dispatch is disabled outside test environments.",
            )
        res = effective_disp.dispatch(name, req.params, token=req.token)
        return res

    @app.post("/internal/clock/advance")
    def advance_clock(req: AdvanceClockRequest) -> dict[str, Any]:
        """Advances the synthetic clock for testing temporal decay and expirations."""
        if not is_internal_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Clock advance is disabled outside test environments.",
            )
        if hasattr(effective_clock, "advance"):
            effective_clock.advance(req.seconds)
        return {
            "advanced_seconds": req.seconds,
            "new_timestamp": effective_clock.now_ts(),
        }

    # -------------------------------------------------------------------------
    # Frontend Static Files Mount
    # -------------------------------------------------------------------------
    ui_dist_path = os.getenv(
        "CT_UI_DIST",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ui", "dist")),
    )
    if os.path.exists(ui_dist_path):
        app.mount("/", StaticFiles(directory=ui_dist_path, html=True), name="ui")

    return app


# Default app instance for standard ASGI servers
app = create_app()
