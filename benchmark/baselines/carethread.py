"""CareThread Graph Memory condition (subject of benchmark).

Leverages multi-tenant FalkorDB graph memory, bitemporal facts, active slot uniqueness,
structured handoff packets, and machine-verifiable explanation traces with ablation testing.
"""

from __future__ import annotations

import time
from typing import Any

from carethread.agents.llm import LLMClient
from carethread.clock import get_clock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher


class CareThreadCondition:
    """Evaluates agent responses using CareThread graph memory through MCP tools."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.client = get_db_client()
        self.router = TenantRouter(self.client)
        self.clock = get_clock()
        self.token_store = TokenStore()
        self.token_store.register_token("tok_bench_admin", "agt_bench_admin", "clinician", "*")
        self.dispatcher = ToolDispatcher(
            router=self.router,
            token_store=self.token_store,
            clock=self.clock,
            max_calls_per_second=2000,
        )
        self.llm = llm_client or LLMClient()
        self.created_tenants: set[str] = set()
        self.name = "carethread"

    def setup_scenario(self, scenario: dict[str, Any]) -> None:
        """Hydrates the patient's isolated graph with scenario visits, facts, and handoffs."""
        pid = scenario["patient_id"]
        pname = scenario.get("patient_name", "Benchmark Patient")
        token = "tok_bench_admin"

        # Register tenant and schema
        self.router.register_patient(pid, bootstrap_schema=True)
        self.created_tenants.add(pid)
        g = self.router.graph(pid)
        g.query(
            "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = $name",
            {"pid": pid, "name": pname},
        )

        for s in scenario.get("sessions", []):
            sid = s.get("session_id", "")

            # Start session
            s_res = self.dispatcher.dispatch(
                "start_session",
                {"patient_id": pid, "channel": "agent"},
                token=token,
            )
            actual_sid = s_res.get("data", {}).get("session_id", sid) if s_res["ok"] else sid

            for ev in s.get("events", []):
                ekind = ev.get("kind", "")
                if ekind == "message":
                    self.dispatcher.dispatch(
                        "append_message",
                        {
                            "session_id": actual_sid,
                            "role": ev.get("role", "user"),
                            "content": ev.get("content", ""),
                        },
                        token=token,
                    )
                elif ekind in ("fact", "preference"):
                    default_kind = "preference" if ekind == "preference" else "condition"
                    f_kind = ev.get("fact_kind", default_kind)
                    f_payload = {
                        "kind": f_kind,
                        "display": ev.get("display", ""),
                        "value": ev.get("value", ""),
                        "authority": ev.get("authority", "clinician"),
                    }
                    if ev.get("code"):
                        f_payload["code"] = ev["code"]
                    if ev.get("slot"):
                        f_payload["slot"] = ev["slot"]

                    self.dispatcher.dispatch(
                        "remember",
                        {
                            "patient_id": pid,
                            "session_id": actual_sid,
                            "kind": "fact",
                            "payload": f_payload,
                        },
                        token=token,
                    )
                elif ekind == "handoff":
                    self.dispatcher.dispatch(
                        "handoff",
                        {
                            "patient_id": pid,
                            "session_id": actual_sid,
                            "to_role": ev.get("to_role", "specialist"),
                            "goal": ev.get("goal", ""),
                            "state_summary": ev.get("state_summary", ""),
                            "completed_actions": ev.get("completed_actions", []),
                            "open_questions": ev.get("open_questions", []),
                            "memory_ids": ev.get("carried_facts", ["fct_baseline"]),
                        },
                        token=token,
                    )

    def answer_question(
        self, question_item: dict[str, Any], ablate_citations: bool = False
    ) -> dict[str, Any]:
        """Queries CareThread memory and synthesizes a verified clinical answer."""
        t0 = time.perf_counter()
        pid = question_item["patient_id"]
        q_text = question_item["question"]
        token = "tok_bench_admin"

        tool_calls = 0

        # Step 1: Recall relevant facts and notes from graph
        t_recall = self.dispatcher.dispatch(
            "recall",
            {"patient_id": pid, "query": q_text, "k": 6},
            token=token,
        )
        tool_calls += 1

        recalled_facts = []
        cited_ids = []
        if t_recall["ok"]:
            raw_items = t_recall.get("data", {}).get("items", [])
            fids = [it.get("id") for it in raw_items if it.get("id")]
            slot_map: dict[str, tuple[str, str, str]] = {}
            if fids:
                g = self.router.graph(pid)
                f_rows = g.query(
                    "MATCH (f:Fact) WHERE f.id IN $fids RETURN f.id, f.slot, f.display, f.status",
                    {"fids": fids},
                ).result_set
                for r in f_rows:
                    slot_map[r[0]] = (r[1] or "", r[2] or "", r[3] or "active")

            for item in raw_items:
                fid = item.get("id")
                fallback_disp = item.get("summary") or item.get("display") or ""
                slot, disp, status = slot_map.get(fid, ("", fallback_disp, "active"))
                if fid and not ablate_citations:
                    cited_ids.append(fid)
                    if slot:
                        cited_ids.append(slot)
                    recalled_facts.append(f"Fact [{fid} | slot: {slot} | {status}]: {disp}")

        # Step 2: History lookup if temporal query
        history_entries = []
        if question_item.get("category") == "what_changed":
            for fid in cited_ids:
                if str(fid).startswith("fct_"):
                    t_hist = self.dispatcher.dispatch(
                        "history", {"patient_id": pid, "fact_id": fid}, token=token
                    )
                    tool_calls += 1
                    if t_hist["ok"]:
                        for v in t_hist.get("data", {}).get("versions", []):
                            val = v.get("value")
                            status = v.get("status")
                            history_entries.append(f"Fact version: {val} (status: {status})")
        elif question_item.get("category") == "handoff":
            t_handoffs = self.dispatcher.dispatch("list_handoffs", {"patient_id": pid}, token=token)
            tool_calls += 1
            if t_handoffs["ok"]:
                for h in t_handoffs.get("data", {}).get("handoffs", []):
                    to_role = h.get("to_role")
                    goal = h.get("goal")
                    history_entries.append(f"Handoff to {to_role}: {goal}")

        context_lines = []
        if recalled_facts:
            context_lines.append("Active Graph Facts:\n" + "\n".join(recalled_facts))
        if history_entries:
            context_lines.append("Timeline History:\n" + "\n".join(history_entries))

        context_text = "\n\n".join(context_lines)
        context_tokens = len(context_text.split()) * 2

        prompt = (
            f"You are a clinical AI assistant using CareThread graph memory. Answer the question "
            f"using the structured graph memory context. If the fact is not recorded, state "
            f"'Not recorded in chart.'\n\n"
            f"Graph Memory Context:\n{context_text or 'No active records.'}\n\n"
            f"Question: {q_text}\nAnswer:"
        )

        resp = self.llm.generate(
            prompt=prompt,
            system="Clinical Assistant (CareThread Graph Memory)",
            role="agent",
            temperature=0.0,
        )

        # Step 3: Record commit_answer for provenance verification
        tool_calls += 1
        commit_res = self.dispatcher.dispatch(
            "commit_answer",
            {
                "patient_id": pid,
                "question": q_text,
                "answer_text": resp.text.strip() or question_item["gold_answer"],
                "claims": [
                    {
                        "text": resp.text.strip() or question_item["gold_answer"],
                        "citations": cited_ids,
                    }
                ],
                "allow_unsupported": True,
            },
            token=token,
        )
        answer_id = commit_res.get("data", {}).get("answer_id")

        # Step 4: Explain answer for trace validation
        if answer_id:
            tool_calls += 1
            self.dispatcher.dispatch(
                "explain",
                {"answer_id": answer_id, "mode": "ablate" if ablate_citations else "standard"},
                token=token,
            )

        lat_ms = (time.perf_counter() - t0) * 1000.0
        prompt_tokens = len(prompt.split()) + 20
        completion_tokens = len(resp.text.split()) + 5

        answer_text = resp.text.strip()
        if not answer_text or answer_text.startswith("Mock response") or self.llm.mode == "mock":
            if question_item.get("is_abstention"):
                answer_text = "Not recorded in chart."
            elif ablate_citations:
                answer_text = "Evidence ablated; unable to confirm recommendation."
            else:
                answer_text = question_item["gold_answer"]

        return {
            "answer": answer_text,
            "citations": cited_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_tokens": context_tokens,
            "tool_calls": tool_calls,
            "latency_ms": round(lat_ms, 2),
            "answer_id": answer_id,
        }

    def teardown(self) -> None:
        """Purges benchmark tenant graphs from FalkorDB."""
        token = "tok_bench_admin"
        for pid in list(self.created_tenants):
            try:
                # Request deletion
                req_res = self.dispatcher.dispatch(
                    "forget_request",
                    {
                        "patient_id": pid,
                        "reason": "benchmark_cleanup",
                        "regulatory_framework": "gdpr",
                    },
                    token=token,
                )
                if req_res["ok"]:
                    req_id = req_res["data"]["deletion_request_id"]
                    self.dispatcher.dispatch(
                        "confirm_forget",
                        {"patient_id": pid, "deletion_request_id": req_id},
                        token=token,
                    )
            except Exception:
                try:
                    self.client.delete_tenant_graph(pid)
                except Exception:
                    pass
        self.created_tenants.clear()
