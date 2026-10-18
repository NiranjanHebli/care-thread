"""Tenant isolation tests verifying clinic_shared graph read-only immutability (T-ISO-06).

Validates:
- All attempts to write, mutate, or inject nodes into clinic_shared fail across all
  code paths available to agents (direct ro_query mutation, MCP dispatcher write tools).
- Node count, edge count, and content checksum of clinic_shared remain strictly unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from redis.exceptions import ResponseError

from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.seed_clinic import DEFAULT_CLINIC_DATA_DIR, seed_clinic
from carethread.db.tenant_router import TenantRouter
from carethread.errors import ValidationError
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher
from carethread.memory.embeddings import HashEmbedder


def compute_clinic_checksum(clinic_graph: Any) -> str:
    """Computes a deterministic SHA-256 digest of all nodes and edges in clinic_shared."""
    res_nodes = clinic_graph.ro_query(
        "MATCH (n) RETURN labels(n)[0] AS lbl, n.id AS id, properties(n) AS props ORDER BY id"
    )
    res_edges = clinic_graph.ro_query(
        "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt ORDER BY src, rel, tgt"
    )

    h = hashlib.sha256()
    for row in res_nodes.result_set:
        props = dict(row[2] or {})
        # Omit volatile or floating-point embeddings for deterministic checksum
        props_clean = {k: v for k, v in sorted(props.items()) if not k.endswith("embedding")}
        chunk = f"{row[0]}:{row[1]}:{json.dumps(props_clean, sort_keys=True)}"
        h.update(chunk.encode("utf-8"))

    for row in res_edges.result_set:
        chunk = f"{row[0]}-{row[1]}->{row[2]}"
        h.update(chunk.encode("utf-8"))

    return h.hexdigest()


@pytest.fixture(scope="module")
def clinic_fixture():
    """Seeds clinic_shared and sets up readers and dispatchers."""
    client = get_db_client()
    embedder = HashEmbedder()
    seed_clinic(
        client=client,
        clinic_data_dir=DEFAULT_CLINIC_DATA_DIR,
        embedder=embedder,
        bootstrap_schema=True,
    )

    router = TenantRouter(client)
    reader = ClinicReader(client)
    clinic_g = client.select_graph(ClinicReader.GRAPH_NAME)

    token_store = TokenStore()
    token_store.register_token("tok_iso_admin", "agt_admin", "admin", "*")
    token_store.register_token("tok_iso_intake", "agt_intake", "intake", "*")

    dispatcher = ToolDispatcher(
        router=router,
        token_store=token_store,
    )

    yield {
        "client": client,
        "router": router,
        "reader": reader,
        "clinic_g": clinic_g,
        "dispatcher": dispatcher,
    }


def test_t_iso_06_clinic_shared_write_attempts_fail(clinic_fixture: dict[str, Any]) -> None:
    """T-ISO-06: All write attempts to clinic_shared fail and content remains immutable."""
    reader: ClinicReader = clinic_fixture["reader"]
    clinic_g = clinic_fixture["clinic_g"]
    disp: ToolDispatcher = clinic_fixture["dispatcher"]
    router: TenantRouter = clinic_fixture["router"]

    # 1. Capture baseline metrics and content checksum
    nodes_before = clinic_g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    edges_before = clinic_g.ro_query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    checksum_before = compute_clinic_checksum(clinic_g)

    assert nodes_before > 0
    assert edges_before > 0

    # 2. Code Path A: Attempt direct mutation via ClinicReader.ro_query
    with pytest.raises((ResponseError, Exception)):
        reader.ro_query("CREATE (n:MaliciousRule {id: 'attack_01', text: 'harmful rule'})")

    with pytest.raises((ResponseError, Exception)):
        reader.ro_query("MATCH (g:Guideline) DELETE g")

    with pytest.raises((ResponseError, Exception)):
        reader.ro_query("MATCH (n) SET n.compromised = true")

    # 3. Code Path B: Attempt to pass 'clinic_shared' as patient_id to TenantRouter
    with pytest.raises(ValidationError) as exc:
        router.graph("clinic_shared")
    assert exc.value.code == "VALIDATION_ERROR"

    # 4. Code Path C: Attempt write tools via MCP ToolDispatcher targeting clinic_shared
    mcp_attempts = [
        ("start_session", {"patient_id": "clinic_shared", "channel": "chat"}),
        (
            "remember",
            {
                "patient_id": "clinic_shared",
                "kind": "fact",
                "payload": {"kind": "condition", "display": "Hacked"},
            },
        ),
        (
            "append_message",
            {
                "patient_id": "clinic_shared",
                "session_id": "ses_fake",
                "role": "user",
                "content": "attack",
            },
        ),
        ("commit_answer", {"patient_id": "clinic_shared", "question": "q", "answer_text": "ans"}),
        (
            "handoff",
            {
                "patient_id": "clinic_shared",
                "goal": "g",
                "state_summary": "s",
                "memory_ids": ["m1"],
            },
        ),
        (
            "forget_request",
            {"patient_id": "clinic_shared", "scope": "all", "reason": "wipe clinic"},
        ),
    ]

    for tool_name, params in mcp_attempts:
        res = disp.dispatch(tool_name, params, token="tok_iso_admin")
        assert res["ok"] is False, (
            f"Tool '{tool_name}' unexpectedly succeeded against clinic_shared!"
        )
        assert res["error"]["code"] == "VALIDATION_ERROR"

    # 5. Verify node and edge counts and checksum remain 100% unchanged
    nodes_after = clinic_g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    edges_after = clinic_g.ro_query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    checksum_after = compute_clinic_checksum(clinic_g)

    assert nodes_after == nodes_before, f"Node count changed from {nodes_before} to {nodes_after}!"
    assert edges_after == edges_before, f"Edge count changed from {edges_before} to {edges_after}!"
    assert checksum_after == checksum_before, "Content checksum of clinic_shared changed!"
