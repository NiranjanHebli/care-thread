"""Tenant isolation tests verifying dense vector search isolation (T-ISO-07).

Validates:
- Seeding identical clinical note text in tenant A and tenant B (yielding identical
  vector embeddings) remains strictly isolated.
- Vector recall in tenant A returns only tenant A's note ID.
- Zero nodes from tenant B are returned in retrieved or rejected sets.
"""

from __future__ import annotations

from typing import Any

import pytest

from carethread.clock import FixedClock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.memory.embeddings import HashEmbedder
from carethread.memory.recall import RecallManager

PID_VEC_A = "p_150a00000007"
PID_VEC_B = "p_150b00000007"
IDENTICAL_NOTE_TEXT = (
    "Patient reports severe allergic reaction to amoxicillin with "
    "facial swelling, dyspnea, and urticaria."
)


@pytest.fixture
def vector_isolation_fixture():
    """Sets up two tenant graphs with identical clinical note text."""
    client = get_db_client()
    router = TenantRouter(client)
    clock = FixedClock(1792190000)
    embedder = HashEmbedder()

    for pid in (PID_VEC_A, PID_VEC_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass
        router.register_patient(pid, bootstrap_schema=True)

    g_a = router.graph(PID_VEC_A)
    g_a.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient Vec A'",
        {"pid": PID_VEC_A},
    )

    g_b = router.graph(PID_VEC_B)
    g_b.query(
        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Patient Vec B'",
        {"pid": PID_VEC_B},
    )

    recall_mgr = RecallManager(router=router, clock=clock, embedder=embedder)

    # Seed identical note text into both graphs
    note_a = recall_mgr.add_note(
        patient_id=PID_VEC_A,
        text=IDENTICAL_NOTE_TEXT,
        kind="clinical",
        author_agent_id="agt_intake_01",
    )

    note_b = recall_mgr.add_note(
        patient_id=PID_VEC_B,
        text=IDENTICAL_NOTE_TEXT,
        kind="clinical",
        author_agent_id="agt_intake_01",
    )

    assert note_a.id != note_b.id, "Generated note IDs must be distinct"

    yield {
        "client": client,
        "router": router,
        "recall_mgr": recall_mgr,
        "note_a_id": note_a.id,
        "note_b_id": note_b.id,
    }

    # Teardown
    for pid in (PID_VEC_A, PID_VEC_B):
        try:
            router.graph(pid).delete()
        except Exception:
            pass
        try:
            router.deregister_patient(pid)
        except Exception:
            pass


def test_t_iso_07_identical_note_vector_recall_isolation(
    vector_isolation_fixture: dict[str, Any],
) -> None:
    """T-ISO-07: Vector recall in tenant A returns only tenant A's note ID, never tenant B's."""
    recall_mgr: RecallManager = vector_isolation_fixture["recall_mgr"]
    note_a_id: str = vector_isolation_fixture["note_a_id"]
    note_b_id: str = vector_isolation_fixture["note_b_id"]

    # 1. Query Tenant A with exact note text
    res_a = recall_mgr.recall(
        patient_id=PID_VEC_A,
        query=IDENTICAL_NOTE_TEXT,
        include_notes=True,
        top_k=5,
        include_rejected=True,
    )

    retrieved_a_ids = [n.id for n in res_a.notes]
    rejected_a_ids = [r.id for r in res_a.rejected]
    all_returned_a_ids = set(retrieved_a_ids + rejected_a_ids)

    # Tenant A must find its own note
    assert note_a_id in retrieved_a_ids, (
        f"Expected note_a_id '{note_a_id}' in tenant A recall results"
    )
    # Tenant A must NEVER see tenant B's note
    assert note_b_id not in all_returned_a_ids, (
        f"Cross-tenant leakage! Note '{note_b_id}' returned in A recall!"
    )

    # 2. Query Tenant B with the same note text
    res_b = recall_mgr.recall(
        patient_id=PID_VEC_B,
        query=IDENTICAL_NOTE_TEXT,
        include_notes=True,
        top_k=5,
        include_rejected=True,
    )

    retrieved_b_ids = [n.id for n in res_b.notes]
    rejected_b_ids = [r.id for r in res_b.rejected]
    all_returned_b_ids = set(retrieved_b_ids + rejected_b_ids)

    # Tenant B must find its own note
    assert note_b_id in retrieved_b_ids, (
        f"Expected note_b_id '{note_b_id}' in tenant B recall results"
    )
    # Tenant B must NEVER see tenant A's note
    assert note_a_id not in all_returned_b_ids, (
        f"Cross-tenant leakage! Note '{note_a_id}' returned in B recall!"
    )
