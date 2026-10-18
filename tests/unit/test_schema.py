"""Unit and integration test suite for FalkorDB schema bootstrap and indexes."""

import uuid

import pytest
from falkordb import FalkorDB
from redis.exceptions import ResponseError

from carethread.db.client import get_db_client
from carethread.db.queries import get_query, list_queries
from carethread.db.schema import (
    PATIENT_RANGE_INDEXES,
    PATIENT_UNIQUE_CONSTRAINTS,
    PATIENT_VECTOR_INDEXES,
    SchemaBootstrapResult,
    bootstrap_all,
    bootstrap_clinic_graph,
    bootstrap_control_graph,
    bootstrap_patient_graph,
    get_existing_constraints,
    get_existing_indexes,
)
from carethread.db.tenant_router import TenantRouter


@pytest.fixture(scope="module")
def db_client() -> FalkorDB:
    return get_db_client()


@pytest.fixture
def temp_graph_name() -> str:
    return f"patient_p_{uuid.uuid4().hex[:12]}"


def test_schema_bootstrap_patient_graph(db_client: FalkorDB, temp_graph_name: str) -> None:
    """Verifies complete bootstrap of patient tenant schema (indexes, constraints, vectors)."""
    graph = db_client.select_graph(temp_graph_name)
    try:
        # First pass: creates all indexes and constraints
        res1 = bootstrap_patient_graph(graph, db_client)
        assert isinstance(res1, SchemaBootstrapResult)
        assert len(res1.errors) == 0, f"Bootstrap errors: {res1.errors}"
        assert res1.range_indexes_created == len(PATIENT_RANGE_INDEXES)
        assert res1.constraints_created == len(PATIENT_UNIQUE_CONSTRAINTS)
        assert res1.vector_indexes_created == len(PATIENT_VECTOR_INDEXES)
        assert res1.fulltext_indexes_created == 1
        assert res1.total_created > 0
        assert res1.total_existed == 0

        # Second pass: idempotent verification
        res2 = bootstrap_patient_graph(graph, db_client)
        assert len(res2.errors) == 0, f"Idempotent pass errors: {res2.errors}"
        assert res2.total_created == 0
        assert res2.range_indexes_existed == len(PATIENT_RANGE_INDEXES)
        assert res2.constraints_existed == len(PATIENT_UNIQUE_CONSTRAINTS)
        assert res2.vector_indexes_existed == len(PATIENT_VECTOR_INDEXES)
        assert res2.fulltext_indexes_existed == 1

        # Check introspected indexes
        indexes = get_existing_indexes(graph)
        indexed_labels = {idx["label"] for idx in indexes}
        assert "Patient" in indexed_labels
        assert "Episode" in indexed_labels
        assert "Fact" in indexed_labels
        assert "Note" in indexed_labels
        assert "Message" in indexed_labels

        # Check introspected constraints
        constraints = get_existing_constraints(graph)
        constraint_labels = {c["label"] for c in constraints}
        assert "Patient" in constraint_labels
        assert "Episode" in constraint_labels
        assert "Fact" in constraint_labels
        assert "Message" in constraint_labels
    finally:
        graph.delete()


def test_unique_node_constraint_enforcement(db_client: FalkorDB, temp_graph_name: str) -> None:
    """Verifies that unique node constraints reject duplicate node IDs."""
    graph = db_client.select_graph(temp_graph_name)
    try:
        bootstrap_patient_graph(graph, db_client)

        # 1. First insert should succeed
        graph.query("CREATE (:Patient {id: 'p_test_unique_01', name: 'Original'})")

        # 2. Second insert with duplicate ID must fail
        with pytest.raises(ResponseError) as exc_info:
            graph.query("CREATE (:Patient {id: 'p_test_unique_01', name: 'Duplicate'})")

        assert "unique constraint violation" in str(exc_info.value).lower()
    finally:
        graph.delete()


def test_compound_unique_constraint_enforcement(db_client: FalkorDB, temp_graph_name: str) -> None:
    """Verifies that compound unique constraint Message(session_id, seq) blocks duplicate seq."""
    graph = db_client.select_graph(temp_graph_name)
    try:
        bootstrap_patient_graph(graph, db_client)

        # 1. Insert message with (session_id: 's1', seq: 1)
        graph.query("CREATE (:Message {id: 'm1', session_id: 's1', seq: 1, content: 'msg 1'})")

        # 2. Insert message with same session_id and different seq: should succeed
        graph.query("CREATE (:Message {id: 'm2', session_id: 's1', seq: 2, content: 'msg 2'})")

        # 3. Insert duplicate (session_id: 's1', seq: 1) must fail
        with pytest.raises(ResponseError) as exc_info:
            graph.query(
                "CREATE (:Message {id: 'm3', session_id: 's1', seq: 1, content: 'duplicate seq'})"
            )

        assert "unique constraint violation" in str(exc_info.value).lower()
    finally:
        graph.delete()


def test_vector_index_search(db_client: FalkorDB, temp_graph_name: str) -> None:
    """Verifies 384-dimensional vector indexing and cosine distance search."""
    graph = db_client.select_graph(temp_graph_name)
    try:
        bootstrap_patient_graph(graph, db_client)

        # Prepare two 384-dimensional vectors
        vec_a = [0.0] * 384
        vec_a[0] = 1.0  # Unit vector along dimension 0

        vec_b = [0.0] * 384
        vec_b[1] = 1.0  # Unit vector along dimension 1 (orthogonal)

        # Insert notes
        graph.query(
            "CREATE (:Note {id: 'note_01', text: 'Note A', embedding: vecf32($emb)})",
            {"emb": vec_a},
        )
        graph.query(
            "CREATE (:Note {id: 'note_02', text: 'Note B', embedding: vecf32($emb)})",
            {"emb": vec_b},
        )

        # Query with vec_a
        query_cypher = (
            "CALL db.idx.vector.queryNodes('Note', 'embedding', 2, vecf32($qvec)) "
            "YIELD node, score RETURN node.id AS id, score"
        )
        res = graph.query(query_cypher, {"qvec": vec_a})
        rows = res.result_set
        assert len(rows) == 2

        # Most similar should be note_01 with distance ~0.0
        assert rows[0][0] == "note_01"
        assert abs(rows[0][1] - 0.0) < 1e-4

        # Orthogonal should be note_02 with distance ~1.0
        assert rows[1][0] == "note_02"
        assert abs(rows[1][1] - 1.0) < 1e-4
    finally:
        graph.delete()


def test_clinic_and_control_graph_bootstraps(db_client: FalkorDB) -> None:
    """Verifies bootstrap of clinic_shared and ct_control graphs."""
    control_g = db_client.select_graph(f"test_ctrl_{uuid.uuid4().hex[:8]}")
    clinic_g = db_client.select_graph(f"test_clnc_{uuid.uuid4().hex[:8]}")
    try:
        ctrl_res = bootstrap_control_graph(control_g, db_client)
        assert len(ctrl_res.errors) == 0
        assert ctrl_res.range_indexes_created == 1
        assert ctrl_res.constraints_created == 1

        clnc_res = bootstrap_clinic_graph(clinic_g, db_client)
        assert len(clnc_res.errors) == 0
        assert clnc_res.range_indexes_created >= 8
        assert clnc_res.constraints_created >= 8
        assert clnc_res.vector_indexes_created == 1
    finally:
        control_g.delete()
        clinic_g.delete()


def test_tenant_router_bootstrap_integration(db_client: FalkorDB) -> None:
    """Verifies TenantRouter.bootstrap_patient and register_patient with bootstrap_schema=True."""
    router = TenantRouter(db_client)
    test_pid = "p_555555555555"

    try:
        # Register with automatic schema bootstrap
        reg = router.register_patient(test_pid, bootstrap_schema=True)
        assert reg["id"] == test_pid
        assert reg["bootstrap_result"] is not None
        assert isinstance(reg["bootstrap_result"], SchemaBootstrapResult)
        assert len(reg["bootstrap_result"].errors) == 0
        assert reg["bootstrap_result"].total_created > 0

        # Second explicit bootstrap call is idempotent
        res2 = router.bootstrap_patient(test_pid)
        assert res2.total_created == 0
        assert res2.total_existed > 0
    finally:
        router.deregister_patient(test_pid)
        try:
            router.graph(test_pid).delete()
        except Exception:
            pass


def test_cypher_query_files_validity() -> None:
    """Verifies that all parameterized .cypher query files exist and can be loaded."""
    queries = list_queries()
    expected_queries = [
        "q_recall_01",
        "q_recall_02",
        "q_recall_03a",
        "q_recall_03b",
        "q_ep_01a",
        "q_ep_01b",
        "q_sup_01",
        "q_sup_02",
        "q_msg_01a",
        "q_msg_01b",
        "q_ho_01a",
        "q_ho_01b",
        "q_ho_02a",
        "q_ho_02b",
        "q_exp_01a",
        "q_exp_01b",
        "q_exp_01c",
        "q_safe_01a",
        "q_safe_01b",
        "q_pb_01",
        "q_cons_01",
        "q_cons_02",
        "q_aud_01",
    ]

    for qname in expected_queries:
        assert qname in queries, f"Missing query file: {qname}.cypher"
        content = get_query(qname)
        assert len(content) > 10, f"Query {qname} is empty or too short"
        assert "MATCH" in content or "CALL" in content, f"Query {qname} missing standard keyword"


def test_bootstrap_all_execution(db_client: FalkorDB) -> None:
    """Verifies that bootstrap_all runs across control, clinic, and tenants."""
    results = bootstrap_all(db_client)
    assert "ct_control" in results
    assert "clinic_shared" in results
    assert len(results["ct_control"].errors) == 0
    assert len(results["clinic_shared"].errors) == 0
