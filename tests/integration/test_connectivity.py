"""FalkorDB connectivity and configuration verification test suite."""

import os
import uuid

import pytest
from falkordb import FalkorDB


@pytest.fixture(scope="module")
def db_client() -> FalkorDB:
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD") or None

    client = FalkorDB(host=host, port=port, password=password)
    return client


def test_falkordb_ping(db_client: FalkorDB) -> None:
    """Verify that FalkorDB responds to ping."""
    pong = db_client.connection.ping()
    assert pong is True or pong == b"PONG" or pong == "PONG"


def test_falkordb_aof_persistence(db_client: FalkorDB) -> None:
    """Verify that Append-Only File (AOF) persistence is enabled."""
    config = db_client.connection.config_get("appendonly")
    assert config.get("appendonly") == "yes" or config.get(b"appendonly") in ("yes", b"yes")


def test_falkordb_graph_crud_and_query(db_client: FalkorDB) -> None:
    """Verify graph creation, Cypher write, Cypher read, and cleanup."""
    graph_name = f"test_conn_{uuid.uuid4().hex[:8]}"
    graph = db_client.select_graph(graph_name)

    try:
        # 1. Create a node
        create_res = graph.query(
            "CREATE (p:Patient {id: 'p_test_01', name: 'Synthea synthetic'}) "
            "RETURN p.id AS id, p.name AS name"
        )
        assert len(create_res.result_set) == 1
        assert create_res.result_set[0][0] == "p_test_01"
        assert create_res.result_set[0][1] == "Synthea synthetic"

        # 2. Query node back
        read_res = graph.query(
            "MATCH (p:Patient {id: $pid}) RETURN p.name AS name",
            {"pid": "p_test_01"},
        )
        assert len(read_res.result_set) == 1
        assert read_res.result_set[0][0] == "Synthea synthetic"

        # 3. Test openCypher aggregation and UNWIND
        agg_res = graph.query("UNWIND [10, 20, 30] AS val RETURN sum(val) AS total")
        assert len(agg_res.result_set) == 1
        assert agg_res.result_set[0][0] == 60
    finally:
        # Cleanup test graph
        graph.delete()
