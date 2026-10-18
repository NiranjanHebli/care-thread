"""Unit and integration test suite for PatientGraphMapper.

Verifies:
- Ingest fixture twice; per-patient node counts equal CSV row counts; no duplicates.
- MERGE on id; batch with UNWIND; ordering per patient.
- Strict node count equality with CSV rows; data_provenance='synthea'.
- Singly-linked chronological NEXT chain with non-negative gap_days.
- Vector search on ingested Episode summary embeddings.
"""

from pathlib import Path

import pytest
from falkordb import FalkorDB

from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.ingest.mapper import PatientGraphMapper, synthea_id_to_patient_id
from carethread.ingest.synthea_csv import load_synthea_dataset
from carethread.memory.embeddings import HashEmbedder

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "synthea_csv"


@pytest.fixture(scope="module")
def db_client() -> FalkorDB:
    return get_db_client()


@pytest.fixture(scope="module")
def router(db_client: FalkorDB) -> TenantRouter:
    return TenantRouter(db_client)


@pytest.fixture(scope="module")
def fixture_dataset():
    return load_synthea_dataset(FIXTURES_DIR)


def test_idempotent_ingestion_and_count_equality(
    db_client: FalkorDB,
    router: TenantRouter,
    fixture_dataset,
) -> None:
    """Ingest fixture twice; node counts equal CSV row counts; zero duplicates."""
    embedder = HashEmbedder()
    mapper = PatientGraphMapper(router, embedder=embedder)

    patient_ids = fixture_dataset.patient_ids
    assert len(patient_ids) == 10

    test_pid_synthea = patient_ids[0]  # John Doe
    test_pid = synthea_id_to_patient_id(test_pid_synthea)
    target_graph = router.graph(test_pid)

    # Pre-clean to ensure pristine state
    router.deregister_patient(test_pid)
    try:
        target_graph.delete()
    except Exception:
        pass

    try:
        # Run 1: Initial ingestion
        res1 = mapper.ingest_patient(test_pid_synthea, fixture_dataset, target_graph)
        assert res1.verified is True
        assert res1.patient_id == test_pid

        # Verify FalkorDB node counts match CSV counts
        expected_counts = fixture_dataset.patient_counts(test_pid_synthea)
        assert res1.episodes_count == expected_counts["encounters"]
        assert res1.conditions_count == expected_counts["conditions"]
        assert res1.medications_count == expected_counts["medications"]
        assert res1.allergies_count == expected_counts["allergies"]
        assert res1.procedures_count == expected_counts["procedures"]
        assert res1.observations_count == expected_counts["observations"]

        # Capture total node count in graph after Run 1
        cnt_res1 = target_graph.query(
            "MATCH (n) RETURN count(n) AS total_nodes, count(DISTINCT n.id) AS unique_ids"
        )
        total_nodes_run1 = cnt_res1.result_set[0][0]
        unique_ids_run1 = cnt_res1.result_set[0][1]

        # All node IDs must be strictly unique
        assert total_nodes_run1 == unique_ids_run1
        assert total_nodes_run1 > 0

        # Run 2: Ingest the exact same patient a second time
        res2 = mapper.ingest_patient(test_pid_synthea, fixture_dataset, target_graph)
        assert res2.verified is True

        # Capture total node count in graph after Run 2
        cnt_res2 = target_graph.query(
            "MATCH (n) RETURN count(n) AS total_nodes, count(DISTINCT n.id) AS unique_ids"
        )
        total_nodes_run2 = cnt_res2.result_set[0][0]
        unique_ids_run2 = cnt_res2.result_set[0][1]

        # ZERO new nodes or duplicate IDs created
        err_msg = (
            f"Idempotency violation: Run 1 created {total_nodes_run1} nodes, "
            f"Run 2 resulted in {total_nodes_run2}"
        )
        assert total_nodes_run2 == total_nodes_run1, err_msg
        assert unique_ids_run2 == unique_ids_run1

    finally:
        # Cleanup
        router.deregister_patient(test_pid)
        try:
            target_graph.delete()
        except Exception:
            pass


def test_next_chain_structure(
    db_client: FalkorDB,
    router: TenantRouter,
    fixture_dataset,
) -> None:
    """Verifies that episodes form a valid chronological NEXT chain with non-negative gap_days."""
    embedder = HashEmbedder()
    mapper = PatientGraphMapper(router, embedder=embedder)

    test_pid_synthea = fixture_dataset.patient_ids[0]
    test_pid = synthea_id_to_patient_id(test_pid_synthea)
    target_graph = router.graph(test_pid)

    try:
        mapper.ingest_patient(test_pid_synthea, fixture_dataset, target_graph)

        # Query all NEXT edges
        chain_res = target_graph.query(
            """
            MATCH (e1:Episode)-[r:NEXT]->(e2:Episode)
            RETURN e1.id AS from_id, e1.start AS from_start,
                   e2.id AS to_id, e2.start AS to_start,
                   r.gap_days AS gap_days
            ORDER BY e1.start ASC
            """
        )

        rows = chain_res.result_set
        expected_chain_len = len(fixture_dataset.encounters_by_patient[test_pid_synthea]) - 1
        assert len(rows) == expected_chain_len

        for row in rows:
            from_start = row[1]
            to_start = row[3]
            gap_days = row[4]
            # Must be chronologically ascending
            assert to_start >= from_start
            assert gap_days >= 0
    finally:
        router.deregister_patient(test_pid)
        try:
            target_graph.delete()
        except Exception:
            pass


def test_data_provenance_and_fact_slots(
    db_client: FalkorDB,
    router: TenantRouter,
    fixture_dataset,
) -> None:
    """Verifies that all nodes have data_provenance='synthea' and facts have valid slots."""
    embedder = HashEmbedder()
    mapper = PatientGraphMapper(router, embedder=embedder)

    test_pid_synthea = fixture_dataset.patient_ids[0]
    test_pid = synthea_id_to_patient_id(test_pid_synthea)
    target_graph = router.graph(test_pid)

    try:
        mapper.ingest_patient(test_pid_synthea, fixture_dataset, target_graph)

        # Check provenance on Patient, Episode, Fact, Observation, ClinicalProcedure
        prov_res = target_graph.query(
            """
            MATCH (n)
            WHERE n.data_provenance <> 'synthea'
            RETURN count(n) AS non_synthea_count
            """
        )
        assert prov_res.result_set[0][0] == 0, "Found nodes without data_provenance='synthea'"

        # Check fact slots
        fact_res = target_graph.query(
            "MATCH (f:Fact) RETURN f.kind AS kind, f.slot AS slot, f.code AS code"
        )
        for row in fact_res.result_set:
            kind, slot, code = row[0], row[1], row[2]
            if kind == "condition":
                assert slot == f"cond:{code}"
            elif kind == "medication":
                assert slot.startswith("med:")
            elif kind == "allergy":
                assert slot == f"allergy:{code}"

    finally:
        router.deregister_patient(test_pid)
        try:
            target_graph.delete()
        except Exception:
            pass


def test_episode_summary_embedding_vector_search(
    db_client: FalkorDB,
    router: TenantRouter,
    fixture_dataset,
) -> None:
    """Verifies Episode summary embeddings are indexed and searchable via vector queries."""
    embedder = HashEmbedder()
    mapper = PatientGraphMapper(router, embedder=embedder)

    test_pid_synthea = fixture_dataset.patient_ids[0]
    test_pid = synthea_id_to_patient_id(test_pid_synthea)
    target_graph = router.graph(test_pid)

    try:
        mapper.ingest_patient(test_pid_synthea, fixture_dataset, target_graph)

        # Retrieve summary of first episode to query with identical embedding
        first_enc = fixture_dataset.encounters_by_patient[test_pid_synthea][0]
        qvec = embedder.embed(first_enc.summary)

        # Execute vector query against Episode.summary_embedding
        vec_res = target_graph.query(
            """
            CALL db.idx.vector.queryNodes('Episode', 'summary_embedding', 2, vecf32($qvec))
            YIELD node, score
            RETURN node.id AS id, node.summary AS summary, score
            """,
            {"qvec": qvec},
        )

        rows = vec_res.result_set
        assert len(rows) > 0
        # Best match should be the first episode with distance ~0.0
        assert rows[0][0] == f"epi_{first_enc.id}"
        assert abs(rows[0][2] - 0.0) < 1e-4

    finally:
        router.deregister_patient(test_pid)
        try:
            target_graph.delete()
        except Exception:
            pass
