"""Unit tests for clinic graph seeding (clinic_shared) and clinical knowledge queries."""

from __future__ import annotations

from pathlib import Path

import pytest
from falkordb import FalkorDB

from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.seed_clinic import DEFAULT_CLINIC_DATA_DIR, ClinicSeeder, seed_clinic
from carethread.memory.embeddings import HashEmbedder


@pytest.fixture(scope="module")
def db_client() -> FalkorDB:
    return get_db_client()


@pytest.fixture(scope="module")
def clinic_reader(db_client: FalkorDB) -> ClinicReader:
    return ClinicReader(db_client)


@pytest.fixture(scope="module")
def seeded_clinic(db_client: FalkorDB):
    embedder = HashEmbedder()
    return seed_clinic(
        client=db_client,
        clinic_data_dir=DEFAULT_CLINIC_DATA_DIR,
        embedder=embedder,
        bootstrap_schema=True,
    )


def test_clinic_seeding_counts_and_verification(seeded_clinic) -> None:
    """Verifies all clinical knowledge entities are seeded with positive counts."""
    assert seeded_clinic.verified is True
    assert seeded_clinic.graph_name == "clinic_shared"
    assert seeded_clinic.guidelines_count == 4
    assert seeded_clinic.rules_count == 11
    assert seeded_clinic.drugs_count == 23
    assert seeded_clinic.drug_classes_count == 12
    assert seeded_clinic.allergen_classes_count == 5
    assert seeded_clinic.allergen_codes_count == 13
    assert seeded_clinic.playbook_templates_count == 4
    assert seeded_clinic.total_nodes >= 70
    assert seeded_clinic.total_edges >= 60


def test_clinic_seeding_idempotency(db_client: FalkorDB) -> None:
    """Running seed_clinic twice must result in identical counts with zero duplicates."""
    embedder = HashEmbedder()
    seeder = ClinicSeeder(client=db_client, embedder=embedder)

    res1 = seeder.seed_from_directory(DEFAULT_CLINIC_DATA_DIR, bootstrap_schema=False)
    res2 = seeder.seed_from_directory(DEFAULT_CLINIC_DATA_DIR, bootstrap_schema=False)

    assert res1.total_nodes == res2.total_nodes
    assert res1.total_edges == res2.total_edges
    assert res1.drugs_count == res2.drugs_count
    assert res1.guidelines_count == res2.guidelines_count


def test_safety_query_q_safe_01a_cross_reactivity(clinic_reader: ClinicReader) -> None:
    """Verifies that q_safe_01a.cypher traverses drug -> class -> allergen class -> member codes."""
    query_file = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "carethread"
        / "db"
        / "queries"
        / "q_safe_01a.cypher"
    )
    query_cypher = query_file.read_text(encoding="utf-8").strip()

    # Cefazolin (1665060) is a cephalosporin that cross-reacts with beta-lactam allergens
    res = clinic_reader.ro_query(query_cypher, {"drug_code": "1665060"})
    rows = res.result_set
    assert len(rows) == 1
    drug_class_id, allergen_class_id, allergen_codes = rows[0]

    assert drug_class_id == "dc_cephalosporins"
    assert allergen_class_id == "alg_beta_lactam"
    # Must include Penicillin and Amoxicillin allergy SNOMED codes
    assert "300916003" in allergen_codes  # Allergy to penicillin
    assert "300913006" in allergen_codes  # Allergy to amoxicillin
    assert "7980" in allergen_codes  # Penicillin class allergy


def test_drug_drug_interaction_query(clinic_reader: ClinicReader) -> None:
    """Verifies drug-drug interaction relationships between Lisinopril and Ibuprofen."""
    query = """
    MATCH (da:Drug {code: $code_a})-[r:INTERACTS_WITH]->(db:Drug {code: $code_b})
    RETURN da.name AS drug_a, db.name AS drug_b, r.severity AS severity, r.effect AS effect
    """
    # 314076 (Lisinopril) and 310965 (Ibuprofen)
    res = clinic_reader.ro_query(query, {"code_a": "314076", "code_b": "310965"})
    assert len(res.result_set) == 1
    row = res.result_set[0]
    assert "lisinopril" in row[0].lower()
    assert "ibuprofen" in row[1].lower()
    assert row[2] == "moderate"
    assert "renal" in row[3].lower() or "antihypertensive" in row[3].lower()


def test_guideline_rules_and_recommends(clinic_reader: ClinicReader) -> None:
    """Verifies that guidelines link to rules, conditions, and recommended drug classes."""
    query = """
    MATCH (g:Guideline {id: 'gdl_htn_aha_2024'})-[:HAS_RULE]->(r:Rule)
    RETURN count(r) AS rule_count, collect(r.kind) AS rule_kinds
    """
    res = clinic_reader.ro_query(query)
    rule_count, rule_kinds = res.result_set[0]
    assert rule_count >= 3
    assert "contraindication" in rule_kinds
    assert "checklist" in rule_kinds

    rec_query = """
    MATCH (g:Guideline {id: 'gdl_htn_aha_2024'})-[:RECOMMENDS]->(dc:DrugClass)
    RETURN collect(dc.id) AS recommended_classes
    """
    rec_res = clinic_reader.ro_query(rec_query)
    classes = rec_res.result_set[0][0]
    assert "dc_ace_inhibitors" in classes
    assert "dc_arbs" in classes


def test_guideline_vector_similarity_search(db_client: FalkorDB) -> None:
    """Verifies that Guideline.embedding vector index supports nearest-neighbor search."""
    embedder = HashEmbedder()
    target_graph = db_client.select_graph("clinic_shared")

    # Retrieve guideline text to query with identical embedding
    res = target_graph.query("MATCH (g:Guideline {id: 'gdl_htn_aha_2024'}) RETURN g.text")
    text = res.result_set[0][0]
    qvec = embedder.embed(text)

    vec_res = target_graph.query(
        """
        CALL db.idx.vector.queryNodes('Guideline', 'embedding', 2, vecf32($qvec))
        YIELD node, score
        RETURN node.id AS id, node.title AS title, score
        """,
        {"qvec": qvec},
    )
    rows = vec_res.result_set
    assert len(rows) > 0
    # Top match should be the hypertension guideline with distance ~0.0
    assert rows[0][0] == "gdl_htn_aha_2024"
    assert abs(rows[0][2] - 0.0) < 1e-4


def test_playbook_templates_seeding(clinic_reader: ClinicReader) -> None:
    """Verifies that PlaybookTemplate nodes are seeded with valid steps."""
    res = clinic_reader.ro_query(
        "MATCH (p:PlaybookTemplate {id: 'pb_ed_chest_pain'}) RETURN p.name, p.episode_type, p.steps"
    )
    assert len(res.result_set) == 1
    name, ep_type, steps_json = res.result_set[0]
    assert "Chest Pain" in name
    assert ep_type == "emergency"
    assert "troponin" in steps_json.lower()
