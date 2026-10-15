"""Unit tests for Synthea CSV parsing, PII stripping, and medication normalization.

Verifies:
- Column allowlist and PII stripping at read time.
- Pure function normalize_med() table-driven test.
- Parsing of committed synthetic fixture dataset (tests/fixtures/synthea_csv).
"""

from pathlib import Path

import pytest

from carethread.ingest.synthea_csv import (
    PATIENTS_ALLOWLIST,
    PROHIBITED_PII_COLUMNS,
    load_synthea_dataset,
    normalize_med,
    parse_synthea_date,
    read_csv_with_allowlist,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "synthea_csv"


# -----------------------------------------------------------------------------
# Table-driven normalize_med() test
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_input", "expected_ingredient"),
    [
        ("Metformin hydrochloride 500 MG Oral Tablet", "metformin"),
        ("Lisinopril 10 MG Oral Tablet", "lisinopril"),
        ("Insulin Glargine 100 UNT/ML Injectable Solution", "insulin glargine"),
        ("Penicillin V potassium 250 MG Oral Tablet", "penicillin"),
        ("Hydrochlorothiazide 25 MG Oral Tablet", "hydrochlorothiazide"),
        ("Atorvastatin 20 MG Oral Tablet", "atorvastatin"),
        ("Simvastatin 40 MG Oral Tablet", "simvastatin"),
        ("Levothyroxine sodium 0.05 MG Oral Tablet", "levothyroxine"),
        ("Albuterol 0.09 MG/ACTUAT Inhalation Aerosol", "albuterol"),
        ("Amoxicillin 500 MG Oral Tablet", "amoxicillin"),
        ("Amoxicillin / Clavulanate 875 MG Oral Tablet", "amoxicillin"),
        ("Metoprolol tartrate 50 MG Oral Tablet", "metoprolol"),
        ("Metoprolol succinate 100 MG Extended Release Oral Tablet", "metoprolol"),
        ("Warfarin sodium 5 MG Oral Tablet", "warfarin"),
        ("Aspirin 81 MG Chewable Tablet", "aspirin"),
        ("Acetaminophen 325 MG Oral Tablet", "acetaminophen"),
        ("Amlodipine 5 MG Oral Tablet", "amlodipine"),
        ("Cephalexin 500 MG Oral Capsule", "cephalexin"),
        ("Ciprofloxacin 500 MG Oral Tablet", "ciprofloxacin"),
        ("Glipizide 5 MG Oral Tablet", "glipizide"),
        ("Ibuprofen 400 MG Oral Tablet", "ibuprofen"),
        # Fallback to first alphabetic token when not in canonical list
        ("NovelDrugX 100 MG Oral Tablet", "noveldrugx"),
        ("ZzzMedication 20 MG", "zzzmedication"),
        ("", ""),
    ],
)
def test_normalize_med_table_driven(raw_input: str, expected_ingredient: str) -> None:
    """Verifies that normalize_med produces canonical active ingredients with fallback."""
    result = normalize_med(raw_input)
    assert result == expected_ingredient, (
        f"normalize_med({raw_input!r}) returned {result!r}, expected {expected_ingredient!r}"
    )


# -----------------------------------------------------------------------------
# Column Allowlist and PII Stripping
# -----------------------------------------------------------------------------


def test_patients_csv_pii_strictly_stripped() -> None:
    """SSN, driver's license, passport, address and unlisted columns are dropped."""
    patients_file = FIXTURES_DIR / "patients.csv"
    assert patients_file.is_file(), f"Fixture missing: {patients_file}"

    rows = read_csv_with_allowlist(patients_file, PATIENTS_ALLOWLIST)
    assert len(rows) == 10

    for row in rows:
        # 1. Allowed columns MUST be present
        assert "Id" in row
        assert "BIRTHDATE" in row
        assert "FIRST" in row
        assert "LAST" in row
        assert "GENDER" in row

        # 2. Prohibited PII columns MUST NEVER be present in the parsed output
        for pii_col in PROHIBITED_PII_COLUMNS:
            assert pii_col not in row, (
                f"Security violation: PII column '{pii_col}' found in parsed row!"
            )


def test_allowlist_drops_unknown_arbitrary_columns(tmp_path: Path) -> None:
    """Verifies that arbitrary unknown columns are ignored and omitted."""
    custom_csv = tmp_path / "custom_test.csv"
    custom_csv.write_text(
        "Id,FIRST,LAST,BIRTHDATE,GENDER,SECRET_KEY,INTERNAL_TRACKING,SSN\n"
        "p001,John,Doe,1980-01-01,M,xyz123,track_456,999-00-1111\n",
        encoding="utf-8",
    )

    rows = read_csv_with_allowlist(custom_csv, PATIENTS_ALLOWLIST)
    assert len(rows) == 1
    row = rows[0]

    assert set(row.keys()) == {"Id", "FIRST", "LAST", "BIRTHDATE", "GENDER"}
    assert "SECRET_KEY" not in row
    assert "INTERNAL_TRACKING" not in row
    assert "SSN" not in row


# -----------------------------------------------------------------------------
# Date Parsing Tests
# -----------------------------------------------------------------------------


def test_parse_synthea_date() -> None:
    """Verifies parsing of ISO dates and timestamps into epoch seconds."""
    # Date only
    ts_date = parse_synthea_date("2023-01-10")
    assert ts_date is not None
    assert ts_date > 0

    # UTC ISO timestamp
    ts_iso = parse_synthea_date("2023-01-10T09:00:00Z")
    assert ts_iso is not None
    assert ts_iso > ts_date

    # None and empty handling
    assert parse_synthea_date(None) is None
    assert parse_synthea_date("") is None
    assert parse_synthea_date("   ") is None


# -----------------------------------------------------------------------------
# Synthetic Fixture Dataset Parsing Tests
# -----------------------------------------------------------------------------


def test_load_synthea_fixture_dataset() -> None:
    """Verifies complete parsing of the committed synthetic fixture dataset."""
    dataset = load_synthea_dataset(FIXTURES_DIR)

    # 1. Ten patients loaded
    assert len(dataset.patients) == 10
    assert len(dataset.patient_ids) == 10

    # 2. Patient 1 has diabetes + penicillin allergy (target demo patient)
    p1_id = "a1000001-0000-0000-0000-000000000001"
    p1 = dataset.get_patient_record(p1_id)
    assert p1 is not None
    assert p1.name == "John Doe"
    assert p1.gender == "M"

    # Patient 1 conditions
    p1_conditions = dataset.conditions_by_patient.get(p1_id, [])
    p1_cond_codes = {c.code for c in p1_conditions}
    assert "44054006" in p1_cond_codes  # Diabetes mellitus type 2

    # Patient 1 allergies
    p1_allergies = dataset.allergies_by_patient.get(p1_id, [])
    p1_allg_codes = {a.code for a in p1_allergies}
    assert "373270004" in p1_allg_codes  # Allergy to penicillin

    # Patient 1 medications
    p1_medications = dataset.medications_by_patient.get(p1_id, [])
    p1_ingredients = {m.ingredient for m in p1_medications}
    assert "metformin" in p1_ingredients
    assert "lisinopril" in p1_ingredients

    # Patient 1 encounters and observations
    p1_encounters = dataset.encounters_by_patient.get(p1_id, [])
    assert len(p1_encounters) >= 2
    # Chronological ordering check
    assert p1_encounters[0].start_ts <= p1_encounters[1].start_ts

    p1_observations = dataset.observations_by_patient.get(p1_id, [])
    assert len(p1_observations) >= 3

    # Counts helper
    counts = dataset.patient_counts(p1_id)
    assert counts["encounters"] == len(p1_encounters)
    assert counts["conditions"] == len(p1_conditions)
    assert counts["medications"] == len(p1_medications)
    assert counts["allergies"] == len(p1_allergies)

    # Total counts
    totals = dataset.total_counts()
    assert totals["patients"] == 10
    assert totals["encounters"] == 12
    assert totals["conditions"] == 11
    assert totals["medications"] == 11
    assert totals["allergies"] == 3
    assert totals["procedures"] == 9
    assert totals["observations"] == 15
    assert totals["providers"] == 3
    assert totals["organizations"] == 2


def test_missing_directory_raises_error() -> None:
    """Verifies that non-existent directory raises NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        load_synthea_dataset("/path/that/does/not/exist")
