"""Synthea CSV parser with strict column allowlist (PII stripping).

Synthea CSV parser and entity model:
- Reads Synthea CSV exports by header name with strict column allowlists.
- Strips PII columns (SSN, DRIVERS, PASSPORT, ADDRESS, etc.) at read time before mapping.
- Pure function `normalize_med()` with canonical ingredient mapping and fallback.
- Strongly typed records and `SyntheaDataset` container grouped by patient.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Column Allowlists
# Any column not in the allowlist is discarded at read time.
# -----------------------------------------------------------------------------

PATIENTS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Id",
        "BIRTHDATE",
        "DEATHDATE",
        "FIRST",
        "LAST",
        "GENDER",
    }
)

ENCOUNTERS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ORGANIZATION",
        "PROVIDER",
        "ENCOUNTERCLASS",
        "CODE",
        "DESCRIPTION",
        "REASONCODE",
        "REASONDESCRIPTION",
    }
)

CONDITIONS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
    }
)

MEDICATIONS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
    }
)

ALLERGIES_ALLOWLIST: frozenset[str] = frozenset(
    {
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "SYSTEM",
        "DESCRIPTION",
        "TYPE",
        "CATEGORY",
    }
)

PROCEDURES_ALLOWLIST: frozenset[str] = frozenset(
    {
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
    }
)

OBSERVATIONS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CATEGORY",
        "CODE",
        "DESCRIPTION",
        "VALUE",
        "UNITS",
        "TYPE",
    }
)

PROVIDERS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Id",
        "ORGANIZATION",
        "NAME",
        "GENDER",
        "SPECIALITY",
        "SPECIALTY",
    }
)

ORGANIZATIONS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Id",
        "NAME",
        "CITY",
    }
)

# PII columns explicitly banned from ever being retained
PROHIBITED_PII_COLUMNS: frozenset[str] = frozenset(
    {
        "SSN",
        "DRIVERS",
        "PASSPORT",
        "ADDRESS",
        "PREFIX",
        "SUFFIX",
        "MAIDEN",
        "MARITAL",
        "RACE",
        "ETHNICITY",
        "BIRTHPLACE",
        "CITY",
        "STATE",
        "COUNTY",
        "FIPS",
        "ZIP",
        "LAT",
        "LON",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
        "INCOME",
        "PHONE",
    }
)


# -----------------------------------------------------------------------------
# Date Parsing Utility
# -----------------------------------------------------------------------------


def parse_synthea_date(val: str | None) -> int | None:
    """Parses a Synthea ISO-8601 date/timestamp string into epoch seconds (int).

    Handles formats:
    - YYYY-MM-DDTHH:MM:SSZ
    - YYYY-MM-DDTHH:MM:SS
    - YYYY-MM-DD
    Returns None if val is empty or None.
    """
    if not val or not val.strip():
        return None
    cleaned = val.strip()

    # If already a numeric timestamp
    if cleaned.isdigit():
        return int(cleaned)

    # Format: 2023-01-10T09:00:00Z or with timezone
    if "T" in cleaned:
        cleaned_iso = cleaned.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            pass

    # Format: 2023-01-10
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        logger.warning(f"Unable to parse date string: {cleaned!r}")
        return None


# -----------------------------------------------------------------------------
# Medication Normalization
# -----------------------------------------------------------------------------

# Canonical ingredient lookup table
_CANONICAL_INGREDIENTS: dict[str, str] = {
    "metformin hydrochloride": "metformin",
    "metformin": "metformin",
    "insulin glargine": "insulin glargine",
    "insulin lispro": "insulin lispro",
    "insulin aspart": "insulin aspart",
    "penicillin v potassium": "penicillin",
    "penicillin g": "penicillin",
    "penicillin": "penicillin",
    "levothyroxine sodium": "levothyroxine",
    "levothyroxine": "levothyroxine",
    "lisinopril": "lisinopril",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "albuterol": "albuterol",
    "atorvastatin": "atorvastatin",
    "simvastatin": "simvastatin",
    "rosuvastatin": "rosuvastatin",
    "metoprolol tartrate": "metoprolol",
    "metoprolol succinate": "metoprolol",
    "metoprolol": "metoprolol",
    "amlodipine": "amlodipine",
    "amoxicillin": "amoxicillin",
    "amoxicillin / clavulanate": "amoxicillin",
    "amoxicillin and clavulanate": "amoxicillin",
    "warfarin sodium": "warfarin",
    "warfarin": "warfarin",
    "acetaminophen": "acetaminophen",
    "aspirin": "aspirin",
    "glipizide": "glipizide",
    "ciprofloxacin": "ciprofloxacin",
    "cephalexin": "cephalexin",
    "ibuprofen": "ibuprofen",
    "gabapentin": "gabapentin",
    "losartan": "losartan",
    "losartan potassium": "losartan",
    "pantoprazole": "pantoprazole",
    "omeprazole": "omeprazole",
    "prednisone": "prednisone",
    "clopidogrel": "clopidogrel",
    "furosemide": "furosemide",
}

_DOSAGE_AND_FORM_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?\s*(mg|mcg|ml|g|unt/ml|actuat|%|meq|hr|hour))"
    r"|(\b(oral|tablet|capsule|injectable|solution|inhalation|aerosol|powder|suspension|"
    r"extended release|delayed release|chewable|topical|cream|ointment|patch|transdermal)\b)",
    re.IGNORECASE,
)


def normalize_med(description: str) -> str:
    """Normalizes a medication description to its active ingredient.

    Pure function with a table-driven unit test; fallback is the first token.

    Examples:
    - 'Metformin hydrochloride 500 MG Oral Tablet' -> 'metformin'
    - 'Lisinopril 10 MG Oral Tablet' -> 'lisinopril'
    - 'Insulin Glargine 100 UNT/ML Injectable Solution' -> 'insulin glargine'
    - 'Penicillin V potassium 250 MG Oral Tablet' -> 'penicillin'
    - 'UnknownDrugX 10 MG Oral Tablet' -> 'unknowndrugx'
    """
    if not description or not description.strip():
        return ""

    raw = description.strip().lower()

    # 1. Match against known canonical phrases (longest first)
    for canonical_phrase, ingredient in sorted(
        _CANONICAL_INGREDIENTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if raw.startswith(canonical_phrase):
            return ingredient

    # 2. Strip dosage and form terms to isolate the active ingredient
    cleaned = _DOSAGE_AND_FORM_PATTERN.sub("", raw).strip()
    cleaned = re.sub(r"[,\-\/]", " ", cleaned).strip()

    # Check cleaned string against known ingredients again
    for canonical_phrase, ingredient in sorted(
        _CANONICAL_INGREDIENTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if canonical_phrase in cleaned.split():
            return ingredient

    # 3. Fallback: first non-numeric alphabetic token
    tokens = [t for t in cleaned.split() if re.search(r"[a-z]", t)]
    if tokens:
        return tokens[0]

    # Extreme fallback to first raw token
    first_token = re.sub(r"[^a-z0-9]", "", raw.split()[0])
    return first_token


# -----------------------------------------------------------------------------
# Record Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientRecord:
    id: str  # Synthea UUID
    birthdate: str
    first: str
    last: str
    gender: str
    deathdate: str | None = None

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}".strip()

    @property
    def birthdate_ts(self) -> int | None:
        return parse_synthea_date(self.birthdate)


@dataclass(frozen=True)
class EncounterRecord:
    id: str
    start: str
    stop: str | None
    patient_id: str
    organization_id: str | None = None
    provider_id: str | None = None
    encounter_class: str | None = None
    code: str | None = None
    description: str | None = None
    reason_code: str | None = None
    reason_description: str | None = None

    @property
    def start_ts(self) -> int | None:
        return parse_synthea_date(self.start)

    @property
    def stop_ts(self) -> int | None:
        return parse_synthea_date(self.stop)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.description:
            parts.append(self.description.strip())
        if self.reason_description:
            parts.append(f"due to {self.reason_description.strip()}")
        return " ".join(parts) or "Clinical Encounter"


@dataclass(frozen=True)
class ConditionRecord:
    start: str
    stop: str | None
    patient_id: str
    encounter_id: str | None
    code: str
    description: str

    @property
    def start_ts(self) -> int | None:
        return parse_synthea_date(self.start)

    @property
    def stop_ts(self) -> int | None:
        return parse_synthea_date(self.stop)

    @property
    def slot(self) -> str:
        return f"cond:{self.code}"


@dataclass(frozen=True)
class MedicationRecord:
    start: str
    stop: str | None
    patient_id: str
    encounter_id: str | None
    code: str
    description: str

    @property
    def start_ts(self) -> int | None:
        return parse_synthea_date(self.start)

    @property
    def stop_ts(self) -> int | None:
        return parse_synthea_date(self.stop)

    @property
    def ingredient(self) -> str:
        return normalize_med(self.description)

    @property
    def slot(self) -> str:
        return f"med:{self.ingredient}"


@dataclass(frozen=True)
class AllergyRecord:
    start: str
    stop: str | None
    patient_id: str
    encounter_id: str | None
    code: str
    system: str | None
    description: str
    type: str | None = None
    category: str | None = None

    @property
    def start_ts(self) -> int | None:
        return parse_synthea_date(self.start)

    @property
    def stop_ts(self) -> int | None:
        return parse_synthea_date(self.stop)

    @property
    def slot(self) -> str:
        return f"allergy:{self.code}"


@dataclass(frozen=True)
class ProcedureRecord:
    start: str
    stop: str | None
    patient_id: str
    encounter_id: str | None
    code: str
    description: str

    @property
    def start_ts(self) -> int | None:
        return parse_synthea_date(self.start)

    @property
    def stop_ts(self) -> int | None:
        return parse_synthea_date(self.stop)


@dataclass(frozen=True)
class ObservationRecord:
    date: str
    patient_id: str
    encounter_id: str | None
    category: str | None
    code: str
    description: str
    value: str
    units: str | None = None
    type: str | None = None

    @property
    def date_ts(self) -> int | None:
        return parse_synthea_date(self.date)


@dataclass(frozen=True)
class ProviderRecord:
    id: str
    organization_id: str | None
    name: str
    gender: str | None = None
    specialty: str | None = None


@dataclass(frozen=True)
class OrganizationRecord:
    id: str
    name: str
    city: str | None = None


# -----------------------------------------------------------------------------
# Synthea Dataset Container
# -----------------------------------------------------------------------------


@dataclass
class SyntheaDataset:
    """In-memory representation of parsed Synthea entities."""

    patients: dict[str, PatientRecord] = field(default_factory=dict)
    encounters_by_patient: dict[str, list[EncounterRecord]] = field(default_factory=dict)
    conditions_by_patient: dict[str, list[ConditionRecord]] = field(default_factory=dict)
    medications_by_patient: dict[str, list[MedicationRecord]] = field(default_factory=dict)
    allergies_by_patient: dict[str, list[AllergyRecord]] = field(default_factory=dict)
    procedures_by_patient: dict[str, list[ProcedureRecord]] = field(default_factory=dict)
    observations_by_patient: dict[str, list[ObservationRecord]] = field(default_factory=dict)
    providers: dict[str, ProviderRecord] = field(default_factory=dict)
    organizations: dict[str, OrganizationRecord] = field(default_factory=dict)

    # Raw row counts per entity type for verification
    raw_row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self.patients.keys())

    def get_patient_record(self, patient_id: str) -> PatientRecord | None:
        return self.patients.get(patient_id)

    def patient_counts(self, patient_id: str) -> dict[str, int]:
        """Returns row counts of all entity types associated with a specific patient."""
        return {
            "encounters": len(self.encounters_by_patient.get(patient_id, [])),
            "conditions": len(self.conditions_by_patient.get(patient_id, [])),
            "medications": len(self.medications_by_patient.get(patient_id, [])),
            "allergies": len(self.allergies_by_patient.get(patient_id, [])),
            "procedures": len(self.procedures_by_patient.get(patient_id, [])),
            "observations": len(self.observations_by_patient.get(patient_id, [])),
        }

    def total_counts(self) -> dict[str, int]:
        """Returns total entity counts across the entire dataset."""
        return {
            "patients": len(self.patients),
            "encounters": sum(len(v) for v in self.encounters_by_patient.values()),
            "conditions": sum(len(v) for v in self.conditions_by_patient.values()),
            "medications": sum(len(v) for v in self.medications_by_patient.values()),
            "allergies": sum(len(v) for v in self.allergies_by_patient.values()),
            "procedures": sum(len(v) for v in self.procedures_by_patient.values()),
            "observations": sum(len(v) for v in self.observations_by_patient.values()),
            "providers": len(self.providers),
            "organizations": len(self.organizations),
        }


# -----------------------------------------------------------------------------
# Low-Level Allowlist CSV Reader
# -----------------------------------------------------------------------------


def read_csv_with_allowlist(
    csv_path: str | Path,
    allowlist: frozenset[str],
) -> list[dict[str, str]]:
    """Reads a CSV file by header name and strictly filters out columns not in the allowlist.

    Guarantees:
    - Extra/unknown columns are dropped.
    - PII columns are stripped before returning.
    - Header names are case-preserved and whitespace trimmed.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    records: list[dict[str, str]] = []

    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Find intersecting headers
        headers_in_file = {h.strip(): h for h in reader.fieldnames if h}
        retained_headers = {h: orig for h, orig in headers_in_file.items() if h in allowlist}

        for row in reader:
            filtered_row: dict[str, str] = {}
            for allowed_header, orig_header in retained_headers.items():
                val = row.get(orig_header, "")
                filtered_row[allowed_header] = val.strip() if val else ""

            records.append(filtered_row)

    return records


# -----------------------------------------------------------------------------
# High-Level Dataset Loader
# -----------------------------------------------------------------------------


def load_synthea_dataset(dir_path: str | Path) -> SyntheaDataset:
    """Loads and parses all available Synthea CSV files from the specified directory.

    Unknown columns are ignored; PII columns are dropped at read time.
    Tolerates missing optional CSV files.
    """
    folder = Path(dir_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Synthea CSV directory not found: {folder}")

    # Auto-detect if Synthea placed CSVs inside a nested 'csv' subfolder
    if (folder / "csv").is_dir() and not (folder / "patients.csv").is_file():
        folder = folder / "csv"

    dataset = SyntheaDataset()

    # 1. Organizations
    org_file = folder / "organizations.csv"
    if org_file.is_file():
        rows = read_csv_with_allowlist(org_file, ORGANIZATIONS_ALLOWLIST)
        dataset.raw_row_counts["organizations"] = len(rows)
        for r in rows:
            org_id = r.get("Id", "")
            if org_id:
                dataset.organizations[org_id] = OrganizationRecord(
                    id=org_id,
                    name=r.get("NAME", ""),
                    city=r.get("CITY") or None,
                )

    # 2. Providers
    prov_file = folder / "providers.csv"
    if prov_file.is_file():
        rows = read_csv_with_allowlist(prov_file, PROVIDERS_ALLOWLIST)
        dataset.raw_row_counts["providers"] = len(rows)
        for r in rows:
            prov_id = r.get("Id", "")
            if prov_id:
                dataset.providers[prov_id] = ProviderRecord(
                    id=prov_id,
                    organization_id=r.get("ORGANIZATION") or None,
                    name=r.get("NAME", ""),
                    gender=r.get("GENDER") or None,
                    specialty=r.get("SPECIALITY") or r.get("SPECIALTY") or None,
                )

    # 3. Patients
    pat_file = folder / "patients.csv"
    if pat_file.is_file():
        rows = read_csv_with_allowlist(pat_file, PATIENTS_ALLOWLIST)
        dataset.raw_row_counts["patients"] = len(rows)
        for r in rows:
            pid = r.get("Id", "")
            if pid:
                dataset.patients[pid] = PatientRecord(
                    id=pid,
                    birthdate=r.get("BIRTHDATE", ""),
                    first=r.get("FIRST", ""),
                    last=r.get("LAST", ""),
                    gender=r.get("GENDER", ""),
                    deathdate=r.get("DEATHDATE") or None,
                )

    # 4. Encounters
    enc_file = folder / "encounters.csv"
    if enc_file.is_file():
        rows = read_csv_with_allowlist(enc_file, ENCOUNTERS_ALLOWLIST)
        dataset.raw_row_counts["encounters"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            enc_id = r.get("Id", "")
            if enc_id and pid:
                rec = EncounterRecord(
                    id=enc_id,
                    start=r.get("START", ""),
                    stop=r.get("STOP") or None,
                    patient_id=pid,
                    organization_id=r.get("ORGANIZATION") or None,
                    provider_id=r.get("PROVIDER") or None,
                    encounter_class=r.get("ENCOUNTERCLASS") or None,
                    code=r.get("CODE") or None,
                    description=r.get("DESCRIPTION") or None,
                    reason_code=r.get("REASONCODE") or None,
                    reason_description=r.get("REASONDESCRIPTION") or None,
                )
                dataset.encounters_by_patient.setdefault(pid, []).append(rec)

    # Sort encounters chronologically for each patient
    for pid in dataset.encounters_by_patient:
        dataset.encounters_by_patient[pid].sort(key=lambda e: e.start_ts or 0)

    # 5. Conditions
    cond_file = folder / "conditions.csv"
    if cond_file.is_file():
        rows = read_csv_with_allowlist(cond_file, CONDITIONS_ALLOWLIST)
        dataset.raw_row_counts["conditions"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            code = r.get("CODE", "")
            if pid and code:
                rec = ConditionRecord(
                    start=r.get("START", ""),
                    stop=r.get("STOP") or None,
                    patient_id=pid,
                    encounter_id=r.get("ENCOUNTER") or None,
                    code=code,
                    description=r.get("DESCRIPTION", ""),
                )
                dataset.conditions_by_patient.setdefault(pid, []).append(rec)

    # 6. Medications
    med_file = folder / "medications.csv"
    if med_file.is_file():
        rows = read_csv_with_allowlist(med_file, MEDICATIONS_ALLOWLIST)
        dataset.raw_row_counts["medications"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            code = r.get("CODE", "")
            if pid and code:
                rec = MedicationRecord(
                    start=r.get("START", ""),
                    stop=r.get("STOP") or None,
                    patient_id=pid,
                    encounter_id=r.get("ENCOUNTER") or None,
                    code=code,
                    description=r.get("DESCRIPTION", ""),
                )
                dataset.medications_by_patient.setdefault(pid, []).append(rec)

    # 7. Allergies
    allg_file = folder / "allergies.csv"
    if allg_file.is_file():
        rows = read_csv_with_allowlist(allg_file, ALLERGIES_ALLOWLIST)
        dataset.raw_row_counts["allergies"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            code = r.get("CODE", "")
            if pid and code:
                rec = AllergyRecord(
                    start=r.get("START", ""),
                    stop=r.get("STOP") or None,
                    patient_id=pid,
                    encounter_id=r.get("ENCOUNTER") or None,
                    code=code,
                    system=r.get("SYSTEM") or None,
                    description=r.get("DESCRIPTION", ""),
                    type=r.get("TYPE") or None,
                    category=r.get("CATEGORY") or None,
                )
                dataset.allergies_by_patient.setdefault(pid, []).append(rec)

    # 8. Procedures
    proc_file = folder / "procedures.csv"
    if proc_file.is_file():
        rows = read_csv_with_allowlist(proc_file, PROCEDURES_ALLOWLIST)
        dataset.raw_row_counts["procedures"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            code = r.get("CODE", "")
            if pid and code:
                rec = ProcedureRecord(
                    start=r.get("START", ""),
                    stop=r.get("STOP") or None,
                    patient_id=pid,
                    encounter_id=r.get("ENCOUNTER") or None,
                    code=code,
                    description=r.get("DESCRIPTION", ""),
                )
                dataset.procedures_by_patient.setdefault(pid, []).append(rec)

    # 9. Observations
    obs_file = folder / "observations.csv"
    if obs_file.is_file():
        rows = read_csv_with_allowlist(obs_file, OBSERVATIONS_ALLOWLIST)
        dataset.raw_row_counts["observations"] = len(rows)
        for r in rows:
            pid = r.get("PATIENT", "")
            code = r.get("CODE", "")
            if pid and code:
                rec = ObservationRecord(
                    date=r.get("DATE", ""),
                    patient_id=pid,
                    encounter_id=r.get("ENCOUNTER") or None,
                    category=r.get("CATEGORY") or None,
                    code=code,
                    description=r.get("DESCRIPTION", ""),
                    value=r.get("VALUE", ""),
                    units=r.get("UNITS") or None,
                    type=r.get("TYPE") or None,
                )
                dataset.observations_by_patient.setdefault(pid, []).append(rec)

    return dataset
