"""Synthea to FalkorDB Graph Mapper and Ingestion Engine.

Ingestion pipeline features:
- Derives valid per-tenant patient IDs (`p_<12 hex chars>`).
- Batches node and edge creation with UNWIND (up to 500 rows).
- Ingestion order: Patient, Provider & Organization, Episodes, NEXT chain,
  Facts (conditions, medications, allergies), Observations, and Procedures.
- Computes Episode summary embeddings (batches of 64).
- Idempotency: MERGE on `id` ensures second run creates 0 duplicate nodes.
- Strict post-ingest verification: asserts FalkorDB node counts match CSV rows.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from falkordb import Graph

from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.errors import DatabaseError, ValidationError
from carethread.ingest.synthea_csv import (
    SyntheaDataset,
)
from carethread.memory.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)

UNWIND_BATCH_SIZE: int = 500
EMBEDDING_BATCH_SIZE: int = 64


def synthea_id_to_patient_id(synthea_id: str) -> str:
    """Converts a Synthea patient UUID to a validated CareThread patient ID (`p_[0-9a-f]{12}`)."""
    digest = hashlib.sha256(synthea_id.strip().encode("utf-8")).hexdigest()[:12]
    return f"p_{digest}"


@dataclass
class IngestPatientResult:
    """Detailed summary of a single patient's graph ingestion."""

    patient_id: str
    synthea_id: str
    graph_name: str
    episodes_count: int = 0
    conditions_count: int = 0
    medications_count: int = 0
    allergies_count: int = 0
    procedures_count: int = 0
    observations_count: int = 0
    providers_count: int = 0
    organizations_count: int = 0
    total_nodes: int = 0
    duration_ms: float = 0.0
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "synthea_id": self.synthea_id,
            "graph_name": self.graph_name,
            "counts": {
                "episodes": self.episodes_count,
                "conditions": self.conditions_count,
                "medications": self.medications_count,
                "allergies": self.allergies_count,
                "procedures": self.procedures_count,
                "observations": self.observations_count,
                "providers": self.providers_count,
                "organizations": self.organizations_count,
            },
            "total_nodes": self.total_nodes,
            "duration_ms": self.duration_ms,
            "verified": self.verified,
        }


@dataclass
class IngestDatasetResult:
    """Overall dataset ingestion outcome."""

    patients_ingested: int = 0
    total_nodes: int = 0
    total_episodes: int = 0
    total_facts: int = 0
    total_observations: int = 0
    total_procedures: int = 0
    duration_s: float = 0.0
    patient_results: list[IngestPatientResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PatientGraphMapper:
    """Maps and writes Synthea entities into isolated FalkorDB tenant graphs."""

    def __init__(
        self,
        router: TenantRouter,
        embedder: Embedder | None = None,
    ) -> None:
        self._router = router
        self._embedder = embedder or get_embedder()

    def ingest_patient(
        self,
        synthea_patient_id: str,
        dataset: SyntheaDataset,
        graph: Graph | None = None,
        bootstrap_schema: bool = True,
    ) -> IngestPatientResult:
        """Ingests all records for a single patient into their isolated graph.

        Maps Synthea CSV records to graph nodes and edges.
        """
        start_time = time.perf_counter()

        patient_rec = dataset.get_patient_record(synthea_patient_id)
        if not patient_rec:
            raise ValidationError(
                f"Patient ID '{synthea_patient_id}' not found in dataset.",
                code="PATIENT_NOT_FOUND",
            )

        patient_id = synthea_id_to_patient_id(synthea_patient_id)
        target_graph = graph or self._router.graph(patient_id)
        graph_name = target_graph.name

        # 1. Register in ct_control & bootstrap schema
        if bootstrap_schema:
            self._router.register_patient(patient_id, bootstrap_schema=True)

        now_ts = int(time.time())

        # 2. Ingest Patient anchor node
        target_graph.query(
            """
            MERGE (p:Patient {id: $pid})
            ON CREATE SET p.synthea_id = $synthea_id,
                          p.name = $name,
                          p.gender = $gender,
                          p.birthdate = $birthdate,
                          p.data_provenance = 'synthea',
                          p.dirty = false,
                          p.created_at = $now
            ON MATCH SET p.name = $name,
                         p.gender = $gender,
                         p.birthdate = $birthdate
            """,
            {
                "pid": patient_id,
                "synthea_id": synthea_patient_id,
                "name": patient_rec.name,
                "gender": patient_rec.gender,
                "birthdate": patient_rec.birthdate,
                "now": now_ts,
            },
        )

        # 3. Referenced Providers & Organizations
        encounters = dataset.encounters_by_patient.get(synthea_patient_id, [])
        ref_prov_ids = {e.provider_id for e in encounters if e.provider_id}
        ref_org_ids = {e.organization_id for e in encounters if e.organization_id}

        orgs_payload = [
            {
                "id": f"org_{org_id}",
                "name": dataset.organizations[org_id].name
                if org_id in dataset.organizations
                else f"Hospital {org_id}",
                "city": dataset.organizations[org_id].city
                if org_id in dataset.organizations
                else "Boston",
            }
            for org_id in sorted(ref_org_ids)
        ]
        if orgs_payload:
            target_graph.query(
                """
                UNWIND $orgs AS o
                MERGE (org:Organization {id: o.id})
                ON CREATE SET org.name = o.name,
                              org.city = o.city,
                              org.data_provenance = 'synthea'
                """,
                {"orgs": orgs_payload},
            )

        provs_payload = [
            {
                "id": f"prov_{prov_id}",
                "name": dataset.providers[prov_id].name
                if prov_id in dataset.providers
                else f"Provider {prov_id}",
                "specialty": dataset.providers[prov_id].specialty
                if prov_id in dataset.providers
                else "Internal Medicine",
            }
            for prov_id in sorted(ref_prov_ids)
        ]
        if provs_payload:
            target_graph.query(
                """
                UNWIND $provs AS pr
                MERGE (prov:Provider {id: pr.id})
                ON CREATE SET prov.name = pr.name,
                              prov.specialty = pr.specialty,
                              prov.data_provenance = 'synthea'
                """,
                {"provs": provs_payload},
            )

        # 4. Ingest Episodes with Summary Embeddings
        episodes_payload: list[dict[str, Any]] = []
        episode_summaries: list[str] = [e.summary for e in encounters]
        episode_embeddings = (
            self._embedder.embed_batch(episode_summaries) if episode_summaries else []
        )

        for idx, enc in enumerate(encounters, start=1):
            ep_id = f"epi_{enc.id}"
            episodes_payload.append(
                {
                    "id": ep_id,
                    "start": enc.start_ts or now_ts,
                    "end": enc.stop_ts,
                    "summary": enc.summary,
                    "embedding": episode_embeddings[idx - 1],
                    "episode_type": enc.encounter_class or "ambulatory",
                    "seq": idx,
                    "provider_id": f"prov_{enc.provider_id}" if enc.provider_id else None,
                    "organization_id": f"org_{enc.organization_id}"
                    if enc.organization_id
                    else None,
                }
            )

        if episodes_payload:
            # Batch merge episodes and link to Patient and Context
            target_graph.query(
                """
                UNWIND $episodes AS ep
                MERGE (e:Episode {id: ep.id})
                ON CREATE SET e.kind = 'synthea_encounter',
                              e.source = 'synthea',
                              e.data_provenance = 'synthea',
                              e.start = ep.start,
                              e.end = ep.end,
                              e.summary = ep.summary,
                              e.summary_embedding = vecf32(ep.embedding),
                              e.outcome = 'completed',
                              e.episode_type = ep.episode_type,
                              e.seq = ep.seq
                WITH e, ep
                MATCH (p:Patient {id: $pid})
                MERGE (p)-[:HAS_EPISODE]->(e)
                """,
                {"episodes": episodes_payload, "pid": patient_id},
            )

            # Link providers
            prov_links = [ep for ep in episodes_payload if ep["provider_id"]]
            if prov_links:
                target_graph.query(
                    """
                    UNWIND $eps AS ep
                    MATCH (e:Episode {id: ep.id}), (prov:Provider {id: ep.provider_id})
                    MERGE (e)-[:WITH_PROVIDER]->(prov)
                    """,
                    {"eps": prov_links},
                )

            # Link organizations
            org_links = [ep for ep in episodes_payload if ep["organization_id"]]
            if org_links:
                target_graph.query(
                    """
                    UNWIND $eps AS ep
                    MATCH (e:Episode {id: ep.id}), (org:Organization {id: ep.organization_id})
                    MERGE (e)-[:AT_ORG]->(org)
                    """,
                    {"eps": org_links},
                )

            # 5. Singly-linked chronological chain (NEXT)
            chains_payload: list[dict[str, Any]] = []
            for i in range(len(episodes_payload) - 1):
                cur_ep = episodes_payload[i]
                nxt_ep = episodes_payload[i + 1]
                gap_days = max(0, int((nxt_ep["start"] - cur_ep["start"]) / 86400))
                chains_payload.append(
                    {
                        "from_id": cur_ep["id"],
                        "to_id": nxt_ep["id"],
                        "gap_days": gap_days,
                    }
                )

            if chains_payload:
                target_graph.query(
                    """
                    UNWIND $chains AS ch
                    MATCH (e1:Episode {id: ch.from_id}), (e2:Episode {id: ch.to_id})
                    MERGE (e1)-[:NEXT {gap_days: ch.gap_days}]->(e2)
                    """,
                    {"chains": chains_payload},
                )

        # 6. Facts: Conditions, Medications, Allergies
        conditions = dataset.conditions_by_patient.get(synthea_patient_id, [])
        cond_payload: list[dict[str, Any]] = []
        for idx, c in enumerate(conditions):
            valid_from = c.start_ts or now_ts
            valid_to = c.stop_ts
            hash_key = f"{c.encounter_id}_{c.code}_{valid_from}_{valid_to}_{idx}".encode()
            cond_id = f"fct_cond_{hashlib.md5(hash_key).hexdigest()[:12]}"
            cond_payload.append(
                {
                    "id": cond_id,
                    "slot": c.slot,
                    "code": c.code,
                    "display": c.description,
                    "value": "active" if valid_to is None else "resolved",
                    "status": "active" if valid_to is None else "superseded",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "superseded_at": valid_to,
                    "encounter_id": f"epi_{c.encounter_id}" if c.encounter_id else None,
                    "op_id": f"op_{cond_id}",
                }
            )

        if cond_payload:
            for i in range(0, len(cond_payload), UNWIND_BATCH_SIZE):
                chunk = cond_payload[i : i + UNWIND_BATCH_SIZE]
                target_graph.query(
                    """
                    UNWIND $batch AS c
                    MERGE (f:Fact {id: c.id})
                    ON CREATE SET f.kind = 'condition',
                                  f.slot = c.slot,
                                  f.code = c.code,
                                  f.system = 'SNOMED-CT',
                                  f.display = c.display,
                                  f.value = c.value,
                                  f.status = c.status,
                                  f.valid_from = c.valid_from,
                                  f.valid_to = c.valid_to,
                                  f.recorded_at = c.valid_from,
                                  f.superseded_at = c.superseded_at,
                                  f.last_confirmed_at = c.valid_from,
                                  f.confidence = 0.95,
                                  f.authority = 'record',
                                  f.evidence_count = 1,
                                  f.stale = false,
                                  f.half_life_days = 365,
                                  f.version = 1,
                                  f.source_episode_id = c.encounter_id,
                                  f.asserted_by = 'synthea_import',
                                  f.op_id = c.op_id,
                                  f.data_provenance = 'synthea'
                    WITH f, c
                    MATCH (p:Patient {id: $pid})
                    MERGE (p)-[:HAS_FACT]->(f)
                    """,
                    {"batch": chunk, "pid": patient_id},
                )

                # Link ASSERTED_IN to Episode
                enc_chunks = [c for c in chunk if c["encounter_id"]]
                if enc_chunks:
                    target_graph.query(
                        """
                        UNWIND $batch AS c
                        MATCH (f:Fact {id: c.id}), (e:Episode {id: c.encounter_id})
                        MERGE (f)-[:ASSERTED_IN]->(e)
                        """,
                        {"batch": enc_chunks},
                    )

        medications = dataset.medications_by_patient.get(synthea_patient_id, [])
        med_payload: list[dict[str, Any]] = []
        for idx, m in enumerate(medications):
            valid_from = m.start_ts or now_ts
            valid_to = m.stop_ts
            hash_key = f"{m.encounter_id}_{m.code}_{valid_from}_{valid_to}_{idx}".encode()
            med_id = f"fct_med_{hashlib.md5(hash_key).hexdigest()[:12]}"
            med_payload.append(
                {
                    "id": med_id,
                    "slot": m.slot,
                    "code": m.code,
                    "display": m.description,
                    "status": "active" if valid_to is None else "superseded",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "superseded_at": valid_to,
                    "encounter_id": f"epi_{m.encounter_id}" if m.encounter_id else None,
                    "op_id": f"op_{med_id}",
                }
            )

        if med_payload:
            for i in range(0, len(med_payload), UNWIND_BATCH_SIZE):
                chunk = med_payload[i : i + UNWIND_BATCH_SIZE]
                target_graph.query(
                    """
                    UNWIND $batch AS m
                    MERGE (f:Fact {id: m.id})
                    ON CREATE SET f.kind = 'medication',
                                  f.slot = m.slot,
                                  f.code = m.code,
                                  f.system = 'RxNorm',
                                  f.display = m.display,
                                  f.value = m.display,
                                  f.status = m.status,
                                  f.valid_from = m.valid_from,
                                  f.valid_to = m.valid_to,
                                  f.recorded_at = m.valid_from,
                                  f.superseded_at = m.superseded_at,
                                  f.last_confirmed_at = m.valid_from,
                                  f.confidence = 0.95,
                                  f.authority = 'record',
                                  f.evidence_count = 1,
                                  f.stale = false,
                                  f.half_life_days = 90,
                                  f.version = 1,
                                  f.source_episode_id = m.encounter_id,
                                  f.asserted_by = 'synthea_import',
                                  f.op_id = m.op_id,
                                  f.data_provenance = 'synthea'
                    WITH f, m
                    MATCH (p:Patient {id: $pid})
                    MERGE (p)-[:HAS_FACT]->(f)
                    """,
                    {"batch": chunk, "pid": patient_id},
                )

                enc_chunks = [m for m in chunk if m["encounter_id"]]
                if enc_chunks:
                    target_graph.query(
                        """
                        UNWIND $batch AS m
                        MATCH (f:Fact {id: m.id}), (e:Episode {id: m.encounter_id})
                        MERGE (f)-[:ASSERTED_IN]->(e)
                        """,
                        {"batch": enc_chunks},
                    )

        allergies = dataset.allergies_by_patient.get(synthea_patient_id, [])
        allg_payload: list[dict[str, Any]] = []
        for idx, a in enumerate(allergies):
            valid_from = a.start_ts or now_ts
            valid_to = a.stop_ts
            hash_key = f"{a.encounter_id}_{a.code}_{valid_from}_{valid_to}_{idx}".encode()
            allg_id = f"fct_allg_{hashlib.md5(hash_key).hexdigest()[:12]}"
            allg_payload.append(
                {
                    "id": allg_id,
                    "slot": a.slot,
                    "code": a.code,
                    "system": a.system or "SNOMED-CT",
                    "display": a.description,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "encounter_id": f"epi_{a.encounter_id}" if a.encounter_id else None,
                    "op_id": f"op_{allg_id}",
                }
            )

        if allg_payload:
            for i in range(0, len(allg_payload), UNWIND_BATCH_SIZE):
                chunk = allg_payload[i : i + UNWIND_BATCH_SIZE]
                target_graph.query(
                    """
                    UNWIND $batch AS a
                    MERGE (f:Fact {id: a.id})
                    ON CREATE SET f.kind = 'allergy',
                                  f.slot = a.slot,
                                  f.code = a.code,
                                  f.system = a.system,
                                  f.display = a.display,
                                  f.value = a.display,
                                  f.status = 'active',
                                  f.valid_from = a.valid_from,
                                  f.valid_to = a.valid_to,
                                  f.recorded_at = a.valid_from,
                                  f.superseded_at = null,
                                  f.last_confirmed_at = a.valid_from,
                                  f.confidence = 0.95,
                                  f.authority = 'record',
                                  f.evidence_count = 1,
                                  f.stale = false,
                                  f.half_life_days = null,
                                  f.version = 1,
                                  f.source_episode_id = a.encounter_id,
                                  f.asserted_by = 'synthea_import',
                                  f.op_id = a.op_id,
                                  f.data_provenance = 'synthea'
                    WITH f, a
                    MATCH (p:Patient {id: $pid})
                    MERGE (p)-[:HAS_FACT]->(f)
                    """,
                    {"batch": chunk, "pid": patient_id},
                )

                enc_chunks = [a for a in chunk if a["encounter_id"]]
                if enc_chunks:
                    target_graph.query(
                        """
                        UNWIND $batch AS a
                        MATCH (f:Fact {id: a.id}), (e:Episode {id: a.encounter_id})
                        MERGE (f)-[:ASSERTED_IN]->(e)
                        """,
                        {"batch": enc_chunks},
                    )

        # 7. Procedures (ClinicalProcedure)
        procedures = dataset.procedures_by_patient.get(synthea_patient_id, [])
        proc_payload: list[dict[str, Any]] = []
        for idx, pr in enumerate(procedures):
            hash_key = f"{pr.encounter_id}_{pr.code}_{pr.start}_{pr.stop}_{idx}".encode()
            proc_id = f"proc_{hashlib.md5(hash_key).hexdigest()[:12]}"
            proc_payload.append(
                {
                    "id": proc_id,
                    "code": pr.code,
                    "display": pr.description,
                    "start": pr.start_ts or now_ts,
                    "end": pr.stop_ts,
                    "encounter_id": f"epi_{pr.encounter_id}" if pr.encounter_id else None,
                }
            )

        if proc_payload:
            for i in range(0, len(proc_payload), UNWIND_BATCH_SIZE):
                chunk = proc_payload[i : i + UNWIND_BATCH_SIZE]
                target_graph.query(
                    """
                    UNWIND $batch AS pr
                    MERGE (cp:ClinicalProcedure {id: pr.id})
                    ON CREATE SET cp.code = pr.code,
                                  cp.display = pr.display,
                                  cp.start = pr.start,
                                  cp.end = pr.end,
                                  cp.data_provenance = 'synthea'
                    """,
                    {"batch": chunk},
                )
                enc_chunks = [pr for pr in chunk if pr["encounter_id"]]
                if enc_chunks:
                    target_graph.query(
                        """
                        UNWIND $batch AS pr
                        MATCH (cp:ClinicalProcedure {id: pr.id}), (e:Episode {id: pr.encounter_id})
                        MERGE (e)-[:PERFORMED]->(cp)
                        """,
                        {"batch": enc_chunks},
                    )

        # 8. Observations
        observations = dataset.observations_by_patient.get(synthea_patient_id, [])
        obs_payload: list[dict[str, Any]] = []
        for idx, o in enumerate(observations):
            hash_key = f"{o.encounter_id}_{o.code}_{o.date}_{o.value}_{idx}".encode()
            obs_id = f"obs_{hashlib.md5(hash_key).hexdigest()[:12]}"
            obs_payload.append(
                {
                    "id": obs_id,
                    "code": o.code,
                    "display": o.description,
                    "value": o.value,
                    "unit": o.units,
                    "ts": o.date_ts or now_ts,
                    "category": o.category or "vital-signs",
                    "encounter_id": f"epi_{o.encounter_id}" if o.encounter_id else None,
                }
            )

        if obs_payload:
            for i in range(0, len(obs_payload), UNWIND_BATCH_SIZE):
                chunk = obs_payload[i : i + UNWIND_BATCH_SIZE]
                target_graph.query(
                    """
                    UNWIND $batch AS o
                    MERGE (obs:Observation {id: o.id})
                    ON CREATE SET obs.code = o.code,
                                  obs.display = o.display,
                                  obs.value = o.value,
                                  obs.unit = o.unit,
                                  obs.ts = o.ts,
                                  obs.category = o.category,
                                  obs.data_provenance = 'synthea'
                    """,
                    {"batch": chunk},
                )
                enc_chunks = [o for o in chunk if o["encounter_id"]]
                if enc_chunks:
                    target_graph.query(
                        """
                        UNWIND $batch AS o
                        MATCH (obs:Observation {id: o.id}), (e:Episode {id: o.encounter_id})
                        MERGE (e)-[:PRODUCED]->(obs)
                        """,
                        {"batch": enc_chunks},
                    )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 9. Post-ingest Verification Invariant
        # Compare FalkorDB node counts with input CSV row counts
        expected_counts = dataset.patient_counts(synthea_patient_id)
        counts_res = target_graph.query(
            """
            OPTIONAL MATCH (p:Patient) WITH count(p) AS patient_cnt
            OPTIONAL MATCH (e:Episode) WITH patient_cnt, count(e) AS episode_cnt
            OPTIONAL MATCH (f:Fact) WITH patient_cnt, episode_cnt, count(f) AS fact_cnt
            OPTIONAL MATCH (cp:ClinicalProcedure)
            WITH patient_cnt, episode_cnt, fact_cnt, count(cp) AS proc_cnt
            OPTIONAL MATCH (obs:Observation)
            WITH patient_cnt, episode_cnt, fact_cnt, proc_cnt, count(obs) AS obs_cnt
            RETURN patient_cnt, episode_cnt, fact_cnt, proc_cnt, obs_cnt
            """
        )
        row = counts_res.result_set[0]
        actual_pat = row[0]
        actual_ep = row[1]
        actual_fact = row[2]
        actual_proc = row[3]
        actual_obs = row[4]

        expected_facts = (
            expected_counts["conditions"]
            + expected_counts["medications"]
            + expected_counts["allergies"]
        )

        mismatches: list[str] = []
        if actual_pat != 1:
            mismatches.append(f"Patient node count: expected 1, found {actual_pat}")
        if actual_ep != expected_counts["encounters"]:
            mismatches.append(
                f"Episode count: expected {expected_counts['encounters']}, found {actual_ep}"
            )
        if actual_fact != expected_facts:
            mismatches.append(f"Fact count: expected {expected_facts}, found {actual_fact}")
        if actual_proc != expected_counts["procedures"]:
            mismatches.append(
                f"Procedure count: expected {expected_counts['procedures']}, found {actual_proc}"
            )
        if actual_obs != expected_counts["observations"]:
            mismatches.append(
                f"Observation count: expected {expected_counts['observations']}, found {actual_obs}"
            )

        if mismatches:
            details = "; ".join(mismatches)
            error_msg = f"Verification failure for patient {patient_id}: {details}"
            logger.error(error_msg)
            raise DatabaseError(error_msg, code="INGEST_VERIFICATION_MISMATCH")

        total_nodes = (
            1
            + actual_ep
            + actual_fact
            + actual_proc
            + actual_obs
            + len(ref_prov_ids)
            + len(ref_org_ids)
        )

        return IngestPatientResult(
            patient_id=patient_id,
            synthea_id=synthea_patient_id,
            graph_name=graph_name,
            episodes_count=actual_ep,
            conditions_count=expected_counts["conditions"],
            medications_count=expected_counts["medications"],
            allergies_count=expected_counts["allergies"],
            procedures_count=actual_proc,
            observations_count=actual_obs,
            providers_count=len(ref_prov_ids),
            organizations_count=len(ref_org_ids),
            total_nodes=total_nodes,
            duration_ms=duration_ms,
            verified=True,
        )


def ingest_dataset(
    dataset: SyntheaDataset,
    router: TenantRouter,
    max_workers: int = 4,
    embedder: Embedder | None = None,
    progress_callback: Any = None,
) -> IngestDatasetResult:
    """Ingests full Synthea dataset across isolated patient graphs with bounded concurrency.

    Concurrent multi-worker ingestion.
    """
    start_time = time.perf_counter()
    mapper = PatientGraphMapper(router, embedder=embedder)
    results: list[IngestPatientResult] = []
    errors: list[str] = []

    patient_ids = dataset.patient_ids
    logger.info(f"Starting ingestion of {len(patient_ids)} patients with {max_workers} workers...")

    # ThreadPool execution with bounded concurrency
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _worker_fn(pid: str) -> IngestPatientResult:
        if max_workers > 1:
            worker_router = TenantRouter(get_db_client())
            worker_mapper = PatientGraphMapper(worker_router, embedder=embedder)
            return worker_mapper.ingest_patient(pid, dataset)
        return mapper.ingest_patient(pid, dataset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {executor.submit(_worker_fn, pid): pid for pid in patient_ids}

        for future in as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                res = future.result()
                results.append(res)
                if progress_callback:
                    progress_callback(res)
            except Exception as e:
                err = f"Failed to ingest patient {pid}: {e}"
                logger.error(err)
                errors.append(err)

    duration_s = time.perf_counter() - start_time

    return IngestDatasetResult(
        patients_ingested=len(results),
        total_nodes=sum(r.total_nodes for r in results),
        total_episodes=sum(r.episodes_count for r in results),
        total_facts=sum(
            r.conditions_count + r.medications_count + r.allergies_count for r in results
        ),
        total_observations=sum(r.observations_count for r in results),
        total_procedures=sum(r.procedures_count for r in results),
        duration_s=duration_s,
        patient_results=results,
        errors=errors,
    )
