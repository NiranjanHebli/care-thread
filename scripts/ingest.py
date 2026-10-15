#!/usr/bin/env bash
"""CareThread Ingestion CLI script.

Execution features:
- Ingests Synthea CSV records into isolated FalkorDB patient graphs (`patient_<id>`).
- Bounded concurrency (<= 4 workers).
- Computes Episode summary embeddings in batches.
- Asserts FalkorDB node counts match source CSV rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from carethread.db.client import get_db_client
from carethread.db.schema import bootstrap_control_graph
from carethread.db.tenant_router import TenantRouter
from carethread.ingest.mapper import ingest_dataset
from carethread.ingest.synthea_csv import load_synthea_dataset
from carethread.memory.embeddings import get_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def resolve_data_dir(specified: str | None) -> Path:
    """Finds best default dataset directory if not explicitly specified."""
    if specified:
        p = Path(specified)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified data directory does not exist: {specified}")

    # Priority 1: Full generated Synthea dataset in data/synthea/csv
    full_data = Path("data/synthea/csv")
    if (full_data / "patients.csv").is_file() or (full_data / "csv" / "patients.csv").is_file():
        return full_data

    # Priority 2: Committed fixture dataset in tests/fixtures/synthea_csv
    fixture_data = Path("tests/fixtures/synthea_csv")
    if fixture_data.is_dir():
        return fixture_data

    raise FileNotFoundError(
        "No Synthea CSV data found. Run `./scripts/synthea_generate.sh` or specify --data-dir."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CareThread Synthea Graph Ingestion CLI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing Synthea CSV files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4).",
    )
    parser.add_argument(
        "--embed-mode",
        type=str,
        default=None,
        choices=["auto", "hash", "fastembed"],
        help="Embedding mode (auto, hash, fastembed).",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional limit on number of patients to ingest.",
    )

    args = parser.parse_args()

    data_path = resolve_data_dir(args.data_dir)
    logger.info(f"Loading Synthea dataset from: {data_path.resolve()}")

    t0 = time.perf_counter()
    dataset = load_synthea_dataset(data_path)
    logger.info(
        f"Loaded dataset in {time.perf_counter() - t0:.2f}s: "
        f"{len(dataset.patients)} patients, "
        f"{sum(len(v) for v in dataset.encounters_by_patient.values())} encounters, "
        f"{sum(len(v) for v in dataset.observations_by_patient.values())} observations."
    )

    if args.max_patients and len(dataset.patients) > args.max_patients:
        selected_ids = set(dataset.patient_ids[: args.max_patients])
        dataset.patients = {k: v for k, v in dataset.patients.items() if k in selected_ids}
        logger.info(f"Restricted ingestion to first {args.max_patients} patients.")

    # Connect to FalkorDB
    client = get_db_client()
    logger.info("Connected to FalkorDB system of record.")

    # Ensure control graph is initialized
    control_g = client.select_graph("ct_control")
    bootstrap_control_graph(control_g, client)

    router = TenantRouter(client)
    embedder = get_embedder(mode=args.embed_mode)
    logger.info(f"Using embedder: {embedder.__class__.__name__} (dim={embedder.dimension})")

    # Execute ingestion with bounded concurrency
    result = ingest_dataset(
        dataset=dataset,
        router=router,
        max_workers=args.workers,
        embedder=embedder,
        progress_callback=lambda res: logger.info(
            f"  [OK] Ingested patient {res.patient_id} ({res.synthea_id[:8]}...): "
            f"{res.episodes_count} eps, "
            f"{res.conditions_count + res.medications_count + res.allergies_count} facts, "
            f"{res.observations_count} obs in {res.duration_ms:.1f}ms"
        ),
    )

    # Print summary
    logger.info("==========================================================")
    logger.info("Ingestion Run Completed:")
    logger.info(f"  Patients Ingested:    {result.patients_ingested}")
    logger.info(f"  Total Graph Nodes:    {result.total_nodes}")
    logger.info(f"  Total Episodes:       {result.total_episodes}")
    logger.info(f"  Total Facts:          {result.total_facts}")
    logger.info(f"  Total Observations:   {result.total_observations}")
    logger.info(f"  Total Procedures:     {result.total_procedures}")
    logger.info(f"  Total Elapsed Time:   {result.duration_s:.2f} seconds")
    logger.info("==========================================================")

    if result.errors:
        logger.error(f"Ingestion finished with {len(result.errors)} errors:")
        for err in result.errors[:10]:
            logger.error(f"  - {err}")
        return 1

    logger.info("All patients successfully verified with zero count mismatches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
