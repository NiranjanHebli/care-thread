#!/usr/bin/env python3
"""CLI entrypoint for seeding the shared clinic knowledge graph (clinic_shared).

Usage:
    python scripts/seed_clinic.py
    python scripts/seed_clinic.py --data-dir data/clinic --embed-mode hash
    make seed-clinic
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from carethread.db.client import check_health, get_db_client
from carethread.db.seed_clinic import DEFAULT_CLINIC_DATA_DIR, seed_clinic
from carethread.memory.embeddings import FastEmbedder, HashEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_clinic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the shared clinic knowledge graph (clinic_shared) from YAML files.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_CLINIC_DATA_DIR),
        help=f"Path to directory containing clinic YAML files (default: {DEFAULT_CLINIC_DATA_DIR})",
    )
    parser.add_argument(
        "--embed-mode",
        choices=["fastembed", "hash"],
        default="hash",
        help="Embedder mode for guideline vector embeddings: 'fastembed' or 'hash' (default: hash)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.is_dir():
        logger.error(f"Clinic data directory does not exist: {data_dir}")
        return 1

    # Verify FalkorDB connectivity
    if not check_health():
        logger.error("Failed to connect to FalkorDB on localhost:6379.")
        return 1

    logger.info("Connected to FalkorDB.")
    embedder = FastEmbedder() if args.embed_mode == "fastembed" else HashEmbedder()
    logger.info(f"Using embedder: {embedder.__class__.__name__} (dim={embedder.dimension})")

    t0 = time.perf_counter()
    try:
        client = get_db_client()
        result = seed_clinic(
            client=client,
            clinic_data_dir=data_dir,
            embedder=embedder,
            bootstrap_schema=True,
        )
    except Exception as e:
        logger.error(f"Clinic seeding failed: {e}")
        return 1

    elapsed = time.perf_counter() - t0
    logger.info("==========================================================")
    logger.info("Clinic Knowledge Graph Seeding Completed:")
    logger.info(f"  Target Graph:         {result.graph_name}")
    logger.info(f"  Total Nodes:          {result.total_nodes}")
    logger.info(f"  Total Edges:          {result.total_edges}")
    logger.info(f"  Guidelines:           {result.guidelines_count}")
    logger.info(f"  Rules:                {result.rules_count}")
    logger.info(f"  Drugs:                {result.drugs_count}")
    logger.info(f"  Drug Classes:         {result.drug_classes_count}")
    logger.info(f"  Allergen Classes:     {result.allergen_classes_count}")
    logger.info(f"  Allergen Codes:       {result.allergen_codes_count}")
    logger.info(f"  Playbook Templates:   {result.playbook_templates_count}")
    logger.info(f"  Elapsed Time:         {elapsed:.2f} seconds")
    logger.info("==========================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
