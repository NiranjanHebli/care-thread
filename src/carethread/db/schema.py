"""FalkorDB Schema bootstrap, index management, and constraint enforcement.

Idempotent schema creation and index management:
- Range indexes (single & compound)
- Unique node constraints (enforced via GRAPH.CONSTRAINT)
- 384-dimensional vector indexes with cosine similarity
- Fulltext indexes
- Idempotent execution and schema introspection
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from falkordb import FalkorDB, Graph
from redis.exceptions import ResponseError

from carethread.db.client import get_db_client

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Patient Graph Schema Specifications
# -----------------------------------------------------------------------------

# Range indexes: (Label, (prop1, ...))
PATIENT_RANGE_INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Anchor & IDs
    ("Patient", ("id",)),
    ("Episode", ("id",)),
    ("Episode", ("start",)),
    ("Episode", ("episode_type",)),
    ("Session", ("id",)),
    ("Message", ("id",)),
    ("Message", ("session_id", "seq")),  # Compound order index
    ("Fact", ("id",)),
    ("Fact", ("slot",)),
    ("Fact", ("kind",)),
    ("Fact", ("status",)),
    ("Fact", ("valid_from",)),
    ("Fact", ("recorded_at",)),
    ("FactCandidate", ("id",)),
    ("Note", ("id",)),
    ("Handoff", ("id",)),
    ("Handoff", ("status",)),
    ("Handoff", ("to_role",)),
    ("OpenQuestion", ("id",)),
    ("Action", ("id",)),
    ("Playbook", ("id",)),
    ("PlaybookStep", ("id",)),
    ("Flag", ("id",)),
    ("Answer", ("id",)),
    ("Agent", ("id",)),
    ("AuditEvent", ("id",)),
    ("AuditEvent", ("ts",)),
    ("ConsolidationRun", ("id",)),
    ("Observation", ("id",)),
    ("ClinicalProcedure", ("id",)),
    ("Provider", ("id",)),
    ("Organization", ("id",)),
)

# Unique Node Constraints: (Label, (prop1, ...))
PATIENT_UNIQUE_CONSTRAINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Patient", ("id",)),
    ("Episode", ("id",)),
    ("Session", ("id",)),
    ("Message", ("id",)),
    ("Message", ("session_id", "seq")),  # Compound uniqueness
    ("Fact", ("id",)),
    ("FactCandidate", ("id",)),
    ("Note", ("id",)),
    ("Handoff", ("id",)),
    ("OpenQuestion", ("id",)),
    ("Action", ("id",)),
    ("Playbook", ("id",)),
    ("PlaybookStep", ("id",)),
    ("Flag", ("id",)),
    ("Answer", ("id",)),
    ("Agent", ("id",)),
    ("AuditEvent", ("id",)),
    ("ConsolidationRun", ("id",)),
    ("Observation", ("id",)),
    ("ClinicalProcedure", ("id",)),
    ("Provider", ("id",)),
    ("Organization", ("id",)),
)

# Vector indexes: (Label, Property, Dimension, SimilarityFunction)
PATIENT_VECTOR_INDEXES: tuple[tuple[str, str, int, str], ...] = (
    ("Note", "embedding", 384, "cosine"),
    ("Message", "embedding", 384, "cosine"),
    ("Episode", "summary_embedding", 384, "cosine"),
    ("Playbook", "embedding", 384, "cosine"),
)

# Optional Fulltext Indexes: (Label, Property)
PATIENT_FULLTEXT_INDEXES: tuple[tuple[str, str], ...] = (("Fact", "display"),)


# -----------------------------------------------------------------------------
# Clinic Shared Graph Schema Specifications
# -----------------------------------------------------------------------------

CLINIC_RANGE_INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Guideline", ("id",)),
    ("Rule", ("id",)),
    ("ConditionCode", ("code",)),
    ("Drug", ("code",)),
    ("DrugClass", ("id",)),
    ("AllergenClass", ("id",)),
    ("AllergenCode", ("code",)),
    ("PlaybookTemplate", ("id",)),
)

CLINIC_UNIQUE_CONSTRAINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Guideline", ("id",)),
    ("Rule", ("id",)),
    ("ConditionCode", ("code",)),
    ("Drug", ("code",)),
    ("DrugClass", ("id",)),
    ("AllergenClass", ("id",)),
    ("AllergenCode", ("code",)),
    ("PlaybookTemplate", ("id",)),
)

CLINIC_VECTOR_INDEXES: tuple[tuple[str, str, int, str], ...] = (
    ("Guideline", "embedding", 384, "cosine"),
)


# -----------------------------------------------------------------------------
# Control Graph Schema Specifications (`ct_control`)
# -----------------------------------------------------------------------------

CONTROL_RANGE_INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (("Tenant", ("id",)),)

CONTROL_UNIQUE_CONSTRAINTS: tuple[tuple[str, tuple[str, ...]], ...] = (("Tenant", ("id",)),)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


@dataclass
class SchemaBootstrapResult:
    """Summary of schema bootstrap execution."""

    graph_name: str
    range_indexes_created: int = 0
    range_indexes_existed: int = 0
    constraints_created: int = 0
    constraints_existed: int = 0
    vector_indexes_created: int = 0
    vector_indexes_existed: int = 0
    fulltext_indexes_created: int = 0
    fulltext_indexes_existed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_created(self) -> int:
        return (
            self.range_indexes_created
            + self.constraints_created
            + self.vector_indexes_created
            + self.fulltext_indexes_created
        )

    @property
    def total_existed(self) -> int:
        return (
            self.range_indexes_existed
            + self.constraints_existed
            + self.vector_indexes_existed
            + self.fulltext_indexes_existed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "total_created": self.total_created,
            "total_existed": self.total_existed,
            "range_indexes": {
                "created": self.range_indexes_created,
                "existed": self.range_indexes_existed,
            },
            "constraints": {
                "created": self.constraints_created,
                "existed": self.constraints_existed,
            },
            "vector_indexes": {
                "created": self.vector_indexes_created,
                "existed": self.vector_indexes_existed,
            },
            "fulltext_indexes": {
                "created": self.fulltext_indexes_created,
                "existed": self.fulltext_indexes_existed,
            },
            "errors": self.errors,
        }


# -----------------------------------------------------------------------------
# Introspection Helpers
# -----------------------------------------------------------------------------


def get_existing_indexes(graph: Graph) -> list[dict[str, Any]]:
    """Introspects existing indexes from FalkorDB using `CALL db.indexes()`."""
    try:
        res = graph.query("CALL db.indexes()")
        indexes: list[dict[str, Any]] = []
        for row in res.result_set:
            # Row format: [label, [properties], types_dict, options_dict, ...]
            label = str(row[0])
            properties = tuple(str(p) for p in row[1])
            types_dict = row[2] if len(row) > 2 and isinstance(row[2], dict) else {}
            indexes.append(
                {
                    "label": label,
                    "properties": properties,
                    "types": types_dict,
                }
            )
        return indexes
    except Exception as e:
        logger.debug(f"Could not introspect indexes for graph {graph.name}: {e}")
        return []


def get_existing_constraints(graph: Graph) -> list[dict[str, Any]]:
    """Introspects existing constraints from FalkorDB using `CALL db.constraints()`."""
    try:
        res = graph.query("CALL db.constraints()")
        constraints: list[dict[str, Any]] = []
        for row in res.result_set:
            # Row format: ['UNIQUE', label, [properties], 'NODE', 'OPERATIONAL']
            c_type = str(row[0])
            label = str(row[1])
            properties = tuple(str(p) for p in row[2])
            constraints.append(
                {
                    "type": c_type,
                    "label": label,
                    "properties": properties,
                }
            )
        return constraints
    except Exception as e:
        logger.debug(f"Could not introspect constraints for graph {graph.name}: {e}")
        return []


# -----------------------------------------------------------------------------
# Index & Constraint Creation Functions
# -----------------------------------------------------------------------------


def create_range_index(
    graph: Graph,
    label: str,
    properties: tuple[str, ...],
) -> bool:
    """Creates a range index for a node label and property/properties.

    Returns True if created, False if it already existed.
    """
    props_str = ", ".join(f"n.{p}" for p in properties)
    cypher = f"CREATE INDEX FOR (n:{label}) ON ({props_str})"
    try:
        graph.query(cypher)
        return True
    except ResponseError as e:
        if "already indexed" in str(e).lower() or "already exists" in str(e).lower():
            return False
        raise


def create_unique_constraint(
    client: FalkorDB,
    graph_name: str,
    label: str,
    properties: tuple[str, ...],
) -> bool:
    """Creates a unique constraint using FalkorDB's native `GRAPH.CONSTRAINT CREATE`.

    Note: FalkorDB requires the corresponding range index to be created first.
    Returns True if created, False if it already existed.
    """
    try:
        client.connection.execute_command(
            "GRAPH.CONSTRAINT",
            "CREATE",
            graph_name,
            "UNIQUE",
            "NODE",
            label,
            "PROPERTIES",
            str(len(properties)),
            *properties,
        )
        return True
    except ResponseError as e:
        err_msg = str(e).lower()
        if "already exists" in err_msg or "constraint already exists" in err_msg:
            return False
        raise


def create_vector_index(
    graph: Graph,
    label: str,
    property_name: str,
    dimension: int = 384,
    similarity: str = "cosine",
) -> bool:
    """Creates a vector index for a node label and property.

    Returns True if created, False if it already existed.
    """
    cypher = (
        f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property_name}) "
        f"OPTIONS {{dimension:{dimension}, similarityFunction:'{similarity}'}}"
    )
    try:
        graph.query(cypher)
        return True
    except ResponseError as e:
        if "already indexed" in str(e).lower() or "already exists" in str(e).lower():
            return False
        raise


def create_fulltext_index(
    graph: Graph,
    label: str,
    property_name: str,
) -> bool:
    """Creates a fulltext index for a node label and property.

    Returns True if created, False if it already existed.
    """
    cypher = f"CALL db.idx.fulltext.createNodeIndex('{label}', '{property_name}')"
    try:
        graph.query(cypher)
        return True
    except ResponseError as e:
        if "already indexed" in str(e).lower() or "already exists" in str(e).lower():
            return False
        raise


# -----------------------------------------------------------------------------
# Bootstrap Procedures
# -----------------------------------------------------------------------------


def bootstrap_graph_schema(
    graph: Graph,
    range_indexes: tuple[tuple[str, tuple[str, ...]], ...],
    unique_constraints: tuple[tuple[str, tuple[str, ...]], ...],
    vector_indexes: tuple[tuple[str, str, int, str], ...],
    fulltext_indexes: tuple[tuple[str, str], ...] = (),
    client: FalkorDB | None = None,
) -> SchemaBootstrapResult:
    """Bootstraps indexes and constraints for any target FalkorDB graph."""
    db_client = client or get_db_client()
    graph_name = graph.name
    result = SchemaBootstrapResult(graph_name=graph_name)

    # 1. Range indexes
    for label, props in range_indexes:
        try:
            created = create_range_index(graph, label, props)
            if created:
                result.range_indexes_created += 1
            else:
                result.range_indexes_existed += 1
        except Exception as e:
            result.errors.append(f"Range index error on {label}{props}: {e}")
            logger.error(f"Failed to create range index on {graph_name}:{label}{props}: {e}")

    # 2. Unique constraints (FalkorDB requires range indexes to exist first)
    for label, props in unique_constraints:
        try:
            created = create_unique_constraint(db_client, graph_name, label, props)
            if created:
                result.constraints_created += 1
            else:
                result.constraints_existed += 1
        except Exception as e:
            result.errors.append(f"Constraint error on {label}{props}: {e}")
            logger.error(f"Failed to create unique constraint on {graph_name}:{label}{props}: {e}")

    # 3. Vector indexes (384-dim, cosine)
    for label, prop, dim, sim in vector_indexes:
        try:
            created = create_vector_index(graph, label, prop, dimension=dim, similarity=sim)
            if created:
                result.vector_indexes_created += 1
            else:
                result.vector_indexes_existed += 1
        except Exception as e:
            result.errors.append(f"Vector index error on {label}.{prop}: {e}")
            logger.error(f"Failed to create vector index on {graph_name}:{label}.{prop}: {e}")

    # 4. Fulltext indexes
    for label, prop in fulltext_indexes:
        try:
            created = create_fulltext_index(graph, label, prop)
            if created:
                result.fulltext_indexes_created += 1
            else:
                result.fulltext_indexes_existed += 1
        except Exception as e:
            result.errors.append(f"Fulltext index error on {label}.{prop}: {e}")
            logger.error(f"Failed to create fulltext index on {graph_name}:{label}.{prop}: {e}")

    return result


def bootstrap_patient_graph(
    graph: Graph,
    client: FalkorDB | None = None,
) -> SchemaBootstrapResult:
    """Bootstraps schema for an isolated per-patient graph (`patient_<id>`)."""
    return bootstrap_graph_schema(
        graph=graph,
        range_indexes=PATIENT_RANGE_INDEXES,
        unique_constraints=PATIENT_UNIQUE_CONSTRAINTS,
        vector_indexes=PATIENT_VECTOR_INDEXES,
        fulltext_indexes=PATIENT_FULLTEXT_INDEXES,
        client=client,
    )


def bootstrap_clinic_graph(
    graph: Graph,
    client: FalkorDB | None = None,
) -> SchemaBootstrapResult:
    """Bootstraps schema for the shared clinic graph (`clinic_shared`)."""
    return bootstrap_graph_schema(
        graph=graph,
        range_indexes=CLINIC_RANGE_INDEXES,
        unique_constraints=CLINIC_UNIQUE_CONSTRAINTS,
        vector_indexes=CLINIC_VECTOR_INDEXES,
        fulltext_indexes=(),
        client=client,
    )


def bootstrap_control_graph(
    graph: Graph,
    client: FalkorDB | None = None,
) -> SchemaBootstrapResult:
    """Bootstraps schema for the control graph (`ct_control`)."""
    return bootstrap_graph_schema(
        graph=graph,
        range_indexes=CONTROL_RANGE_INDEXES,
        unique_constraints=CONTROL_UNIQUE_CONSTRAINTS,
        vector_indexes=(),
        fulltext_indexes=(),
        client=client,
    )


def bootstrap_all(client: FalkorDB | None = None) -> dict[str, SchemaBootstrapResult]:
    """Bootstraps system graphs (`ct_control`, `clinic_shared`) and all registered tenant graphs."""
    db_client = client or get_db_client()
    results: dict[str, SchemaBootstrapResult] = {}

    # 1. Control graph
    control_g = db_client.select_graph("ct_control")
    results["ct_control"] = bootstrap_control_graph(control_g, db_client)

    # 2. Clinic graph
    clinic_g = db_client.select_graph("clinic_shared")
    results["clinic_shared"] = bootstrap_clinic_graph(clinic_g, db_client)

    # 3. Registered patient tenant graphs
    try:
        from carethread.db.tenant_router import TenantRouter

        router = TenantRouter(db_client)
        tenants = router.list_tenants()
        for pid in tenants:
            patient_g = router.graph(pid)
            results[f"patient_{pid}"] = bootstrap_patient_graph(patient_g, db_client)
    except Exception as e:
        logger.warning(f"Could not iterate tenants for bootstrap: {e}")

    return results


def main() -> None:
    """CLI entrypoint for running schema bootstrap via `python -m carethread.db.schema`."""
    parser = argparse.ArgumentParser(description="Bootstrap FalkorDB schemas for CareThread.")
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional specific patient ID (e.g. p_0123456789ab) to bootstrap.",
    )
    args = parser.parse_args()

    client = get_db_client()
    print("Connecting to FalkorDB...")

    if args.patient_id:
        from carethread.db.tenant_router import TenantRouter

        router = TenantRouter(client)
        g = router.graph(args.patient_id)
        res = bootstrap_patient_graph(g, client)
        print(
            f"Bootstrapped patient graph {g.name}: "
            f"created={res.total_created}, existed={res.total_existed}"
        )
        if res.errors:
            print(f"Errors encountered: {res.errors}")
    else:
        results = bootstrap_all(client)
        for gname, res in results.items():
            print(
                f"Graph '{gname}': {res.total_created} created, "
                f"{res.total_existed} already existed, {len(res.errors)} errors"
            )


if __name__ == "__main__":
    main()
