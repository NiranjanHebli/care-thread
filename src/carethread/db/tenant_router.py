"""Tenant routing and registry management for isolated patient graphs."""

import re
import time
from typing import Any

from falkordb import FalkorDB, Graph

from carethread.db.clinic_reader import ClinicReader
from carethread.errors import ValidationError

# Patient ID format: prefix 'p_' followed by exactly 12 lowercase hex characters
PATIENT_ID_REGEX = re.compile(r"^p_[0-9a-f]{12}$")
CONTROL_GRAPH_NAME = "ct_control"


class TenantRouter:
    """Manages routing to isolated per-patient graphs and tenant registry in `ct_control`.

    Every patient record is stored in an isolated FalkorDB graph (`patient_<id>`).
    This class is the only authorized gateway for obtaining patient graph handles.
    """

    def __init__(self, client: FalkorDB) -> None:
        self._client = client
        self._control_graph = client.select_graph(CONTROL_GRAPH_NAME)

    @staticmethod
    def validate_patient_id(patient_id: str) -> str:
        """Validates that patient_id strictly matches the required format `^p_[0-9a-f]{12}$`.

        Raises ValidationError with code 'VALIDATION_ERROR' if invalid.
        """
        if not isinstance(patient_id, str) or not PATIENT_ID_REGEX.match(patient_id):
            raise ValidationError(
                f"Invalid patient_id format: {patient_id!r}. "
                f"Must strictly match pattern '{PATIENT_ID_REGEX.pattern}'.",
                code="VALIDATION_ERROR",
                details={
                    "patient_id": str(patient_id),
                    "expected_pattern": PATIENT_ID_REGEX.pattern,
                },
            )
        return patient_id

    def graph(self, patient_id: str) -> Graph:
        """Returns the isolated FalkorDB Graph handle for a patient.

        This is the sole authorized entrypoint for constructing patient graph handles.
        Validates patient_id before interacting with FalkorDB.
        """
        valid_pid = self.validate_patient_id(patient_id)
        graph_name = f"patient_{valid_pid}"
        return self._client.select_graph(graph_name)

    def get_clinic_reader(self) -> ClinicReader:
        """Returns a strictly read-only reader for the shared clinic graph."""
        return ClinicReader(self._client)

    def bootstrap_patient(self, patient_id: str) -> Any:
        """Bootstraps schema (range indexes, constraints, vector indexes) for the patient graph.

        Idempotent operation.
        """
        from carethread.db.schema import bootstrap_patient_graph

        g = self.graph(patient_id)
        return bootstrap_patient_graph(g, self._client)

    # -------------------------------------------------------------------------
    # Tenant Registry Operations (in `ct_control`)
    # -------------------------------------------------------------------------

    def register_patient(
        self,
        patient_id: str,
        metadata: dict[str, Any] | None = None,
        bootstrap_schema: bool = False,
    ) -> dict[str, Any]:
        """Registers a patient in the ct_control tenant registry.

        Idempotent: merges the node if it already exists.
        Optionally bootstraps schema for the patient graph if bootstrap_schema is True.
        """
        valid_pid = self.validate_patient_id(patient_id)
        now_ts = int(time.time())
        meta = metadata or {}

        query = (
            "MERGE (p:Tenant {id: $pid}) "
            "ON CREATE SET p.created_at = $now, p.status = 'active', p.graph_name = $graph_name "
            "RETURN p.id AS id, p.status AS status, p.created_at AS created_at"
        )
        res = self._control_graph.query(
            query,
            {
                "pid": valid_pid,
                "now": now_ts,
                "graph_name": f"patient_{valid_pid}",
            },
        )
        row = res.result_set[0]

        bootstrap_res = None
        if bootstrap_schema:
            bootstrap_res = self.bootstrap_patient(valid_pid)

        return {
            "id": row[0],
            "status": row[1],
            "created_at": row[2],
            "graph_name": f"patient_{valid_pid}",
            "metadata": meta,
            "bootstrap_result": bootstrap_res,
        }

    def is_registered(self, patient_id: str) -> bool:
        """Checks if a patient is registered in the ct_control registry."""
        valid_pid = self.validate_patient_id(patient_id)
        query = "MATCH (p:Tenant {id: $pid}) RETURN count(p) AS count"
        res = self._control_graph.query(query, {"pid": valid_pid})
        return bool(res.result_set and res.result_set[0][0] > 0)

    def list_tenants(self) -> list[str]:
        """Lists all registered patient tenant IDs in ct_control."""
        query = "MATCH (p:Tenant) RETURN p.id AS id ORDER BY p.id"
        res = self._control_graph.query(query)
        return [row[0] for row in res.result_set]

    def deregister_patient(self, patient_id: str) -> bool:
        """Removes a patient from the active tenant registry in ct_control."""
        valid_pid = self.validate_patient_id(patient_id)
        query = "MATCH (p:Tenant {id: $pid}) DELETE p RETURN count(p) AS deleted"
        res = self._control_graph.query(query, {"pid": valid_pid})
        return bool(res.result_set and res.result_set[0][0] > 0)
