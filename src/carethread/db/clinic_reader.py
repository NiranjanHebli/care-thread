"""Strictly read-only reader for the shared clinic knowledge graph."""

from typing import Any

from falkordb import FalkorDB, QueryResult


class ClinicReader:
    """Read-only access wrapper for the shared clinic knowledge graph (`clinic_shared`).

    Enforces read-only querying at the application and driver levels.
    Exposes only `ro_query` (invoking FalkorDB's GRAPH.RO_QUERY).
    Deliberately omits any write, delete, update, or general query methods.
    """

    GRAPH_NAME = "clinic_shared"

    def __init__(self, client: FalkorDB) -> None:
        self._client = client
        self._graph = client.select_graph(self.GRAPH_NAME)

    @property
    def graph_name(self) -> str:
        """Name of the backing shared clinic graph."""
        return self.GRAPH_NAME

    def ro_query(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Executes a strictly read-only query against the clinic_shared graph.

        Uses GRAPH.RO_QUERY under the hood to ensure read-only transactions.
        """
        return self._graph.ro_query(query, params or {})
