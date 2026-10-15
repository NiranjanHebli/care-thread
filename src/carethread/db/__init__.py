"""Database package exports."""

from typing import TYPE_CHECKING

from carethread.db.client import check_health, get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.tenant_router import TenantRouter

if TYPE_CHECKING:
    from carethread.db.schema import (
        SchemaBootstrapResult,
        bootstrap_all,
        bootstrap_clinic_graph,
        bootstrap_control_graph,
        bootstrap_patient_graph,
    )

__all__ = [
    "get_db_client",
    "check_health",
    "TenantRouter",
    "ClinicReader",
    "bootstrap_patient_graph",
    "bootstrap_clinic_graph",
    "bootstrap_control_graph",
    "bootstrap_all",
    "SchemaBootstrapResult",
]


def __getattr__(name: str):
    if name in {
        "bootstrap_patient_graph",
        "bootstrap_clinic_graph",
        "bootstrap_control_graph",
        "bootstrap_all",
        "SchemaBootstrapResult",
    }:
        import carethread.db.schema as schema_mod

        return getattr(schema_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
