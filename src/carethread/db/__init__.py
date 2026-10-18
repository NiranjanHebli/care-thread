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
    from carethread.db.seed_clinic import (
        ClinicSeeder,
        ClinicSeedResult,
        seed_clinic,
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
    "seed_clinic",
    "ClinicSeeder",
    "ClinicSeedResult",
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
    if name in {"seed_clinic", "ClinicSeeder", "ClinicSeedResult"}:
        import carethread.db.seed_clinic as seed_mod

        return getattr(seed_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
