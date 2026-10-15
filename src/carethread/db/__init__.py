"""Database package exports."""

from carethread.db.client import check_health, get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.tenant_router import TenantRouter

__all__ = [
    "get_db_client",
    "check_health",
    "TenantRouter",
    "ClinicReader",
]
