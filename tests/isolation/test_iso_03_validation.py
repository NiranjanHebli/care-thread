"""Tenant isolation tests verifying malicious patient_id input rejection (T-ISO-03).

Validates:
- Malicious patient_id values (SQL/Cypher injection, directory traversal,
  missing prefixes, empty strings, oversized 10 KB payloads, invalid hex)
  are strictly rejected by validation with VALIDATION_ERROR.
- A spy on the database client verifies that none of these inputs ever reach FalkorDB.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from falkordb import FalkorDB

from carethread.db.tenant_router import TenantRouter
from carethread.errors import ValidationError

MALICIOUS_PATIENT_IDS = [
    "p_x'; MATCH (n) RETURN n",
    "../patient_B",
    "patient_B",
    "",
    "a" * 10240,  # 10 KB string
    "p_1234567890",  # too short
    "p_1234567890abcdef",  # too long
    "p_1234567890AB",  # uppercase hex
    "p_1234567890ag",  # non-hex character 'g'
    "p_1234 567890a",  # contains space
    None,  # type: ignore[list-item]
    12345,  # type: ignore[list-item]
]


def test_t_iso_03_malicious_patient_ids_rejected_with_spy() -> None:
    """T-ISO-03: Malicious patient_id inputs rejected with VALIDATION_ERROR before DB."""
    spy_client = MagicMock(spec=FalkorDB)
    router = TenantRouter(spy_client)

    # Initial control graph was selected during initialization
    initial_call_count = spy_client.select_graph.call_count

    for bad_id in MALICIOUS_PATIENT_IDS:
        with pytest.raises(ValidationError) as exc_info:
            router.graph(bad_id)  # type: ignore[arg-type]

        assert exc_info.value.code == "VALIDATION_ERROR", (
            f"Expected code 'VALIDATION_ERROR' for {bad_id!r}, got {exc_info.value.code}"
        )

        # Verification by spy: no graph was selected for this malicious input
        assert spy_client.select_graph.call_count == initial_call_count, (
            f"FalkorDB was contacted for malicious input: {bad_id!r}"
        )


def test_t_iso_03_router_register_rejects_malicious_ids() -> None:
    """T-ISO-03: Registration interface also strictly rejects malicious patient IDs with spy."""
    spy_client = MagicMock(spec=FalkorDB)
    router = TenantRouter(spy_client)

    initial_query_count = spy_client.select_graph.return_value.query.call_count

    for bad_id in MALICIOUS_PATIENT_IDS:
        with pytest.raises(ValidationError) as exc_info:
            router.register_patient(bad_id)  # type: ignore[arg-type]

        assert exc_info.value.code == "VALIDATION_ERROR"
        assert spy_client.select_graph.return_value.query.call_count == initial_query_count
