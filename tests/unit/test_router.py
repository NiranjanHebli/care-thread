"""Unit tests for TenantRouter, ClinicReader, and graph isolation security."""

from unittest.mock import MagicMock

import pytest
from falkordb import FalkorDB

from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
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


def test_malicious_patient_ids_rejected_with_spy() -> None:
    """Malicious patient_id values are rejected by validation with VALIDATION_ERROR.

    A spy on the client verifies that none of these values ever reach FalkorDB.
    """
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


def test_valid_patient_ids_create_correct_graph_handles() -> None:
    """Valid patient IDs matching ^p_[0-9a-f]{12}$ return patient_<id> graph handle."""
    mock_client = MagicMock(spec=FalkorDB)
    router = TenantRouter(mock_client)

    valid_ids = [
        "p_0123456789ab",
        "p_abcdef012345",
        "p_000000000000",
        "p_ffffffffffff",
    ]

    for pid in valid_ids:
        router.graph(pid)
        mock_client.select_graph.assert_called_with(f"patient_{pid}")


def test_clinic_reader_strictly_read_only() -> None:
    """ClinicReader exposes only ro_query. No write method exists."""
    mock_client = MagicMock(spec=FalkorDB)
    reader = ClinicReader(mock_client)

    # Verify read method is exposed
    assert hasattr(reader, "ro_query")
    assert callable(reader.ro_query)

    # Verify that write methods DO NOT exist on the wrapper
    prohibited_methods = [
        "query",
        "write",
        "execute_write",
        "create",
        "delete",
        "update",
        "mutate",
    ]
    for method in prohibited_methods:
        assert not hasattr(reader, method), f"ClinicReader illegally exposes write method: {method}"

    # Verify ro_query delegates to underlying graph ro_query
    reader.ro_query("MATCH (g:Guideline) RETURN g")
    mock_client.select_graph.return_value.ro_query.assert_called_once_with(
        "MATCH (g:Guideline) RETURN g", {}
    )


def test_ct_control_tenant_registry_integration() -> None:
    """Verifies live tenant registration, idempotency, and listing in ct_control."""
    client = get_db_client()
    router = TenantRouter(client)

    test_pid = "p_a1b2c3d4e5f6"

    try:
        # 1. Register patient
        reg_result = router.register_patient(test_pid, metadata={"source": "test"})
        assert reg_result["id"] == test_pid
        assert reg_result["status"] == "active"
        assert reg_result["graph_name"] == f"patient_{test_pid}"

        # 2. Check registration
        assert router.is_registered(test_pid) is True

        # 3. Verify in tenant list
        tenants = router.list_tenants()
        assert test_pid in tenants

        # 4. Idempotent re-registration
        re_reg = router.register_patient(test_pid)
        assert re_reg["id"] == test_pid

    finally:
        # Cleanup
        router.deregister_patient(test_pid)
        assert router.is_registered(test_pid) is False
