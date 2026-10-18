"""Standard error codes, exceptions, and response envelopes for the MCP protocol.

Provides:
- Standardized MCP tool response envelopes ({ok: True, data: ..., request_id: ...})
- Error serialization matching error codes
- Re-exports of domain exceptions
"""

from __future__ import annotations

from typing import Any

from carethread.errors import (
    AnswerNotFoundError,
    AuditError,
    AuditEventNotFoundError,
    AuthenticationError,
    CareThreadError,
    ClaimUnsupportedError,
    ConflictStaleVersionError,
    ConsolidationError,
    DatabaseError,
    DeletionRequestNotFoundError,
    EpisodeNotFoundError,
    FactNotFoundError,
    FlagNotFoundError,
    ForgetConfirmationRequiredError,
    ForgetNotAllowedError,
    GraphUnavailableError,
    HandoffExpiredError,
    HandoffIncompleteError,
    HandoffNotFoundError,
    HandoffNotOpenError,
    HandoffStateError,
    MemoryIdNotFoundError,
    NoteNotFoundError,
    PatientNotFoundError,
    PlaybookNotFoundError,
    PlaybookStateError,
    RateLimitError,
    RequestExpiredError,
    RetrievalExpiredError,
    RoleNotAllowedError,
    SessionClosedError,
    SessionNotFoundError,
    TenantMismatchError,
    ValidationError,
)


def success_envelope(data: Any, request_id: str) -> dict[str, Any]:
    """Wraps successful tool execution results in the standard MCP envelope."""
    return {
        "ok": True,
        "data": data,
        "request_id": request_id,
    }


def error_envelope(
    code: str,
    message: str,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wraps error execution details in the standard MCP envelope."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
        "request_id": request_id,
    }


__all__ = [
    "AnswerNotFoundError",
    "AuditError",
    "AuditEventNotFoundError",
    "AuthenticationError",
    "CareThreadError",
    "ClaimUnsupportedError",
    "ConflictStaleVersionError",
    "ConsolidationError",
    "DatabaseError",
    "DeletionRequestNotFoundError",
    "EpisodeNotFoundError",
    "FactNotFoundError",
    "FlagNotFoundError",
    "ForgetConfirmationRequiredError",
    "ForgetNotAllowedError",
    "GraphUnavailableError",
    "HandoffExpiredError",
    "HandoffIncompleteError",
    "HandoffNotFoundError",
    "HandoffNotOpenError",
    "HandoffStateError",
    "MemoryIdNotFoundError",
    "NoteNotFoundError",
    "PatientNotFoundError",
    "PlaybookNotFoundError",
    "PlaybookStateError",
    "RateLimitError",
    "RequestExpiredError",
    "RetrievalExpiredError",
    "RoleNotAllowedError",
    "SessionClosedError",
    "SessionNotFoundError",
    "TenantMismatchError",
    "ValidationError",
    "error_envelope",
    "success_envelope",
]
