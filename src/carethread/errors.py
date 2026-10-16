"""CareThread standardized error classes and exception hierarchy."""

from typing import Any


class CareThreadError(Exception):
    """Base exception for all CareThread domain errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to standardized API error dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(CareThreadError):
    """Raised when request payload or parameter validation fails."""

    def __init__(
        self,
        message: str,
        code: str = "VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class TenantMismatchError(CareThreadError):
    """Raised when an operation targets a tenant outside authorized scope."""

    def __init__(
        self,
        message: str = "Target patient is outside authorized tenant scope",
        code: str = "TENANT_MISMATCH",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class PatientNotFoundError(CareThreadError):
    """Raised when a specified patient graph or registry entry does not exist."""

    def __init__(
        self,
        message: str = "Patient not found",
        code: str = "PATIENT_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class SessionNotFoundError(CareThreadError):
    """Raised when a specified session does not exist."""

    def __init__(
        self,
        message: str = "Session not found",
        code: str = "SESSION_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class EpisodeNotFoundError(CareThreadError):
    """Raised when a specified episode does not exist."""

    def __init__(
        self,
        message: str = "Episode not found",
        code: str = "EPISODE_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class FactNotFoundError(CareThreadError):
    """Raised when a specified fact does not exist."""

    def __init__(
        self,
        message: str = "Fact not found",
        code: str = "FACT_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ConflictStaleVersionError(CareThreadError):
    """Raised when compare-and-swap (CAS) supersession detects an expected_version mismatch."""

    def __init__(
        self,
        message: str = "Fact version conflict: expected_version does not match current version.",
        code: str = "CONFLICT_STALE_VERSION",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class NoteNotFoundError(CareThreadError):
    """Raised when a specified note does not exist."""

    def __init__(
        self,
        message: str = "Note not found",
        code: str = "NOTE_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class FlagNotFoundError(CareThreadError):
    """Raised when a specified safety flag does not exist."""

    def __init__(
        self,
        message: str = "Flag not found",
        code: str = "FLAG_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HandoffNotFoundError(CareThreadError):
    """Raised when a specified handoff packet does not exist."""

    def __init__(
        self,
        message: str = "Handoff not found",
        code: str = "HANDOFF_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HandoffStateError(CareThreadError):
    """Raised when an invalid handoff state transition or CAS conflict occurs."""

    def __init__(
        self,
        message: str = "Invalid handoff state transition",
        code: str = "HANDOFF_STATE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class DatabaseError(CareThreadError):
    """Raised when graph database operation or verification fails."""

    def __init__(
        self,
        message: str = "Database operation failed",
        code: str = "DATABASE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class AuditError(CareThreadError):
    """Raised when an audit logging operation fails."""

    def __init__(
        self,
        message: str = "Audit logging operation failed",
        code: str = "AUDIT_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class AuditEventNotFoundError(CareThreadError):
    """Raised when a requested audit event is not found."""

    def __init__(
        self,
        message: str = "Audit event not found",
        code: str = "AUDIT_EVENT_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
