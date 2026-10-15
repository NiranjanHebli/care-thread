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
