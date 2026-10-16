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


class PlaybookNotFoundError(CareThreadError):
    """Raised when a specified playbook does not exist."""

    def __init__(
        self,
        message: str = "Playbook not found",
        code: str = "PLAYBOOK_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class PlaybookStateError(CareThreadError):
    """Raised when an invalid playbook state transition or operation occurs."""

    def __init__(
        self,
        message: str = "Invalid playbook state transition",
        code: str = "PLAYBOOK_STATE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ConsolidationError(CareThreadError):
    """Raised when a memory consolidation run fails."""

    def __init__(
        self,
        message: str = "Memory consolidation operation failed",
        code: str = "CONSOLIDATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ForgetNotAllowedError(CareThreadError):
    """Raised when a forget or deletion request operation is not permitted."""

    def __init__(
        self,
        message: str = "Forget request operation not permitted",
        code: str = "FORGET_NOT_ALLOWED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ForgetConfirmationRequiredError(CareThreadError):
    """Raised when confirmation token is missing, invalid, or unverified."""

    def __init__(
        self,
        message: str = "Confirmation required or invalid token provided",
        code: str = "FORGET_CONFIRMATION_REQUIRED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class RequestExpiredError(CareThreadError):
    """Raised when a deletion request has expired past its validity window."""

    def __init__(
        self,
        message: str = "Request has expired",
        code: str = "REQUEST_EXPIRED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class DeletionRequestNotFoundError(CareThreadError):
    """Raised when a specified deletion request cannot be found."""

    def __init__(
        self,
        message: str = "Deletion request not found",
        code: str = "DELETION_REQUEST_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class AuthenticationError(CareThreadError):
    """Raised when authentication credentials are missing or invalid."""

    def __init__(
        self,
        message: str = "Authentication required or credentials invalid",
        code: str = "AUTH_REQUIRED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class RoleNotAllowedError(CareThreadError):
    """Raised when caller role is not authorized for requested tool or action."""

    def __init__(
        self,
        message: str = "Role not authorized for requested operation",
        code: str = "ROLE_NOT_ALLOWED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class SessionClosedError(CareThreadError):
    """Raised when an operation targets a session that is already closed."""

    def __init__(
        self,
        message: str = "Session is already closed",
        code: str = "SESSION_CLOSED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class MemoryIdNotFoundError(CareThreadError):
    """Raised when one or more referenced memory IDs cannot be found."""

    def __init__(
        self,
        message: str = "Referenced memory identifier not found",
        code: str = "MEMORY_ID_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class RateLimitError(CareThreadError):
    """Raised when request rate exceeds allowed throughput quotas."""

    def __init__(
        self,
        message: str = "Request rate limit exceeded",
        code: str = "RATE_LIMITED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class GraphUnavailableError(CareThreadError):
    """Raised when graph database connectivity or transaction fails."""

    def __init__(
        self,
        message: str = "Graph storage service is unavailable",
        code: str = "GRAPH_UNAVAILABLE",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class EmbeddingUnavailableError(CareThreadError):
    """Raised when embedding model or computation is unavailable."""

    def __init__(
        self,
        message: str = "Embedding service is unavailable",
        code: str = "EMBEDDING_UNAVAILABLE",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HandoffIncompleteError(CareThreadError):
    """Raised when handoff creation packet is missing required attributes."""

    def __init__(
        self,
        message: str = "Handoff packet is incomplete",
        code: str = "HANDOFF_INCOMPLETE",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HandoffNotOpenError(CareThreadError):
    """Raised when an operation expects a handoff to be in open status."""

    def __init__(
        self,
        message: str = "Handoff is not in open status",
        code: str = "HANDOFF_NOT_OPEN",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HandoffExpiredError(CareThreadError):
    """Raised when an operation targets an expired handoff."""

    def __init__(
        self,
        message: str = "Handoff has expired past its TTL",
        code: str = "HANDOFF_EXPIRED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class RetrievalExpiredError(CareThreadError):
    """Raised when an answer commits against an expired or evicted retrieval."""

    def __init__(
        self,
        message: str = "Retrieval context has expired",
        code: str = "RETRIEVAL_EXPIRED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ClaimUnsupportedError(CareThreadError):
    """Raised when an answer contains claims lacking verifiable support."""

    def __init__(
        self,
        message: str = "Answer claims lack supporting provenance citations",
        code: str = "CLAIM_UNSUPPORTED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class AnswerNotFoundError(CareThreadError):
    """Raised when a cited or queried answer node is not found."""

    def __init__(
        self,
        message: str = "Answer not found",
        code: str = "ANSWER_NOT_FOUND",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
