"""Retention and deletion memory alias re-exporting forgetting facilities."""

from carethread.memory.forgetting import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SALT,
    REQUEST_TTL_SECONDS,
    VALID_SCOPES,
    DeletionRequestRecord,
    ForgettingManager,
    TombstoneRecord,
    compute_patient_hash,
    generate_confirm_token,
    hash_confirm_token,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_SALT",
    "REQUEST_TTL_SECONDS",
    "VALID_SCOPES",
    "DeletionRequestRecord",
    "ForgettingManager",
    "TombstoneRecord",
    "compute_patient_hash",
    "generate_confirm_token",
    "hash_confirm_token",
]
