"""Authentication, token resolution, and role-based access control for MCP tools.

Manages:
- Token mapping from configuration (e.g. config/agents.yaml or CT_TOKENS_FILE)
- Caller identity extraction (agent_id, role, patient_scope)
- Tenant scope boundary validation
- Tool permission verification and authority level enforcement
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from carethread.errors import (
    AuthenticationError,
    RoleNotAllowedError,
    TenantMismatchError,
    ValidationError,
)

logger = logging.getLogger(__name__)

DEFAULT_TOKENS_FILE_ENV: str = "CT_TOKENS_FILE"
DEFAULT_TOKENS_FILE: str = "config/agents.yaml"

# Tool permission mapping per role
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "intake": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "forget_request",
    },
    "specialist": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "forget_request",
    },
    "pharmacy": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "forget_request",
    },
    "clinician": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "consolidate",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "resolve_flag",
        "confirm_fact",
        "forget_request",
    },
    "clinician_admin": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "consolidate",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "resolve_flag",
        "confirm_fact",
        "forget_request",
        "confirm_forget",
    },
    "admin": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "remember",
        "recall",
        "history",
        "handoff",
        "list_handoffs",
        "accept_handoff",
        "complete_handoff",
        "consolidate",
        "record_outcome",
        "commit_answer",
        "explain",
        "list_flags",
        "resolve_flag",
        "confirm_fact",
        "forget_request",
        "confirm_forget",
    },
    "consolidator": {
        "start_session",
        "end_session",
        "append_message",
        "get_session_messages",
        "recall",
        "history",
        "consolidate",
        "record_outcome",
        "explain",
        "list_flags",
    },
}

# Authority caps per role for fact writes
ROLE_AUTHORITY_CAPS: dict[str, set[str]] = {
    "intake": {"patient_reported", "agent_inferred"},
    "specialist": {"agent_inferred", "pending_review"},
    "pharmacy": {"agent_inferred", "record", "pharmacy"},
    "clinician": {"clinician", "record", "agent_inferred", "patient_reported"},
    "clinician_admin": {"clinician", "record", "agent_inferred", "patient_reported"},
    "admin": {"clinician", "record", "agent_inferred", "patient_reported"},
}


@dataclass
class CallerIdentity:
    """Represents the authenticated caller identity for an MCP session or request."""

    token: str
    agent_id: str
    role: str
    patient_scope: list[str] | str = "*"

    def can_access_patient(self, patient_id: str) -> bool:
        """Determines if the caller has access permissions for the target patient tenant."""
        if self.patient_scope == "*" or "*" in self.patient_scope:
            return True
        if isinstance(self.patient_scope, list):
            return patient_id in self.patient_scope
        return patient_id == self.patient_scope


class TokenStore:
    """Loads, caches, and verifies bearer tokens against configured agent permissions."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._tokens: dict[str, CallerIdentity] = {}
        target_path = (
            Path(config_path)
            if config_path is not None
            else Path(os.environ.get(DEFAULT_TOKENS_FILE_ENV, DEFAULT_TOKENS_FILE))
        )
        self.load_config(target_path)

    def load_config(self, path: Path) -> None:
        """Loads token configuration from a YAML file."""
        if not path.is_file():
            logger.debug(f"Tokens configuration file '{path}' not found. Using defaults.")
            self._load_fallback_tokens()
            return

        try:
            raw_text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw_text) or {}
            tokens_dict = data.get("tokens", {})
            for tok, meta in tokens_dict.items():
                self.register_token(
                    token=str(tok),
                    agent_id=str(meta.get("agent_id", "unknown_agent")),
                    role=str(meta.get("role", "intake")),
                    patient_scope=meta.get("patient_scope", "*"),
                )
        except Exception as e:
            logger.warning(f"Error loading tokens config from '{path}': {e}. Using fallbacks.")
            self._load_fallback_tokens()

    def _load_fallback_tokens(self) -> None:
        """Provides default dev tokens when config file is absent."""
        defaults = [
            ("tok_intake_01", "agt_intake_01", "intake", "*"),
            ("tok_specialist_01", "agt_specialist_01", "specialist", "*"),
            ("tok_pharmacy_01", "agt_pharmacy_01", "pharmacy", "*"),
            ("tok_clinician_01", "agt_clinician_01", "clinician", "*"),
            ("tok_admin_01", "agt_admin_01", "clinician_admin", "*"),
            ("tok_consolidator_01", "agt_consolidator_01", "consolidator", "*"),
        ]
        for tok, aid, role, scope in defaults:
            self.register_token(tok, aid, role, scope)

    def register_token(
        self,
        token: str,
        agent_id: str,
        role: str,
        patient_scope: list[str] | str = "*",
    ) -> None:
        """Registers a token mapping in memory."""
        self._tokens[token.strip()] = CallerIdentity(
            token=token.strip(),
            agent_id=agent_id.strip(),
            role=role.strip().lower(),
            patient_scope=patient_scope,
        )

    def resolve_token(self, token: str | None) -> CallerIdentity:
        """Resolves a bearer token string to a verified CallerIdentity.

        Raises:
            AuthenticationError: If token is missing, blank, or unrecognized.
        """
        if not token or not token.strip():
            raise AuthenticationError(
                "Authentication required: missing or empty bearer token.",
                details={"token": str(token)},
            )

        clean_token = token.strip()
        identity = self._tokens.get(clean_token)
        if identity is None:
            raise AuthenticationError(
                f"Invalid or unrecognized authentication token: {clean_token!r}",
                details={"token": clean_token},
            )
        return identity

    def check_permission(
        self,
        caller: CallerIdentity,
        tool_name: str,
        patient_id: str | None = None,
        authority: str | None = None,
    ) -> None:
        """Enforces role-based permissions, patient scoping, and authority limits.

        Raises:
            RoleNotAllowedError: If caller's role is not authorized for tool_name.
            TenantMismatchError: If target patient is outside caller's authorized scope.
            ValidationError: If requested write authority exceeds role permissions.
        """
        clean_tool = tool_name.strip()
        allowed_tools = ROLE_PERMISSIONS.get(caller.role, set())
        if clean_tool not in allowed_tools:
            raise RoleNotAllowedError(
                f"Role '{caller.role}' is not authorized to execute tool '{clean_tool}'.",
                details={
                    "agent_id": caller.agent_id,
                    "role": caller.role,
                    "tool": clean_tool,
                    "allowed_tools": sorted(allowed_tools),
                },
            )

        if patient_id and not caller.can_access_patient(patient_id):
            raise TenantMismatchError(
                f"Target patient '{patient_id}' is outside authorized tenant scope for caller.",
                details={
                    "agent_id": caller.agent_id,
                    "role": caller.role,
                    "target_patient_id": patient_id,
                    "patient_scope": caller.patient_scope,
                },
            )

        if authority:
            clean_auth = authority.strip().lower()
            allowed_auths = ROLE_AUTHORITY_CAPS.get(caller.role, set())
            if clean_auth not in allowed_auths:
                raise ValidationError(
                    f"Authority '{authority}' exceeds permissions for role '{caller.role}'.",
                    code="ROLE_NOT_ALLOWED",
                    details={
                        "role": caller.role,
                        "requested_authority": authority,
                        "allowed_authorities": sorted(allowed_auths),
                    },
                )
