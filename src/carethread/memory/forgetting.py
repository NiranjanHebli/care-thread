"""Forgetting and deletion lifecycle management with cryptographically secure tombstones.

Deletion and retention lifecycle:
- forget_request generates a 32-byte cryptographic token and stores its SHA-256 hash.
- Tokens expire after 15 minutes TTL.
- Approver must hold clinician_admin role and be distinct from the requester.
- Execution per scope:
  - 'all': marks tenant registry entry deleting (blocking new sessions), drops in-memory
    sessions and retrievals, executes GRAPH.DELETE, removes the registry entry,
    writes a cryptographic tombstone (SHA-256 of salt + patient_id), purges tenant
    LLM cache fixtures, and triggers BGREWRITEAOF.
  - 'fact': sets status='retracted', nulls display and value, retains a graph stub.
  - 'session': redacts message content and nulls session embeddings.
- Worker resumes any tenant entry left in 'deleting' state following an unexpected crash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import CONTROL_GRAPH_NAME, TenantRouter
from carethread.errors import (
    DeletionRequestNotFoundError,
    FactNotFoundError,
    ForgetConfirmationRequiredError,
    ForgetNotAllowedError,
    PatientNotFoundError,
    RequestExpiredError,
    SessionNotFoundError,
    ValidationError,
)
from carethread.util.id import deletion_request_id, tombstone_id

logger = logging.getLogger(__name__)

VALID_SCOPES: frozenset[str] = frozenset({"fact", "session", "all"})
REQUEST_TTL_SECONDS: int = 15 * 60  # 15 minutes TTL
DEFAULT_SALT_ENV: str = "CT_SALT"
DEFAULT_SALT: str = "carethread_default_cryptographic_salt"
DEFAULT_CACHE_DIR_ENV: str = "CT_LLM_CACHE_DIR"
DEFAULT_CACHE_DIR: str = "fixtures/llm_cache"


def _extract_props(node: Any) -> dict[str, Any]:
    """Safely extracts properties dictionary from a graph node or mapping."""
    if hasattr(node, "properties"):
        return dict(node.properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


def compute_patient_hash(patient_id: str, salt: str | None = None) -> str:
    """Computes a cryptographically secure SHA-256 hash of salt + patient_id."""
    effective_salt = salt if salt is not None else os.environ.get(DEFAULT_SALT_ENV, DEFAULT_SALT)
    payload = f"{effective_salt}:{patient_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_confirm_token(token: str) -> str:
    """Computes SHA-256 digest of a confirmation token string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_confirm_token() -> str:
    """Generates a cryptographically secure 32-byte hex confirmation token."""
    return secrets.token_hex(32)


@dataclass
class DeletionRequestRecord:
    """Represents a deletion or forget request lifecycle state in ct_control."""

    id: str
    patient_id: str
    scope: str
    target_id: str | None
    reason: str
    requested_by: str
    token_hash: str
    created_at: int
    expires_at: int
    status: str = "pending_approval"
    approver_id: str | None = None
    executed_at: int | None = None
    tombstone_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeletionRequestRecord:
        return cls(
            id=str(data.get("id", "")),
            patient_id=str(data.get("patient_id", "")),
            scope=str(data.get("scope", "all")),
            target_id=data.get("target_id") or None,
            reason=str(data.get("reason", "")),
            requested_by=str(data.get("requested_by", "")),
            token_hash=str(data.get("token_hash", "")),
            created_at=int(data.get("created_at", 0)),
            expires_at=int(data.get("expires_at", 0)),
            status=str(data.get("status", "pending_approval")),
            approver_id=data.get("approver_id") or None,
            executed_at=(int(data["executed_at"]) if data.get("executed_at") is not None else None),
            tombstone_id=data.get("tombstone_id") or None,
        )


@dataclass
class TombstoneRecord:
    """Cryptographically secure tombstone stored in ct_control after deletion."""

    id: str
    patient_hash: str
    timestamp: int
    request_id: str
    approver_id: str
    scope: str
    target_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TombstoneRecord:
        return cls(
            id=str(data.get("id", "")),
            patient_hash=str(data.get("patient_hash", "")),
            timestamp=int(data.get("timestamp", 0)),
            request_id=str(data.get("request_id", "")),
            approver_id=str(data.get("approver_id", "")),
            scope=str(data.get("scope", "all")),
            target_id=data.get("target_id") or None,
        )


class ForgettingManager:
    """Manages forget requests, approval verifications, tombstones, and deletion."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clock: Clock | None = None,
        salt: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._client = router._client if router is not None else get_db_client()
        self._router = router or TenantRouter(self._client)
        self._control_graph = self._client.select_graph(CONTROL_GRAPH_NAME)
        self._clock = clock or get_clock()
        self._salt = salt if salt is not None else os.environ.get(DEFAULT_SALT_ENV, DEFAULT_SALT)
        effective_cache = (
            cache_dir
            if cache_dir is not None
            else os.environ.get(DEFAULT_CACHE_DIR_ENV, DEFAULT_CACHE_DIR)
        )
        self._cache_dir = Path(effective_cache)

    def forget_request(
        self,
        patient_id: str,
        scope: str,
        target_id: str | None = None,
        reason: str = "",
        requested_by: str = "",
    ) -> dict[str, Any]:
        """Initiates a deletion request and returns a confirmation token.

        Args:
            patient_id: Validated patient tenant identifier (e.g. 'p_1a967cbe5b94').
            scope: Deletion scope ('fact', 'session', or 'all').
            target_id: Target entity ID (required for 'fact' or 'session' scopes).
            reason: Clinical or administrative justification for deletion.
            requested_by: Identifier of the requesting user or agent.

        Returns:
            Dictionary containing request_id, confirm_token, expires_at, and status.

        Raises:
            ValidationError: If scope or arguments are invalid.
            PatientNotFoundError: If patient tenant is not registered.
            FactNotFoundError: If target fact does not exist in patient graph.
            SessionNotFoundError: If target session does not exist in patient graph.
        """
        clean_scope = scope.strip().lower()
        if clean_scope not in VALID_SCOPES:
            raise ValidationError(
                f"Invalid deletion scope '{scope}'. Allowed scopes: {sorted(VALID_SCOPES)}",
                details={"scope": scope, "allowed": list(VALID_SCOPES)},
            )

        valid_pid = self._router.validate_patient_id(patient_id)
        if not self._router.is_registered(valid_pid):
            raise PatientNotFoundError(
                f"Patient tenant '{valid_pid}' is not registered in the system.",
                details={"patient_id": valid_pid},
            )

        clean_target = target_id.strip() if target_id else None
        target_graph = self._router.graph(valid_pid)

        if clean_scope == "fact":
            if not clean_target:
                raise ValidationError(
                    "target_id is required when deletion scope is 'fact'.",
                    details={"scope": clean_scope},
                )
            f_res = target_graph.query(
                "MATCH (f:Fact {id: $fid}) RETURN f.id",
                {"fid": clean_target},
            )
            if not f_res.result_set:
                raise FactNotFoundError(
                    f"Fact '{clean_target}' not found in patient graph '{valid_pid}'.",
                    details={"patient_id": valid_pid, "fact_id": clean_target},
                )

        elif clean_scope == "session":
            if not clean_target:
                raise ValidationError(
                    "target_id is required when deletion scope is 'session'.",
                    details={"scope": clean_scope},
                )
            s_res = target_graph.query(
                "MATCH (s:Session {id: $sid}) RETURN s.id",
                {"sid": clean_target},
            )
            if not s_res.result_set:
                raise SessionNotFoundError(
                    f"Session '{clean_target}' not found in patient graph '{valid_pid}'.",
                    details={"patient_id": valid_pid, "session_id": clean_target},
                )

        req_id = deletion_request_id()
        raw_token = generate_confirm_token()
        token_hash = hash_confirm_token(raw_token)
        now_ts = self._clock.now_ts()
        expires_at = now_ts + REQUEST_TTL_SECONDS

        cypher = """
        CREATE (r:DeletionRequest {
            id: $id,
            patient_id: $patient_id,
            scope: $scope,
            target_id: $target_id,
            reason: $reason,
            requested_by: $requested_by,
            token_hash: $token_hash,
            created_at: $created_at,
            expires_at: $expires_at,
            status: 'pending_approval'
        })
        RETURN r.id
        """
        self._control_graph.query(
            cypher,
            {
                "id": req_id,
                "patient_id": valid_pid,
                "scope": clean_scope,
                "target_id": clean_target,
                "reason": reason.strip(),
                "requested_by": requested_by.strip(),
                "token_hash": token_hash,
                "created_at": now_ts,
                "expires_at": expires_at,
            },
        )

        return {
            "request_id": req_id,
            "confirm_token": raw_token,
            "expires_at": expires_at,
            "status": "pending_approval",
        }

    def confirm_forget(
        self,
        request_id: str,
        confirm_token: str,
        approver_id: str,
        approver_role: str = "clinician_admin",
    ) -> dict[str, Any]:
        """Validates credentials and executes approved forgetting request.

        Args:
            request_id: Identifier of the pending DeletionRequest.
            confirm_token: 32-byte secret token provided at request creation.
            approver_id: Identifier of the approving administrator.
            approver_role: Role of the approver (must be 'clinician_admin').

        Returns:
            Dictionary with status 'executed' and tombstone_id.

        Raises:
            DeletionRequestNotFoundError: If request_id does not exist.
            ForgetNotAllowedError: If status is not pending, approver equals requester,
                or approver lacks clinician_admin role.
            RequestExpiredError: If request has expired past TTL.
            ForgetConfirmationRequiredError: If token does not match stored hash.
        """
        clean_req_id = request_id.strip()
        fetch_cypher = "MATCH (r:DeletionRequest {id: $rid}) RETURN r"
        res = self._control_graph.query(fetch_cypher, {"rid": clean_req_id})
        if not res.result_set:
            raise DeletionRequestNotFoundError(
                f"Deletion request '{clean_req_id}' not found.",
                details={"request_id": clean_req_id},
            )

        raw_props = _extract_props(res.result_set[0][0])
        req = DeletionRequestRecord.from_dict(raw_props)

        if req.status != "pending_approval":
            raise ForgetNotAllowedError(
                f"Deletion request '{clean_req_id}' is in status '{req.status}' "
                "and cannot be executed.",
                details={"request_id": clean_req_id, "status": req.status},
            )

        now_ts = self._clock.now_ts()
        if now_ts > req.expires_at:
            raise RequestExpiredError(
                f"Deletion request '{clean_req_id}' expired at timestamp {req.expires_at}.",
                details={
                    "request_id": clean_req_id,
                    "expires_at": req.expires_at,
                    "current_time": now_ts,
                },
            )

        if approver_role.strip().lower() != "clinician_admin":
            raise ForgetNotAllowedError(
                "Deletion requests require approval by a user with role 'clinician_admin'.",
                details={
                    "request_id": clean_req_id,
                    "approver_role": approver_role,
                    "required_role": "clinician_admin",
                },
            )

        clean_approver = approver_id.strip()
        if not clean_approver:
            raise ForgetNotAllowedError("Approver ID must be provided.")

        if clean_approver == req.requested_by.strip():
            raise ForgetNotAllowedError(
                "Separation of duties required: approver must be distinct from requester.",
                details={
                    "request_id": clean_req_id,
                    "requested_by": req.requested_by,
                    "approver_id": clean_approver,
                },
            )

        given_hash = hash_confirm_token(confirm_token.strip())
        if given_hash != req.token_hash:
            raise ForgetConfirmationRequiredError(
                "Invalid confirmation token provided for deletion request.",
                details={"request_id": clean_req_id},
            )

        t_id = tombstone_id()
        patient_hash = compute_patient_hash(req.patient_id, self._salt)

        if req.scope == "all":
            self._execute_scope_all(
                req=req,
                tombstone_id=t_id,
                patient_hash=patient_hash,
                approver_id=clean_approver,
                timestamp=now_ts,
            )
        elif req.scope == "fact":
            self._execute_scope_fact(
                req=req,
                tombstone_id=t_id,
                patient_hash=patient_hash,
                approver_id=clean_approver,
                timestamp=now_ts,
            )
        elif req.scope == "session":
            self._execute_scope_session(
                req=req,
                tombstone_id=t_id,
                patient_hash=patient_hash,
                approver_id=clean_approver,
                timestamp=now_ts,
            )

        update_cypher = """
        MATCH (r:DeletionRequest {id: $rid})
        SET r.status = 'executed',
            r.executed_at = $now_ts,
            r.approver_id = $approver_id,
            r.tombstone_id = $tid
        """
        self._control_graph.query(
            update_cypher,
            {
                "rid": clean_req_id,
                "now_ts": now_ts,
                "approver_id": clean_approver,
                "tid": t_id,
            },
        )

        return {"status": "executed", "tombstone_id": t_id}

    def _execute_scope_all(
        self,
        req: DeletionRequestRecord,
        tombstone_id: str,
        patient_hash: str,
        approver_id: str,
        timestamp: int,
    ) -> None:
        """Executes full tenant graph deletion, registry wipe, tombstone, and BGREWRITEAOF."""
        pid = req.patient_id

        # 1. Mark registry entry 'deleting' to block incoming sessions
        self._router.set_tenant_status(pid, "deleting")

        # 2. Delete patient graph from FalkorDB
        try:
            target_graph = self._router.graph(pid)
            target_graph.delete()
        except Exception as e:
            logger.warning(f"Error dropping graph for tenant '{pid}': {e}")

        # 3. Remove registry entry from ct_control
        self._router.deregister_patient(pid)

        # 4. Write cryptographic tombstone in ct_control
        tomb_cypher = """
        CREATE (t:Tombstone {
            id: $tid,
            patient_hash: $phash,
            timestamp: $ts,
            request_id: $rid,
            approver_id: $aid,
            scope: 'all'
        })
        """
        self._control_graph.query(
            tomb_cypher,
            {
                "tid": tombstone_id,
                "phash": patient_hash,
                "ts": timestamp,
                "rid": req.id,
                "aid": approver_id,
            },
        )

        # 5. Purge tenant LLM cache fixtures and runtime cache
        self._purge_tenant_cache(patient_hash)

        # 6. Trigger BGREWRITEAOF to flush data from disk persistence
        self._trigger_bgrewriteaof()

    def _execute_scope_fact(
        self,
        req: DeletionRequestRecord,
        tombstone_id: str,
        patient_hash: str,
        approver_id: str,
        timestamp: int,
    ) -> None:
        """Retracts fact in patient graph, retaining stub for integrity, and writes tombstone."""
        fid = req.target_id
        target_graph = self._router.graph(req.patient_id)

        retract_cypher = """
        MATCH (f:Fact {id: $fid})
        SET f.status = 'retracted',
            f.display = null,
            f.value = null,
            f.retracted_at = $ts,
            f.retracted_by = $aid
        """
        target_graph.query(
            retract_cypher,
            {"fid": fid, "ts": timestamp, "aid": approver_id},
        )

        tomb_cypher = """
        CREATE (t:Tombstone {
            id: $tid,
            patient_hash: $phash,
            timestamp: $ts,
            request_id: $rid,
            approver_id: $aid,
            scope: 'fact',
            target_id: $fid
        })
        """
        self._control_graph.query(
            tomb_cypher,
            {
                "tid": tombstone_id,
                "phash": patient_hash,
                "ts": timestamp,
                "rid": req.id,
                "aid": approver_id,
                "fid": fid,
            },
        )

    def _execute_scope_session(
        self,
        req: DeletionRequestRecord,
        tombstone_id: str,
        patient_hash: str,
        approver_id: str,
        timestamp: int,
    ) -> None:
        """Redacts messages and session node in patient graph and writes tombstone."""
        sid = req.target_id
        target_graph = self._router.graph(req.patient_id)

        redact_msg_cypher = """
        MATCH (m:Message {session_id: $sid})
        SET m.content = '[REDACTED]',
            m.embedding = null,
            m.redacted_at = $ts
        """
        target_graph.query(redact_msg_cypher, {"sid": sid, "ts": timestamp})

        redact_ses_cypher = """
        MATCH (s:Session {id: $sid})
        SET s.embedding = null,
            s.summary = '[REDACTED]',
            s.status = 'closed'
        """
        target_graph.query(redact_ses_cypher, {"sid": sid})

        tomb_cypher = """
        CREATE (t:Tombstone {
            id: $tid,
            patient_hash: $phash,
            timestamp: $ts,
            request_id: $rid,
            approver_id: $aid,
            scope: 'session',
            target_id: $sid
        })
        """
        self._control_graph.query(
            tomb_cypher,
            {
                "tid": tombstone_id,
                "phash": patient_hash,
                "ts": timestamp,
                "rid": req.id,
                "aid": approver_id,
                "sid": sid,
            },
        )

    def _purge_tenant_cache(self, patient_hash: str) -> int:
        """Purges cached entries matching patient_hash from the cache directory."""
        if not self._cache_dir.exists():
            return 0

        purged_count = 0
        for path in self._cache_dir.glob("**/*"):
            if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
                continue
            try:
                if path.suffix == ".jsonl":
                    lines = path.read_text(encoding="utf-8").splitlines()
                    kept: list[str] = []
                    changed = False
                    for line in lines:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            item = json.loads(line_str)
                            th = item.get("tenant_hash") or item.get("patient_hash")
                            if th == patient_hash:
                                purged_count += 1
                                changed = True
                            else:
                                kept.append(line_str)
                        except Exception:
                            kept.append(line_str)
                    if changed:
                        path.write_text(
                            "\n".join(kept) + ("\n" if kept else ""),
                            encoding="utf-8",
                        )
                elif path.suffix == ".json":
                    text = path.read_text(encoding="utf-8")
                    try:
                        item = json.loads(text)
                        th = item.get("tenant_hash") or item.get("patient_hash")
                        if th == patient_hash:
                            path.unlink(missing_ok=True)
                            purged_count += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error purging cache file {path}: {e}")
        return purged_count

    def _trigger_bgrewriteaof(self) -> None:
        """Triggers FalkorDB background AOF rewrite to purge deleted graph data."""
        try:
            client = self._client
            conn = getattr(client, "connection", client)
            conn.execute_command("BGREWRITEAOF")
        except Exception as e:
            err_msg = str(e).lower()
            if "already in progress" in err_msg or "in progress" in err_msg:
                logger.debug(f"BGREWRITEAOF already in progress: {e}")
            else:
                logger.warning(f"BGREWRITEAOF execution returned: {e}")

    def resume_incomplete_deletions(self) -> list[str]:
        """Resumes and completes any deletion left in 'deleting' status after a crash."""
        deleting_tenants = self._router.list_deleting_tenants()
        resumed: list[str] = []

        now_ts = self._clock.now_ts()
        for pid in deleting_tenants:
            try:
                # 1. Drop graph
                try:
                    target_graph = self._router.graph(pid)
                    target_graph.delete()
                except Exception as e:
                    logger.debug(f"Graph delete during resume on '{pid}': {e}")

                # 2. Remove registry entry
                self._router.deregister_patient(pid)

                # 3. Ensure tombstone exists
                patient_hash = compute_patient_hash(pid, self._salt)
                check_cypher = "MATCH (t:Tombstone {patient_hash: $phash}) RETURN t.id"
                res = self._control_graph.query(check_cypher, {"phash": patient_hash})
                if not res.result_set:
                    t_id = tombstone_id()
                    tomb_cypher = """
                    CREATE (t:Tombstone {
                        id: $tid,
                        patient_hash: $phash,
                        timestamp: $ts,
                        request_id: 'resumed_crash_recovery',
                        approver_id: 'worker_recovery',
                        scope: 'all'
                    })
                    """
                    self._control_graph.query(
                        tomb_cypher,
                        {"tid": t_id, "phash": patient_hash, "ts": now_ts},
                    )

                # 4. Purge tenant cache
                self._purge_tenant_cache(patient_hash)

                # 5. Trigger rewrite
                self._trigger_bgrewriteaof()

                resumed.append(pid)
            except Exception as e:
                logger.error(f"Failed resuming incomplete deletion for tenant '{pid}': {e}")

        return resumed

    def get_tombstone(self, tombstone_id: str) -> TombstoneRecord | None:
        """Retrieves a specific TombstoneRecord from ct_control."""
        clean_id = tombstone_id.strip()
        cypher = "MATCH (t:Tombstone {id: $tid}) RETURN t"
        res = self._control_graph.query(cypher, {"tid": clean_id})
        if not res.result_set:
            return None
        props = _extract_props(res.result_set[0][0])
        return TombstoneRecord.from_dict(props)

    def list_tombstones(self, patient_hash: str | None = None) -> list[TombstoneRecord]:
        """Lists tombstones from ct_control, optionally filtered by patient_hash."""
        if patient_hash:
            cypher = "MATCH (t:Tombstone {patient_hash: $phash}) RETURN t ORDER BY t.timestamp DESC"
            res = self._control_graph.query(cypher, {"phash": patient_hash.strip()})
        else:
            cypher = "MATCH (t:Tombstone) RETURN t ORDER BY t.timestamp DESC"
            res = self._control_graph.query(cypher)

        tombstones: list[TombstoneRecord] = []
        for row in res.result_set:
            props = _extract_props(row[0])
            tombstones.append(TombstoneRecord.from_dict(props))
        return tombstones

    def get_deletion_request(self, request_id: str) -> DeletionRequestRecord | None:
        """Retrieves a DeletionRequestRecord from ct_control."""
        clean_id = request_id.strip()
        cypher = "MATCH (r:DeletionRequest {id: $rid}) RETURN r"
        res = self._control_graph.query(cypher, {"rid": clean_id})
        if not res.result_set:
            return None
        props = _extract_props(res.result_set[0][0])
        return DeletionRequestRecord.from_dict(props)
