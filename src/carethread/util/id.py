"""ULID-based identifier generators with standardized domain prefixes.

Standardized domain-prefixed identifier generation:
- Standard 26-character Crockford Base32 ULIDs
- Lowercase domain-prefixed identifiers (e.g. ses_<ulid>, msg_<ulid>, epi_<ulid>)
- Deterministic extraction of UTC timestamps from ULIDs
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import ulid

# Pattern for validating prefixed ULIDs: <prefix>_<26 base32 chars>
_PREFIXED_ULID_REGEX = re.compile(r"^[a-z0-9]+_[0-9a-hj-km-np-za-km-z]{26}$", re.IGNORECASE)


def generate_ulid(lowercase: bool = True) -> str:
    """Generate a standard 26-character Crockford Base32 ULID.

    Args:
        lowercase: When True (default), returns lowercase representation.

    Returns:
        26-character ULID string.
    """
    raw = ulid.new().str
    return raw.lower() if lowercase else raw


def ulid_prefixed(prefix: str, lowercase: bool = True) -> str:
    """Generate a domain-prefixed ULID string.

    Example:
        ulid_prefixed('ses') -> 'ses_01m508wp7tvxrgy3zz2rb5nghf'

    Args:
        prefix: Lowercase entity prefix without underscore (e.g., 'ses', 'msg', 'epi').
        lowercase: When True (default), lowercase suffix is returned.

    Returns:
        Formatted string: f"{prefix}_{ulid}".
    """
    clean_prefix = prefix.strip().rstrip("_").lower()
    return f"{clean_prefix}_{generate_ulid(lowercase=lowercase)}"


def parse_ulid_timestamp(identifier: str) -> datetime:
    """Extracts UTC datetime from a ULID or prefixed ULID string.

    Args:
        identifier: Either a 26-char ULID or a prefixed identifier (e.g. 'ses_01...').

    Returns:
        Timezone-aware datetime in UTC.
    """
    raw_ulid = identifier.split("_")[-1].strip().upper()
    parsed = ulid.from_str(raw_ulid)
    epoch_ms = parsed.timestamp().int
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)


def is_valid_prefixed_ulid(identifier: str, expected_prefix: str | None = None) -> bool:
    """Checks whether an identifier is a valid domain-prefixed ULID."""
    if not isinstance(identifier, str):
        return False
    if not _PREFIXED_ULID_REGEX.match(identifier):
        return False
    if expected_prefix is not None:
        prefix = identifier.split("_")[0]
        return prefix.lower() == expected_prefix.lower()
    return True


# Domain identifier helpers
def session_id() -> str:
    """Generate a new Session identifier (ses_<ulid>)."""
    return ulid_prefixed("ses")


def message_id() -> str:
    """Generate a new Message identifier (msg_<ulid>)."""
    return ulid_prefixed("msg")


def episode_id() -> str:
    """Generate a new Episode identifier (epi_<ulid>)."""
    return ulid_prefixed("epi")


def fact_id() -> str:
    """Generate a new Fact identifier (fct_<ulid>)."""
    return ulid_prefixed("fct")


def note_id() -> str:
    """Generate a new Note identifier (not_<ulid>)."""
    return ulid_prefixed("not")


def handoff_id() -> str:
    """Generate a new Handoff identifier (hnd_<ulid>)."""
    return ulid_prefixed("hnd")


def action_id() -> str:
    """Generate a new Action identifier (act_<ulid>)."""
    return ulid_prefixed("act")


def flag_id() -> str:
    """Generate a new Flag identifier (flg_<ulid>)."""
    return ulid_prefixed("flg")


def answer_id() -> str:
    """Generate a new Answer identifier (ans_<ulid>)."""
    return ulid_prefixed("ans")


def audit_id() -> str:
    """Generate a new AuditEvent identifier (aud_<ulid>)."""
    return ulid_prefixed("aud")


def playbook_id() -> str:
    """Generate a new Playbook identifier (pb_<ulid>)."""
    return ulid_prefixed("pb")


def playbook_step_id() -> str:
    """Generate a new PlaybookStep identifier (pbs_<ulid>)."""
    return ulid_prefixed("pbs")


def candidate_id() -> str:
    """Generate a new FactCandidate identifier (can_<ulid>)."""
    return ulid_prefixed("can")


def consolidation_id() -> str:
    """Generate a new ConsolidationRun identifier (cns_<ulid>)."""
    return ulid_prefixed("cns")


def deletion_request_id() -> str:
    """Generate a new DeletionRequest identifier (del_<ulid>)."""
    return ulid_prefixed("del")


def tombstone_id() -> str:
    """Generate a new Tombstone identifier (tmb_<ulid>)."""
    return ulid_prefixed("tmb")
