"""Unit tests for ULID generation and domain-prefixed identifier helpers."""

from datetime import datetime, timezone

from carethread.util.id import (
    action_id,
    answer_id,
    audit_id,
    candidate_id,
    consolidation_id,
    deletion_request_id,
    episode_id,
    fact_id,
    flag_id,
    generate_ulid,
    handoff_id,
    is_valid_prefixed_ulid,
    message_id,
    note_id,
    parse_ulid_timestamp,
    playbook_id,
    playbook_step_id,
    session_id,
    tombstone_id,
    ulid_prefixed,
)


def test_generate_ulid_format() -> None:
    u1 = generate_ulid(lowercase=True)
    assert len(u1) == 26
    assert u1 == u1.lower()
    assert u1.isalnum()

    u2 = generate_ulid(lowercase=False)
    assert len(u2) == 26
    assert u2 == u2.upper()
    assert u2.isalnum()


def test_ulid_prefixed() -> None:
    sid = ulid_prefixed("ses")
    assert sid.startswith("ses_")
    assert len(sid) == 30  # 4 prefix chars + 26 ulid
    assert is_valid_prefixed_ulid(sid, expected_prefix="ses")

    mid = ulid_prefixed("msg")
    assert mid.startswith("msg_")
    assert is_valid_prefixed_ulid(mid, expected_prefix="msg")


def test_domain_helpers() -> None:
    assert is_valid_prefixed_ulid(session_id(), "ses")
    assert is_valid_prefixed_ulid(message_id(), "msg")
    assert is_valid_prefixed_ulid(episode_id(), "epi")
    assert is_valid_prefixed_ulid(fact_id(), "fct")
    assert is_valid_prefixed_ulid(note_id(), "not")
    assert is_valid_prefixed_ulid(handoff_id(), "hnd")
    assert is_valid_prefixed_ulid(action_id(), "act")
    assert is_valid_prefixed_ulid(flag_id(), "flg")
    assert is_valid_prefixed_ulid(answer_id(), "ans")
    assert is_valid_prefixed_ulid(audit_id(), "aud")
    assert is_valid_prefixed_ulid(playbook_id(), "pb")
    assert is_valid_prefixed_ulid(playbook_step_id(), "pbs")
    assert is_valid_prefixed_ulid(candidate_id(), "can")
    assert is_valid_prefixed_ulid(consolidation_id(), "cns")
    assert is_valid_prefixed_ulid(deletion_request_id(), "del")
    assert is_valid_prefixed_ulid(tombstone_id(), "tmb")


def test_parse_ulid_timestamp() -> None:
    before = datetime.now(timezone.utc).timestamp()
    uid = generate_ulid()
    after = datetime.now(timezone.utc).timestamp()

    parsed_dt = parse_ulid_timestamp(uid)
    assert parsed_dt.tzinfo == timezone.utc
    parsed_ts = parsed_dt.timestamp()

    # ULID timestamp precision is milliseconds, should fall within 2-second test window
    assert before - 1.0 <= parsed_ts <= after + 1.0

    # Also works for prefixed IDs
    prefixed = f"ses_{uid}"
    assert parse_ulid_timestamp(prefixed) == parsed_dt
