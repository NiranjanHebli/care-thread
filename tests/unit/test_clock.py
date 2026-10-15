"""Unit tests for injectable Clock abstraction (SystemClock and FixedClock)."""

from datetime import datetime, timezone

from carethread.clock import FixedClock, SystemClock, get_clock, reset_clock, set_clock


def test_system_clock_returns_utc() -> None:
    clock = SystemClock()
    now = clock.now()
    assert now.tzinfo is not None
    assert now.tzinfo == timezone.utc
    assert isinstance(clock.now_ts(), int)
    assert clock.now_ts() > 1700000000
    assert "T" in clock.now_iso()


def test_fixed_clock_initialization() -> None:
    dt = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    clock = FixedClock(dt)
    assert clock.now() == dt
    assert clock.now_ts() == int(dt.timestamp())
    assert "2026-10-15T12:00:00" in clock.now_iso()


def test_fixed_clock_advance() -> None:
    clock = FixedClock("2026-10-15T12:00:00Z")
    start_ts = clock.now_ts()

    clock.advance(60)
    assert clock.now_ts() == start_ts + 60

    clock.advance(3600)
    assert clock.now_ts() == start_ts + 3660


def test_fixed_clock_set_time() -> None:
    clock = FixedClock(1700000000)
    assert clock.now_ts() == 1700000000

    clock.set_time(1700005000)
    assert clock.now_ts() == 1700005000


def test_global_clock_injection() -> None:
    reset_clock()
    default_clock = get_clock()
    assert isinstance(default_clock, SystemClock)

    fixed = FixedClock(1750000000)
    set_clock(fixed)
    assert get_clock().now_ts() == 1750000000

    reset_clock()
    assert isinstance(get_clock(), SystemClock)
