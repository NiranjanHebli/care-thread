"""Injectable Clock abstraction for deterministic time and reproducible clinical timelines.

Clock abstraction for clinical timeline operations:
- Protocol-based Clock interface
- SystemClock for production runtime
- FixedClock for deterministic testing and event replay
- Global clock accessors with test overrides
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for time providers."""

    def now(self) -> datetime:
        """Returns the current timezone-aware UTC datetime."""
        ...

    def now_ts(self) -> int:
        """Returns the current Unix timestamp in whole seconds."""
        ...

    def now_iso(self) -> str:
        """Returns the current time formatted as ISO 8601 UTC string."""
        ...


class SystemClock:
    """Standard system clock using real UTC time."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_ts(self) -> int:
        return int(self.now().timestamp())

    def now_iso(self) -> str:
        return self.now().isoformat()


class FixedClock:
    """Deterministic clock initialized with a fixed time for reproducible tests."""

    def __init__(self, initial: datetime | str | int | float | None = None) -> None:
        if initial is None:
            self._current = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
        elif isinstance(initial, datetime):
            if initial.tzinfo is None:
                self._current = initial.replace(tzinfo=timezone.utc)
            else:
                self._current = initial.astimezone(timezone.utc)
        elif isinstance(initial, (int, float)):
            self._current = datetime.fromtimestamp(initial, tz=timezone.utc)
        elif isinstance(initial, str):
            clean = initial.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(clean)
            if parsed.tzinfo is None:
                self._current = parsed.replace(tzinfo=timezone.utc)
            else:
                self._current = parsed.astimezone(timezone.utc)
        else:
            raise TypeError(f"Unsupported initial time type: {type(initial)}")

    def now(self) -> datetime:
        return self._current

    def now_ts(self) -> int:
        return int(self._current.timestamp())

    def now_iso(self) -> str:
        return self._current.isoformat()

    def set_time(self, new_time: datetime | str | int | float) -> None:
        """Update current time to a specific point."""
        temp = FixedClock(new_time)
        self._current = temp.now()

    def advance(self, seconds: float | int = 1) -> None:
        """Advance the fixed clock forward by the given number of seconds."""
        ts = self._current.timestamp() + float(seconds)
        self._current = datetime.fromtimestamp(ts, tz=timezone.utc)


_GLOBAL_CLOCK: Clock = SystemClock()


def get_clock() -> Clock:
    """Returns the globally configured Clock instance."""
    return _GLOBAL_CLOCK


def set_clock(clock: Clock) -> None:
    """Sets the globally configured Clock instance (useful for tests)."""
    global _GLOBAL_CLOCK
    _GLOBAL_CLOCK = clock


def reset_clock() -> None:
    """Resets the globally configured Clock instance back to SystemClock."""
    global _GLOBAL_CLOCK
    _GLOBAL_CLOCK = SystemClock()
