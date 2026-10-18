"""CareThread package root."""

from carethread.clock import Clock, FixedClock, SystemClock, get_clock, set_clock

__version__ = "0.1.0"

__all__ = ["__version__", "Clock", "FixedClock", "SystemClock", "get_clock", "set_clock"]
