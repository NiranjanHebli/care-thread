"""CareThread memory layer package."""

from carethread.memory.embeddings import (
    Embedder,
    FastEmbedder,
    HashEmbedder,
    get_embedder,
)
from carethread.memory.episodes import (
    EpisodeManager,
    EpisodeRecord,
)
from carethread.memory.facts import (
    FactManager,
    FactRecord,
    compute_slot,
)
from carethread.memory.recall import (
    NoteRecord,
    RecalledFact,
    RecalledGuideline,
    RecalledNote,
    RecallManager,
    RecallResult,
    RejectedCandidate,
)
from carethread.memory.safety import (
    SafetyCheckResult,
    SafetyFlagRecord,
    SafetyManager,
)
from carethread.memory.sessions import (
    MessageRecord,
    SessionManager,
    SessionRecord,
)

__all__ = [
    "Embedder",
    "FastEmbedder",
    "HashEmbedder",
    "get_embedder",
    "EpisodeManager",
    "EpisodeRecord",
    "FactManager",
    "FactRecord",
    "MessageRecord",
    "NoteRecord",
    "RecalledFact",
    "RecalledGuideline",
    "RecalledNote",
    "RecallManager",
    "RecallResult",
    "RejectedCandidate",
    "SafetyCheckResult",
    "SafetyFlagRecord",
    "SafetyManager",
    "SessionManager",
    "SessionRecord",
    "compute_slot",
]
