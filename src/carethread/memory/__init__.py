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
    "SessionManager",
    "SessionRecord",
    "compute_slot",
]
