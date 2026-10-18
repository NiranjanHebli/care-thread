"""Pre-defined Cypher query loader and registry."""

from functools import lru_cache
from pathlib import Path

_QUERIES_DIR = Path(__file__).parent


@lru_cache(maxsize=64)
def get_query(name: str) -> str:
    """Loads a .cypher query by name (with or without .cypher extension).

    Raises FileNotFoundError if query does not exist.
    """
    file_name = name if name.endswith(".cypher") else f"{name}.cypher"
    path = _QUERIES_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Cypher query file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def list_queries() -> list[str]:
    """Returns sorted list of available query file names."""
    return sorted(p.stem for p in _QUERIES_DIR.glob("*.cypher"))


__all__ = ["get_query", "list_queries"]
