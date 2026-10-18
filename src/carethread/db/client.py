"""FalkorDB client connection factory and management."""

import os
from typing import Any

from falkordb import FalkorDB

_CLIENT_INSTANCE: FalkorDB | None = None


def get_db_client(
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
    username: str | None = None,
    ssl: bool | None = None,
    **kwargs: Any,
) -> FalkorDB:
    """Creates or returns a FalkorDB client configured via params or environment variables.

    Reads defaults from:
    - FALKORDB_HOST (default: localhost)
    - FALKORDB_PORT (default: 6379)
    - FALKORDB_PASSWORD (default: None)
    - FALKORDB_USER / FALKORDB_USERNAME (default: None)
    - FALKORDB_SSL (default: False)
    """
    resolved_host = host or os.getenv("FALKORDB_HOST", "localhost")
    resolved_port = int(port or os.getenv("FALKORDB_PORT", "6379"))
    resolved_password = (
        password if password is not None else (os.getenv("FALKORDB_PASSWORD") or None)
    )
    resolved_username = (
        username
        if username is not None
        else (os.getenv("FALKORDB_USER") or os.getenv("FALKORDB_USERNAME") or None)
    )

    if ssl is not None:
        resolved_ssl = ssl
    else:
        ssl_env = os.getenv("FALKORDB_SSL", "").strip().lower()
        resolved_ssl = ssl_env in ("1", "true", "yes")

    client = FalkorDB(
        host=resolved_host,
        port=resolved_port,
        password=resolved_password,
        username=resolved_username,
        ssl=resolved_ssl,
        **kwargs,
    )
    return client


def check_health(client: FalkorDB | None = None) -> bool:
    """Verifies that FalkorDB responds to ping."""
    target_client = client or get_db_client()
    try:
        pong = target_client.connection.ping()
        return bool(pong is True or pong in (b"PONG", "PONG"))
    except Exception:
        return False
