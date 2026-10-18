"""CareThread Model Context Protocol (MCP) server, dispatcher, and authentication."""

from carethread.mcp.auth import (
    CallerIdentity,
    TokenStore,
)
from carethread.mcp.dispatcher import ToolDispatcher
from carethread.mcp.errors import (
    error_envelope,
    success_envelope,
)
from carethread.mcp.server import (
    FastMCP,
    create_mcp_server,
)

__all__ = [
    "CallerIdentity",
    "FastMCP",
    "TokenStore",
    "ToolDispatcher",
    "create_mcp_server",
    "error_envelope",
    "success_envelope",
]
