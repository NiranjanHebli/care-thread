"""Standard I/O transport entrypoint for CareThread MCP server.

Enables stdio-based integration with external agent frameworks and opencode.
Run via:
    python -m carethread.mcp.stdio
"""

from __future__ import annotations

import asyncio
import logging
import sys

from carethread.mcp.server import create_mcp_server

logger = logging.getLogger(__name__)


async def run_stdio_server() -> None:
    """Runs the CareThread MCP server over standard input and output streams."""
    server = create_mcp_server()
    await server.run_stdio_async()


def main() -> None:
    """Synchronous command line entrypoint for stdio MCP execution."""
    try:
        asyncio.run(run_stdio_server())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in CareThread stdio server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
