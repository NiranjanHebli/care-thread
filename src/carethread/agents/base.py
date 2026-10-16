"""Base class for CareThread clinical agents interacting with MCP ToolDispatcher."""

from __future__ import annotations

import logging
from typing import Any

from carethread.agents.llm import LLMClient
from carethread.errors import CareThreadError
from carethread.mcp.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


class BaseClinicalAgent:
    """Foundational agent base providing authenticated tool invocation and LLM reasoning."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        token: str,
        dispatcher: ToolDispatcher,
        llm: LLMClient | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.token = token
        self.dispatcher = dispatcher
        self.llm = llm or LLMClient()

    def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatches an authenticated tool request through the CareThread ToolDispatcher.

        Raises:
            CareThreadError or RuntimeError if the tool invocation fails.
        """
        envelope = self.dispatcher.dispatch(
            tool_name=tool_name,
            args=args,
            token=self.token,
        )
        if not envelope.get("ok"):
            err = envelope.get("error", {})
            err_msg = err.get("message", "Tool invocation failed")
            err_code = err.get("code", "INTERNAL_ERROR")
            err_details = err.get("details")
            logger.error(
                f"Agent [{self.agent_id}:{self.role}] tool '{tool_name}' failed: {err_msg}"
            )
            raise CareThreadError(err_msg, code=err_code, details=err_details)

        return envelope.get("data", {})
