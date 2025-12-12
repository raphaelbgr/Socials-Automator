"""Tool executor for running AI-requested tools.

Executes tools based on AI function calls and reports results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Awaitable

from .definitions import ToolResult, AVAILABLE_TOOLS

# Logger for tool execution
_logger = logging.getLogger("ai_calls")


# Type for tool call callback (for CLI display)
ToolCallCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


class ToolExecutor:
    """Executes tools requested by AI agents.

    Handles web_search, news_search, and other tools that AI can call.
    Reports progress and results via callback for CLI display.
    """

    def __init__(
        self,
        callback: ToolCallCallback = None,
        post_id: str | None = None,
    ):
        """Initialize the tool executor.

        Args:
            callback: Optional callback for tool execution events.
            post_id: Post ID for logging context.
        """
        self.callback = callback
        self.post_id = post_id or "unknown"
        self._web_searcher = None
        self._total_tool_calls = 0

    async def _get_web_searcher(self):
        """Lazy-load the web searcher."""
        if self._web_searcher is None:
            from ..research import WebSearcher
            self._web_searcher = WebSearcher(timeout=10, max_results_per_query=5)
        return self._web_searcher

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit a tool event if callback is set."""
        if self.callback:
            await self.callback(event)

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments parsed from AI function call.

        Returns:
            ToolResult with execution outcome.
        """
        self._total_tool_calls += 1
        start_time = time.time()

        # Emit tool_call_start event
        await self._emit_event({
            "type": "tool_call_start",
            "tool_name": tool_name,
            "arguments": arguments,
            "call_number": self._total_tool_calls,
        })

        args_str = json.dumps(arguments, ensure_ascii=False)
        _logger.info(
            f"POST:{self.post_id} | TOOL_CALL_START | tool:{tool_name} | "
            f"call_num:{self._total_tool_calls} | args:{args_str[:200]}"
        )

        result: ToolResult

        try:
            if tool_name == "web_search":
                result = await self._execute_web_search(arguments)
            elif tool_name == "news_search":
                result = await self._execute_news_search(arguments)
            else:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_name}",
                )

        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )
            _logger.error(f"POST:{self.post_id} | TOOL_ERROR | tool:{tool_name} | error:{e}")

        # Add timing
        result.duration_ms = int((time.time() - start_time) * 1000)

        # Emit tool_call_complete event
        await self._emit_event({
            "type": "tool_call_complete",
            "tool_name": tool_name,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "metadata": result.metadata,
            "error": result.error,
        })

        metadata_str = json.dumps(result.metadata, ensure_ascii=False)
        result_preview = str(result.result)[:100] if result.result else ""
        _logger.info(
            f"POST:{self.post_id} | TOOL_CALL_END | tool:{tool_name} | "
            f"success:{result.success} | duration:{result.duration_ms}ms | "
            f"metadata:{metadata_str[:150]} | result_preview:{result_preview}..."
        )

        return result

    async def _execute_web_search(self, args: dict[str, Any]) -> ToolResult:
        """Execute web search tool.

        Args:
            args: {"queries": [...], "max_results": int}

        Returns:
            ToolResult with search results.
        """
        queries = args.get("queries", [])
        max_results = args.get("max_results", 5)

        if not queries:
            return ToolResult(
                tool_name="web_search",
                success=False,
                result=None,
                error="No queries provided",
            )

        try:
            searcher = await self._get_web_searcher()
            results = await searcher.parallel_search(queries, max_results=max_results)

            # Build result context
            context = results.to_context_string(max_sources=10)

            return ToolResult(
                tool_name="web_search",
                success=True,
                result=context,
                metadata={
                    "queries": queries,
                    "total_results": results.total_results,
                    "unique_sources": len(results.all_sources),
                    "domains": results.unique_domains[:5],
                    "duration_ms": results.duration_ms,
                },
            )

        except ImportError:
            return ToolResult(
                tool_name="web_search",
                success=False,
                result=None,
                error="Web search unavailable: ddgs not installed",
            )

    async def _execute_news_search(self, args: dict[str, Any]) -> ToolResult:
        """Execute news search tool.

        Args:
            args: {"queries": [...], "max_results": int}

        Returns:
            ToolResult with news results.
        """
        queries = args.get("queries", [])
        max_results = args.get("max_results", 3)

        if not queries:
            return ToolResult(
                tool_name="news_search",
                success=False,
                result=None,
                error="No queries provided",
            )

        try:
            searcher = await self._get_web_searcher()
            results = await searcher.parallel_news_search(queries, max_results=max_results)

            # Build result context
            context = results.to_context_string(max_sources=8)

            return ToolResult(
                tool_name="news_search",
                success=True,
                result=context,
                metadata={
                    "queries": queries,
                    "total_results": results.total_results,
                    "unique_sources": len(results.all_sources),
                    "domains": results.unique_domains[:5],
                    "duration_ms": results.duration_ms,
                },
            )

        except ImportError:
            return ToolResult(
                tool_name="news_search",
                success=False,
                result=None,
                error="News search unavailable: ddgs not installed",
            )

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolResult]:
        """Execute multiple tool calls (potentially in parallel).

        Args:
            tool_calls: List of tool calls from AI response.
                        Each has: {"id": str, "function": {"name": str, "arguments": str}}

        Returns:
            List of ToolResults in same order as input.
        """
        results = []

        # Execute sequentially for now (can be parallelized later)
        for call in tool_calls:
            func = call.get("function", {})
            tool_name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            # Parse arguments
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            result = await self.execute(tool_name, arguments)
            results.append(result)

        return results

    @property
    def total_tool_calls(self) -> int:
        """Get total number of tool calls executed."""
        return self._total_tool_calls
