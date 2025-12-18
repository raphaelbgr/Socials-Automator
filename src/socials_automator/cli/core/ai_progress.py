"""AI progress display for CLI.

Shows real-time AI call progress with:
- Provider and model being used
- Task being performed
- Duration of each call
- Failed providers (if any)

Usage:
    from socials_automator.cli.core.ai_progress import AIProgressDisplay

    display = AIProgressDisplay(console)
    text_provider = TextProvider(event_callback=display.handle_event)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console


@dataclass
class AICallStats:
    """Statistics for AI calls during a session."""

    total_calls: int = 0
    total_duration: float = 0.0
    total_cost: float = 0.0
    successful_calls: int = 0
    failed_calls: int = 0
    providers_used: set[str] = field(default_factory=set)
    failed_providers: list[str] = field(default_factory=list)


class AIProgressDisplay:
    """Display AI call progress in CLI.

    Handles events from TextProvider and displays them in a consistent format.

    Event types handled:
    - text_call: AI call starting
    - text_response: AI call completed
    - text_skip: Provider skipped (offline, no API key, etc.)
    - text_error: AI call failed
    """

    def __init__(
        self,
        console: Console,
        verbose: bool = True,
        show_preview: bool = False,
    ):
        """Initialize progress display.

        Args:
            console: Rich console for output.
            verbose: If True, show detailed progress. If False, show minimal.
            show_preview: If True, show response previews.
        """
        self.console = console
        self.verbose = verbose
        self.show_preview = show_preview
        self.stats = AICallStats()
        self._current_task: Optional[str] = None
        self._call_start_time: Optional[float] = None
        self._header_shown = False

    def _show_header(self) -> None:
        """Show AI calls header once."""
        if not self._header_shown:
            self.console.print()
            self.console.print("[bold]>>> AI CALLS[/bold]")
            self._header_shown = True

    async def handle_event(self, event: dict[str, Any]) -> None:
        """Handle an AI event from TextProvider.

        Args:
            event: Event dict with type and data.
        """
        event_type = event.get("type", "")

        if event_type == "text_call":
            await self._handle_call_start(event)
        elif event_type == "text_response":
            await self._handle_response(event)
        elif event_type == "text_skip":
            await self._handle_skip(event)
        elif event_type == "text_error":
            await self._handle_error(event)

    async def _handle_call_start(self, event: dict[str, Any]) -> None:
        """Handle AI call starting."""
        self._show_header()

        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        task = event.get("task", "")
        failed = event.get("failed_providers", [])

        self._current_task = task
        self._call_start_time = time.time()

        # Show failed providers if any
        if failed and self.verbose:
            failed_str = ", ".join(failed)
            self.console.print(f"  [dim]Skipped: {failed_str}[/dim]")

        # Show call info
        task_str = f" ({task})" if task else ""
        model_short = self._shorten_model_name(model)

        if self.verbose:
            self.console.print(
                f"  [cyan][AI][/cyan] {provider}/{model_short}{task_str}...",
                end="",
            )

    async def _handle_response(self, event: dict[str, Any]) -> None:
        """Handle AI response received."""
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        duration = event.get("duration_seconds", 0.0)
        cost = event.get("cost_usd", 0.0)
        task = event.get("task", self._current_task)

        # Update stats
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.total_duration += duration
        self.stats.total_cost += cost
        self.stats.providers_used.add(provider)

        # Format duration
        if duration < 1:
            duration_str = f"{duration*1000:.0f}ms"
        else:
            duration_str = f"{duration:.1f}s"

        if self.verbose:
            # Complete the line with timing
            self.console.print(f" [green][OK][/green] {duration_str}")
        else:
            # Compact output
            model_short = self._shorten_model_name(model)
            task_str = f" ({task})" if task else ""
            self.console.print(
                f"  [green][OK][/green] {provider}/{model_short}{task_str} - {duration_str}"
            )

    async def _handle_skip(self, event: dict[str, Any]) -> None:
        """Handle provider skipped."""
        provider = event.get("provider", "unknown")
        reason = event.get("reason", "unknown")

        self.stats.failed_providers.append(f"{provider}({reason})")

        # Don't show individual skips in verbose mode (shown in batch)
        if not self.verbose:
            self.console.print(
                f"  [dim][SKIP][/dim] {provider}: {reason}"
            )

    async def _handle_error(self, event: dict[str, Any]) -> None:
        """Handle AI call error."""
        self._show_header()

        provider = event.get("provider", "unknown")
        error = event.get("error", "unknown error")
        task = event.get("task", self._current_task)

        self.stats.failed_calls += 1

        task_str = f" ({task})" if task else ""

        if self.verbose:
            # Complete the line with error
            self.console.print(f" [red][FAIL][/red]")
            self.console.print(f"        {error}")
        else:
            self.console.print(
                f"  [red][FAIL][/red] {provider}{task_str}: {error}"
            )

    def _shorten_model_name(self, model: str) -> str:
        """Shorten model name for display.

        Args:
            model: Full model name.

        Returns:
            Shortened model name.
        """
        # Common abbreviations
        replacements = {
            "llama-3.3-70b-versatile": "llama-3.3-70b",
            "llama-3.1-8b-instant": "llama-3.1-8b",
            "gemini-2.0-flash-exp": "gemini-2.0-flash",
            "gpt-4o-mini": "gpt-4o-mini",
            "local-model": "local",
        }

        for long, short in replacements.items():
            if long in model:
                return short

        # Truncate if too long
        if len(model) > 25:
            return model[:22] + "..."

        return model

    def show_summary(self) -> None:
        """Show summary of all AI calls."""
        if self.stats.total_calls == 0:
            return

        self.console.print()
        self.console.print("[bold]>>> AI SUMMARY[/bold]")

        # Providers used
        providers = ", ".join(sorted(self.stats.providers_used))
        self.console.print(f"  Providers: [cyan]{providers}[/cyan]")

        # Calls stats
        self.console.print(
            f"  Calls: [green]{self.stats.successful_calls}[/green] "
            f"successful, [red]{self.stats.failed_calls}[/red] failed"
        )

        # Duration
        if self.stats.total_duration < 60:
            duration_str = f"{self.stats.total_duration:.1f}s"
        else:
            mins = int(self.stats.total_duration // 60)
            secs = self.stats.total_duration % 60
            duration_str = f"{mins}m {secs:.0f}s"

        self.console.print(f"  Total time: [yellow]{duration_str}[/yellow]")

        # Cost (if any)
        if self.stats.total_cost > 0:
            self.console.print(f"  Est. cost: [yellow]${self.stats.total_cost:.4f}[/yellow]")

    def reset(self) -> None:
        """Reset statistics for a new session."""
        self.stats = AICallStats()
        self._current_task = None
        self._call_start_time = None
        self._header_shown = False


# Singleton for easy access
_progress_display: Optional[AIProgressDisplay] = None


def get_ai_progress_display(console: Console) -> AIProgressDisplay:
    """Get or create the global AI progress display.

    Args:
        console: Rich console for output.

    Returns:
        AIProgressDisplay instance.
    """
    global _progress_display
    if _progress_display is None:
        _progress_display = AIProgressDisplay(console)
    return _progress_display


def reset_ai_progress_display() -> None:
    """Reset the global AI progress display."""
    global _progress_display
    if _progress_display is not None:
        _progress_display.reset()
    _progress_display = None
