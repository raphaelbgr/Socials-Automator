"""Clean CLI display system for video pipeline.

Provides:
- Clean console output with only important milestones
- Detailed file logging for debugging
- No duplicate messages
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text


class LogLevel(Enum):
    """Log levels for CLI display."""
    DEBUG = "debug"      # Only to file
    DETAIL = "detail"    # Only to file (verbose progress)
    INFO = "info"        # Console + file
    STEP = "step"        # Console + file (step headers)
    SUCCESS = "success"  # Console + file
    WARNING = "warning"  # Console + file
    ERROR = "error"      # Console + file


class PipelineDisplay:
    """Handles all CLI display for video pipeline.

    Console output: Clean, milestone-focused
    File output: Detailed for debugging
    """

    ICONS = {
        LogLevel.DEBUG: "[.]",
        LogLevel.DETAIL: "[.]",
        LogLevel.INFO: " ",
        LogLevel.STEP: "[>]",
        LogLevel.SUCCESS: "[OK]",
        LogLevel.WARNING: "[!]",
        LogLevel.ERROR: "[X]",
    }

    COLORS = {
        LogLevel.DEBUG: "dim",
        LogLevel.DETAIL: "dim",
        LogLevel.INFO: "white",
        LogLevel.STEP: "cyan",
        LogLevel.SUCCESS: "green",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "red",
    }

    def __init__(
        self,
        console: Optional[Console] = None,
        show_timestamps: bool = True,
        verbose: bool = False,
    ):
        """Initialize pipeline display.

        Args:
            console: Rich console instance.
            show_timestamps: Show timestamps on each line.
            verbose: Show detailed progress (usually only in file).
        """
        self.console = console or Console()
        self.show_timestamps = show_timestamps
        self.verbose = verbose
        self._current_step: Optional[str] = None
        self._step_number: int = 0
        self._total_steps: int = 0
        self._start_time: Optional[datetime] = None
        self._file_logger = self._setup_file_logging()

    def _setup_file_logging(self) -> logging.Logger:
        """Setup file-only logging for detailed logs."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger("video.pipeline.file")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # Remove any existing handlers
        logger.propagate = False  # Don't propagate to parent loggers

        file_handler = logging.FileHandler(
            log_dir / "video_pipeline.log",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(file_handler)

        return logger

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _log_to_file(self, level: LogLevel, message: str, step_name: Optional[str] = None) -> None:
        """Log message to file only."""
        msg = f"{step_name}: {message}" if step_name else message
        if level == LogLevel.ERROR:
            self._file_logger.error(msg)
        elif level == LogLevel.WARNING:
            self._file_logger.warning(msg)
        elif level == LogLevel.DEBUG or level == LogLevel.DETAIL:
            self._file_logger.debug(msg)
        else:
            self._file_logger.info(msg)

    def _print_to_console(
        self,
        level: LogLevel,
        message: str,
        step_name: Optional[str] = None,
    ) -> None:
        """Print formatted message to console."""
        parts = []

        if self.show_timestamps:
            parts.append(f"[dim]{self._timestamp()}[/dim] ")

        icon = self.ICONS.get(level, " ")
        color = self.COLORS.get(level, "white")

        if icon.strip():
            parts.append(f"[{color}]{icon}[/{color}] ")

        if step_name:
            parts.append(f"[bold cyan]{step_name}:[/bold cyan] ")

        parts.append(f"[{color}]{message}[/{color}]")

        self.console.print(Text.from_markup("".join(parts)))

    def log(
        self,
        level: LogLevel,
        message: str,
        step_name: Optional[str] = None,
    ) -> None:
        """Log a message.

        DEBUG/DETAIL: File only (unless verbose)
        Others: Console + File
        """
        # Always log to file
        self._log_to_file(level, message, step_name)

        # Console output based on level
        if level in (LogLevel.DEBUG, LogLevel.DETAIL):
            if self.verbose:
                self._print_to_console(level, message, step_name)
        else:
            self._print_to_console(level, message, step_name)

    def debug(self, message: str, step_name: Optional[str] = None) -> None:
        """Log debug message (file only unless verbose)."""
        self.log(LogLevel.DEBUG, message, step_name)

    def detail(self, message: str, step_name: Optional[str] = None) -> None:
        """Log detailed progress (file only unless verbose)."""
        self.log(LogLevel.DETAIL, message, step_name)

    def info(self, message: str, step_name: Optional[str] = None) -> None:
        """Log info message (console + file)."""
        self.log(LogLevel.INFO, message, step_name)

    def step(self, message: str, step_name: Optional[str] = None) -> None:
        """Log step progress (console + file)."""
        self.log(LogLevel.STEP, message, step_name)

    def success(self, message: str, step_name: Optional[str] = None) -> None:
        """Log success message (console + file)."""
        self.log(LogLevel.SUCCESS, message, step_name)

    def warning(self, message: str, step_name: Optional[str] = None) -> None:
        """Log warning message (console + file)."""
        self.log(LogLevel.WARNING, message, step_name)

    def error(self, message: str, step_name: Optional[str] = None) -> None:
        """Log error message (console + file)."""
        self.log(LogLevel.ERROR, message, step_name)

    def start_pipeline(self, profile_name: str, total_steps: int = 9) -> None:
        """Display pipeline start banner."""
        self._start_time = datetime.now()
        self._step_number = 0
        self._total_steps = total_steps

        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Video Reel Generation[/bold cyan]\n"
            f"Profile: [yellow]{profile_name}[/yellow]\n"
            f"Started: [dim]{self._start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
        ))

    def start_step(self, step_name: str, description: str) -> None:
        """Mark start of a pipeline step."""
        self._step_number += 1
        self._current_step = step_name

        self.console.print()
        self.console.print(f"[dim]{'─' * 60}[/dim]")
        self.console.print(
            f"[bold white]Step {self._step_number}/{self._total_steps}:[/bold white] "
            f"[bold cyan]{step_name}[/bold cyan]"
        )
        self.console.print(f"[dim]{description}[/dim]")
        self.console.print(f"[dim]{'─' * 60}[/dim]")

        self._log_to_file(LogLevel.INFO, f"=== Step {self._step_number}: {step_name} - {description} ===")

    def end_pipeline(self, output_path: Optional[Path] = None, success: bool = True) -> None:
        """Display pipeline completion banner."""
        end_time = datetime.now()
        duration = end_time - self._start_time if self._start_time else None
        duration_str = str(duration).split(".")[0] if duration else "unknown"

        self.console.print()

        if success:
            content = f"[bold green]Video Generated Successfully![/bold green]\n\n"
            content += f"Duration: {duration_str}\n"
            if output_path:
                content += f"Output: [cyan]{output_path}[/cyan]"

            self.console.print(Panel(content, border_style="green", title="[green]Complete[/green]"))
            self._log_to_file(LogLevel.SUCCESS, f"=== COMPLETE: Video generated in {duration_str} -> {output_path} ===")
        else:
            self.console.print(Panel(
                f"[bold red]Pipeline Failed[/bold red]\n\n"
                f"Duration: {duration_str}\n"
                f"[dim]Check logs/video_pipeline.log for details[/dim]",
                border_style="red",
                title="[red]Error[/red]",
            ))
            self._log_to_file(LogLevel.ERROR, f"=== FAILED: Pipeline failed after {duration_str} ===")

    def show_topic(self, topic: str, pillar: str) -> None:
        """Display selected topic."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]{topic}[/bold]\n"
            f"[dim]Pillar: {pillar}[/dim]",
            title="[cyan]Topic[/cyan]",
            border_style="cyan",
        ))

    def show_script(self, title: str, segments: int, duration: float) -> None:
        """Display script summary."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]{title}[/bold]\n"
            f"Segments: {segments} | Duration: {duration:.0f}s",
            title="[cyan]Script[/cyan]",
            border_style="cyan",
        ))

    def show_cache_stats(self, hits: int, misses: int) -> None:
        """Display cache statistics."""
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        self.info(f"Cache: {hits} hits, {misses} downloads ({hit_rate:.0f}% cached)")

    # =========================================================================
    # AI Progress Display
    # =========================================================================

    def _init_ai_tracking(self) -> None:
        """Initialize AI call tracking."""
        if not hasattr(self, "_ai_calls"):
            self._ai_calls: int = 0
            self._ai_duration: float = 0.0
            self._ai_cost: float = 0.0
            self._ai_providers: set[str] = set()
            self._ai_failed: list[str] = []
            self._ai_header_shown: bool = False

    async def ai_event_callback(self, event: dict) -> None:
        """Handle AI events from TextProvider.

        This is passed to TextProvider as the event_callback parameter.

        Args:
            event: Event dict with type and data.
        """
        import time as time_module  # Avoid shadowing
        self._init_ai_tracking()

        event_type = event.get("type", "")

        if event_type == "text_call":
            await self._handle_ai_call(event)
        elif event_type == "text_response":
            await self._handle_ai_response(event)
        elif event_type == "text_skip":
            await self._handle_ai_skip(event)
        elif event_type == "text_error":
            await self._handle_ai_error(event)

    async def _handle_ai_call(self, event: dict) -> None:
        """Handle AI call starting."""
        if not self._ai_header_shown:
            self.console.print()
            self.console.print("[bold]>>> AI CALLS[/bold]")
            self._ai_header_shown = True

        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        task = event.get("task", "")
        failed = event.get("failed_providers", [])

        # Show skipped providers (if any)
        if failed:
            self._ai_failed.extend(failed)
            failed_str = ", ".join(failed)
            self.console.print(f"  [dim]Skipped: {failed_str}[/dim]")

        # Show call info
        task_str = f" ({task})" if task else ""
        model_short = self._shorten_model_name(model)

        self.console.print(
            f"  [cyan][AI][/cyan] {provider}/{model_short}{task_str}...",
            end="",
        )
        self._log_to_file(LogLevel.DEBUG, f"AI call: {provider}/{model} - {task}")

    async def _handle_ai_response(self, event: dict) -> None:
        """Handle AI response received."""
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        duration = event.get("duration_seconds", 0.0)
        cost = event.get("cost_usd", 0.0)

        # Update stats
        self._ai_calls += 1
        self._ai_duration += duration
        self._ai_cost += cost
        self._ai_providers.add(provider)

        # Format duration
        if duration < 1:
            duration_str = f"{duration*1000:.0f}ms"
        else:
            duration_str = f"{duration:.1f}s"

        # Complete the line
        self.console.print(f" [green][OK][/green] {duration_str}")
        self._log_to_file(
            LogLevel.DEBUG,
            f"AI response: {provider}/{model} - {duration:.2f}s, ${cost:.4f}"
        )

    async def _handle_ai_skip(self, event: dict) -> None:
        """Handle provider skipped."""
        provider = event.get("provider", "unknown")
        reason = event.get("reason", "unknown")
        self._ai_failed.append(f"{provider}({reason})")

    async def _handle_ai_error(self, event: dict) -> None:
        """Handle AI call error."""
        provider = event.get("provider", "unknown")
        error = event.get("error", "unknown error")

        self.console.print(f" [red][FAIL][/red]")
        self.console.print(f"        [red]{error}[/red]")
        self._log_to_file(LogLevel.ERROR, f"AI error: {provider} - {error}")

    def _shorten_model_name(self, model: str) -> str:
        """Shorten model name for display."""
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
        if len(model) > 25:
            return model[:22] + "..."
        return model

    def show_ai_summary(self) -> None:
        """Show summary of AI calls at end of pipeline."""
        self._init_ai_tracking()

        if self._ai_calls == 0:
            return

        self.console.print()
        self.console.print("[bold]>>> AI SUMMARY[/bold]")

        providers = ", ".join(sorted(self._ai_providers))
        self.console.print(f"  Providers: [cyan]{providers}[/cyan]")
        self.console.print(f"  Calls: [green]{self._ai_calls}[/green]")

        if self._ai_duration < 60:
            duration_str = f"{self._ai_duration:.1f}s"
        else:
            mins = int(self._ai_duration // 60)
            secs = self._ai_duration % 60
            duration_str = f"{mins}m {secs:.0f}s"

        self.console.print(f"  Total AI time: [yellow]{duration_str}[/yellow]")

        if self._ai_cost > 0:
            self.console.print(f"  Est. cost: [yellow]${self._ai_cost:.4f}[/yellow]")


# Global display instance
_display: Optional[PipelineDisplay] = None


def get_display() -> PipelineDisplay:
    """Get global pipeline display instance."""
    global _display
    if _display is None:
        _display = PipelineDisplay()
    return _display


def set_display(display: PipelineDisplay) -> None:
    """Set global pipeline display instance."""
    global _display
    _display = display


def setup_display(
    show_timestamps: bool = True,
    verbose: bool = False,
) -> PipelineDisplay:
    """Setup and return a new pipeline display."""
    display = PipelineDisplay(
        show_timestamps=show_timestamps,
        verbose=verbose,
    )
    set_display(display)
    return display
