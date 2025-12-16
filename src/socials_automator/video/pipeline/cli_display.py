"""Clean CLI display system for video pipeline.

Provides timestamped, formatted logging for all pipeline steps.
Uses Rich for beautiful terminal output with full transparency.
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


class LogLevel(Enum):
    """Log levels for CLI display."""
    DEBUG = "debug"
    INFO = "info"
    STEP = "step"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class PipelineDisplay:
    """Handles all CLI display for video pipeline.

    Provides:
    - Timestamped logging for every action
    - Step-by-step progress tracking
    - Clean, formatted output
    - Full transparency of what the program is doing
    """

    # Step icons (ASCII-compatible)
    ICONS = {
        LogLevel.DEBUG: "[.]",
        LogLevel.INFO: "[i]",
        LogLevel.STEP: "[>]",
        LogLevel.SUCCESS: "[OK]",
        LogLevel.WARNING: "[!]",
        LogLevel.ERROR: "[X]",
    }

    # Colors for each level
    COLORS = {
        LogLevel.DEBUG: "dim",
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
            console: Rich console instance (creates new if not provided).
            show_timestamps: Whether to show timestamps on each line.
            verbose: Whether to show debug messages.
        """
        self.console = console or Console()
        self.show_timestamps = show_timestamps
        self.verbose = verbose
        self._current_step: Optional[str] = None
        self._step_number: int = 0
        self._total_steps: int = 0
        self._start_time: Optional[datetime] = None

        # Setup file logging
        self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Setup file logging for detailed logs."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Create file handler
        log_file = log_dir / "video_pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Format with timestamps
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        # Add to root logger for video pipeline
        logger = logging.getLogger("video.pipeline")
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _format_line(
        self,
        level: LogLevel,
        message: str,
        step_name: Optional[str] = None,
    ) -> Text:
        """Format a log line with timestamp and styling.

        Args:
            level: Log level.
            message: Message to display.
            step_name: Optional step name prefix.

        Returns:
            Formatted Rich Text object.
        """
        parts = []

        # Timestamp
        if self.show_timestamps:
            parts.append(f"[dim]{self._timestamp()}[/dim] ")

        # Icon
        icon = self.ICONS.get(level, "[?]")
        color = self.COLORS.get(level, "white")
        parts.append(f"[{color}]{icon}[/{color}] ")

        # Step name
        if step_name:
            parts.append(f"[bold cyan]{step_name}[/bold cyan]: ")

        # Message
        parts.append(f"[{color}]{message}[/{color}]")

        return Text.from_markup("".join(parts))

    def log(
        self,
        level: LogLevel,
        message: str,
        step_name: Optional[str] = None,
    ) -> None:
        """Log a message with formatting.

        Args:
            level: Log level.
            message: Message to display.
            step_name: Optional step name prefix.
        """
        if level == LogLevel.DEBUG and not self.verbose:
            return

        line = self._format_line(level, message, step_name)
        self.console.print(line)

        # Also log to file
        logger = logging.getLogger("video.pipeline")
        log_msg = f"{step_name}: {message}" if step_name else message

        if level == LogLevel.ERROR:
            logger.error(log_msg)
        elif level == LogLevel.WARNING:
            logger.warning(log_msg)
        elif level == LogLevel.DEBUG:
            logger.debug(log_msg)
        else:
            logger.info(log_msg)

    def debug(self, message: str, step_name: Optional[str] = None) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, step_name)

    def info(self, message: str, step_name: Optional[str] = None) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, step_name)

    def step(self, message: str, step_name: Optional[str] = None) -> None:
        """Log step progress message."""
        self.log(LogLevel.STEP, message, step_name)

    def success(self, message: str, step_name: Optional[str] = None) -> None:
        """Log success message."""
        self.log(LogLevel.SUCCESS, message, step_name)

    def warning(self, message: str, step_name: Optional[str] = None) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, step_name)

    def error(self, message: str, step_name: Optional[str] = None) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, step_name)

    def start_pipeline(self, profile_name: str, total_steps: int = 10) -> None:
        """Display pipeline start banner.

        Args:
            profile_name: Name of the profile.
            total_steps: Total number of pipeline steps.
        """
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
        self.console.print()

    def start_step(self, step_name: str, description: str) -> None:
        """Mark the start of a pipeline step.

        Args:
            step_name: Name of the step.
            description: Description of what the step does.
        """
        self._step_number += 1
        self._current_step = step_name

        header = f"Step {self._step_number}/{self._total_steps}: {step_name}"
        self.console.print()
        self.console.print(f"[bold white]{'-' * 60}[/bold white]")
        self.console.print(f"[bold cyan]{header}[/bold cyan]")
        self.console.print(f"[dim]{description}[/dim]")
        self.console.print(f"[bold white]{'-' * 60}[/bold white]")

        self.log(LogLevel.STEP, f"Starting: {description}", step_name)

    def end_step(self, step_name: str, summary: str) -> None:
        """Mark the end of a pipeline step.

        Args:
            step_name: Name of the step.
            summary: Summary of what was accomplished.
        """
        self.log(LogLevel.SUCCESS, summary, step_name)

    def end_pipeline(self, output_path: Optional[Path] = None, success: bool = True) -> None:
        """Display pipeline completion banner.

        Args:
            output_path: Path to the output file.
            success: Whether the pipeline completed successfully.
        """
        end_time = datetime.now()
        duration = end_time - self._start_time if self._start_time else None
        duration_str = str(duration).split(".")[0] if duration else "unknown"

        self.console.print()

        if success:
            content = (
                f"[bold green]Video Generated Successfully![/bold green]\n\n"
                f"[bold]Duration:[/bold] {duration_str}\n"
            )
            if output_path:
                content += f"[bold]Output:[/bold] {output_path}\n"

            self.console.print(Panel(content, border_style="green", title="Complete"))
        else:
            self.console.print(Panel(
                f"[bold red]Pipeline Failed[/bold red]\n\n"
                f"[bold]Duration:[/bold] {duration_str}\n"
                f"[dim]Check logs/video_pipeline.log for details[/dim]",
                border_style="red",
                title="Error",
            ))

    def show_topic(self, topic: str, pillar: str) -> None:
        """Display selected topic.

        Args:
            topic: The selected topic.
            pillar: Content pillar name.
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold]{topic}[/bold]\n"
            f"[dim]Pillar: {pillar}[/dim]",
            title="[cyan]Selected Topic[/cyan]",
            border_style="cyan",
        ))

    def show_script(self, title: str, segments: int, duration: float) -> None:
        """Display script summary.

        Args:
            title: Script title.
            segments: Number of segments.
            duration: Total duration in seconds.
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold]{title}[/bold]\n"
            f"Segments: {segments}\n"
            f"Duration: {duration:.0f}s",
            title="[cyan]Script Planned[/cyan]",
            border_style="cyan",
        ))

    def show_clips_table(self, clips: list[dict]) -> None:
        """Display table of downloaded clips.

        Args:
            clips: List of clip info dicts.
        """
        table = Table(title="Downloaded Clips", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Pexels ID", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Source", style="dim")

        for i, clip in enumerate(clips, 1):
            table.add_row(
                str(i),
                str(clip.get("pexels_id", "?")),
                f"{clip.get('duration', 0):.1f}s",
                clip.get("source", "pexels")[:30],
            )

        self.console.print()
        self.console.print(table)

    def show_cache_stats(self, hits: int, misses: int) -> None:
        """Display Pexels cache statistics.

        Args:
            hits: Number of cache hits.
            misses: Number of cache misses.
        """
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0

        self.info(
            f"Cache: {hits} hits, {misses} misses ({hit_rate:.0f}% hit rate)",
            "PexelsCache"
        )

    def show_validation_result(
        self,
        step_name: str,
        is_valid: bool,
        score: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> None:
        """Display validation result.

        Args:
            step_name: Name of the validation step.
            is_valid: Whether validation passed.
            score: Optional score (e.g., 7/10).
            feedback: Optional feedback message.
        """
        if is_valid:
            score_str = f" (score: {score}/10)" if score else ""
            self.success(f"Validation passed{score_str}", step_name)
        else:
            score_str = f" (score: {score}/10)" if score else ""
            self.warning(f"Validation failed{score_str}: {feedback}", step_name)

    def show_retry(self, step_name: str, attempt: int, max_attempts: int) -> None:
        """Display retry attempt.

        Args:
            step_name: Name of the step being retried.
            attempt: Current attempt number.
            max_attempts: Maximum number of attempts.
        """
        self.warning(f"Retrying ({attempt}/{max_attempts})...", step_name)


# Global display instance for easy access
_display: Optional[PipelineDisplay] = None


def get_display() -> PipelineDisplay:
    """Get the global pipeline display instance."""
    global _display
    if _display is None:
        _display = PipelineDisplay()
    return _display


def set_display(display: PipelineDisplay) -> None:
    """Set the global pipeline display instance."""
    global _display
    _display = display


def setup_display(
    show_timestamps: bool = True,
    verbose: bool = False,
) -> PipelineDisplay:
    """Setup and return a new pipeline display.

    Args:
        show_timestamps: Whether to show timestamps.
        verbose: Whether to show debug messages.

    Returns:
        Configured PipelineDisplay instance.
    """
    display = PipelineDisplay(
        show_timestamps=show_timestamps,
        verbose=verbose,
    )
    set_display(display)
    return display


class StepLogger:
    """Context manager for logging pipeline steps.

    Provides a clean way to log step start/end with proper formatting.

    Usage:
        display = get_display()
        with StepLogger(display, "TopicSelector", "Selecting topic from profile"):
            # Do work
            display.info("Found 5 content pillars")
            # More work
    """

    def __init__(
        self,
        display: PipelineDisplay,
        step_name: str,
        description: str,
    ):
        self.display = display
        self.step_name = step_name
        self.description = description
        self._success_message: Optional[str] = None

    def __enter__(self) -> "StepLogger":
        self.display.start_step(self.step_name, self.description)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            if self._success_message:
                self.display.end_step(self.step_name, self._success_message)
        else:
            self.display.error(f"Failed: {exc_val}", self.step_name)
        return False  # Don't suppress exceptions

    def set_success(self, message: str) -> None:
        """Set the success message for when the step completes."""
        self._success_message = message

    def log(self, message: str) -> None:
        """Log a message within this step."""
        self.display.info(message, self.step_name)

    def debug(self, message: str) -> None:
        """Log a debug message within this step."""
        self.display.debug(message, self.step_name)

    def warning(self, message: str) -> None:
        """Log a warning within this step."""
        self.display.warning(message, self.step_name)
