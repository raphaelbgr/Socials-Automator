"""Debug logger for video pipeline.

Saves comprehensive debug information to a log.txt file in each post's output folder.
Captures all parameters, timing, and debug info for troubleshooting.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class PipelineDebugLogger:
    """Collects and saves debug information for a pipeline run."""

    def __init__(self):
        """Initialize debug logger."""
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._entries: list[str] = []
        self._params: dict[str, Any] = {}
        self._step_times: dict[str, dict] = {}
        self._current_step: Optional[str] = None
        self._current_step_start: Optional[datetime] = None

    def start(self, **params) -> None:
        """Mark pipeline start and record initial parameters.

        Args:
            **params: Pipeline configuration parameters.
        """
        self._start_time = datetime.now()
        self._params = params
        self._entries = []
        self._step_times = {}

        self._add_header("PIPELINE START")
        self._add_line(f"Started: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._add_line("")

        self._add_header("CONFIGURATION")
        for key, value in sorted(params.items()):
            self._add_line(f"  {key}: {value}")
        self._add_line("")

    def start_step(self, step_name: str) -> None:
        """Mark start of a pipeline step.

        Args:
            step_name: Name of the step.
        """
        self._current_step = step_name
        self._current_step_start = datetime.now()
        self._add_line(f"[{self._timestamp()}] START: {step_name}")

    def end_step(self, step_name: str, details: Optional[dict] = None) -> None:
        """Mark end of a pipeline step.

        Args:
            step_name: Name of the step.
            details: Optional details to log.
        """
        end_time = datetime.now()
        duration = None

        if self._current_step == step_name and self._current_step_start:
            duration = end_time - self._current_step_start
            duration_str = str(duration).split(".")[0]
            self._step_times[step_name] = {
                "start": self._current_step_start,
                "end": end_time,
                "duration": duration_str,
            }
            self._add_line(f"[{self._timestamp()}] END: {step_name} ({duration_str})")
        else:
            self._add_line(f"[{self._timestamp()}] END: {step_name}")

        if details:
            for key, value in details.items():
                self._add_line(f"    {key}: {value}")

        self._current_step = None
        self._current_step_start = None

    def log(self, message: str, level: str = "INFO") -> None:
        """Add a log entry.

        Args:
            message: Log message.
            level: Log level (INFO, DEBUG, WARNING, ERROR).
        """
        self._add_line(f"[{self._timestamp()}] {level}: {message}")

    def log_topic(self, topic: str, pillar: str, keywords: list[str]) -> None:
        """Log selected topic details.

        Args:
            topic: Topic text.
            pillar: Content pillar name.
            keywords: Associated keywords.
        """
        self._add_line("")
        self._add_header("TOPIC")
        self._add_line(f"  Topic: {topic}")
        self._add_line(f"  Pillar: {pillar}")
        self._add_line(f"  Keywords: {', '.join(keywords[:5])}")

    def log_script(self, title: str, segments: int, duration: float, narration: str) -> None:
        """Log script details.

        Args:
            title: Script title.
            segments: Number of segments.
            duration: Total duration in seconds.
            narration: Full narration text.
        """
        self._add_line("")
        self._add_header("SCRIPT")
        self._add_line(f"  Title: {title}")
        self._add_line(f"  Segments: {segments}")
        self._add_line(f"  Duration: {duration:.1f}s")
        self._add_line(f"  Word count: {len(narration.split())}")
        self._add_line("")
        self._add_line("  --- Narration ---")
        # Wrap narration text
        words = narration.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 80:
                self._add_line(line)
                line = "  " + word
            else:
                line += " " + word if line != "  " else word
        if line.strip():
            self._add_line(line)
        self._add_line("  --- End Narration ---")

    def log_video_search(self, segment_results: list[dict]) -> None:
        """Log video search results.

        Args:
            segment_results: List of search results per segment.
        """
        self._add_line("")
        self._add_header("VIDEO SEARCH RESULTS")
        for result in segment_results:
            seg_idx = result.get("segment_index", "?")
            keywords = result.get("keywords_used", [])
            video = result.get("video", {})
            pexels_id = video.get("id", "?")
            duration = video.get("duration", 0)
            self._add_line(f"  Segment {seg_idx}: pexels_id={pexels_id}, duration={duration}s, keywords={keywords[:3]}")

    def log_cache_stats(self, hits: int, misses: int) -> None:
        """Log cache statistics.

        Args:
            hits: Number of cache hits.
            misses: Number of cache misses.
        """
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        self._add_line("")
        self._add_header("CACHE STATS")
        self._add_line(f"  Hits: {hits}")
        self._add_line(f"  Misses: {misses}")
        self._add_line(f"  Hit rate: {hit_rate:.0f}%")

    def log_voice(self, backend: str, voice: str, timestamps_count: int) -> None:
        """Log voice generation details.

        Args:
            backend: TTS backend used.
            voice: Voice name/ID.
            timestamps_count: Number of word timestamps.
        """
        self._add_line("")
        self._add_header("VOICE GENERATION")
        self._add_line(f"  Backend: {backend}")
        self._add_line(f"  Voice: {voice}")
        self._add_line(f"  Word timestamps: {timestamps_count}")

    def log_error(self, error: str, step: Optional[str] = None) -> None:
        """Log an error.

        Args:
            error: Error message.
            step: Optional step where error occurred.
        """
        self._add_line("")
        self._add_header("ERROR")
        if step:
            self._add_line(f"  Step: {step}")
        self._add_line(f"  Error: {error}")

    def end(self, success: bool, output_path: Optional[Path] = None) -> None:
        """Mark pipeline end.

        Args:
            success: Whether pipeline succeeded.
            output_path: Path to output video if successful.
        """
        self._end_time = datetime.now()
        duration = self._end_time - self._start_time if self._start_time else None
        duration_str = str(duration).split(".")[0] if duration else "unknown"

        self._add_line("")
        self._add_header("STEP TIMING SUMMARY")
        for step_name, times in self._step_times.items():
            self._add_line(f"  {step_name}: {times['duration']}")

        self._add_line("")
        self._add_header("PIPELINE END")
        self._add_line(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        self._add_line(f"  Total duration: {duration_str}")
        if output_path:
            self._add_line(f"  Output: {output_path}")
        self._add_line(f"  Ended: {self._end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def save(self, output_dir: Path) -> Path:
        """Save debug log to file.

        Args:
            output_dir: Directory to save log file.

        Returns:
            Path to saved log file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = output_dir / "debug_log.txt"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._entries))

        return log_path

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _add_header(self, title: str) -> None:
        """Add a section header."""
        self._entries.append(f"{'=' * 60}")
        self._entries.append(f" {title}")
        self._entries.append(f"{'=' * 60}")

    def _add_line(self, line: str) -> None:
        """Add a line to the log."""
        self._entries.append(line)
