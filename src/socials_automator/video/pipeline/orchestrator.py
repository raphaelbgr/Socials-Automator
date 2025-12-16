"""Video generation pipeline orchestrator.

Coordinates all pipeline steps to generate a complete video:
1. Select topic from profile
2. Research topic via web search
3. Plan video script
4. Generate voiceover (before video for timing sync)
5. Search for stock videos
6. Download video clips
7. Assemble video
8. Render subtitles
9. Generate caption and hashtags
10. Output final video
"""

import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .base import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    ProfileMetadata,
)
from .caption_generator import CaptionGenerator
from .cli_display import PipelineDisplay, setup_display
from .debug_logger import PipelineDebugLogger
from .script_planner import ScriptPlanner
from .subtitle_renderer import SubtitleRenderer
from .topic_researcher import TopicResearcher
from .topic_selector import TopicSelector
from .video_assembler import VideoAssembler
from .video_downloader import VideoDownloader
from .video_searcher import VideoSearcher
from .voice_generator import VoiceGenerator


# Type alias for progress callback
ProgressCallback = Callable[[str, float, str], None]

# Step descriptions for clean logging
STEP_DESCRIPTIONS = {
    "TopicSelector": "Analyzing profile and selecting topic based on content pillars",
    "TopicResearcher": "Researching topic via web search for accurate content",
    "ScriptPlanner": "Planning video script with AI-generated narration",
    "VoiceGenerator": "Generating voiceover audio (determines final video duration)",
    "VideoSearcher": "Searching Pexels for relevant stock footage",
    "VideoDownloader": "Downloading video clips (with cache optimization)",
    "VideoAssembler": "Assembling clips into 9:16 vertical video",
    "SubtitleRenderer": "Rendering karaoke-style subtitles and adding audio",
    "CaptionGenerator": "Generating Instagram caption and hashtags with AI validation",
}


class VideoPipeline:
    """Orchestrates the complete video generation pipeline."""

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        voice: str = "rvc_adam",
        text_ai: Optional[str] = None,
        video_matcher: str = "pexels",
        subtitle_size: int = 80,
        target_duration: float = 60.0,
        progress_callback: Optional[ProgressCallback] = None,
        show_timestamps: bool = True,
        verbose: bool = False,
    ):
        """Initialize video pipeline.

        Args:
            pexels_api_key: Pexels API key. Reads from env if not provided.
            voice: Voice preset or name for TTS.
            text_ai: Text AI provider (lmstudio, openai, etc.). None uses templates.
            video_matcher: Video source ('pexels'). Reserved for future sources.
            subtitle_size: Subtitle font size in pixels (default 80).
            target_duration: Target video duration in seconds (default 60).
            progress_callback: Optional callback for progress updates.
                Signature: (stage: str, progress: float, message: str)
            show_timestamps: Whether to show timestamps in CLI output.
            verbose: Whether to show debug messages.
        """
        self.logger = logging.getLogger("video.pipeline")
        self.progress_callback = progress_callback
        self.text_ai = text_ai
        self.video_matcher = video_matcher
        self.target_duration = target_duration
        self.voice = voice
        self.subtitle_size = subtitle_size

        # Initialize debug logger
        self.debug_logger = PipelineDebugLogger()

        # Setup CLI display
        self.display = setup_display(
            show_timestamps=show_timestamps,
            verbose=verbose,
        )

        # Create AI client if text_ai is specified
        ai_client = None
        if text_ai:
            ai_client = self._create_ai_client(text_ai)
            self.logger.info(f"Using text AI: {text_ai}")
            self.display.info(f"Text AI provider: {text_ai}")

        # Initialize pipeline steps
        # Note: VoiceGenerator runs BEFORE VideoAssembler so we get actual speech timing
        # The narration audio is the source of truth for final video duration
        self.steps: list[PipelineStep] = [
            TopicSelector(ai_client=ai_client),
            TopicResearcher(),
            ScriptPlanner(ai_client=ai_client, target_duration=target_duration),
            VoiceGenerator(voice=voice),  # Generate voice - this determines actual duration
            VideoSearcher(api_key=pexels_api_key, ai_client=ai_client),  # AI for unique keywords
            VideoDownloader(),
            VideoAssembler(),  # Uses audio duration as source of truth
            SubtitleRenderer(font_size=subtitle_size),
            CaptionGenerator(ai_client=ai_client),  # Generate caption and hashtags
        ]

        # Set display on all steps
        for step in self.steps:
            step.set_display(self.display)

    def _create_ai_client(self, provider: str) -> Optional[object]:
        """Create AI client for enhanced topic/script generation.

        Args:
            provider: Provider name (lmstudio, openai, etc.)

        Returns:
            AI client or None if unavailable.
        """
        try:
            # Import from socials_automator.providers (go up 3 levels: pipeline -> video -> socials_automator)
            from socials_automator.providers.text import TextProvider
            return TextProvider(provider_override=provider)
        except ImportError as e:
            self.logger.warning(
                f"Could not import TextProvider for {provider}: {e}. "
                "Using template-based generation."
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"Could not create AI client for {provider}: {e}. "
                "Using template-based generation."
            )
            return None

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)

    async def generate(
        self,
        profile_path: Path,
        output_dir: Optional[Path] = None,
        post_id: Optional[str] = None,
    ) -> Path:
        """Generate a complete video from profile.

        Args:
            profile_path: Path to profile directory containing metadata.json.
            output_dir: Output directory for final video. If None, uses temp.
            post_id: Optional post ID. Auto-generated if not provided.

        Returns:
            Path to final video file.

        Raises:
            PipelineError: If any pipeline step fails.
        """
        # Generate post ID
        if post_id is None:
            now = datetime.now()
            post_id = f"{now.strftime('%Y%m%d-%H%M%S')}"

        # Load profile metadata
        self.display.info("Loading profile metadata...")
        metadata_path = Path(profile_path) / "metadata.json"

        if not metadata_path.exists():
            self.display.error(f"Profile metadata not found: {metadata_path}")
            raise PipelineError(f"Profile metadata not found: {metadata_path}")

        profile = ProfileMetadata.from_file(metadata_path)

        # Start debug logger with configuration
        self.debug_logger.start(
            profile=profile.display_name,
            profile_path=str(profile_path),
            post_id=post_id,
            voice=self.voice,
            text_ai=self.text_ai or "templates",
            video_matcher=self.video_matcher,
            subtitle_size=self.subtitle_size,
            target_duration=self.target_duration,
        )

        # Start pipeline display
        self.display.start_pipeline(profile.display_name, total_steps=len(self.steps))
        self.display.info(f"Post ID: {post_id}")

        # Create temp directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp(prefix=f"video_{post_id}_"))
        self.display.debug(f"Temp directory: {temp_dir}")

        # Determine output directory
        if output_dir is None:
            output_dir = temp_dir / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.display.info(f"Output: {output_dir}")

        # Create pipeline context
        context = PipelineContext(
            profile=profile,
            post_id=post_id,
            output_dir=output_dir,
            temp_dir=temp_dir,
        )

        # Execute pipeline steps
        total_steps = len(self.steps)

        try:
            for i, step in enumerate(self.steps):
                step_name = step.name
                progress = i / total_steps
                description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                # Start step with description
                self.display.start_step(step_name, description)
                self.debug_logger.start_step(step_name)

                self._update_progress(step_name, progress, f"Starting {step_name}...")

                context = await step.execute(context)

                # Show specific info after certain steps and log to debug
                if step_name == "TopicSelector" and context.topic:
                    self.display.show_topic(context.topic.topic, context.topic.pillar_name)
                    self.debug_logger.log_topic(
                        context.topic.topic,
                        context.topic.pillar_name,
                        context.topic.keywords,
                    )
                    self.debug_logger.end_step(step_name)

                elif step_name == "ScriptPlanner" and context.script:
                    self.display.show_script(
                        context.script.title,
                        len(context.script.segments),
                        context.script.total_duration,
                    )
                    self.debug_logger.log_script(
                        context.script.title,
                        len(context.script.segments),
                        context.script.total_duration,
                        context.script.full_narration,
                    )
                    self.debug_logger.end_step(step_name, {
                        "segments": len(context.script.segments),
                        "duration": f"{context.script.total_duration:.1f}s",
                    })

                elif step_name == "VoiceGenerator":
                    self.debug_logger.log_voice(
                        getattr(step, 'backend', 'unknown'),
                        getattr(step, 'voice', 'unknown'),
                        len(getattr(context, '_word_timestamps', [])) if hasattr(context, '_word_timestamps') else 0,
                    )
                    self.debug_logger.end_step(step_name)

                elif step_name == "VideoSearcher":
                    search_results = getattr(context, '_search_results', [])
                    if search_results:
                        self.debug_logger.log_video_search(search_results)
                    self.debug_logger.end_step(step_name, {
                        "clips_found": len(search_results) if search_results else 0,
                    })

                elif step_name == "VideoDownloader" and hasattr(step, '_cache_hits'):
                    hits = getattr(step, '_cache_hits', 0)
                    misses = getattr(step, '_cache_misses', 0)
                    self.display.show_cache_stats(hits, misses)
                    self.debug_logger.log_cache_stats(hits, misses)
                    self.debug_logger.end_step(step_name, {
                        "cache_hits": hits,
                        "cache_misses": misses,
                    })

                else:
                    self.debug_logger.end_step(step_name)

                self._update_progress(
                    step_name,
                    (i + 1) / total_steps,
                    f"Completed {step_name}",
                )

            # Pipeline complete
            if context.final_video_path:
                # Copy metadata to output
                if context.metadata:
                    metadata_out = output_dir / "metadata.json"
                    with open(metadata_out, "w", encoding="utf-8") as f:
                        json.dump(
                            context.metadata.model_dump(),
                            f,
                            indent=2,
                            default=str,
                        )
                    self.display.info(f"Metadata saved: {metadata_out.name}")

                # Save debug log
                self.debug_logger.end(success=True, output_path=context.final_video_path)
                debug_log_path = self.debug_logger.save(output_dir)
                self.display.info(f"Debug log saved: {debug_log_path.name}")

                self.display.end_pipeline(context.final_video_path, success=True)
                return context.final_video_path
            else:
                self.debug_logger.end(success=False)
                self.debug_logger.save(output_dir)
                self.display.end_pipeline(success=False)
                raise PipelineError("Pipeline completed but no output video")

        except Exception as e:
            self.display.error(f"Pipeline failed: {e}")
            self.debug_logger.log_error(str(e))
            self.debug_logger.end(success=False)
            try:
                self.debug_logger.save(output_dir)
            except Exception:
                pass  # Don't fail if we can't save debug log
            self.display.end_pipeline(success=False)
            raise PipelineError(f"Video generation failed: {e}") from e

        finally:
            # Optionally clean up temp directory
            # Keeping it for debugging; add cleanup flag if needed
            self.display.debug(f"Temp files preserved at: {temp_dir}")

    def generate_sync(
        self,
        profile_path: Path,
        output_dir: Optional[Path] = None,
        post_id: Optional[str] = None,
    ) -> Path:
        """Synchronous wrapper for generate().

        Args:
            profile_path: Path to profile directory.
            output_dir: Output directory.
            post_id: Optional post ID.

        Returns:
            Path to final video.
        """
        import asyncio
        return asyncio.run(self.generate(profile_path, output_dir, post_id))


class DdgsLogFilter(logging.Filter):
    """Filter that reformats ddgs error logs into clean messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Reformat ddgs error logs to be more readable."""
        if record.name == "ddgs.ddgs" and "Error in engine" in str(record.msg):
            try:
                import re
                msg = str(record.msg)

                # Extract engine name and error type
                engine_match = re.search(r"Error in engine (\w+):", msg)
                engine = engine_match.group(1) if engine_match else "unknown"

                # Determine error type
                if "timed out" in msg.lower():
                    error_type = "TIMEOUT"
                elif "403" in msg or "forbidden" in msg.lower():
                    error_type = "BLOCKED"
                elif "404" in msg:
                    error_type = "NOT FOUND"
                elif "connection" in msg.lower():
                    error_type = "CONNECTION"
                else:
                    error_type = "ERROR"

                # Create clean message
                record.msg = f"{engine:12} [{error_type}]"

            except Exception:
                pass  # Keep original message if parsing fails

        return True


class PrimpLogFilter(logging.Filter):
    """Filter that reformats primp HTTP response logs into clean search logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Reformat primp logs to show clean search info."""
        if record.name == "primp" and "response:" in record.msg:
            try:
                # Parse: "response: https://... 200"
                parts = record.msg.split()
                if len(parts) >= 3:
                    url = parts[1]
                    status = parts[2]

                    # Extract domain
                    from urllib.parse import urlparse, parse_qs, unquote
                    parsed = urlparse(url)
                    domain = parsed.netloc.replace("www.", "").replace("en.", "")

                    # Shorten common domains
                    domain_map = {
                        "wikipedia.org": "wikipedia",
                        "search.yahoo.com": "yahoo",
                        "mojeek.com": "mojeek",
                        "duckduckgo.com": "duckduckgo",
                        "google.com": "google",
                        "bing.com": "bing",
                    }
                    for full, short in domain_map.items():
                        if full in domain:
                            domain = short
                            break

                    # Extract search query
                    query_params = parse_qs(parsed.query)
                    search_term = ""
                    for param in ["q", "search", "p", "query"]:
                        if param in query_params:
                            search_term = unquote(query_params[param][0])
                            break

                    # Truncate long search terms
                    if len(search_term) > 50:
                        search_term = search_term[:47] + "..."

                    # Format status
                    status_icon = "[OK]" if status == "200" else f"[{status}]"

                    # Create clean message
                    if search_term:
                        record.msg = f"{domain:12} {status_icon:6} {search_term}"
                    else:
                        record.msg = f"{domain:12} {status_icon:6} (api call)"

            except Exception:
                pass  # Keep original message if parsing fails

        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging for video pipeline.

    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set specific loggers
    logging.getLogger("video.pipeline").setLevel(level)

    # Add filter to reformat primp (HTTP client) logs
    primp_logger = logging.getLogger("primp")
    primp_logger.addFilter(PrimpLogFilter())

    # Add filter to reformat ddgs (DuckDuckGo) error logs
    ddgs_logger = logging.getLogger("ddgs.ddgs")
    ddgs_logger.addFilter(DdgsLogFilter())
