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


class VideoPipeline:
    """Orchestrates the complete video generation pipeline."""

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        voice: str = "rvc_adam",
        text_ai: Optional[str] = None,
        video_matcher: str = "pexels",
        subtitle_size: int = 80,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize video pipeline.

        Args:
            pexels_api_key: Pexels API key. Reads from env if not provided.
            voice: Voice preset or name for TTS.
            text_ai: Text AI provider (lmstudio, openai, etc.). None uses templates.
            video_matcher: Video source ('pexels'). Reserved for future sources.
            subtitle_size: Subtitle font size in pixels (default 80).
            progress_callback: Optional callback for progress updates.
                Signature: (stage: str, progress: float, message: str)
        """
        self.logger = logging.getLogger("video.pipeline")
        self.progress_callback = progress_callback
        self.text_ai = text_ai
        self.video_matcher = video_matcher

        # Create AI client if text_ai is specified
        ai_client = None
        if text_ai:
            ai_client = self._create_ai_client(text_ai)
            self.logger.info(f"Using text AI: {text_ai}")

        # Initialize pipeline steps
        # Note: VoiceGenerator runs BEFORE VideoAssembler so we get actual speech timing
        # to sync video clips with sentence endings
        self.steps: list[PipelineStep] = [
            TopicSelector(ai_client=ai_client),
            TopicResearcher(),
            ScriptPlanner(ai_client=ai_client),
            VoiceGenerator(voice=voice),  # Generate voice first to get actual timing
            VideoSearcher(api_key=pexels_api_key),
            VideoDownloader(),
            VideoAssembler(),  # Now uses actual voice timing for segment durations
            SubtitleRenderer(font_size=subtitle_size),
            CaptionGenerator(ai_client=ai_client),  # Generate caption and hashtags
        ]

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

    def _log(self, message: str) -> None:
        """Log message to console and logger."""
        self.logger.info(message)
        print(f"[VIDEO] {message}")

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
        self._log("=" * 60)
        self._log("VIDEO GENERATION PIPELINE")
        self._log("=" * 60)

        # Generate post ID
        if post_id is None:
            now = datetime.now()
            post_id = f"{now.strftime('%Y%m%d-%H%M%S')}"

        self._log(f"Post ID: {post_id}")

        # Load profile metadata
        self._log("Loading profile metadata...")
        metadata_path = Path(profile_path) / "metadata.json"

        if not metadata_path.exists():
            raise PipelineError(f"Profile metadata not found: {metadata_path}")

        profile = ProfileMetadata.from_file(metadata_path)
        self._log(f"Profile: {profile.display_name}")

        # Create temp directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp(prefix=f"video_{post_id}_"))
        self._log(f"Temp directory: {temp_dir}")

        # Determine output directory
        if output_dir is None:
            output_dir = temp_dir / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Output directory: {output_dir}")

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

                self._log("-" * 40)
                self._log(f"STEP {i + 1}/{total_steps}: {step_name}")
                self._log("-" * 40)

                self._update_progress(step_name, progress, f"Starting {step_name}...")

                context = await step.execute(context)

                self._update_progress(
                    step_name,
                    (i + 1) / total_steps,
                    f"Completed {step_name}",
                )

            # Pipeline complete
            self._log("=" * 60)
            self._log("PIPELINE COMPLETE")
            self._log("=" * 60)

            if context.final_video_path:
                self._log(f"Final video: {context.final_video_path}")

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
                    self._log(f"Metadata: {metadata_out}")

                return context.final_video_path
            else:
                raise PipelineError("Pipeline completed but no output video")

        except Exception as e:
            self._log(f"Pipeline failed: {e}")
            raise PipelineError(f"Video generation failed: {e}") from e

        finally:
            # Optionally clean up temp directory
            # Keeping it for debugging; add cleanup flag if needed
            self._log(f"Temp files preserved at: {temp_dir}")

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
