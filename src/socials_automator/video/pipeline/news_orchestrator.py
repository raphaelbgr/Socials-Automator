"""News-based video generation pipeline orchestrator.

Coordinates all pipeline steps to generate a news briefing video:
1. Fetch news from RSS feeds and web search
2. Curate and rank news stories with AI
3. Generate video script from curated stories
4. [PARALLEL] Generate voiceover + Search/download stock videos
5. Assemble video
6. Render subtitles
7. Generate caption and hashtags
8. Output final video

This orchestrator replaces TopicSelector/TopicResearcher with
NewsAggregator/NewsCurator for news-based content.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .base import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    ProfileMetadata,
    TopicInfo,
    ResearchResult,
)
from .caption_generator import CaptionGenerator
from .cli_display import PipelineDisplay, setup_display
from .debug_logger import PipelineDebugLogger
from .news_script_planner import NewsScriptPlanner
from .subtitle_renderer import SubtitleRenderer
from .thumbnail_generator import ThumbnailGenerator
from .video_assembler import VideoAssembler
from .video_downloader import VideoDownloader
from .video_searcher import VideoSearcher
from .voice_generator import VoiceGenerator

# News module imports
from ...news.aggregator import NewsAggregator
from ...news.curator import NewsCurator, CurationConfig
from ...news.models import NewsBrief, NewsEdition, NewsCategory

# GPU-accelerated renderers (optional)
try:
    from .gpu_utils import validate_gpu_setup, GPUInfo
    from .video_renderer_gpu import GPUVideoAssembler, GPUSubtitleRenderer
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUInfo = None


# Type alias for progress callback
ProgressCallback = Callable[[str, float, str], None]

# Step descriptions for clean logging
STEP_DESCRIPTIONS = {
    "NewsAggregator": "Fetching news from RSS feeds and web search",
    "NewsCurator": "Curating and ranking stories with AI",
    "NewsScriptPlanner": "Planning video script from curated news",
    "VoiceGenerator": "Generating voiceover audio",
    "VideoSearcher": "Searching Pexels for relevant stock footage",
    "VideoDownloader": "Downloading video clips",
    "VideoAssembler": "Assembling clips into 9:16 vertical video",
    "GPUVideoAssembler": "Assembling clips into 9:16 vertical video (GPU NVENC)",
    "SubtitleRenderer": "Rendering karaoke-style subtitles and adding audio",
    "GPUSubtitleRenderer": "Rendering subtitles and adding audio (GPU NVENC)",
    "ThumbnailGenerator": "Generating thumbnail for Instagram",
    "CaptionGenerator": "Generating Instagram caption and hashtags",
}


class NewsPipelineContext(PipelineContext):
    """Extended pipeline context with news-specific data."""

    news_brief: Optional[NewsBrief] = None
    thumbnail_text: Optional[str] = None  # Custom thumbnail text (teaser list for news)


class NewsPipeline:
    """Orchestrates news-based video generation pipeline.

    Similar to VideoPipeline but uses news aggregation and curation
    instead of topic selection and research.
    """

    # Duration validation constants
    MAX_DURATION_MULTIPLIER = 1.5
    MAX_DURATION_RETRIES = 5

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        voice: str = "rvc_adam",
        voice_rate: str = "+0%",
        voice_pitch: str = "+0Hz",
        text_ai: Optional[str] = None,
        subtitle_size: int = 80,
        subtitle_font: str = "Montserrat-Bold.ttf",
        target_duration: float = 60.0,
        progress_callback: Optional[ProgressCallback] = None,
        show_timestamps: bool = True,
        verbose: bool = False,
        gpu_accelerate: bool = False,
        gpu_index: Optional[int] = None,
        # News-specific options
        edition: Optional[NewsEdition] = None,
        story_count: Optional[int] = None,  # None = auto (AI decides)
        max_news_age_hours: int = 24,
        news_categories: Optional[list[NewsCategory]] = None,
        profile_name: Optional[str] = None,  # For theme history tracking
        profile_path: Optional[Path] = None,  # For profile-scoped data storage
    ):
        """Initialize news pipeline.

        Args:
            pexels_api_key: Pexels API key.
            voice: Voice preset for TTS.
            voice_rate: Speech rate adjustment.
            voice_pitch: Pitch adjustment.
            text_ai: Text AI provider (for curation and scripts).
            subtitle_size: Subtitle font size.
            subtitle_font: Subtitle font file.
            target_duration: Target video duration in seconds.
            progress_callback: Progress callback function.
            show_timestamps: Show timestamps in CLI output.
            verbose: Show debug messages.
            gpu_accelerate: Enable GPU acceleration.
            gpu_index: GPU index to use.
            edition: News edition (morning, midday, evening, night).
            story_count: Number of stories per video (None = auto).
            max_news_age_hours: Maximum article age in hours.
            news_categories: Filter to specific categories.
            profile_name: Profile name for theme history tracking.
            profile_path: Profile directory for profile-scoped data storage.
        """
        self.profile_name = profile_name
        self.profile_path = profile_path
        self.logger = logging.getLogger("video.news_pipeline")
        self.progress_callback = progress_callback
        self.text_ai = text_ai
        self.target_duration = target_duration
        self.voice = voice
        self.subtitle_size = subtitle_size
        self.subtitle_font = subtitle_font
        self.gpu_accelerate = gpu_accelerate
        self.gpu_index = gpu_index
        self.gpu_info: Optional[GPUInfo] = None

        # News-specific settings
        self.edition = edition
        self.story_count = story_count
        self.max_news_age_hours = max_news_age_hours
        self.news_categories = news_categories

        # Initialize debug logger
        self.debug_logger = PipelineDebugLogger()

        # Setup CLI display
        self.display = setup_display(
            show_timestamps=show_timestamps,
            verbose=verbose,
        )

        # Validate and setup GPU if requested
        if gpu_accelerate:
            if not GPU_AVAILABLE:
                self.display.error("GPU acceleration requested but GPU modules not available")
                self.display.info("Falling back to CPU rendering")
                self.gpu_accelerate = False
            else:
                success, message, gpu = validate_gpu_setup(gpu_index)
                if success:
                    self.gpu_info = gpu
                    self.display.info(f"GPU acceleration enabled: {gpu.name} (GPU {gpu.index})")
                else:
                    self.display.error(f"GPU setup failed: {message}")
                    self.display.info("Falling back to CPU rendering")
                    self.gpu_accelerate = False

        # Initialize news components
        self.news_aggregator = NewsAggregator(
            profile_path=profile_path,
            profile_name=profile_name,
            use_dynamic_queries=True,  # AI-generated queries based on topic history
        )
        self.news_curator = NewsCurator(
            config=CurationConfig(
                stories_per_brief=story_count,
                target_duration=target_duration,  # For auto story count calculation
                provider_override=text_ai,
                profile_name=profile_name,  # For theme/story/topic history tracking
            )
        )

        # Script planner (news-specific)
        self.script_planner = NewsScriptPlanner(
            target_duration=target_duration,
            preferred_provider=text_ai,
        )

        # Voice generation
        self.voice_step = VoiceGenerator(voice=voice, rate=voice_rate, pitch=voice_pitch)

        # Video search and download
        self.video_search_step = VideoSearcher(api_key=pexels_api_key, ai_client=None)
        self.video_download_step = VideoDownloader()

        # Assembly and post-processing
        if self.gpu_accelerate and GPU_AVAILABLE and self.gpu_info:
            self.assembly_step = GPUVideoAssembler(gpu=self.gpu_info)
            self.subtitle_step = GPUSubtitleRenderer(
                gpu=self.gpu_info,
                font=subtitle_font,
                font_size=subtitle_size,
            )
        else:
            self.assembly_step = VideoAssembler()
            self.subtitle_step = SubtitleRenderer(
                font_size=subtitle_size,
                font_name=subtitle_font,
            )

        # Use larger font for thumbnails (90px) for better visibility on Instagram grid
        # Regular reels use 72px, but news hooks are often longer so we use 90px
        self.thumbnail_step = ThumbnailGenerator(font=subtitle_font, font_size=90)
        self.caption_step = CaptionGenerator(preferred_provider=text_ai)

        # Connect all steps to the display system for logging
        self._connect_step_displays()

        # Track all steps for progress
        self.steps = [
            "NewsAggregator",
            "NewsCurator",
            "NewsScriptPlanner",
            "VoiceGenerator",
            "VideoSearcher",
            "VideoDownloader",
            "VideoAssembler",
            "ThumbnailGenerator",
            "SubtitleRenderer",
            "CaptionGenerator",
        ]

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Failed to get audio duration: {e}")
            return 0.0

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)

    def _connect_step_displays(self) -> None:
        """Connect all pipeline steps to the display system.

        This enables organized step logging (>>> PREPROCESSING, >>> ENCODING, etc.)
        to appear in the CLI output.
        """
        steps_to_connect = [
            self.script_planner,
            self.voice_step,
            self.video_search_step,
            self.video_download_step,
            self.assembly_step,
            self.subtitle_step,
            self.thumbnail_step,
            self.caption_step,
        ]
        for step in steps_to_connect:
            if hasattr(step, 'set_display'):
                step.set_display(self.display)

    async def generate(
        self,
        profile_path: Path,
        output_dir: Optional[Path] = None,
        post_id: Optional[str] = None,
    ) -> Path:
        """Generate a news briefing video.

        Args:
            profile_path: Path to profile directory.
            output_dir: Output directory for final video.
            post_id: Optional post ID.

        Returns:
            Path to final video file.
        """
        # Generate post ID with edition
        if post_id is None:
            now = datetime.now()
            edition = self.edition or NewsEdition.from_hour(now.hour)
            post_id = f"{now.strftime('%d-%H%M')}-{edition.value}"

        # Load profile metadata
        self.display.info("Loading profile metadata...")
        metadata_path = Path(profile_path) / "metadata.json"

        if not metadata_path.exists():
            self.display.error(f"Profile metadata not found: {metadata_path}")
            raise PipelineError(f"Profile metadata not found: {metadata_path}")

        profile = ProfileMetadata.from_file(metadata_path)

        # Start debug logger
        self.debug_logger.start(
            profile=profile.display_name,
            profile_path=str(profile_path),
            post_id=post_id,
            voice=self.voice,
            text_ai=self.text_ai or "templates",
            video_matcher="pexels",
            subtitle_size=self.subtitle_size,
            target_duration=self.target_duration,
        )

        # Start pipeline display
        self.display.start_pipeline(profile.display_name, total_steps=len(self.steps))
        self.display.info(f"Post ID: {post_id}")
        self.display.info(f"Mode: News Briefing ({self.edition or 'auto'} edition)")

        # Create temp directory (system temp for pipeline working files)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"news_{post_id}_"))
        self.display.debug(f"Temp directory: {temp_dir}")

        # Create post-specific subfolder in project temp for ffmpeg operations
        from socials_automator.constants import get_temp_dir
        project_temp_subdir = get_temp_dir() / post_id
        project_temp_subdir.mkdir(parents=True, exist_ok=True)
        # Set env vars so ffmpeg and other tools use this post-specific folder
        os.environ["TEMP"] = str(project_temp_subdir)
        os.environ["TMP"] = str(project_temp_subdir)

        # Determine output directory
        if output_dir is None:
            output_dir = temp_dir / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.display.info(f"Output: {output_dir}")

        # Create pipeline context
        context = NewsPipelineContext(
            profile=profile,
            post_id=post_id,
            output_dir=output_dir,
            temp_dir=temp_dir,
        )

        total_steps = len(self.steps)
        step_counter = 0

        try:
            # =================================================================
            # Phase 1: News Aggregation (with global source rotation)
            # =================================================================
            step_name = "NewsAggregator"
            description = STEP_DESCRIPTIONS.get(step_name, "Fetching news")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Fetching global news...")

            # Use new rotation-based fetch for global sources
            aggregation = await self.news_aggregator.fetch_with_rotation(
                max_age_hours=self.max_news_age_hours,
                max_feeds=40,
                max_articles=50,
            )

            # Enhanced logging with region/language info
            self.display.info(
                f"Fetched {aggregation.total_articles} articles "
                f"(RSS: {aggregation.rss_articles_count}, Search: {aggregation.search_articles_count})"
            )

            # Show source diversity
            if aggregation.regions_used:
                regions_str = ", ".join(sorted(aggregation.regions_used))
                self.display.info(f"Regions: {regions_str}")

            if aggregation.languages_used:
                langs_str = ", ".join(sorted(aggregation.languages_used))
                self.display.info(f"Languages: {langs_str}")

            if aggregation.query_batch > 0:
                self.display.debug(f"Query batch: {aggregation.query_batch}")

            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 2: News Curation
            # =================================================================
            step_name = "NewsCurator"
            description = STEP_DESCRIPTIONS.get(step_name, "Curating news")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Curating stories...")

            edition = self.edition or NewsEdition.from_hour(datetime.now().hour)
            news_brief = await self.news_curator.curate(
                aggregation=aggregation,
                edition=edition,
                story_count=self.story_count,
            )

            context.news_brief = news_brief
            # Set thumbnail teaser text (e.g., "JUST IN\n-> Story 1\n-> Story 2")
            context.thumbnail_text = news_brief.get_thumbnail_text()

            # Create TopicInfo for compatibility with rest of pipeline
            context.topic = TopicInfo(
                topic=news_brief.theme,
                pillar_id="news_briefing",
                pillar_name=edition.display_name,
                keywords=[s.headline for s in news_brief.stories],
                search_queries=[],
            )

            # Create ResearchResult for compatibility
            context.research = ResearchResult(
                topic=news_brief.theme,
                summary=f"{edition.display_name} covering {news_brief.story_count} stories",
                key_points=[s.summary for s in news_brief.stories],
                sources=[{"name": s.source_name, "url": s.original_url} for s in news_brief.stories],
            )

            self.display.show_topic(news_brief.theme, edition.display_name)
            self.display.info(f"Selected {news_brief.story_count} stories from {news_brief.sources_cited}")
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 3: Script Planning
            # =================================================================
            step_name = "NewsScriptPlanner"
            description = STEP_DESCRIPTIONS.get(step_name, "Planning script")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Planning script...")

            script = await self.script_planner.plan_from_brief(
                brief=news_brief,
                profile_name=profile.display_name or profile.name,
                profile_handle=profile.instagram_handle,
            )
            context.script = script

            self.display.info(
                f"Script planned: {len(script.segments)} segments, "
                f"~{len(script.full_narration.split())} words"
            )
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 4: Duration Validation Loop (Script + Voice)
            # =================================================================
            duration_attempt = 0
            max_acceptable_duration = self.target_duration * self.MAX_DURATION_MULTIPLIER

            while duration_attempt < self.MAX_DURATION_RETRIES:
                duration_attempt += 1

                # Generate voice
                step_name = "VoiceGenerator"
                if duration_attempt > 1:
                    self.display.info(f"Regenerating voice (attempt {duration_attempt})...")
                else:
                    description = STEP_DESCRIPTIONS.get(step_name, "Generating voice")
                    self.display.start_step(step_name, description)

                self.debug_logger.start_step(step_name)
                self._update_progress(step_name, step_counter / total_steps, "Generating voiceover...")

                context = await self.voice_step.execute(context)

                # Check audio duration and SET THE DURATION CONTRACT
                if context.audio_path and context.audio_path.exists():
                    actual_duration = self._get_audio_duration(context.audio_path)

                    # === CRITICAL: Set the duration contract ===
                    # This is the source of truth for video assembly and subtitle rendering
                    context.required_video_duration = actual_duration
                    self.display.info(f"Audio duration: {actual_duration:.1f}s (target: {self.target_duration}s)")
                    self.display.info("")
                    self.display.info("=" * 60)
                    self.display.info(f"  [Duration Contract] SET TO {actual_duration:.1f}s")
                    self.display.info("  This is the source of truth for all video steps")
                    self.display.info("=" * 60)
                    self.display.info("")

                    if actual_duration <= max_acceptable_duration:
                        self.debug_logger.end_step(step_name)
                        break
                    else:
                        self.display.warning(
                            f"Audio too long ({actual_duration:.1f}s > {max_acceptable_duration:.1f}s), "
                            "regenerating..."
                        )
                        # For news, we don't regenerate the script - just accept longer duration
                        # News content is factual and can't be arbitrarily shortened
                        self.display.info("Accepting longer duration for news content")
                        self.debug_logger.end_step(step_name)
                        break
                else:
                    # Fallback to target duration if no audio (shouldn't happen)
                    context.required_video_duration = self.target_duration
                    self.display.warning(f"[Duration Contract] No audio file - using target: {self.target_duration:.1f}s")
                    self.debug_logger.end_step(step_name)
                    break

            step_counter += 1

            # =================================================================
            # Phase 5: Parallel Video Search and Download
            # =================================================================
            step_name = "VideoSearcher"
            description = STEP_DESCRIPTIONS.get(step_name, "Searching videos")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Searching for videos...")

            context = await self.video_search_step.execute(context)
            self.display.info(f"Found {len(context.clips)} video clips")
            self.debug_logger.end_step(step_name)
            step_counter += 1

            step_name = "VideoDownloader"
            description = STEP_DESCRIPTIONS.get(step_name, "Downloading videos")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Downloading clips...")

            context = await self.video_download_step.execute(context)
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 6: Video Assembly
            # =================================================================
            step_name = "VideoAssembler" if not self.gpu_accelerate else "GPUVideoAssembler"
            description = STEP_DESCRIPTIONS.get(step_name, "Assembling video")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Assembling video...")

            context = await self.assembly_step.execute(context)
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 7: Thumbnail Generation (before subtitles)
            # =================================================================
            step_name = "ThumbnailGenerator"
            description = STEP_DESCRIPTIONS.get(step_name, "Generating thumbnail")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Generating thumbnail...")

            context = await self.thumbnail_step.execute(context)
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 8: Subtitle Rendering
            # =================================================================
            step_name = "SubtitleRenderer" if not self.gpu_accelerate else "GPUSubtitleRenderer"
            description = STEP_DESCRIPTIONS.get(step_name, "Rendering subtitles")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Rendering subtitles...")

            context = await self.subtitle_step.execute(context)
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Phase 9: Caption Generation
            # =================================================================
            step_name = "CaptionGenerator"
            description = STEP_DESCRIPTIONS.get(step_name, "Generating caption")
            self.display.start_step(step_name, description)
            self.debug_logger.start_step(step_name)
            self._update_progress(step_name, step_counter / total_steps, "Generating caption...")

            context = await self.caption_step.execute(context)
            self.debug_logger.end_step(step_name)
            step_counter += 1

            # =================================================================
            # Done
            # =================================================================
            self._update_progress("Complete", 1.0, "Video generation complete!")

            # Save metadata
            if context.metadata:
                metadata_out = output_dir / "metadata.json"
                with open(metadata_out, "w", encoding="utf-8") as f:
                    # Add news-specific metadata
                    metadata_dict = context.metadata.model_dump()
                    metadata_dict["news_brief"] = news_brief.to_dict() if news_brief else None
                    json.dump(metadata_dict, f, indent=2, default=str)

            # Copy final video to output
            if context.final_video_path:
                final_output = output_dir / "final.mp4"
                if context.final_video_path != final_output:
                    shutil.copy2(context.final_video_path, final_output)
                context.final_video_path = final_output

            self.display.end_pipeline(context.final_video_path, success=True)
            self.debug_logger.end(success=True, output_path=context.final_video_path)

            return context.final_video_path

        except Exception as e:
            self.display.error(f"Pipeline failed: {e}")
            self.debug_logger.end(success=False)
            raise PipelineError(f"News pipeline failed: {e}") from e

        finally:
            # Cleanup system temp directory
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

            # Cleanup project temp subfolder (ffmpeg temp files)
            if project_temp_subdir.exists():
                try:
                    shutil.rmtree(project_temp_subdir)
                    self.display.debug(f"Cleaned up temp folder: {project_temp_subdir.name}")
                except Exception:
                    pass

    def generate_sync(
        self,
        profile_path: Path,
        output_dir: Optional[Path] = None,
        post_id: Optional[str] = None,
    ) -> Path:
        """Synchronous wrapper for generate()."""
        return asyncio.run(self.generate(profile_path, output_dir, post_id))


# =============================================================================
# Convenience Functions
# =============================================================================

def is_news_profile(profile_path: Path) -> bool:
    """Check if a profile is configured for news content.

    Looks for 'news_sources' key in metadata.json.
    """
    metadata_path = profile_path / "metadata.json"
    if not metadata_path.exists():
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "news_sources" in data
    except Exception:
        return False
