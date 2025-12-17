"""Video generation pipeline orchestrator.

Coordinates all pipeline steps to generate a complete video:
1. Select topic from profile
2. Research topic via web search
3. Plan video script
4. [PARALLEL] Generate voiceover + Search/download stock videos
5. Assemble video
6. Render subtitles
7. Generate caption and hashtags
8. Output final video

Parallelization: After script planning, VoiceGenerator and VideoSearcher+VideoDownloader
run in parallel since both only depend on the script, not each other.
"""

import asyncio
import json
import logging
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
)
from .caption_generator import CaptionGenerator
from .cli_display import PipelineDisplay, setup_display
from .debug_logger import PipelineDebugLogger
from .script_planner import ScriptPlanner
from .subtitle_renderer import SubtitleRenderer
from .thumbnail_generator import ThumbnailGenerator
from .topic_researcher import TopicResearcher
from .topic_selector import TopicSelector
from .video_assembler import VideoAssembler
from .video_downloader import VideoDownloader
from .video_searcher import VideoSearcher
from .voice_generator import VoiceGenerator

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
    "TopicSelector": "Analyzing profile and selecting topic based on content pillars",
    "TopicResearcher": "Researching topic via web search for accurate content",
    "ScriptPlanner": "Planning video script with AI-generated narration",
    "VoiceGenerator": "Generating voiceover audio (determines final video duration)",
    "VideoSearcher": "Searching Pexels for relevant stock footage",
    "VideoDownloader": "Downloading video clips (with cache optimization)",
    "VideoAssembler": "Assembling clips into 9:16 vertical video",
    "GPUVideoAssembler": "Assembling clips into 9:16 vertical video (GPU NVENC)",
    "SubtitleRenderer": "Rendering karaoke-style subtitles and adding audio",
    "GPUSubtitleRenderer": "Rendering subtitles and adding audio (GPU NVENC)",
    "ThumbnailGenerator": "Generating thumbnail with hook text for Instagram",
    "CaptionGenerator": "Generating Instagram caption and hashtags with AI validation",
}


class VideoPipeline:
    """Orchestrates the complete video generation pipeline."""

    # Duration validation constants
    MAX_DURATION_MULTIPLIER = 1.5  # Accept up to 1.5x target duration (e.g., 60s target -> 90s max)
    MAX_DURATION_RETRIES = 10  # Maximum script regeneration attempts for duration

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file using ffprobe.

        Args:
            audio_path: Path to audio file.

        Returns:
            Duration in seconds.
        """
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
                timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.display.error(f"Failed to get audio duration: {e}")
            return 0.0

    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        voice: str = "rvc_adam",
        voice_rate: str = "+0%",
        voice_pitch: str = "+0Hz",
        text_ai: Optional[str] = None,
        video_matcher: str = "pexels",
        subtitle_size: int = 80,
        subtitle_font: str = "Montserrat-Bold.ttf",
        target_duration: float = 60.0,
        progress_callback: Optional[ProgressCallback] = None,
        show_timestamps: bool = True,
        verbose: bool = False,
        gpu_accelerate: bool = False,
        gpu_index: Optional[int] = None,
    ):
        """Initialize video pipeline.

        Args:
            pexels_api_key: Pexels API key. Reads from env if not provided.
            voice: Voice preset or name for TTS.
            voice_rate: Speech rate adjustment (e.g., '+12%' for excited).
            voice_pitch: Pitch adjustment (e.g., '+3Hz' for excited).
            text_ai: Text AI provider (lmstudio, openai, etc.). None uses templates.
            video_matcher: Video source ('pexels'). Reserved for future sources.
            subtitle_size: Subtitle font size in pixels (default 80).
            subtitle_font: Subtitle font file from /fonts folder (default Montserrat-Bold.ttf).
            target_duration: Target video duration in seconds (default 60).
            progress_callback: Optional callback for progress updates.
                Signature: (stage: str, progress: float, message: str)
            show_timestamps: Whether to show timestamps in CLI output.
            verbose: Whether to show debug messages.
            gpu_accelerate: Enable GPU acceleration with NVENC (requires NVIDIA GPU).
            gpu_index: GPU index to use (0, 1, etc.). Auto-selects if not specified.
        """
        self.logger = logging.getLogger("video.pipeline")
        self.progress_callback = progress_callback
        self.text_ai = text_ai
        self.video_matcher = video_matcher
        self.target_duration = target_duration
        self.voice = voice
        self.subtitle_size = subtitle_size
        self.subtitle_font = subtitle_font
        self.gpu_accelerate = gpu_accelerate
        self.gpu_index = gpu_index
        self.gpu_info: Optional[GPUInfo] = None

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

        # Create AI client if text_ai is specified
        self.ai_client = None
        if text_ai:
            self.ai_client = self._create_ai_client(text_ai)
            self.logger.info(f"Using text AI: {text_ai}")
            self.display.info(f"Text AI provider: {text_ai}")

        # Initialize pipeline steps
        # Structured for parallel execution after script planning

        # Sequential: Topic selection -> Research -> Script planning
        self.sequential_steps: list[PipelineStep] = [
            TopicSelector(ai_client=self.ai_client),
            TopicResearcher(),
            ScriptPlanner(ai_client=self.ai_client, target_duration=target_duration),
        ]

        # Parallel branch A: Voice generation (determines final duration)
        self.voice_step = VoiceGenerator(voice=voice, rate=voice_rate, pitch=voice_pitch)

        # Parallel branch B: Video search and download
        self.video_search_step = VideoSearcher(api_key=pexels_api_key, ai_client=self.ai_client)
        self.video_download_step = VideoDownloader()

        # Sequential: Assembly -> Thumbnail -> Subtitles -> Caption (need results from both parallel branches)
        # ThumbnailGenerator runs BEFORE SubtitleRenderer to get clean frames without text
        # Use GPU renderers if GPU acceleration is enabled
        if self.gpu_accelerate and self.gpu_info:
            self.display.info("Using GPU-accelerated video assembly and subtitle rendering (NVENC)")
            self.final_steps: list[PipelineStep] = [
                GPUVideoAssembler(gpu=self.gpu_info),  # GPU NVENC assembly
                ThumbnailGenerator(font=subtitle_font, font_size=72),  # Thumbnail from clean video (no subtitles)
                GPUSubtitleRenderer(gpu=self.gpu_info, font=subtitle_font, font_size=subtitle_size),  # GPU with ASS karaoke
                CaptionGenerator(ai_client=self.ai_client),
            ]
        else:
            self.final_steps: list[PipelineStep] = [
                VideoAssembler(),  # Uses audio duration as source of truth
                ThumbnailGenerator(font=subtitle_font, font_size=72),  # Thumbnail from clean video (no subtitles)
                SubtitleRenderer(font=subtitle_font, font_size=subtitle_size),
                CaptionGenerator(ai_client=self.ai_client),
            ]

        # Collect all steps for display setup and counting
        self.steps: list[PipelineStep] = (
            self.sequential_steps
            + [self.voice_step, self.video_search_step, self.video_download_step]
            + self.final_steps
        )

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
        step_counter = 0

        try:
            # Phase 1: Sequential steps - Topic selection and Research (NOT ScriptPlanner yet)
            for step in self.sequential_steps[:-1]:  # Skip ScriptPlanner, we'll run it in the duration loop
                step_name = step.name
                progress = step_counter / total_steps
                description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                self.display.start_step(step_name, description)
                self.debug_logger.start_step(step_name)
                self._update_progress(step_name, progress, f"Starting {step_name}...")

                context = await step.execute(context)

                # Log step-specific info
                if step_name == "TopicSelector" and context.topic:
                    self.display.show_topic(context.topic.topic, context.topic.pillar_name)
                    self.debug_logger.log_topic(
                        context.topic.topic,
                        context.topic.pillar_name,
                        context.topic.keywords,
                    )
                else:
                    self.debug_logger.end_step(step_name)

                step_counter += 1
                self._update_progress(step_name, step_counter / total_steps, f"Completed {step_name}")

            # Phase 1.5: Duration validation loop - ScriptPlanner + VoiceGenerator
            # Regenerate script if audio duration is off target
            script_planner = self.sequential_steps[-1]  # ScriptPlanner
            duration_attempt = 0
            duration_feedback: Optional[dict] = None  # Feedback for AI about duration adjustment

            while duration_attempt < self.MAX_DURATION_RETRIES:
                duration_attempt += 1

                # Run ScriptPlanner (with duration feedback if retrying)
                step_name = script_planner.name
                progress = step_counter / total_steps
                description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                if duration_attempt > 1:
                    self.display.info(f"Regenerating script (attempt {duration_attempt}/{self.MAX_DURATION_RETRIES}) - adjusting for duration...")
                    # Pass duration feedback to script planner
                    if hasattr(script_planner, 'duration_feedback'):
                        script_planner.duration_feedback = duration_feedback

                self.display.start_step(step_name, description)
                self.debug_logger.start_step(step_name)
                self._update_progress(step_name, progress, f"Starting {step_name}...")

                context = await script_planner.execute(context)

                if context.script:
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

                step_counter += 1
                self._update_progress(step_name, step_counter / total_steps, f"Completed {step_name}")

                # Run VoiceGenerator to get actual audio duration
                voice_step = self.voice_step
                step_name = voice_step.name
                description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                self.display.start_step(step_name, description)
                self.debug_logger.start_step(step_name)

                context = await voice_step.execute(context)

                self.debug_logger.log_voice(
                    getattr(voice_step, 'backend', 'unknown'),
                    getattr(voice_step, 'voice', 'unknown'),
                    len(getattr(context, '_word_timestamps', [])) if hasattr(context, '_word_timestamps') else 0,
                )
                self.debug_logger.end_step(step_name)

                # Check actual audio duration
                if context.audio_path and context.audio_path.exists():
                    actual_duration = self._get_audio_duration(context.audio_path)
                    max_duration = self.target_duration * self.MAX_DURATION_MULTIPLIER

                    self.display.info(f"Audio duration: {actual_duration:.1f}s (target: {self.target_duration:.1f}s, max: {max_duration:.0f}s)")

                    # Accept any duration under max (target * 1.5)
                    if actual_duration <= max_duration:
                        if actual_duration <= self.target_duration:
                            self.display.info(f"Duration OK ({actual_duration:.1f}s <= {self.target_duration:.0f}s target)")
                        else:
                            self.display.info(f"Duration acceptable ({actual_duration:.1f}s <= {max_duration:.0f}s max)")
                        break
                    else:
                        # Too long - need shorter script
                        over_by = actual_duration - max_duration
                        word_count = len(context.script.full_narration.split())
                        # Calculate target words to hit target duration (not max) for better results
                        target_ratio = self.target_duration / actual_duration
                        target_words = int(word_count * target_ratio)
                        duration_feedback = {
                            "issue": "too_long",
                            "actual_duration": actual_duration,
                            "target_duration": self.target_duration,
                            "max_duration": max_duration,
                            "current_words": word_count,
                            "target_words": target_words,
                            "message": f"Script is {over_by:.1f}s over the {max_duration:.0f}s limit. Reduce from {word_count} to ~{target_words} words to hit {self.target_duration:.0f}s.",
                        }
                        self.display.warning(f"Script too long ({actual_duration:.1f}s > {max_duration:.0f}s) - will regenerate shorter")

                        if duration_attempt >= self.MAX_DURATION_RETRIES:
                            self.display.warning(f"Max duration retries ({self.MAX_DURATION_RETRIES}) reached - using current script")
                            break

                        # Clean up audio files before regenerating
                        if context.audio_path and context.audio_path.exists():
                            context.audio_path.unlink()
                        if context.srt_path and context.srt_path.exists():
                            context.srt_path.unlink()
                        context.audio_path = None
                        context.srt_path = None
                else:
                    self.display.warning("Could not verify audio duration")
                    break

            step_counter += 1  # For voice step

            # Phase 2: Video search/download with validation loop
            self.display.info("Running video search and download...")

            async def video_branch(ctx: PipelineContext) -> PipelineContext:
                """Search and download video clips until we have enough unique coverage.

                Strategy:
                1. Deduplicate all clips by pexels_id
                2. Keep searching with AI-generated keywords until video duration > audio duration
                3. Merge new unique clips with existing collection
                4. Stop when we have enough or hit max retries
                """
                # Get audio duration using ffprobe (ctx has audio_path, not audio object)
                audio_duration = self._get_audio_duration(ctx.audio_path) if ctx.audio_path else 60.0
                max_retries = 10
                retry_count = 0

                # Track ALL unique clips by pexels_id (the source of truth)
                unique_clips_by_id: dict[int, object] = {}  # pexels_id -> VideoClipInfo
                all_used_keywords: set[str] = set()

                while retry_count < max_retries:
                    # Video search
                    search_step = self.video_search_step
                    step_name = search_step.name
                    description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                    if retry_count == 0:
                        self.display.start_step(step_name, description)
                        self.debug_logger.start_step(step_name)
                    else:
                        self.display.info(f"Searching for more unique videos (attempt {retry_count + 1}/{max_retries})...")

                    ctx = await search_step.execute(ctx)

                    search_results = getattr(ctx, '_search_results', [])
                    if search_results:
                        self.debug_logger.log_video_search(search_results)

                    if retry_count == 0:
                        self.debug_logger.end_step(step_name, {
                            "clips_found": len(search_results) if search_results else 0,
                        })

                    # Video download
                    download_step = self.video_download_step
                    step_name = download_step.name
                    description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                    if retry_count == 0:
                        self.display.start_step(step_name, description)
                        self.debug_logger.start_step(step_name)

                    ctx = await download_step.execute(ctx)

                    if retry_count == 0 and hasattr(download_step, '_cache_hits'):
                        hits = getattr(download_step, '_cache_hits', 0)
                        misses = getattr(download_step, '_cache_misses', 0)
                        self.display.show_cache_stats(hits, misses)
                        self.debug_logger.log_cache_stats(hits, misses)
                        self.debug_logger.end_step(step_name, {
                            "cache_hits": hits,
                            "cache_misses": misses,
                        })

                    # Merge new clips - deduplicate by pexels_id
                    new_clips_added = 0
                    for clip in ctx.clips:
                        if clip.pexels_id not in unique_clips_by_id:
                            unique_clips_by_id[clip.pexels_id] = clip
                            new_clips_added += 1
                            # Track keywords used
                            for kw in clip.keywords_used:
                                all_used_keywords.add(kw.lower())

                    # Calculate total unique video duration
                    all_clips = list(unique_clips_by_id.values())
                    total_video_duration = sum(clip.duration_seconds for clip in all_clips)

                    self.display.info(
                        f"Video coverage: {total_video_duration:.1f}s / {audio_duration:.1f}s needed "
                        f"({len(all_clips)} unique clips, +{new_clips_added} new)"
                    )

                    # Check if we have enough video
                    if total_video_duration >= audio_duration:
                        self.display.info(f"[OK] Sufficient unique video coverage ({total_video_duration:.1f}s >= {audio_duration:.1f}s)")
                        break

                    # Not enough - need more videos
                    retry_count += 1

                    if retry_count >= max_retries:
                        self.display.warning(
                            f"Max retries reached. Only {total_video_duration:.1f}s of unique video for {audio_duration:.1f}s audio. "
                            "Videos may be reused/looped."
                        )
                        break

                    # Calculate shortfall
                    shortfall = audio_duration - total_video_duration
                    self.display.info(f"Need {shortfall:.1f}s more unique video, generating new search keywords...")

                    # Ask AI for DIFFERENT keywords (avoiding all previously used)
                    if self.ai_client and ctx.script:
                        try:
                            used_kw_list = ', '.join(sorted(all_used_keywords)[:20])  # Limit to avoid huge prompt
                            prompt = f"""Generate stock video search keywords for a video about: {ctx.topic.topic if ctx.topic else 'technology'}

CRITICAL: Generate COMPLETELY DIFFERENT keywords. DO NOT use any of these already-used keywords:
{used_kw_list}

I need {shortfall:.0f} more seconds of unique video footage.

Generate 8 NEW 2-word search phrases that will find DIFFERENT videos.
Think creatively - try related concepts, different angles, abstract visuals.
Examples of good variety: "sunrise timelapse", "hands typing", "city aerial", "water droplets", "light particles"

Return ONLY the keywords, one per line, no numbers or bullets."""

                            response = await self.ai_client.generate(prompt)
                            new_keywords = [k.strip() for k in response.strip().split('\n') if k.strip() and k.lower() not in all_used_keywords]

                            # Update script segments with new keywords (spread across segments)
                            if new_keywords and ctx.script:
                                self.display.info(f"AI generated {len(new_keywords)} new keywords: {new_keywords[:3]}...")
                                for i, segment in enumerate(ctx.script.segments):
                                    # Assign different keywords to each segment
                                    kw_idx = i % len(new_keywords)
                                    segment.keywords = [new_keywords[kw_idx]] + new_keywords[(kw_idx + 1) % len(new_keywords):kw_idx + 3]
                        except Exception as e:
                            self.display.info(f"Could not generate new keywords: {e}")
                            # Fallback: use generic abstract keywords
                            fallback_keywords = ["abstract motion", "light effects", "digital waves", "gradient flow", "particle system"]
                            if ctx.script:
                                for i, segment in enumerate(ctx.script.segments):
                                    segment.keywords = [fallback_keywords[i % len(fallback_keywords)]]

                # Store all unique clips (sorted by segment index for consistent ordering)
                ctx.clips = sorted(all_clips, key=lambda c: c.segment_index)
                return ctx

            # Run video branch (voice already done in duration loop above)
            video_ctx = await video_branch(context)

            # Merge video results into context (audio already set in duration loop)
            context.clips = video_ctx.clips

            step_counter += 2  # search + download

            # Phase 3: Sequential final steps (Assembly, Subtitles, Caption)
            for step in self.final_steps:
                step_name = step.name
                progress = step_counter / total_steps
                description = STEP_DESCRIPTIONS.get(step_name, f"Executing {step_name}")

                self.display.start_step(step_name, description)
                self.debug_logger.start_step(step_name)
                self._update_progress(step_name, progress, f"Starting {step_name}...")

                context = await step.execute(context)

                self.debug_logger.end_step(step_name)
                step_counter += 1
                self._update_progress(step_name, step_counter / total_steps, f"Completed {step_name}")

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

                # Get the full message including args if present
                # ddgs logs with: logger.info("Error in engine %s: %r", engine.name, ex)
                if record.args:
                    # Extract engine from args (first arg is engine name)
                    engine = str(record.args[0]) if record.args else "unknown"
                    # Get the exception from args to determine error type
                    ex_str = str(record.args[1]) if len(record.args) > 1 else ""
                else:
                    msg = str(record.msg)
                    engine_match = re.search(r"Error in engine (\w+):", msg)
                    engine = engine_match.group(1) if engine_match else "unknown"
                    ex_str = msg

                # Determine error type from exception string
                ex_lower = ex_str.lower()
                if "timed out" in ex_lower or "timeout" in ex_lower:
                    error_type = "TIMEOUT"
                elif "403" in ex_str or "forbidden" in ex_lower:
                    error_type = "BLOCKED"
                elif "404" in ex_str:
                    error_type = "NOT FOUND"
                elif "connection" in ex_lower:
                    error_type = "CONNECTION"
                else:
                    error_type = "ERROR"

                # Create clean message and clear args to prevent formatting error
                record.msg = f"{engine:12} [{error_type}]"
                record.args = ()  # Clear args to prevent %-style formatting

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


class ColoredToolFormatter(logging.Formatter):
    """Formatter that adds colors to tool/logger names."""

    # ANSI color codes
    COLORS = {
        "primp": "\033[36m",       # Cyan - HTTP client
        "ddgs": "\033[33m",        # Yellow - DuckDuckGo
        "ddgs.ddgs": "\033[33m",   # Yellow - DuckDuckGo
        "moviepy": "\033[35m",     # Magenta - MoviePy
        "ffmpeg": "\033[32m",      # Green - FFmpeg
        "edge_tts": "\033[34m",    # Blue - Edge TTS
        "httpx": "\033[36m",       # Cyan - HTTP
        "pexels": "\033[32m",      # Green - Pexels
        "video.pipeline": "\033[37m",  # White - Pipeline
    }
    RESET = "\033[0m"

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Get color for this logger
        color = self.RESET
        for name_prefix, col in self.COLORS.items():
            if record.name.startswith(name_prefix):
                color = col
                break

        # Shorten logger name for display
        short_name = record.name.split(".")[-1]
        name_map = {
            "primp": "http",
            "ddgs": "search",
            "moviepy": "moviepy",
            "ffmpeg": "ffmpeg",
            "edge_tts": "tts",
            "httpx": "http",
            "pipeline": "pipeline",
        }
        for key, short in name_map.items():
            if key in record.name.lower():
                short_name = short
                break

        # Format with color
        original_name = record.name
        record.name = f"{color}{short_name:8}{self.RESET}"

        result = super().format(record)

        # Restore original name
        record.name = original_name
        return result


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging for video pipeline.

    Args:
        level: Logging level.
    """
    # Create colored formatter
    formatter = ColoredToolFormatter(
        fmt="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Setup root logger with colored output
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set specific loggers
    logging.getLogger("video.pipeline").setLevel(level)

    # Add filter to reformat primp (HTTP client) logs
    primp_logger = logging.getLogger("primp")
    primp_logger.addFilter(PrimpLogFilter())

    # Add filter to reformat ddgs (DuckDuckGo) error logs
    ddgs_logger = logging.getLogger("ddgs.ddgs")
    ddgs_logger.addFilter(DdgsLogFilter())
