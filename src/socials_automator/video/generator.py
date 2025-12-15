"""Main video generation orchestrator."""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from .assembler import VideoAssembler
from .config import VideoGeneratorConfig
from .models import (
    GenerationProgress,
    ScriptGenerationError,
    VideoGenerationError,
    VideoOutput,
    VideoScript,
)
from .stock_footage import StockFootageService
from .subtitles import SubtitleRenderer
from .tts import TTSGenerator

logger = logging.getLogger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[[GenerationProgress], None]


class VideoGenerator:
    """Main orchestrator for video generation pipeline.

    Pipeline:
    1. Generate script (AI) -> VideoScript
    2. Generate voiceover (edge-tts) -> audio + word timestamps
    3. Download stock footage (Pexels) -> video clips
    4. Assemble video (MoviePy) -> combined video
    5. Add subtitles (pycaps/MoviePy) -> final video
    """

    def __init__(
        self,
        config: Optional[VideoGeneratorConfig] = None,
        script_generator: Optional[Callable] = None,
    ):
        """Initialize video generator.

        Args:
            config: Video generation configuration.
            script_generator: Optional custom script generator function.
                Should take (topic: str, config: VideoGeneratorConfig)
                and return VideoScript.
        """
        self.config = config or VideoGeneratorConfig.default()
        self.script_generator = script_generator

        # Initialize components
        self.tts = TTSGenerator(self.config.tts)
        self.footage = StockFootageService(self.config.pexels)
        self.assembler = VideoAssembler(self.config.output)
        self.subtitles = SubtitleRenderer(self.config.subtitles)

    async def generate(
        self,
        topic: str,
        output_dir: Path,
        script: Optional[VideoScript] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> VideoOutput:
        """Generate complete video from topic.

        Args:
            topic: Topic for video content.
            output_dir: Directory for output files.
            script: Optional pre-generated script. If not provided,
                will generate using script_generator.
            progress_callback: Optional callback for progress updates.

        Returns:
            VideoOutput with all generated file paths.

        Raises:
            VideoGenerationError: If generation fails.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = output_dir / "clips"

        def update_progress(stage: str, progress: float, message: str, **kwargs):
            if progress_callback:
                progress_callback(
                    GenerationProgress(
                        stage=stage,
                        progress=progress,
                        message=message,
                        **kwargs,
                    )
                )

        try:
            # Stage 1: Get or generate script
            update_progress("script", 0.0, "Generating video script...")

            if script is None:
                if self.script_generator is None:
                    raise ScriptGenerationError(
                        "No script provided and no script_generator configured"
                    )
                script = await self._generate_script(topic)

            # Save script
            script_path = output_dir / "script.json"
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(script.model_dump(), f, indent=2)

            update_progress("script", 1.0, "Script generated")
            logger.info(f"Script: {script.title}")

            # Stage 2: Generate voiceover
            update_progress("tts", 0.0, "Generating voiceover...")

            voiceover = await self.tts.generate(
                text=script.full_narration,
                output_dir=output_dir,
                filename="voiceover",
            )

            update_progress("tts", 1.0, f"Voiceover: {voiceover.duration_seconds:.1f}s")
            logger.info(f"Voiceover duration: {voiceover.duration_seconds:.1f}s")

            # Stage 3: Download stock footage
            update_progress("footage", 0.0, "Downloading stock footage...")

            scenes = [
                (scene.video_keywords, scene.duration_seconds)
                for scene in script.scenes
            ]
            total_scenes = len(scenes)

            clips = []
            for i, (keywords, duration) in enumerate(scenes):
                update_progress(
                    "footage",
                    i / total_scenes,
                    f"Downloading scene {i + 1}/{total_scenes}...",
                    current_scene=i + 1,
                    total_scenes=total_scenes,
                )

                clip = await self.footage.find_video(
                    keywords=keywords,
                    target_duration=duration,
                    scene_index=i + 1,
                    output_dir=clips_dir,
                )
                clips.append(clip)

            update_progress("footage", 1.0, f"Downloaded {len(clips)} clips")
            logger.info(f"Downloaded {len(clips)} video clips")

            # Stage 4: Assemble video
            update_progress("assembly", 0.0, "Assembling video...")

            assembled_path = output_dir / "assembled.mp4"
            scene_durations = [scene.duration_seconds for scene in script.scenes]

            self.assembler.assemble(
                clips=clips,
                audio_path=voiceover.audio_path,
                output_path=assembled_path,
                scene_durations=scene_durations,
            )

            update_progress("assembly", 1.0, "Video assembled")
            logger.info(f"Assembled video: {assembled_path}")

            # Stage 5: Add subtitles
            update_progress("subtitles", 0.0, "Adding subtitles...")

            final_path = output_dir / "final.mp4"

            self.subtitles.render(
                video_path=assembled_path,
                srt_path=voiceover.srt_path,
                output_path=final_path,
            )

            update_progress("subtitles", 1.0, "Subtitles added")
            logger.info(f"Final video: {final_path}")

            # Stage 6: Create thumbnail
            update_progress("thumbnail", 0.0, "Creating thumbnail...")

            thumbnail_path = output_dir / "thumbnail.jpg"
            self.assembler.create_thumbnail(
                video_path=final_path,
                output_path=thumbnail_path,
                time=2.0,  # 2 seconds in
            )

            update_progress("thumbnail", 1.0, "Thumbnail created")

            # Complete
            update_progress("complete", 1.0, "Video generation complete!")

            return VideoOutput(
                script_path=script_path,
                audio_path=voiceover.audio_path,
                srt_path=voiceover.srt_path,
                clips_dir=clips_dir,
                assembled_path=assembled_path,
                final_path=final_path,
                thumbnail_path=thumbnail_path,
                duration_seconds=voiceover.duration_seconds,
                resolution=self.config.output.resolution,
            )

        except VideoGenerationError:
            raise
        except Exception as e:
            raise VideoGenerationError(f"Video generation failed: {e}") from e
        finally:
            await self.footage.close()

    async def _generate_script(self, topic: str) -> VideoScript:
        """Generate script using configured generator.

        Args:
            topic: Topic for video.

        Returns:
            Generated VideoScript.
        """
        if asyncio.iscoroutinefunction(self.script_generator):
            return await self.script_generator(topic, self.config)
        else:
            return self.script_generator(topic, self.config)

    def generate_sync(
        self,
        topic: str,
        output_dir: Path,
        script: Optional[VideoScript] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> VideoOutput:
        """Synchronous wrapper for generate().

        Args:
            topic: Topic for video content.
            output_dir: Directory for output files.
            script: Optional pre-generated script.
            progress_callback: Optional callback for progress updates.

        Returns:
            VideoOutput with all generated file paths.
        """
        return asyncio.run(
            self.generate(topic, output_dir, script, progress_callback)
        )

    async def generate_from_script(
        self,
        script: VideoScript,
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> VideoOutput:
        """Generate video from pre-made script.

        Args:
            script: Video script to use.
            output_dir: Directory for output files.
            progress_callback: Optional callback for progress updates.

        Returns:
            VideoOutput with all generated file paths.
        """
        return await self.generate(
            topic="",  # Not used when script is provided
            output_dir=output_dir,
            script=script,
            progress_callback=progress_callback,
        )


def create_output_directory(
    base_dir: Path,
    profile: str,
    post_id: Optional[str] = None,
) -> Path:
    """Create output directory for video files.

    Creates structure: base_dir/reels/YYYY/MM/generated/{post_id}/

    Args:
        base_dir: Base profile directory.
        profile: Profile name.
        post_id: Optional post ID. Auto-generated if not provided.

    Returns:
        Path to output directory.
    """
    now = datetime.now()

    if post_id is None:
        post_id = f"{now.day:02d}-{now.hour:02d}{now.minute:02d}"

    output_dir = (
        base_dir
        / "profiles"
        / profile
        / "reels"
        / str(now.year)
        / f"{now.month:02d}"
        / "generated"
        / post_id
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Example script generator for testing
def create_sample_script(topic: str, config: VideoGeneratorConfig) -> VideoScript:
    """Create a sample script for testing.

    Args:
        topic: Topic for the video.
        config: Video generator configuration.

    Returns:
        Sample VideoScript.
    """
    return VideoScript(
        title=f"5 Tips About {topic}",
        hook=f"Stop scrolling! Here are 5 things about {topic} you need to know.",
        scenes=[
            {
                "text": f"First, let's understand what {topic} really means.",
                "duration_seconds": 10,
                "video_keywords": ["technology", "abstract"],
            },
            {
                "text": "Tip number one: Start with the basics.",
                "duration_seconds": 10,
                "video_keywords": ["learning", "education"],
            },
            {
                "text": "Tip two: Practice makes perfect.",
                "duration_seconds": 10,
                "video_keywords": ["practice", "training"],
            },
            {
                "text": "Tip three: Don't be afraid to make mistakes.",
                "duration_seconds": 10,
                "video_keywords": ["growth", "progress"],
            },
            {
                "text": "Tip four: Learn from experts in the field.",
                "duration_seconds": 10,
                "video_keywords": ["expert", "professional"],
            },
        ],
        cta="Follow for more tips! Link in bio.",
        total_duration=config.target_duration,
    )
