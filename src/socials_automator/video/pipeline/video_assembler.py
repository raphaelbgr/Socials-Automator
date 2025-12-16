"""Video assembly using MoviePy.

Assembles downloaded clips into a 1-minute video with:
- Proper 9:16 cropping
- 1080x1920 resolution
- Segment timing
- Metadata output (SRT-like structure)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import (
    IVideoAssembler,
    PipelineContext,
    VideoAssemblyError,
    VideoClipInfo,
    VideoMetadata,
    VideoScript,
)


class VideoAssembler(IVideoAssembler):
    """Assembles video clips into final video."""

    # Target output settings
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30
    DURATION = 60.0  # Fixed 1-minute video duration - audio/subtitles adapt to this

    def __init__(self):
        """Initialize video assembler."""
        super().__init__()

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute video assembly step.

        Args:
            context: Pipeline context with clips and script.

        Returns:
            Updated context with assembled video.
        """
        if not context.clips:
            raise VideoAssemblyError("No clips available for assembly")
        if not context.script:
            raise VideoAssemblyError("No script available for assembly")

        self.log_start(f"Assembling {len(context.clips)} clips into 1-minute video")

        try:
            output_path = context.temp_dir / "assembled.mp4"

            assembled_path, metadata = await self.assemble(
                context.clips,
                context.script,
                output_path,
            )

            context.assembled_video_path = assembled_path
            context.metadata = metadata

            # Save metadata to file
            metadata_path = context.temp_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2, default=str)

            self.log_success(f"Assembled video: {assembled_path}")
            return context

        except Exception as e:
            self.log_error(f"Video assembly failed: {e}")
            raise VideoAssemblyError(f"Failed to assemble video: {e}") from e

    async def assemble(
        self,
        clips: list[VideoClipInfo],
        script: VideoScript,
        output_path: Path,
    ) -> tuple[Path, VideoMetadata]:
        """Assemble clips into final video with metadata.

        Args:
            clips: List of video clips.
            script: Video script with timing.
            output_path: Output video path.

        Returns:
            Tuple of (video_path, metadata).
        """
        try:
            # MoviePy 2.x imports
            from moviepy import (
                VideoFileClip,
                concatenate_videoclips,
            )
        except ImportError:
            try:
                # Fallback to MoviePy 1.x imports
                from moviepy.editor import (
                    VideoFileClip,
                    concatenate_videoclips,
                )
            except ImportError as e:
                raise VideoAssemblyError(
                    "moviepy is not installed. Run: pip install moviepy"
                ) from e

        self.log_progress("Loading and processing clips...")

        # Sort clips by segment index
        sorted_clips = sorted(clips, key=lambda c: c.segment_index)

        # Video is ALWAYS 60 seconds - narration must fill this time
        # Use segment durations from script (validated to fill ~60s)
        segment_durations = [s.duration_seconds for s in script.segments]

        # Scale segment durations to exactly fill 60 seconds
        total_segment_time = sum(segment_durations)
        if total_segment_time > 0:
            scale = self.DURATION / total_segment_time
            segment_durations = [d * scale for d in segment_durations]
            self.log_progress(f"Scaled {len(segment_durations)} segments to fill {self.DURATION}s")

        # Process each clip
        video_clips = []
        segment_metadata = []
        current_time = 0.0

        for clip_info, duration in zip(sorted_clips, segment_durations):
            self.log_progress(f"Processing segment {clip_info.segment_index}...")

            clip = VideoFileClip(str(clip_info.path))

            # Crop to 9:16
            clip = self._crop_to_9_16(clip)

            # Resize to target resolution (resized for MoviePy 2.x, resize for 1.x)
            if hasattr(clip, 'resized'):
                clip = clip.resized((self.WIDTH, self.HEIGHT))
            else:
                clip = clip.resize((self.WIDTH, self.HEIGHT))

            # Adjust duration
            clip = self._adjust_duration(clip, duration)

            video_clips.append(clip)

            # Track segment timing for metadata
            segment_metadata.append({
                "index": clip_info.segment_index,
                "start_time": current_time,
                "end_time": current_time + duration,
                "duration": duration,
                "pexels_id": clip_info.pexels_id,
                "source_url": clip_info.source_url,
                "keywords": clip_info.keywords_used,
            })

            current_time += duration

        self.log_progress("Concatenating clips...")

        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Ensure video is exactly 60 seconds
        if final_video.duration > self.DURATION:
            if hasattr(final_video, 'subclipped'):
                final_video = final_video.subclipped(0, self.DURATION)
            else:
                final_video = final_video.subclip(0, self.DURATION)

        self.log_progress(f"Exporting to {output_path}...")

        # Export without audio (audio will be added in voice generation step)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            fps=self.FPS,
            codec="libx264",
            audio=False,
            logger=None,
        )

        # Cleanup
        final_video.close()
        for clip in video_clips:
            clip.close()

        # Create metadata
        metadata = VideoMetadata(
            post_id=output_path.parent.name,
            title=script.title,
            topic=script.title,
            duration_seconds=self.DURATION,
            segments=segment_metadata,
            clips_used=[
                {
                    "segment_index": c.segment_index,
                    "pexels_id": c.pexels_id,
                    "source_url": c.source_url,
                    "title": c.title,
                }
                for c in sorted_clips
            ],
            narration=script.full_narration,
        )

        return output_path, metadata

    def _crop_to_9_16(self, clip):
        """Center crop video to 9:16 aspect ratio.

        Args:
            clip: MoviePy VideoFileClip.

        Returns:
            Cropped clip.
        """
        w, h = clip.size
        current_ratio = w / h
        target_ratio = 9 / 16

        if abs(current_ratio - target_ratio) < 0.01:
            return clip

        # Use cropped for MoviePy 2.x, crop for 1.x
        crop_method = clip.cropped if hasattr(clip, 'cropped') else clip.crop

        if current_ratio > target_ratio:
            # Video is too wide, crop width
            new_width = int(h * target_ratio)
            x_center = w // 2
            clip = crop_method(
                x1=x_center - new_width // 2,
                x2=x_center + new_width // 2,
            )
        else:
            # Video is too tall, crop height
            new_height = int(w / target_ratio)
            y_center = h // 2
            clip = crop_method(
                y1=y_center - new_height // 2,
                y2=y_center + new_height // 2,
            )

        return clip

    def _adjust_duration(self, clip, target_duration: float):
        """Adjust clip duration to target.

        Args:
            clip: MoviePy VideoFileClip.
            target_duration: Target duration in seconds.

        Returns:
            Adjusted clip.
        """
        if clip.duration >= target_duration:
            # Clip is longer, select middle segment
            start_time = (clip.duration - target_duration) / 2
            # Use subclipped for MoviePy 2.x, subclip for 1.x
            if hasattr(clip, 'subclipped'):
                return clip.subclipped(start_time, start_time + target_duration)
            else:
                return clip.subclip(start_time, start_time + target_duration)
        else:
            # Clip is shorter, loop it
            try:
                # MoviePy 2.x uses vfx.Loop
                from moviepy import vfx
                n_loops = int(target_duration / clip.duration) + 1
                looped = clip.with_effects([vfx.Loop(n_loops)])
                # Trim to exact duration
                if hasattr(looped, 'subclipped'):
                    return looped.subclipped(0, target_duration)
                else:
                    return looped.subclip(0, target_duration)
            except (ImportError, AttributeError):
                # Fallback to MoviePy 1.x loop method
                return clip.loop(duration=target_duration)
