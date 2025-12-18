"""Video assembly using MoviePy.

Assembles downloaded clips into a video with:
- Proper 9:16 cropping
- 1080x1920 resolution
- Duration matching the narration audio (source of truth)
- Metadata output (SRT-like structure)
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress MoviePy frame read warnings (non-critical, uses last valid frame)
warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")

from socials_automator.constants import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    get_temp_dir,
)
from .base import (
    ArtifactStatus,
    ArtifactsInfo,
    IVideoAssembler,
    PipelineContext,
    VideoAssemblyError,
    VideoClipInfo,
    VideoMetadata,
    VideoScript,
)


class VideoAssembler(IVideoAssembler):
    """Assembles video clips into final video."""

    # Target output settings (from constants)
    WIDTH = VIDEO_WIDTH
    HEIGHT = VIDEO_HEIGHT
    FPS = VIDEO_FPS

    def __init__(self):
        """Initialize video assembler."""
        super().__init__()
        # Set MoviePy temp directory to avoid files in project root
        self._setup_moviepy_temp()

    def _setup_moviepy_temp(self) -> None:
        """Configure MoviePy to use project temp directory."""
        temp_dir = get_temp_dir()
        # Set environment variables for temp files
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)

    def _get_temp_audiofile_path(self, output_path: Path) -> str:
        """Get temp audio file path for MoviePy write operation.

        Args:
            output_path: The output video path.

        Returns:
            Path string for temp audio file.
        """
        temp_dir = get_temp_dir()
        # Use .m4a extension for AAC codec compatibility (not .mp3)
        temp_audio = temp_dir / f"{output_path.stem}_TEMP_audio.m4a"
        return str(temp_audio)

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            from moviepy import AudioFileClip
        except ImportError:
            from moviepy.editor import AudioFileClip

        audio = AudioFileClip(str(audio_path))
        duration = audio.duration
        audio.close()
        return duration

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute video assembly step.

        Args:
            context: Pipeline context with clips, script, and audio.

        Returns:
            Updated context with assembled video.
        """
        if not context.clips:
            raise VideoAssemblyError("No clips available for assembly")
        if not context.script:
            raise VideoAssemblyError("No script available for assembly")
        if not context.audio_path:
            raise VideoAssemblyError("No audio available - narration is source of truth for duration")

        # Get actual audio duration - this is the source of truth for video length
        audio_duration = self._get_audio_duration(context.audio_path)
        self.log_progress(f"Audio duration: {audio_duration:.1f}s (source of truth)")

        self.log_start(f"Assembling {len(context.clips)} clips to match {audio_duration:.1f}s narration")

        try:
            output_path = context.temp_dir / "assembled.mp4"

            assembled_path, metadata = await self.assemble(
                context.clips,
                context.script,
                output_path,
                audio_duration,
            )

            context.assembled_video_path = assembled_path
            context.metadata = metadata

            # Save metadata to file
            metadata_path = context.temp_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2, default=str)

            self.log_success(f"Assembled video: {assembled_path} ({audio_duration:.1f}s)")
            return context

        except Exception as e:
            self.log_error(f"Video assembly failed: {e}")
            raise VideoAssemblyError(f"Failed to assemble video: {e}") from e

    async def assemble(
        self,
        clips: list[VideoClipInfo],
        script: VideoScript,
        output_path: Path,
        target_duration: float,
    ) -> tuple[Path, VideoMetadata]:
        """Assemble clips into final video with metadata.

        Args:
            clips: List of video clips.
            script: Video script with timing.
            output_path: Output video path.
            target_duration: Target video duration (from audio).

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

        self.log_detail("Loading and processing clips...")

        # Deduplicate clips by pexels_id (keep first occurrence, preserve segment order)
        seen_pexels_ids = set()
        unique_clips = []
        for clip in clips:
            if clip.pexels_id not in seen_pexels_ids:
                seen_pexels_ids.add(clip.pexels_id)
                unique_clips.append(clip)
            else:
                self.log_detail(f"Skipping duplicate pexels_id={clip.pexels_id} for segment {clip.segment_index}")

        if len(unique_clips) < len(clips):
            self.log_progress(f"Deduplicated: {len(clips)} clips -> {len(unique_clips)} unique clips")

        # Sort clips by segment index
        sorted_clips = sorted(unique_clips, key=lambda c: c.segment_index)

        # Process clips and keep adding until we have enough footage
        # to cover the entire audio duration
        video_clips = []
        segment_metadata = []
        current_duration = 0.0
        clip_index = 0

        self.log_detail(f"Need {target_duration:.1f}s of video to match narration")

        while current_duration < target_duration and clip_index < len(sorted_clips):
            clip_info = sorted_clips[clip_index]
            self.log_detail(f"Processing clip {clip_index + 1}/{len(sorted_clips)}...")

            clip = VideoFileClip(str(clip_info.path))
            original_duration = clip.duration

            # Crop to 9:16
            clip = self._crop_to_9_16(clip)

            # Resize to target resolution
            if hasattr(clip, 'resized'):
                clip = clip.resized((self.WIDTH, self.HEIGHT))
            else:
                clip = clip.resize((self.WIDTH, self.HEIGHT))

            # Calculate how much time we still need
            remaining_time = target_duration - current_duration

            if clip.duration > remaining_time:
                # Clip is longer than needed - trim it to exactly fill remaining time
                self.log_detail(f"  Trimming clip from {clip.duration:.1f}s to {remaining_time:.1f}s")
                if hasattr(clip, 'subclipped'):
                    clip = clip.subclipped(0, remaining_time)
                else:
                    clip = clip.subclip(0, remaining_time)
                clip_duration = remaining_time
            else:
                # Use full clip
                clip_duration = clip.duration
                self.log_detail(f"  Using full clip: {clip_duration:.1f}s")

            video_clips.append(clip)

            # Track segment timing for metadata
            segment_metadata.append({
                "index": clip_info.segment_index,
                "start_time": current_duration,
                "end_time": current_duration + clip_duration,
                "duration": clip_duration,
                "pexels_id": clip_info.pexels_id,
                "source_url": clip_info.source_url,
                "keywords": clip_info.keywords_used,
            })

            current_duration += clip_duration
            clip_index += 1
            self.log_detail(f"  Total video so far: {current_duration:.1f}s / {target_duration:.1f}s")

        # Check if we have enough footage - KEEP LOOPING until we fill the audio duration
        if current_duration < target_duration:
            remaining_time = target_duration - current_duration
            self.log_progress(f"[WARNING] Not enough unique video ({current_duration:.1f}s) for audio ({target_duration:.1f}s)")
            self.log_progress(f"[WARNING] Will reuse clips to fill {remaining_time:.1f}s gap - videos may repeat!")

            # Keep adding clips until we reach the target duration
            extend_index = 0
            max_iterations = 20  # Safety limit
            reused_ids = set()

            while current_duration < target_duration and extend_index < max_iterations:
                # Rotate through clips, starting from middle to avoid obvious repetition
                # Skip first and last clips on first pass to reduce visible repetition
                if len(sorted_clips) >= 3:
                    # Start from middle, then rotate through all
                    offset = len(sorted_clips) // 2
                    clip_idx = (extend_index + offset) % len(sorted_clips)
                else:
                    clip_idx = extend_index % len(sorted_clips)

                reuse_clip_info = sorted_clips[clip_idx]
                clip = VideoFileClip(str(reuse_clip_info.path))

                # Crop and resize
                clip = self._crop_to_9_16(clip)
                if hasattr(clip, 'resized'):
                    clip = clip.resized((self.WIDTH, self.HEIGHT))
                else:
                    clip = clip.resize((self.WIDTH, self.HEIGHT))

                remaining_time = target_duration - current_duration

                # Use different sections of the clip on each reuse to reduce visual repetition
                # First use: start from 0, second use: start from 1/3, third use: start from 2/3
                section_offset = (extend_index % 3) * (clip.duration / 3)
                section_offset = min(section_offset, max(0, clip.duration - remaining_time))

                if clip.duration > remaining_time:
                    # Clip is longer than needed - use section to fill remaining time
                    end_time = min(section_offset + remaining_time, clip.duration)
                    if hasattr(clip, 'subclipped'):
                        clip = clip.subclipped(section_offset, end_time)
                    else:
                        clip = clip.subclip(section_offset, end_time)
                    clip_duration = end_time - section_offset
                else:
                    # Use full clip
                    clip_duration = clip.duration

                video_clips.append(clip)
                current_duration += clip_duration
                extend_index += 1
                self.log_detail(f"  Added clip {clip_idx + 1} ({clip_duration:.1f}s) -> total: {current_duration:.1f}s / {target_duration:.1f}s")

        self.log_progress(f"Processing {len(video_clips)} clips -> {current_duration:.1f}s total")
        self.log_detail("Concatenating clips...")

        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Trim video to match audio duration exactly
        if final_video.duration > target_duration:
            self.log_progress(f"Trimming video from {final_video.duration:.1f}s to {target_duration:.1f}s")
            if hasattr(final_video, 'subclipped'):
                final_video = final_video.subclipped(0, target_duration)
            else:
                final_video = final_video.subclip(0, target_duration)

        self.log_progress(f"Exporting {target_duration:.1f}s video to {output_path}...")

        # Export without audio (audio will be added in subtitle renderer step)
        # Optimized FFmpeg settings for faster encoding
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            fps=self.FPS,
            codec="libx264",
            audio=False,
            preset="medium",  # Better quality (was "fast")
            logger=None,
            ffmpeg_params=["-crf", "18"],  # High quality (was 26)
        )

        # Cleanup
        final_video.close()
        for clip in video_clips:
            clip.close()

        # Create metadata with artifact tracking
        artifacts = ArtifactsInfo(
            video=ArtifactStatus(
                status="ok",
                file=str(output_path.name),
            ),
            # Other artifacts will be set by their respective pipeline steps
            voiceover=ArtifactStatus(status="pending"),
            subtitles=ArtifactStatus(status="pending"),
            thumbnail=ArtifactStatus(status="pending"),
            caption=ArtifactStatus(status="pending"),
            hashtags=ArtifactStatus(status="pending"),
        )

        metadata = VideoMetadata(
            post_id=output_path.parent.name,
            title=script.title,
            topic=script.title,
            duration_seconds=target_duration,
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
            artifacts=artifacts,
        )

        return output_path, metadata

    def _detect_black_bars(self, clip, threshold: int = 30, min_bar_ratio: float = 0.02) -> tuple[int, int, int, int]:
        """Detect black bars in video by sampling frames.

        Analyzes edges of frames to find consistent black bars (pillarbox or letterbox).

        Args:
            clip: MoviePy VideoFileClip.
            threshold: Maximum pixel value to consider "black" (0-255).
            min_bar_ratio: Minimum bar size as ratio of dimension to detect.

        Returns:
            Tuple of (left, top, right, bottom) crop values in pixels.
            Values represent how much to crop from each edge.
        """
        import numpy as np

        w, h = clip.size

        # Sample frames at 25%, 50%, 75% of video duration
        sample_times = [clip.duration * p for p in [0.25, 0.5, 0.75]]
        sample_times = [min(t, clip.duration - 0.1) for t in sample_times]

        # Collect bar measurements from each sample
        left_bars = []
        right_bars = []
        top_bars = []
        bottom_bars = []

        for t in sample_times:
            try:
                frame = clip.get_frame(t)
            except Exception:
                continue

            # Convert to grayscale for simpler analysis
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame

            # Detect left black bar
            left = 0
            for x in range(w // 4):  # Check up to 25% of width
                col_mean = np.mean(gray[:, x])
                if col_mean > threshold:
                    break
                left = x + 1
            left_bars.append(left)

            # Detect right black bar
            right = 0
            for x in range(w - 1, w - w // 4 - 1, -1):
                col_mean = np.mean(gray[:, x])
                if col_mean > threshold:
                    break
                right = w - x
            right_bars.append(right)

            # Detect top black bar
            top = 0
            for y in range(h // 4):  # Check up to 25% of height
                row_mean = np.mean(gray[y, :])
                if row_mean > threshold:
                    break
                top = y + 1
            top_bars.append(top)

            # Detect bottom black bar
            bottom = 0
            for y in range(h - 1, h - h // 4 - 1, -1):
                row_mean = np.mean(gray[y, :])
                if row_mean > threshold:
                    break
                bottom = h - y
            bottom_bars.append(bottom)

        if not left_bars:
            return (0, 0, 0, 0)

        # Use minimum detected bar size (most conservative)
        # This avoids cropping actual content
        left = min(left_bars)
        right = min(right_bars)
        top = min(top_bars)
        bottom = min(bottom_bars)

        # Only return bar values that exceed minimum ratio
        min_horizontal = int(w * min_bar_ratio)
        min_vertical = int(h * min_bar_ratio)

        left = left if left >= min_horizontal else 0
        right = right if right >= min_horizontal else 0
        top = top if top >= min_vertical else 0
        bottom = bottom if bottom >= min_vertical else 0

        return (left, top, right, bottom)

    def _crop_to_9_16(self, clip):
        """Center crop video to 9:16 aspect ratio, removing black bars first.

        Args:
            clip: MoviePy VideoFileClip.

        Returns:
            Cropped clip.
        """
        w, h = clip.size

        # Use cropped for MoviePy 2.x, crop for 1.x
        crop_method = clip.cropped if hasattr(clip, 'cropped') else clip.crop

        # Step 1: Detect and remove black bars
        left, top, right, bottom = self._detect_black_bars(clip)

        if left > 0 or top > 0 or right > 0 or bottom > 0:
            self.log_detail(f"  Removing black bars: L={left} T={top} R={right} B={bottom}")
            clip = crop_method(
                x1=left,
                y1=top,
                x2=w - right,
                y2=h - bottom,
            )
            w, h = clip.size
            # Need to refresh crop_method after modification
            crop_method = clip.cropped if hasattr(clip, 'cropped') else clip.crop

        # Step 2: Crop to 9:16 aspect ratio
        current_ratio = w / h
        target_ratio = 9 / 16

        if abs(current_ratio - target_ratio) < 0.01:
            return clip

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
        """Adjust clip duration to target by trimming.

        Args:
            clip: MoviePy VideoFileClip.
            target_duration: Target duration in seconds.

        Returns:
            Adjusted clip with correct duration.
        """
        if clip.duration >= target_duration:
            # Clip is longer - select middle segment for better visuals
            start_time = (clip.duration - target_duration) / 2
            if hasattr(clip, 'subclipped'):
                return clip.subclipped(start_time, start_time + target_duration)
            else:
                return clip.subclip(start_time, start_time + target_duration)
        else:
            # Clip is shorter than needed - use the full clip
            # The video searcher should find long enough clips, but if not,
            # we just use what we have (no looping, no slow motion)
            self.log_progress(
                f"Clip is {clip.duration:.1f}s but need {target_duration:.1f}s - using full clip"
            )
            return clip
