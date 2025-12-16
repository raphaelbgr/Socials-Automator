"""Video assembly using MoviePy."""

import logging
import os
from pathlib import Path
from typing import Optional

from socials_automator.constants import get_temp_dir
from .config import OutputConfig
from .models import VideoAssemblyError, VideoClip

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assemble video clips with audio using MoviePy."""

    def __init__(self, config: Optional[OutputConfig] = None):
        """Initialize video assembler.

        Args:
            config: Output configuration. Uses defaults if not provided.
        """
        self.config = config or OutputConfig()
        # Set MoviePy temp directory to avoid files in project root
        self._setup_moviepy_temp()

    def _setup_moviepy_temp(self) -> None:
        """Configure MoviePy to use project temp directory."""
        temp_dir = get_temp_dir()
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)

    def _get_temp_audiofile_path(self, output_path: Path) -> str:
        """Get temp audio file path for MoviePy write operation."""
        temp_dir = get_temp_dir()
        temp_audio = temp_dir / f"{output_path.stem}_TEMP_audio.mp3"
        return str(temp_audio)

    def assemble(
        self,
        clips: list[VideoClip],
        audio_path: Path,
        output_path: Path,
        scene_durations: Optional[list[float]] = None,
    ) -> Path:
        """Assemble video clips with audio.

        Args:
            clips: List of VideoClip objects to concatenate.
            audio_path: Path to audio file.
            output_path: Path for output video.
            scene_durations: Optional list of duration for each scene.
                If not provided, divides total duration equally.

        Returns:
            Path to assembled video.

        Raises:
            VideoAssemblyError: If assembly fails.
        """
        if not clips:
            raise VideoAssemblyError("No clips provided for assembly")

        try:
            from moviepy.editor import (
                AudioFileClip,
                VideoFileClip,
                concatenate_videoclips,
            )
        except ImportError as e:
            raise VideoAssemblyError(
                "moviepy is not installed. Run: pip install moviepy"
            ) from e

        try:
            # Load audio to get duration
            audio = AudioFileClip(str(audio_path))
            total_duration = min(audio.duration, self.config.duration)

            # Calculate scene durations
            if scene_durations is None:
                clip_duration = total_duration / len(clips)
                scene_durations = [clip_duration] * len(clips)
            else:
                # Normalize durations to fit total
                scale = total_duration / sum(scene_durations)
                scene_durations = [d * scale for d in scene_durations]

            logger.info(
                f"Assembling {len(clips)} clips, total duration: {total_duration:.1f}s"
            )

            # Process each clip
            video_clips = []
            for clip_info, duration in zip(clips, scene_durations):
                clip = VideoFileClip(str(clip_info.path))

                # Crop to 9:16 if needed
                clip = self._crop_to_9_16(clip)

                # Resize to target resolution
                clip = clip.resize(self.config.resolution)

                # Set duration (loop if too short, trim if too long)
                clip = self._adjust_duration(clip, duration)

                video_clips.append(clip)

            # Concatenate all clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            # Add audio
            final_video = final_video.set_audio(audio.subclip(0, total_duration))

            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting to: {output_path}")
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec=self.config.codec,
                audio_codec=self.config.audio_codec,
                bitrate=self.config.bitrate,
                logger=None,  # Suppress moviepy progress bar
                temp_audiofile=self._get_temp_audiofile_path(output_path),
            )

            # Cleanup
            final_video.close()
            audio.close()
            for clip in video_clips:
                clip.close()

            logger.info(f"Assembly complete: {output_path}")
            return output_path

        except Exception as e:
            raise VideoAssemblyError(f"Video assembly failed: {e}") from e

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
            # Already correct ratio
            return clip

        if current_ratio > target_ratio:
            # Video is too wide, crop width
            new_width = int(h * target_ratio)
            x_center = w // 2
            clip = clip.crop(
                x1=x_center - new_width // 2,
                x2=x_center + new_width // 2,
            )
        else:
            # Video is too tall, crop height
            new_height = int(w / target_ratio)
            y_center = h // 2
            clip = clip.crop(
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
            # Clip is longer, select best segment (prefer middle)
            start_time = (clip.duration - target_duration) / 2
            return clip.subclip(start_time, start_time + target_duration)
        else:
            # Clip is shorter, loop it
            if clip.duration < target_duration * 0.5:
                # Much shorter, loop to fill
                return clip.loop(duration=target_duration)
            else:
                # Close enough, just use as-is with freeze frame at end
                from moviepy.editor import concatenate_videoclips

                freeze_duration = target_duration - clip.duration
                freeze = clip.to_ImageClip(t=clip.duration - 0.1).set_duration(
                    freeze_duration
                )
                return concatenate_videoclips([clip, freeze])

    def create_thumbnail(
        self,
        video_path: Path,
        output_path: Path,
        time: float = 0.5,
    ) -> Path:
        """Extract thumbnail from video.

        Args:
            video_path: Path to video file.
            output_path: Path for thumbnail image.
            time: Time in seconds to extract frame.

        Returns:
            Path to thumbnail image.
        """
        try:
            from moviepy.editor import VideoFileClip
        except ImportError as e:
            raise VideoAssemblyError(
                "moviepy is not installed. Run: pip install moviepy"
            ) from e

        try:
            clip = VideoFileClip(str(video_path))

            # Get frame at specified time
            frame_time = min(time, clip.duration - 0.1)
            frame = clip.get_frame(frame_time)

            # Save as image
            from PIL import Image

            img = Image.fromarray(frame)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path), quality=95)

            clip.close()
            logger.info(f"Thumbnail saved: {output_path}")
            return output_path

        except Exception as e:
            raise VideoAssemblyError(f"Thumbnail creation failed: {e}") from e


def select_clip_segment(
    clip_path: Path,
    needed_duration: float,
) -> tuple[float, float]:
    """Calculate best segment from a longer clip.

    Args:
        clip_path: Path to video file.
        needed_duration: Desired duration in seconds.

    Returns:
        Tuple of (start_time, end_time) in seconds.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        raise VideoAssemblyError(
            "moviepy is not installed. Run: pip install moviepy"
        ) from e

    clip = VideoFileClip(str(clip_path))
    clip_duration = clip.duration
    clip.close()

    if clip_duration <= needed_duration:
        return (0, clip_duration)

    # Prefer middle section (usually more interesting)
    start_time = (clip_duration - needed_duration) / 2
    return (start_time, start_time + needed_duration)


def get_video_info(video_path: Path) -> dict:
    """Get video file information.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary with video information.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        raise VideoAssemblyError(
            "moviepy is not installed. Run: pip install moviepy"
        ) from e

    clip = VideoFileClip(str(video_path))
    info = {
        "duration": clip.duration,
        "width": clip.size[0],
        "height": clip.size[1],
        "fps": clip.fps,
        "aspect_ratio": clip.size[0] / clip.size[1],
    }
    clip.close()
    return info
