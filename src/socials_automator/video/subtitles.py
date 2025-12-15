"""Subtitle rendering for karaoke-style animated subtitles."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import SubtitleError, SubtitleStyle, WordTimestamp

logger = logging.getLogger(__name__)


@dataclass
class SRTEntry:
    """Single SRT subtitle entry."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0


def parse_srt(srt_path: Path) -> list[SRTEntry]:
    """Parse SRT file into entries.

    Args:
        srt_path: Path to SRT file.

    Returns:
        List of SRT entries.
    """
    entries = []

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (entry separator)
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            times = lines[1]
            text = " ".join(lines[2:])

            # Parse timestamp: 00:00:00,000 --> 00:00:00,000
            match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                times,
            )

            if match:
                start_ms = (
                    int(match.group(1)) * 3600000
                    + int(match.group(2)) * 60000
                    + int(match.group(3)) * 1000
                    + int(match.group(4))
                )
                end_ms = (
                    int(match.group(5)) * 3600000
                    + int(match.group(6)) * 60000
                    + int(match.group(7)) * 1000
                    + int(match.group(8))
                )

                entries.append(
                    SRTEntry(
                        index=index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text,
                    )
                )
        except (ValueError, IndexError):
            continue

    return entries


def word_timestamps_to_srt(
    timestamps: list[WordTimestamp],
    output_path: Path,
) -> Path:
    """Convert word timestamps to SRT format.

    Args:
        timestamps: List of word timestamps.
        output_path: Path to save SRT file.

    Returns:
        Path to SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, ts in enumerate(timestamps, 1):
            start = _ms_to_srt_time(ts.start_ms)
            end = _ms_to_srt_time(ts.end_ms)
            f.write(f"{i}\n{start} --> {end}\n{ts.word}\n\n")

    return output_path


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format.

    Args:
        ms: Time in milliseconds.

    Returns:
        SRT timestamp string (HH:MM:SS,mmm).
    """
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class SubtitleRenderer:
    """Render karaoke-style subtitles on video."""

    def __init__(self, style: Optional[SubtitleStyle] = None):
        """Initialize subtitle renderer.

        Args:
            style: Subtitle styling configuration.
        """
        self.style = style or SubtitleStyle()

    def render(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Add karaoke-style subtitles to video.

        Tries pycaps first, falls back to MoviePy TextClip.

        Args:
            video_path: Path to input video.
            srt_path: Path to SRT file with word timestamps.
            output_path: Path for output video.

        Returns:
            Path to video with subtitles.

        Raises:
            SubtitleError: If rendering fails.
        """
        # Try pycaps first
        try:
            return self._render_with_pycaps(video_path, srt_path, output_path)
        except (ImportError, Exception) as e:
            logger.warning(f"pycaps not available: {e}, using MoviePy fallback")

        # Fallback to MoviePy
        return self._render_with_moviepy(video_path, srt_path, output_path)

    def _render_with_pycaps(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render subtitles using pycaps.

        Args:
            video_path: Path to input video.
            srt_path: Path to SRT file.
            output_path: Path for output video.

        Returns:
            Path to output video.
        """
        try:
            from pycaps import render_video
        except ImportError as e:
            raise ImportError(
                "pycaps is not installed. Run: pip install pycaps"
            ) from e

        output_path.parent.mkdir(parents=True, exist_ok=True)

        style_dict = {
            "font": self.style.font,
            "font_size": self.style.font_size,
            "color": self.style.color,
            "highlight_color": self.style.highlight_color,
            "stroke_color": self.style.stroke_color,
            "stroke_width": self.style.stroke_width,
            "position": self.style.position.value,
            "animation": self.style.animation.value,
        }

        render_video(
            input_video=str(video_path),
            subtitles=str(srt_path),
            output_video=str(output_path),
            style=style_dict,
        )

        logger.info(f"Subtitles rendered with pycaps: {output_path}")
        return output_path

    def _render_with_moviepy(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render subtitles using MoviePy (fallback).

        Args:
            video_path: Path to input video.
            srt_path: Path to SRT file.
            output_path: Path for output video.

        Returns:
            Path to output video.
        """
        try:
            from moviepy.editor import (
                CompositeVideoClip,
                TextClip,
                VideoFileClip,
            )
        except ImportError as e:
            raise SubtitleError(
                "moviepy is not installed. Run: pip install moviepy"
            ) from e

        try:
            # Parse SRT
            entries = parse_srt(srt_path)

            if not entries:
                raise SubtitleError("No subtitle entries found in SRT file")

            # Load video
            video = VideoFileClip(str(video_path))
            video_width, video_height = video.size

            # Position mapping
            position_map = {
                "top": ("center", 100),
                "center": ("center", video_height // 2),
                "bottom": ("center", video_height - 200),
            }
            position = position_map.get(
                self.style.position.value,
                ("center", video_height - 200),
            )

            # Create text clips for each word
            text_clips = []

            for entry in entries:
                try:
                    txt_clip = (
                        TextClip(
                            entry.text,
                            fontsize=self.style.font_size,
                            color=self.style.highlight_color,
                            font=self.style.font,
                            stroke_color=self.style.stroke_color,
                            stroke_width=self.style.stroke_width,
                        )
                        .set_position(position)
                        .set_start(entry.start_seconds)
                        .set_duration(entry.end_seconds - entry.start_seconds)
                    )
                    text_clips.append(txt_clip)
                except Exception as e:
                    logger.warning(f"Failed to create text clip for '{entry.text}': {e}")
                    continue

            if not text_clips:
                raise SubtitleError("Could not create any subtitle clips")

            # Composite video with subtitles
            final = CompositeVideoClip([video] + text_clips)

            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final.write_videofile(
                str(output_path),
                fps=video.fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )

            # Cleanup
            final.close()
            video.close()
            for clip in text_clips:
                clip.close()

            logger.info(f"Subtitles rendered with MoviePy: {output_path}")
            return output_path

        except Exception as e:
            raise SubtitleError(f"Subtitle rendering failed: {e}") from e


def group_words_into_phrases(
    timestamps: list[WordTimestamp],
    words_per_phrase: int = 4,
    max_duration_ms: int = 3000,
) -> list[tuple[str, int, int]]:
    """Group word timestamps into phrases for subtitle display.

    Args:
        timestamps: List of word timestamps.
        words_per_phrase: Maximum words per phrase.
        max_duration_ms: Maximum phrase duration in milliseconds.

    Returns:
        List of (phrase_text, start_ms, end_ms) tuples.
    """
    if not timestamps:
        return []

    phrases = []
    current_words = []
    current_start = timestamps[0].start_ms

    for ts in timestamps:
        current_words.append(ts.word)
        duration = ts.end_ms - current_start

        # Check if we should end this phrase
        if len(current_words) >= words_per_phrase or duration >= max_duration_ms:
            phrase = " ".join(current_words)
            phrases.append((phrase, current_start, ts.end_ms))
            current_words = []
            if ts != timestamps[-1]:
                # Set start for next phrase
                next_idx = timestamps.index(ts) + 1
                if next_idx < len(timestamps):
                    current_start = timestamps[next_idx].start_ms

    # Add remaining words
    if current_words:
        phrase = " ".join(current_words)
        phrases.append((phrase, current_start, timestamps[-1].end_ms))

    return phrases
