"""Subtitle rendering with karaoke-style highlighting.

Combines video, audio, and subtitles into final output
with word-by-word highlighting synchronized to the voiceover.
"""

import os
from pathlib import Path
from typing import Optional

from socials_automator.constants import (
    VIDEO_FPS,
    SUBTITLE_FONT_SIZE_DEFAULT,
    SUBTITLE_FONT_NAME,
    SUBTITLE_FONT_COLOR,
    SUBTITLE_HIGHLIGHT_COLOR,
    SUBTITLE_STROKE_COLOR,
    SUBTITLE_STROKE_WIDTH,
    SUBTITLE_MAX_WORDS_PER_LINE,
    get_temp_dir,
)
from .base import (
    ISubtitleRenderer,
    PipelineContext,
    SubtitleRenderError,
)


class SubtitleRenderer(ISubtitleRenderer):
    """Renders karaoke-style subtitles on video."""

    def __init__(
        self,
        font: str = SUBTITLE_FONT_NAME,
        font_size: int = SUBTITLE_FONT_SIZE_DEFAULT,
        color: str = SUBTITLE_FONT_COLOR,
        highlight_color: str = SUBTITLE_HIGHLIGHT_COLOR,
        stroke_color: str = SUBTITLE_STROKE_COLOR,
        stroke_width: int = SUBTITLE_STROKE_WIDTH,
        position: str = "bottom",
        profile_handle: Optional[str] = None,
        horizontal_margin: float = 0.10,  # 10% margin on each side
        max_words_per_line: int = SUBTITLE_MAX_WORDS_PER_LINE,
    ):
        """Initialize subtitle renderer.

        Args:
            font: Font family.
            font_size: Font size in pixels (default 80).
            color: Default text color.
            highlight_color: Highlighted word color.
            stroke_color: Text stroke/outline color.
            stroke_width: Stroke width in pixels.
            position: Subtitle position (top, center, bottom).
            profile_handle: Instagram handle for watermark (e.g., @ai.for.mortals).
            horizontal_margin: Horizontal margin as fraction of video width (0.10 = 10%).
            max_words_per_line: Max words per line before wrapping.
        """
        super().__init__()
        self.font = font
        self.font_size = font_size
        self.color = color
        self.highlight_color = highlight_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.position = position
        self.profile_handle = profile_handle
        self.horizontal_margin = horizontal_margin
        self.max_words_per_line = max_words_per_line
        # Set MoviePy temp directory to avoid files in project root
        self._setup_moviepy_temp()

    def _setup_moviepy_temp(self) -> None:
        """Configure MoviePy to use project temp directory."""
        temp_dir = get_temp_dir()
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

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute subtitle rendering step.

        Args:
            context: Pipeline context with video, audio, and SRT.

        Returns:
            Updated context with final video.
        """
        if not context.assembled_video_path:
            raise SubtitleRenderError("No assembled video available")
        if not context.audio_path:
            raise SubtitleRenderError("No audio available")
        if not context.srt_path:
            raise SubtitleRenderError("No SRT file available")

        # Get profile handle from context if not set
        if not self.profile_handle and context.profile.id:
            self.profile_handle = f"@{context.profile.id}"

        self.log_start("Rendering subtitles and combining audio...")

        try:
            final_path = context.output_dir / "final.mp4"

            result_path = await self.render_subtitles(
                context.assembled_video_path,
                context.audio_path,
                context.srt_path,
                final_path,
            )

            context.final_video_path = result_path

            self.log_success(f"Final video: {result_path}")
            return context

        except Exception as e:
            self.log_error(f"Subtitle rendering failed: {e}")
            raise SubtitleRenderError(f"Failed to render subtitles: {e}") from e

    async def render_subtitles(
        self,
        video_path: Path,
        audio_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render karaoke-style subtitles on video.

        Args:
            video_path: Path to video without audio.
            audio_path: Path to audio file.
            srt_path: Path to SRT file with word timestamps.
            output_path: Path for output video.

        Returns:
            Path to final video.
        """
        # Try pycaps first, fall back to MoviePy
        try:
            return await self._render_with_pycaps(
                video_path, audio_path, srt_path, output_path
            )
        except Exception as e:
            self.log_detail(f"pycaps not available: {e}, using MoviePy")
            return await self._render_with_moviepy(
                video_path, audio_path, srt_path, output_path
            )

    async def _render_with_pycaps(
        self,
        video_path: Path,
        audio_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render subtitles using pycaps.

        Args:
            video_path: Input video path.
            audio_path: Audio path.
            srt_path: SRT path.
            output_path: Output path.

        Returns:
            Path to output video.
        """
        try:
            from pycaps import render_video
        except ImportError as e:
            raise ImportError("pycaps not available") from e

        self.log_detail("Using pycaps for karaoke subtitles...")

        # First combine video with audio using MoviePy
        video_with_audio = await self._add_audio_to_video(video_path, audio_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        style = {
            "font": self.font,
            "font_size": self.font_size,
            "color": self.color,
            "highlight_color": self.highlight_color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "position": self.position,
            "animation": "pop",
        }

        render_video(
            input_video=str(video_with_audio),
            subtitles=str(srt_path),
            output_video=str(output_path),
            style=style,
        )

        # Clean up temp file
        if video_with_audio != video_path:
            video_with_audio.unlink(missing_ok=True)

        return output_path

    async def _render_with_moviepy(
        self,
        video_path: Path,
        audio_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render subtitles using MoviePy (fallback).

        Args:
            video_path: Input video path.
            audio_path: Audio path.
            srt_path: SRT path.
            output_path: Output path.

        Returns:
            Path to output video.
        """
        try:
            # MoviePy 2.x imports
            from moviepy import (
                AudioFileClip,
                CompositeVideoClip,
                TextClip,
                VideoFileClip,
            )
        except ImportError:
            try:
                # Fallback to MoviePy 1.x imports
                from moviepy.editor import (
                    AudioFileClip,
                    CompositeVideoClip,
                    TextClip,
                    VideoFileClip,
                )
            except ImportError as e:
                raise SubtitleRenderError(
                    "moviepy is not installed. Run: pip install moviepy"
                ) from e

        self.log_detail("Using MoviePy for karaoke subtitles...")

        # Load video and audio
        video = VideoFileClip(str(video_path))
        audio = AudioFileClip(str(audio_path))

        # Set audio on video (with_audio for MoviePy 2.x, set_audio for 1.x)
        if hasattr(video, 'with_audio'):
            video = video.with_audio(audio)
        else:
            video = video.set_audio(audio)

        # Parse SRT and create text clips
        self.log_detail("Parsing SRT file...")
        entries = self._parse_srt(srt_path)

        if not entries:
            self.log_detail("No subtitle entries, exporting without subtitles")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video.write_videofile(
                str(output_path),
                fps=VIDEO_FPS,
                codec="libx264",
                audio_codec="aac",
                preset="fast",
                logger=None,
                temp_audiofile=self._get_temp_audiofile_path(output_path),
                ffmpeg_params=["-crf", "26"],
            )
            video.close()
            audio.close()
            return output_path

        # Create text clips
        video_width, video_height = video.size

        # Position subtitles - in safe zone to avoid being cut off
        # Instagram/TikTok safe zone: top 10% and bottom 35% may have UI overlays
        # Account for 2-line subtitles: font_size * 2.5 (2 lines + spacing)
        max_text_height = int(self.font_size * 2.5)

        if self.position == "top":
            pos_y = int(video_height * 0.12)
        elif self.position == "center":
            pos_y = (video_height // 2) - (max_text_height // 2)
        else:  # bottom - position so 2-line text stays in safe zone
            # Position at ~55% from top to prevent bottom clipping
            pos_y = int(video_height * 0.55) - (max_text_height // 2)

        self.log_detail(f"Creating karaoke clips for {len(entries)} words...")

        # Create karaoke-style subtitle clips with word highlighting
        text_clips = self._create_karaoke_clips_with_highlight(
            entries, video_width, video_height, pos_y
        )

        self.log_detail(f"Created {len(text_clips)} subtitle clips")

        # Create watermark clips if profile handle is set
        watermark_clips = []
        if self.profile_handle:
            self.log_detail(f"Adding watermark: {self.profile_handle}")
            watermark_clips = self._create_watermark_clips(
                video_width, video_height, video.duration
            )
            self.log_detail(f"Created {len(watermark_clips)} watermark clips")

        self.log_progress("Rendering subtitles...")

        # Composite all layers
        all_clips = [video] + text_clips + watermark_clips
        if len(all_clips) > 1:
            final = CompositeVideoClip(all_clips)
        else:
            final = video

        # Export with optimized FFmpeg settings
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            logger=None,
            temp_audiofile=self._get_temp_audiofile_path(output_path),
            ffmpeg_params=["-crf", "26"],
        )

        # Cleanup
        final.close()
        video.close()
        audio.close()
        for clip in text_clips + watermark_clips:
            clip.close()

        return output_path

    def _create_karaoke_clips_with_highlight(
        self,
        entries: list[tuple[str, float, float]],
        video_width: int,
        video_height: int,
        pos_y: int,
    ) -> list:
        """Create karaoke-style clips with word highlighting.

        Shows phrase with current word highlighted in yellow, others in white.
        Uses PIL for precise multi-color text rendering.

        Args:
            entries: Word entries (text, start, end).
            video_width: Video width for sizing.
            video_height: Video height.
            pos_y: Y position for subtitles.

        Returns:
            List of text clips.
        """
        try:
            from moviepy import ImageClip
        except ImportError:
            from moviepy.editor import ImageClip

        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        text_clips = []

        # Load font for PIL - try project fonts folder first, then system fonts
        pil_font = None

        # Find project root (where fonts/ folder is)
        project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up from pipeline/video/socials_automator/src
        fonts_dir = project_root / "fonts"

        # Build font paths to try - handle both with and without .ttf extension
        font_base = self.font.replace('.ttf', '').replace('.TTF', '')
        font_paths_to_try = [
            fonts_dir / f"{font_base}.ttf",  # Project fonts folder (e.g., Montserrat-Bold.ttf)
            fonts_dir / self.font,  # Try exact name in fonts folder
            self.font,  # Try as-is (full path or system font)
            f"C:/Windows/Fonts/{font_base.lower()}.ttf",  # Windows fonts folder
        ]

        for font_path in font_paths_to_try:
            try:
                pil_font = ImageFont.truetype(str(font_path), self.font_size)
                self.log_detail(f"Loaded font: {font_path}")
                break
            except Exception:
                continue

        if pil_font is None:
            self.log_detail(f"Warning: Could not load font '{self.font}', using default (subtitles may be small)")
            pil_font = ImageFont.load_default()

        # Group into 3-word phrases
        phrases = self._group_into_phrases(entries, words_per_phrase=3)

        # Track word index
        word_index = 0

        for phrase_text, phrase_start, phrase_end in phrases:
            words_in_phrase = phrase_text.split()
            num_words = len(words_in_phrase)

            # Get word entries for this phrase
            phrase_word_entries = []
            for _ in range(num_words):
                if word_index < len(entries):
                    phrase_word_entries.append(entries[word_index])
                    word_index += 1

            if not phrase_word_entries:
                continue

            # For each word, create an image with that word highlighted
            for i, (word_text, word_start, word_end) in enumerate(phrase_word_entries):
                word_duration = word_end - word_start
                if word_duration <= 0.02:
                    continue

                try:
                    # Create image for this frame
                    img_clip = self._create_karaoke_frame(
                        words_in_phrase,
                        i,  # highlight index
                        pil_font,
                        video_width,
                    )

                    if img_clip is not None:
                        # Convert to MoviePy clip
                        clip = ImageClip(img_clip)

                        # Apply timing and position
                        if hasattr(clip, 'with_position'):
                            clip = clip.with_position(("center", pos_y))
                            clip = clip.with_start(word_start)
                            clip = clip.with_duration(word_duration)
                        else:
                            clip = clip.set_position(("center", pos_y))
                            clip = clip.set_start(word_start)
                            clip = clip.set_duration(word_duration)

                        text_clips.append(clip)

                except Exception as e:
                    self.log_detail(f"Skipping word '{word_text}': {e}")
                    continue

        return text_clips

    def _create_karaoke_frame(
        self,
        words: list[str],
        highlight_index: int,
        font: "ImageFont.FreeTypeFont",
        video_width: int,
    ) -> "np.ndarray":
        """Create a single karaoke frame with one word highlighted.

        Supports 2-line text if needed to fit within horizontal margins.

        Args:
            words: List of words in the phrase.
            highlight_index: Index of word to highlight in yellow.
            font: PIL font object.
            video_width: Video width for margin calculation.

        Returns:
            RGBA numpy array for the frame.
        """
        from PIL import Image, ImageDraw
        import numpy as np

        # Colors
        white = (255, 255, 255, 255)
        yellow = (255, 215, 0, 255)  # Gold
        black = (0, 0, 0, 255)

        # Calculate available width respecting margins
        margin_px = int(video_width * self.horizontal_margin)
        max_text_width = video_width - (margin_px * 2)

        # Build uppercase words
        words_upper = [w.upper() for w in words]

        # Calculate text size for dummy measurements
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        # Check if we need 2 lines - measure full phrase width
        full_phrase = " ".join(words_upper)
        full_bbox = dummy_draw.textbbox((0, 0), full_phrase, font=font)
        full_width = full_bbox[2] - full_bbox[0]

        # Split into lines if needed
        if full_width > max_text_width and len(words) > 1:
            # Split into 2 lines - try to balance
            mid = len(words) // 2
            # Find best split point
            best_split = mid
            best_diff = float('inf')
            for split in range(1, len(words)):
                line1 = " ".join(words_upper[:split])
                line2 = " ".join(words_upper[split:])
                bbox1 = dummy_draw.textbbox((0, 0), line1, font=font)
                bbox2 = dummy_draw.textbbox((0, 0), line2, font=font)
                w1 = bbox1[2] - bbox1[0]
                w2 = bbox2[2] - bbox2[0]
                diff = abs(w1 - w2)
                if diff < best_diff and max(w1, w2) <= max_text_width:
                    best_diff = diff
                    best_split = split
            lines = [words_upper[:best_split], words_upper[best_split:]]
            highlight_line = 0 if highlight_index < best_split else 1
            highlight_in_line = highlight_index if highlight_line == 0 else highlight_index - best_split
        else:
            lines = [words_upper]
            highlight_line = 0
            highlight_in_line = highlight_index

        # Measure line dimensions
        line_heights = []
        line_widths = []
        for line_words in lines:
            line_text = " ".join(line_words)
            bbox = dummy_draw.textbbox((0, 0), line_text, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        # Create image sized for all lines
        # Use generous padding - especially bottom for descenders (g, p, y, etc.)
        padding_horizontal = 20
        padding_top = 20
        padding_bottom = int(self.font_size * 0.35)  # 35% of font size for descenders
        line_spacing = 10
        total_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        max_line_width = max(line_widths)

        img_width = max_line_width + padding_horizontal * 2
        img_height = total_height + padding_top + padding_bottom

        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw each line centered
        y_offset = padding_top
        word_global_index = 0

        for line_idx, line_words in enumerate(lines):
            line_text = " ".join(line_words)
            line_bbox = dummy_draw.textbbox((0, 0), line_text, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_height = line_bbox[3] - line_bbox[1]

            # Center this line horizontally
            x_offset = padding_horizontal + (max_line_width - line_width) // 2

            for word_idx, word_upper in enumerate(line_words):
                # Add space before word (except first in line)
                if word_idx > 0:
                    space_bbox = draw.textbbox((0, 0), " ", font=font)
                    x_offset += space_bbox[2] - space_bbox[0]

                # Choose color - highlight if this is the current word
                is_highlighted = (line_idx == highlight_line and word_idx == highlight_in_line)
                color = yellow if is_highlighted else white

                # Draw stroke (outline) first
                for dx in range(-self.stroke_width, self.stroke_width + 1):
                    for dy in range(-self.stroke_width, self.stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x_offset + dx, y_offset + dy), word_upper, font=font, fill=black)

                # Draw text
                draw.text((x_offset, y_offset), word_upper, font=font, fill=color)

                # Move x for next word
                word_bbox = draw.textbbox((0, 0), word_upper, font=font)
                x_offset += word_bbox[2] - word_bbox[0]

                word_global_index += 1

            y_offset += line_height + line_spacing

        # Convert to numpy array
        return np.array(img)

    def _create_watermark_clips(
        self,
        video_width: int,
        video_height: int,
        duration: float,
    ) -> list:
        """Create watermark clips that move every 10 seconds.

        Args:
            video_width: Video width.
            video_height: Video height.
            duration: Video duration in seconds.

        Returns:
            List of watermark text clips.
        """
        try:
            from moviepy import TextClip
        except ImportError:
            from moviepy.editor import TextClip

        watermark_clips = []

        # Watermark positions (in safe zone, avoiding edges that get cut off)
        # Safe zone: 10% from edges, avoid bottom 30%
        margin_x = int(video_width * 0.05)
        margin_top = int(video_height * 0.15)
        margin_mid = int(video_height * 0.40)

        positions = [
            (margin_x, margin_top),                          # Top-left
            (video_width - 220, margin_top),                 # Top-right
            (margin_x, margin_mid),                          # Middle-left
            (video_width - 220, margin_mid),                 # Middle-right
            (video_width // 2 - 80, margin_top),             # Top-center
        ]

        interval = 10.0  # Change position every 10 seconds
        current_time = 0.0
        pos_index = 0

        while current_time < duration:
            segment_duration = min(interval, duration - current_time)
            pos = positions[pos_index % len(positions)]

            try:
                # Create watermark text
                try:
                    # MoviePy 2.x style
                    wm_clip = TextClip(
                        font=self.font,
                        text=self.profile_handle,
                        font_size=28,
                        color="white",
                        stroke_color="black",
                        stroke_width=1,
                        margin=(5, 5),  # Add margin to prevent cutoff
                    )
                except TypeError:
                    # MoviePy 1.x style
                    wm_clip = TextClip(
                        self.profile_handle,
                        fontsize=28,
                        color="white",
                        font=self.font,
                        stroke_color="black",
                        stroke_width=1,
                        method="label",
                    )

                # Apply timing and position
                if hasattr(wm_clip, 'with_position'):
                    wm_clip = wm_clip.with_position(pos)
                    wm_clip = wm_clip.with_start(current_time)
                    wm_clip = wm_clip.with_duration(segment_duration)
                else:
                    wm_clip = wm_clip.set_position(pos)
                    wm_clip = wm_clip.set_start(current_time)
                    wm_clip = wm_clip.set_duration(segment_duration)

                watermark_clips.append(wm_clip)

            except Exception as e:
                self.log_detail(f"Skipping watermark at {current_time}s: {e}")

            current_time += interval
            pos_index += 1

        return watermark_clips

    async def _add_audio_to_video(
        self,
        video_path: Path,
        audio_path: Path,
    ) -> Path:
        """Add audio to video file.

        Args:
            video_path: Video without audio.
            audio_path: Audio file.

        Returns:
            Path to video with audio.
        """
        try:
            # MoviePy 2.x imports
            from moviepy import AudioFileClip, VideoFileClip
        except ImportError:
            try:
                # Fallback to MoviePy 1.x imports
                from moviepy.editor import AudioFileClip, VideoFileClip
            except ImportError as e:
                raise SubtitleRenderError("moviepy not available") from e

        output_path = video_path.parent / "video_with_audio.mp4"

        video = VideoFileClip(str(video_path))
        audio = AudioFileClip(str(audio_path))

        # Trim audio to video length if needed
        if audio.duration > video.duration:
            if hasattr(audio, 'subclipped'):
                audio = audio.subclipped(0, video.duration)
            else:
                audio = audio.subclip(0, video.duration)

        # Set audio (with_audio for MoviePy 2.x, set_audio for 1.x)
        if hasattr(video, 'with_audio'):
            video = video.with_audio(audio)
        else:
            video = video.set_audio(audio)

        video.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            logger=None,
            temp_audiofile=self._get_temp_audiofile_path(output_path),
            ffmpeg_params=["-crf", "26"],
        )

        video.close()
        audio.close()

        return output_path

    def _parse_srt(self, srt_path: Path) -> list[tuple[str, float, float]]:
        """Parse SRT file into entries.

        Args:
            srt_path: Path to SRT file.

        Returns:
            List of (text, start_seconds, end_seconds) tuples.
        """
        import re

        entries = []

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"\n\n+", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                times = lines[1]
                text = " ".join(lines[2:])

                match = re.match(
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                    times,
                )

                if match:
                    start_s = (
                        int(match.group(1)) * 3600
                        + int(match.group(2)) * 60
                        + int(match.group(3))
                        + int(match.group(4)) / 1000
                    )
                    end_s = (
                        int(match.group(5)) * 3600
                        + int(match.group(6)) * 60
                        + int(match.group(7))
                        + int(match.group(8)) / 1000
                    )

                    entries.append((text, start_s, end_s))

            except (ValueError, IndexError):
                continue

        return entries

    def _group_into_phrases(
        self,
        entries: list[tuple[str, float, float]],
        words_per_phrase: int = 3,
    ) -> list[tuple[str, float, float]]:
        """Group SRT entries into phrases.

        Args:
            entries: List of (text, start, end) tuples.
            words_per_phrase: Words per phrase.

        Returns:
            Grouped phrases.
        """
        if not entries:
            return []

        phrases = []
        current_words = []
        phrase_start = None

        for text, start, end in entries:
            # When starting a new phrase, record its start time
            if not current_words:
                phrase_start = start
            current_words.append(text)

            if len(current_words) >= words_per_phrase:
                phrase = " ".join(current_words)
                phrases.append((phrase, phrase_start, end))
                current_words = []
                # phrase_start will be set on next iteration

        # Add remaining words
        if current_words and phrase_start is not None:
            phrase = " ".join(current_words)
            phrases.append((phrase, phrase_start, entries[-1][2]))

        return phrases
