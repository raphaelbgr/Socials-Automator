"""Thumbnail generator for Instagram Reels.

Generates a cover image with hook text for Instagram Reels.
The text is positioned within the 1:1 grid safe zone so it's
visible in all Instagram views (grid, feed, full reel).

Safe Zones:
    - Full Reel: 1080x1920 (9:16)
    - Feed Preview: 1080x1350 (4:5) - centered
    - Grid View: 1080x1080 (1:1) - center square (profile thumbnail)

Text should be centered in the grid safe zone with margins.
"""

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from socials_automator.constants import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    SUBTITLE_FONT_NAME,
)
from .base import (
    ArtifactStatus,
    PipelineContext,
    PipelineStep,
    ThumbnailGenerationError,
)


# Safe zone dimensions
GRID_SAFE_ZONE_SIZE = 1080  # 1:1 square
FEED_PREVIEW_HEIGHT = 1350  # 4:5 aspect

# Text styling
DEFAULT_FONT_SIZE = 54  # Default size (use --font-size to customize)
DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_STROKE_COLOR = "#000000"
DEFAULT_STROKE_WIDTH = 3  # Stroke width for text outline
TEXT_HORIZONTAL_MARGIN = 0.08  # 8% margin on each side (reduced for larger text)
TEXT_VERTICAL_MARGIN = 0.12  # 12% margin top/bottom within safe zone
MAX_CHARS_PER_LINE = 18  # Force line breaks for readability
MAX_WORDS = 10  # Maximum words in thumbnail title
MAX_LINES = 3  # Maximum lines of text


class ThumbnailGenerator(PipelineStep):
    """Generates thumbnail with hook text for Instagram Reels."""

    def __init__(
        self,
        font: str = SUBTITLE_FONT_NAME,
        font_size: int = DEFAULT_FONT_SIZE,
        text_color: str = DEFAULT_TEXT_COLOR,
        stroke_color: str = DEFAULT_STROKE_COLOR,
        stroke_width: int = DEFAULT_STROKE_WIDTH,
        frame_time: float = 1.5,  # Extract frame at 1.5 seconds
    ):
        """Initialize thumbnail generator.

        Args:
            font: Font file name from /fonts folder.
            font_size: Font size in pixels.
            text_color: Text color (hex).
            stroke_color: Stroke/outline color (hex).
            stroke_width: Stroke width in pixels.
            frame_time: Time in seconds to extract frame from video.
        """
        super().__init__("ThumbnailGenerator")
        self.font = font
        self.font_size = font_size
        self.text_color = text_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.frame_time = frame_time

    def _get_font_path(self) -> Path:
        """Get path to font file."""
        # Try project fonts directory
        project_root = Path(__file__).parent.parent.parent.parent.parent
        fonts_dir = project_root / "fonts"

        font_path = fonts_dir / self.font
        if font_path.exists():
            return font_path

        # Try with .ttf extension
        if not self.font.endswith(".ttf"):
            font_path = fonts_dir / f"{self.font}.ttf"
            if font_path.exists():
                return font_path

        # Fallback to Montserrat-Bold
        fallback = fonts_dir / "Montserrat-Bold.ttf"
        if fallback.exists():
            return fallback

        raise ThumbnailGenerationError(f"Font not found: {self.font}")

    def _extract_frame(self, video_path: Path, time_seconds: float) -> Image.Image:
        """Extract a frame from video at specified time.

        Args:
            video_path: Path to video file.
            time_seconds: Time in seconds to extract frame.

        Returns:
            PIL Image of the frame.
        """
        try:
            from moviepy import VideoFileClip
        except ImportError:
            from moviepy.editor import VideoFileClip

        clip = None
        try:
            clip = VideoFileClip(str(video_path))

            # Ensure time is within video duration
            if time_seconds >= clip.duration:
                time_seconds = clip.duration * 0.1  # Use 10% into video

            # Get frame as numpy array
            frame = clip.get_frame(time_seconds)

            # Convert to PIL Image
            return Image.fromarray(frame)

        finally:
            if clip:
                clip.close()

    def _wrap_text(
        self,
        text: str,
        max_chars: int = MAX_CHARS_PER_LINE,
        max_words: int = MAX_WORDS,
        max_lines: int = MAX_LINES,
    ) -> list[str]:
        """Wrap text into lines for better readability.

        Args:
            text: Text to wrap.
            max_chars: Maximum characters per line.
            max_words: Maximum total words (truncate if exceeded).
            max_lines: Maximum number of lines (truncate if exceeded).

        Returns:
            List of lines.
        """
        words = text.split()

        # Limit to max_words
        if len(words) > max_words:
            words = words[:max_words]

        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            # +1 for space between words
            if current_length + word_length + (1 if current_line else 0) <= max_chars:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    # Check if we've hit max lines
                    if len(lines) >= max_lines:
                        return lines
                current_line = [word]
                current_length = word_length

        if current_line and len(lines) < max_lines:
            lines.append(" ".join(current_line))

        return lines

    def _render_text_on_image(
        self,
        image: Image.Image,
        text: str,
    ) -> Image.Image:
        """Render hook text centered in the grid safe zone.

        The grid safe zone is the center 1080x1080 square of the 1080x1920 video.
        Text is centered within this zone with margins.

        Args:
            image: Base image (1080x1920).
            text: Hook text to render.

        Returns:
            Image with text overlay.
        """
        # Create a copy to draw on
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = self._get_font_path()
        try:
            font = ImageFont.truetype(str(font_path), self.font_size)
        except Exception as e:
            self.log_detail(f"Font load error: {e}, using default")
            font = ImageFont.load_default()

        # Calculate safe zone boundaries
        # Grid safe zone: center 1080x1080 square
        safe_zone_top = (VIDEO_HEIGHT - GRID_SAFE_ZONE_SIZE) // 2  # 420
        safe_zone_bottom = safe_zone_top + GRID_SAFE_ZONE_SIZE  # 1500

        # Apply margins within safe zone
        margin_h = int(VIDEO_WIDTH * TEXT_HORIZONTAL_MARGIN)
        margin_v = int(GRID_SAFE_ZONE_SIZE * TEXT_VERTICAL_MARGIN)

        text_area_left = margin_h
        text_area_right = VIDEO_WIDTH - margin_h
        text_area_top = safe_zone_top + margin_v
        text_area_bottom = safe_zone_bottom - margin_v
        text_area_width = text_area_right - text_area_left

        # Wrap text into lines
        lines = self._wrap_text(text.upper())  # Uppercase for impact

        # Calculate total text height
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        line_spacing = self.font_size * 0.3  # 30% of font size
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Calculate starting Y position (vertically centered in text area)
        text_area_height = text_area_bottom - text_area_top
        start_y = text_area_top + (text_area_height - total_height) // 2

        # Draw each line centered
        current_y = start_y
        for i, line in enumerate(lines):
            line_width = line_widths[i]
            line_height = line_heights[i]

            # Center horizontally
            x = (VIDEO_WIDTH - line_width) // 2

            # Draw stroke/outline (draw text multiple times offset)
            for dx in range(-self.stroke_width, self.stroke_width + 1):
                for dy in range(-self.stroke_width, self.stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            (x + dx, current_y + dy),
                            line,
                            font=font,
                            fill=self.stroke_color,
                        )

            # Draw main text
            draw.text(
                (x, current_y),
                line,
                font=font,
                fill=self.text_color,
            )

            current_y += line_height + line_spacing

        return img

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Generate thumbnail with hook text.

        Uses the ASSEMBLED video (before subtitles) to avoid text-over-text.

        Args:
            context: Pipeline context with assembled video and script.

        Returns:
            Updated context with thumbnail path.
        """
        self.log_start("Generating thumbnail with hook text...")

        # Use assembled video (NO subtitles) - this step runs BEFORE SubtitleRenderer
        # This ensures we get a clean frame without any text overlay
        if not context.assembled_video_path:
            self.log_detail("No assembled video path, skipping thumbnail")
            return context

        video_path = context.assembled_video_path

        if not context.script or not context.script.hook:
            self.log_detail("No hook text available, skipping thumbnail")
            return context

        try:
            # Extract frame from assembled video (before subtitles)
            self.log_detail(f"Extracting frame at {self.frame_time}s from assembled video (no subtitles)...")
            frame_image = self._extract_frame(
                video_path,
                self.frame_time,
            )

            # Render hook text on frame
            hook_text = context.script.hook
            self.log_detail(f"Hook: {hook_text[:50]}...")
            thumbnail = self._render_text_on_image(frame_image, hook_text)

            # Save thumbnail
            thumbnail_path = context.output_dir / "thumbnail.jpg"
            thumbnail.save(thumbnail_path, "JPEG", quality=95)

            self.log_progress(f"Thumbnail saved: {thumbnail_path.name}")

            # Store in context for later use
            # Note: We add thumbnail_path as an extra attribute
            context._thumbnail_path = thumbnail_path

            # Update artifact tracking
            if context.metadata:
                context.metadata.artifacts.thumbnail = ArtifactStatus(
                    status="ok",
                    file=thumbnail_path.name,
                )

            return context

        except Exception as e:
            self.log_detail(f"Thumbnail generation failed: {e}")
            # Update artifact tracking with error
            if context.metadata:
                context.metadata.artifacts.thumbnail = ArtifactStatus(
                    status="failed",
                    error=str(e),
                )
            # Don't fail the pipeline, thumbnail is optional
            return context
