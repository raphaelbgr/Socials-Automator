"""Video-related constants for the Socials Automator.

This module contains all constants related to video generation including:
- Resolution and aspect ratio settings
- Frame rates and duration limits
- Codec and format settings
- Subtitle styling defaults

AI CONTEXT:
-----------
These constants are used throughout the video pipeline to ensure consistent
output across all generated videos. The primary target platform is Instagram
Reels/TikTok, which requires 9:16 vertical video format.

MODIFICATION GUIDE:
------------------
- VIDEO_WIDTH/HEIGHT: Change for different resolutions (keep 9:16 ratio)
- VIDEO_FPS: 30 is standard, 60 for smoother motion
- SUBTITLE_* settings: Adjust for different text styles
"""

from typing import Final

# =============================================================================
# VIDEO RESOLUTION
# =============================================================================
# Target resolution for Instagram Reels / TikTok
# 9:16 aspect ratio (vertical/portrait orientation)

VIDEO_WIDTH: Final[int] = 1080
"""Output video width in pixels. Standard HD width for mobile."""

VIDEO_HEIGHT: Final[int] = 1920
"""Output video height in pixels. 9:16 aspect ratio with 1080 width."""

VIDEO_ASPECT_RATIO: Final[tuple[int, int]] = (9, 16)
"""Aspect ratio as (width, height) tuple. 9:16 is vertical/portrait."""

VIDEO_ASPECT_RATIO_FLOAT: Final[float] = 9 / 16
"""Aspect ratio as decimal (0.5625). Used for crop calculations."""


# =============================================================================
# FRAME RATE AND TIMING
# =============================================================================

VIDEO_FPS: Final[int] = 30
"""Frames per second. 30 is standard for social media."""

VIDEO_MIN_DURATION_SECONDS: Final[float] = 15.0
"""Minimum video duration in seconds. Instagram Reels minimum."""

VIDEO_MAX_DURATION_SECONDS: Final[float] = 180.0
"""Maximum video duration in seconds (3 minutes). Platform limit."""

VIDEO_DEFAULT_DURATION_SECONDS: Final[float] = 60.0
"""Default target duration in seconds. Sweet spot for engagement."""


# =============================================================================
# VIDEO CODEC AND FORMAT
# =============================================================================

VIDEO_CODEC: Final[str] = "libx264"
"""Video codec for encoding. H.264 is widely compatible."""

VIDEO_FORMAT: Final[str] = "mp4"
"""Output container format. MP4 is universal."""

AUDIO_CODEC: Final[str] = "aac"
"""Audio codec for encoding. AAC is standard for MP4."""

AUDIO_SAMPLE_RATE: Final[int] = 44100
"""Audio sample rate in Hz. CD quality."""


# =============================================================================
# SUBTITLE STYLING
# =============================================================================
# Karaoke-style subtitle defaults for vertical video

SUBTITLE_FONT_SIZE_DEFAULT: Final[int] = 80
"""Default subtitle font size in pixels. Optimized for mobile viewing."""

SUBTITLE_FONT_SIZE_MIN: Final[int] = 40
"""Minimum subtitle font size."""

SUBTITLE_FONT_SIZE_MAX: Final[int] = 120
"""Maximum subtitle font size."""

SUBTITLE_FONT_NAME: Final[str] = "Montserrat-Bold.ttf"
"""Default font for subtitles. Montserrat Bold is #1 for social media (61% of videos).
Available fonts in /fonts folder:
- Montserrat-Bold.ttf (RECOMMENDED - most popular for TikTok/Reels)
- Montserrat-ExtraBold.ttf (even bolder)
- Poppins-Bold.ttf (clean, highly readable)
- BebasNeue-Regular.ttf (condensed, modern)
- Impact.ttf (classic, bold)
"""

SUBTITLE_FONT_COLOR: Final[str] = "white"
"""Default text color for subtitles."""

SUBTITLE_STROKE_COLOR: Final[str] = "black"
"""Outline/stroke color for subtitle text."""

SUBTITLE_STROKE_WIDTH: Final[int] = 4
"""Stroke width in pixels for text outline."""

SUBTITLE_HIGHLIGHT_COLOR: Final[str] = "#FFFF00"
"""Highlight color for karaoke-style current word (yellow)."""

SUBTITLE_POSITION_Y_PERCENT: Final[float] = 0.75
"""Vertical position as percentage from top (0.75 = 75% down)."""

SUBTITLE_MAX_WORDS_PER_LINE: Final[int] = 4
"""Maximum words to display per subtitle line."""


# =============================================================================
# VIDEO CLIP SETTINGS
# =============================================================================

CLIP_MIN_DURATION_SECONDS: Final[float] = 3.0
"""Minimum duration for a single video clip segment."""

CLIP_MAX_DURATION_SECONDS: Final[float] = 15.0
"""Maximum duration for a single video clip before switching."""

CLIP_CROSSFADE_DURATION: Final[float] = 0.0
"""Crossfade duration between clips. 0 = hard cut."""

SEGMENTS_DEFAULT_COUNT: Final[int] = 8
"""Default number of video segments in a reel."""

SEGMENTS_MIN_COUNT: Final[int] = 4
"""Minimum number of video segments."""

SEGMENTS_MAX_COUNT: Final[int] = 12
"""Maximum number of video segments."""


# =============================================================================
# WATERMARK SETTINGS
# =============================================================================

WATERMARK_ENABLED: Final[bool] = True
"""Whether to add watermark to videos."""

WATERMARK_OPACITY: Final[float] = 0.7
"""Watermark opacity (0.0 = invisible, 1.0 = fully opaque)."""

WATERMARK_POSITION: Final[str] = "bottom_right"
"""Watermark position on video."""

WATERMARK_MARGIN: Final[int] = 20
"""Margin from edge in pixels."""


# =============================================================================
# IMAGE OVERLAY SETTINGS
# =============================================================================
# Settings for image overlays that appear during narration

IMAGE_OVERLAY_MARGIN_X: Final[int] = 40
"""Horizontal margin from left/right edge in pixels."""

IMAGE_OVERLAY_MAX_HEIGHT: Final[int] = 600
"""Maximum image container height in pixels."""

IMAGE_OVERLAY_MARGIN_BOTTOM: Final[int] = 20
"""Margin above subtitle area in pixels."""

IMAGE_OVERLAY_SUBTITLE_Y: Final[int] = 1400
"""Y position where subtitles start (approx). Used to position overlay above."""

IMAGE_OVERLAY_BLUR_STRENGTH: Final[int] = 20
"""Blur strength for frosted glass backdrop effect."""

IMAGE_OVERLAY_BORDER_RADIUS: Final[int] = 20
"""Border radius for rounded corners in pixels."""

# Animation durations (in seconds)
IMAGE_OVERLAY_POP_IN_DURATION: Final[float] = 0.3
"""Duration of pop-in animation in seconds."""

IMAGE_OVERLAY_POP_OUT_DURATION: Final[float] = 0.2
"""Duration of pop-out animation in seconds."""

# Animation keyframe values (scale factor)
IMAGE_OVERLAY_POP_OVERSHOOT: Final[float] = 1.1
"""Scale overshoot for bounce effect (1.0 = no overshoot)."""

IMAGE_OVERLAY_POP_BOUNCE: Final[float] = 0.95
"""Scale bounce-back value after overshoot."""
