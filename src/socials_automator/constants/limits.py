"""Limit constants for the Socials Automator.

This module contains all limits and constraints:
- API rate limits and quotas
- Content length limits
- Retry and timeout settings
- Validation thresholds

AI CONTEXT:
-----------
These limits are derived from platform requirements (Instagram, TikTok) and
API provider constraints (Pexels, OpenAI, etc.). Exceeding these limits may
result in rejected content or API errors.

MODIFICATION GUIDE:
------------------
- API_* limits: Check provider documentation before changing
- INSTAGRAM_* limits: Based on Instagram Graph API requirements
- RETRY_* settings: Adjust for different network conditions
- VALIDATION_* thresholds: Tune for content quality requirements
"""

from typing import Final

# =============================================================================
# INSTAGRAM LIMITS
# =============================================================================
# Based on Instagram Graph API documentation

INSTAGRAM_CAPTION_MAX_LENGTH: Final[int] = 2200
"""Maximum caption length in characters for Instagram posts."""

INSTAGRAM_HASHTAG_MAX_COUNT: Final[int] = 30
"""Maximum number of hashtags allowed per post."""

INSTAGRAM_CAROUSEL_MIN_SLIDES: Final[int] = 2
"""Minimum slides in a carousel post."""

INSTAGRAM_CAROUSEL_MAX_SLIDES: Final[int] = 10
"""Maximum slides in a carousel post."""

INSTAGRAM_POSTS_PER_DAY_LIMIT: Final[int] = 25
"""Conservative daily posting limit to avoid rate limits."""

INSTAGRAM_REEL_MIN_DURATION: Final[float] = 3.0
"""Minimum reel duration in seconds."""

INSTAGRAM_REEL_MAX_DURATION: Final[float] = 90.0
"""Maximum reel duration in seconds (standard accounts)."""

INSTAGRAM_REEL_MAX_DURATION_EXTENDED: Final[float] = 180.0
"""Maximum reel duration for accounts with extended access."""


# =============================================================================
# PEXELS API LIMITS
# =============================================================================

PEXELS_REQUESTS_PER_MONTH: Final[int] = 20000
"""Monthly request limit for Pexels free tier."""

PEXELS_REQUESTS_PER_HOUR: Final[int] = 200
"""Hourly request limit for Pexels API."""

PEXELS_VIDEOS_PER_SEARCH: Final[int] = 15
"""Number of videos to request per search query."""

PEXELS_MIN_VIDEO_DURATION: Final[float] = 5.0
"""Minimum video duration in seconds to consider."""

PEXELS_MAX_VIDEO_DURATION: Final[float] = 60.0
"""Maximum video duration in seconds to download."""


# =============================================================================
# AI PROVIDER LIMITS
# =============================================================================

AI_MAX_TOKENS_DEFAULT: Final[int] = 4096
"""Default maximum tokens for AI responses."""

AI_MAX_RETRIES: Final[int] = 3
"""Maximum retries for failed AI calls."""

AI_TIMEOUT_SECONDS: Final[int] = 60
"""Timeout for AI API calls in seconds."""

AI_TEMPERATURE_DEFAULT: Final[float] = 0.7
"""Default temperature for AI generation (creativity level)."""


# =============================================================================
# CONTENT GENERATION LIMITS
# =============================================================================

CONTENT_TITLE_MAX_LENGTH: Final[int] = 100
"""Maximum length for content titles."""

CONTENT_SLIDE_TEXT_MAX_LENGTH: Final[int] = 150
"""Maximum text length per slide."""

CONTENT_NARRATION_MAX_LENGTH: Final[int] = 2000
"""Maximum narration text length for TTS."""

CONTENT_KEYWORD_MAX_COUNT: Final[int] = 10
"""Maximum keywords per content piece."""


# =============================================================================
# RETRY AND TIMEOUT SETTINGS
# =============================================================================

RETRY_MAX_ATTEMPTS: Final[int] = 3
"""Maximum retry attempts for recoverable errors."""

RETRY_BASE_DELAY_SECONDS: Final[float] = 1.0
"""Base delay between retries (exponential backoff)."""

RETRY_MAX_DELAY_SECONDS: Final[float] = 30.0
"""Maximum delay between retries."""

TIMEOUT_HTTP_CONNECT: Final[int] = 10
"""HTTP connection timeout in seconds."""

TIMEOUT_HTTP_READ: Final[int] = 60
"""HTTP read timeout in seconds."""

TIMEOUT_VIDEO_DOWNLOAD: Final[int] = 120
"""Timeout for video file downloads in seconds."""

TIMEOUT_VIDEO_RENDER: Final[int] = 600
"""Timeout for video rendering operations in seconds (10 min)."""


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================
# Thresholds for AI validation and quality checks

VALIDATION_CAPTION_MIN_LENGTH: Final[int] = 50
"""Minimum caption length to pass validation."""

VALIDATION_CAPTION_MAX_LENGTH: Final[int] = 1000
"""Maximum caption length for validation (stricter than Instagram limit)."""

VALIDATION_CAPTION_MIN_SCORE: Final[int] = 7
"""Minimum AI validation score (out of 10) to pass."""

VALIDATION_MIN_TOOLS_MENTIONED: Final[int] = 3
"""Minimum specific tools/tips that must be mentioned in caption."""

VALIDATION_MAX_GENERIC_PHRASES: Final[int] = 0
"""Maximum allowed generic phrases in validated content."""


# =============================================================================
# BATCH AND QUEUE LIMITS
# =============================================================================

BATCH_GENERATE_MAX_COUNT: Final[int] = 10
"""Maximum posts to generate in a single batch."""

QUEUE_MAX_PENDING: Final[int] = 50
"""Maximum posts allowed in pending queue."""

LOOP_MIN_INTERVAL_SECONDS: Final[int] = 60
"""Minimum interval between loop iterations."""

LOOP_DEFAULT_DELAY_SECONDS: Final[int] = 3
"""Default delay between successful generations in loop mode."""


# =============================================================================
# FILE SIZE LIMITS
# =============================================================================

FILE_MAX_VIDEO_SIZE_MB: Final[int] = 650
"""Maximum video file size in MB (Instagram limit ~650MB)."""

FILE_MAX_IMAGE_SIZE_MB: Final[int] = 8
"""Maximum image file size in MB."""

FILE_MAX_AUDIO_SIZE_MB: Final[int] = 50
"""Maximum audio file size in MB."""


# =============================================================================
# CACHE LIMITS
# =============================================================================

CACHE_MAX_AGE_DAYS: Final[int] = 30
"""Maximum age for cached items before cleanup."""

CACHE_MAX_SIZE_GB: Final[float] = 10.0
"""Maximum cache size in GB before cleanup."""

CACHE_MAX_VIDEOS: Final[int] = 500
"""Maximum number of cached videos."""
