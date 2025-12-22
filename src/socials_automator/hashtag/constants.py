"""Hashtag constants and limits.

Instagram reduced the hashtag limit from 30 to 5 in December 2025.
This module provides a central place for this limit to be configured.
"""

# Instagram maximum hashtags per post (December 2025 limit)
INSTAGRAM_MAX_HASHTAGS: int = 5

# Minimum hashtags to keep (for fallback scenarios)
MIN_HASHTAGS: int = 1

# Default fallback hashtags when generation fails
DEFAULT_FALLBACK_HASHTAGS: list[str] = [
    "viral",
    "reels",
    "fyp",
]
