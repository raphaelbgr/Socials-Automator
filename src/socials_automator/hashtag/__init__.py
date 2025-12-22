"""Hashtag validation and sanitization module.

Provides:
- INSTAGRAM_MAX_HASHTAGS: Current Instagram hashtag limit (5 as of Dec 2025)
- HashtagSanitizer: Class for counting, trimming, and removing hashtags
- HashtagValidator: Pipeline step for validating hashtags before upload
"""

from .constants import INSTAGRAM_MAX_HASHTAGS, DEFAULT_FALLBACK_HASHTAGS
from .sanitizer import HashtagSanitizer
from .validator import HashtagValidationResult, validate_hashtags_in_caption

__all__ = [
    "INSTAGRAM_MAX_HASHTAGS",
    "DEFAULT_FALLBACK_HASHTAGS",
    "HashtagSanitizer",
    "HashtagValidationResult",
    "validate_hashtags_in_caption",
]
