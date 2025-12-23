"""Hashtag sanitization utilities.

Provides HashtagSanitizer class with methods for:
- Counting hashtags in text
- Extracting hashtags from text
- Trimming hashtags to a maximum count
- Removing all hashtags from text
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .constants import INSTAGRAM_MAX_HASHTAGS


@dataclass
class SanitizeResult:
    """Result of a hashtag sanitization operation."""

    original_text: str
    sanitized_text: str
    original_count: int
    final_count: int
    removed_count: int
    removed_hashtags: list[str]
    kept_hashtags: list[str]

    @property
    def was_modified(self) -> bool:
        """Check if the text was modified."""
        return self.original_text != self.sanitized_text

    @property
    def exceeded_limit(self) -> bool:
        """Check if original exceeded the limit."""
        return self.removed_count > 0


class HashtagSanitizer:
    """Sanitizer for Instagram hashtags.

    Handles counting, extraction, trimming, and removal of hashtags
    from caption text. Uses the Instagram limit from constants.

    Example:
        sanitizer = HashtagSanitizer(max_hashtags=5)
        result = sanitizer.trim_hashtags("Caption #one #two #three #four #five #six #seven")
        print(result.sanitized_text)  # "Caption #one #two #three #four #five"
        print(result.removed_hashtags)  # ["#six", "#seven"]
    """

    # Regex pattern for hashtags: # followed by alphanumeric chars (including unicode)
    HASHTAG_PATTERN = re.compile(r'#[\w\u0080-\uFFFF]+', re.UNICODE)

    def __init__(self, max_hashtags: Optional[int] = None):
        """Initialize sanitizer.

        Args:
            max_hashtags: Maximum allowed hashtags. Defaults to INSTAGRAM_MAX_HASHTAGS.
        """
        self.max_hashtags = max_hashtags if max_hashtags is not None else INSTAGRAM_MAX_HASHTAGS

    def count_hashtags(self, text: str) -> int:
        """Count the number of hashtags in text.

        Args:
            text: Text to count hashtags in.

        Returns:
            Number of hashtags found.
        """
        if not text:
            return 0
        return len(self.HASHTAG_PATTERN.findall(text))

    def extract_hashtags(self, text: str) -> list[str]:
        """Extract all hashtags from text.

        Args:
            text: Text to extract hashtags from.

        Returns:
            List of hashtags (including # prefix).
        """
        if not text:
            return []
        return self.HASHTAG_PATTERN.findall(text)

    def trim_hashtags(self, text: str, max_count: Optional[int] = None) -> SanitizeResult:
        """Trim hashtags to maximum allowed count, keeping the first N.

        Args:
            text: Text with hashtags.
            max_count: Maximum hashtags to keep. Defaults to self.max_hashtags.

        Returns:
            SanitizeResult with trimmed text and details.
        """
        if not text:
            return SanitizeResult(
                original_text="",
                sanitized_text="",
                original_count=0,
                final_count=0,
                removed_count=0,
                removed_hashtags=[],
                kept_hashtags=[],
            )

        limit = max_count if max_count is not None else self.max_hashtags
        hashtags = self.extract_hashtags(text)
        original_count = len(hashtags)

        if original_count <= limit:
            # No trimming needed
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                original_count=original_count,
                final_count=original_count,
                removed_count=0,
                removed_hashtags=[],
                kept_hashtags=hashtags,
            )

        # Keep first N hashtags, remove the rest
        kept_hashtags = hashtags[:limit]
        removed_hashtags = hashtags[limit:]

        # Build new text by removing excess hashtags
        sanitized_text = text
        for tag in removed_hashtags:
            # Remove the hashtag and any surrounding whitespace
            # Be careful to only remove exact matches
            sanitized_text = re.sub(
                rf'\s*{re.escape(tag)}(?=\s|$)',
                '',
                sanitized_text,
                count=1,  # Only remove first occurrence
            )

        # Clean up any double spaces but preserve newlines
        # Only collapse multiple spaces, not newlines
        sanitized_text = re.sub(r'[ \t]+', ' ', sanitized_text)  # Collapse spaces/tabs only
        sanitized_text = re.sub(r' ?\n ?', '\n', sanitized_text)  # Clean space around newlines
        sanitized_text = sanitized_text.strip()

        return SanitizeResult(
            original_text=text,
            sanitized_text=sanitized_text,
            original_count=original_count,
            final_count=limit,
            removed_count=len(removed_hashtags),
            removed_hashtags=removed_hashtags,
            kept_hashtags=kept_hashtags,
        )

    def remove_all_hashtags(self, text: str) -> SanitizeResult:
        """Remove all hashtags from text.

        Args:
            text: Text with hashtags.

        Returns:
            SanitizeResult with all hashtags removed.
        """
        if not text:
            return SanitizeResult(
                original_text="",
                sanitized_text="",
                original_count=0,
                final_count=0,
                removed_count=0,
                removed_hashtags=[],
                kept_hashtags=[],
            )

        hashtags = self.extract_hashtags(text)
        original_count = len(hashtags)

        if original_count == 0:
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                original_count=0,
                final_count=0,
                removed_count=0,
                removed_hashtags=[],
                kept_hashtags=[],
            )

        # Remove all hashtags
        sanitized_text = self.HASHTAG_PATTERN.sub('', text)

        # Clean up whitespace but preserve newlines
        sanitized_text = re.sub(r'[ \t]+', ' ', sanitized_text)  # Collapse spaces/tabs only
        sanitized_text = re.sub(r' ?\n ?', '\n', sanitized_text)  # Clean space around newlines
        sanitized_text = sanitized_text.strip()

        return SanitizeResult(
            original_text=text,
            sanitized_text=sanitized_text,
            original_count=original_count,
            final_count=0,
            removed_count=original_count,
            removed_hashtags=hashtags,
            kept_hashtags=[],
        )

    def validate(self, text: str) -> tuple[bool, int, str]:
        """Validate hashtag count against limit.

        Args:
            text: Text to validate.

        Returns:
            Tuple of (is_valid, count, message).
        """
        count = self.count_hashtags(text)
        is_valid = count <= self.max_hashtags

        if is_valid:
            message = f"Hashtag count OK ({count}/{self.max_hashtags})"
        else:
            over = count - self.max_hashtags
            message = f"Too many hashtags: {count} (limit: {self.max_hashtags}, over by {over})"

        return is_valid, count, message
