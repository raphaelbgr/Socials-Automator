"""Hashtag validation for pipeline integration.

Provides validation functions and result types for use in
generate-reel and upload-reel pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import INSTAGRAM_MAX_HASHTAGS
from .sanitizer import HashtagSanitizer, SanitizeResult

_logger = logging.getLogger("hashtag_validator")


@dataclass
class HashtagValidationResult:
    """Result of hashtag validation on a reel."""

    is_valid: bool
    original_count: int
    final_count: int
    was_trimmed: bool
    removed_hashtags: list[str]
    caption_before: str
    caption_after: str
    error: Optional[str] = None

    @property
    def message(self) -> str:
        """Get a human-readable status message."""
        if self.error:
            return f"[X] Hashtag validation error: {self.error}"
        if self.was_trimmed:
            return f"[!] Trimmed hashtags: {self.original_count} -> {self.final_count} (removed {len(self.removed_hashtags)})"
        if self.original_count == 0:
            return "[OK] No hashtags in caption"
        return f"[OK] Hashtag count valid ({self.original_count}/{INSTAGRAM_MAX_HASHTAGS})"


def validate_hashtags_in_caption(
    caption: str,
    max_hashtags: Optional[int] = None,
    auto_trim: bool = True,
) -> HashtagValidationResult:
    """Validate and optionally trim hashtags in a caption.

    Args:
        caption: Caption text to validate.
        max_hashtags: Maximum allowed hashtags. Defaults to INSTAGRAM_MAX_HASHTAGS.
        auto_trim: If True, automatically trim excess hashtags.

    Returns:
        HashtagValidationResult with validation status and trimmed caption.
    """
    limit = max_hashtags if max_hashtags is not None else INSTAGRAM_MAX_HASHTAGS
    sanitizer = HashtagSanitizer(max_hashtags=limit)

    try:
        is_valid, count, _ = sanitizer.validate(caption)

        if is_valid:
            return HashtagValidationResult(
                is_valid=True,
                original_count=count,
                final_count=count,
                was_trimmed=False,
                removed_hashtags=[],
                caption_before=caption,
                caption_after=caption,
            )

        if not auto_trim:
            return HashtagValidationResult(
                is_valid=False,
                original_count=count,
                final_count=count,
                was_trimmed=False,
                removed_hashtags=[],
                caption_before=caption,
                caption_after=caption,
                error=f"Caption has {count} hashtags, limit is {limit}",
            )

        # Auto-trim hashtags
        result: SanitizeResult = sanitizer.trim_hashtags(caption)

        _logger.info(
            f"HASHTAG_TRIM | original={result.original_count} | "
            f"final={result.final_count} | removed={result.removed_hashtags}"
        )

        return HashtagValidationResult(
            is_valid=True,
            original_count=result.original_count,
            final_count=result.final_count,
            was_trimmed=True,
            removed_hashtags=result.removed_hashtags,
            caption_before=result.original_text,
            caption_after=result.sanitized_text,
        )

    except Exception as e:
        _logger.error(f"Hashtag validation failed: {e}")
        return HashtagValidationResult(
            is_valid=False,
            original_count=0,
            final_count=0,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before=caption,
            caption_after=caption,
            error=str(e),
        )


def validate_reel_hashtags(
    reel_path: Path,
    max_hashtags: Optional[int] = None,
    auto_trim: bool = True,
    update_file: bool = True,
) -> HashtagValidationResult:
    """Validate hashtags in a reel's caption+hashtags.txt file.

    Args:
        reel_path: Path to reel folder.
        max_hashtags: Maximum allowed hashtags. Defaults to INSTAGRAM_MAX_HASHTAGS.
        auto_trim: If True, automatically trim excess hashtags.
        update_file: If True and trimming occurred, update the file.

    Returns:
        HashtagValidationResult with validation status.
    """
    caption_path = reel_path / "caption+hashtags.txt"

    if not caption_path.exists():
        return HashtagValidationResult(
            is_valid=False,
            original_count=0,
            final_count=0,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before="",
            caption_after="",
            error="caption+hashtags.txt not found",
        )

    try:
        caption = caption_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return HashtagValidationResult(
            is_valid=False,
            original_count=0,
            final_count=0,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before="",
            caption_after="",
            error=f"Failed to read caption file: {e}",
        )

    result = validate_hashtags_in_caption(
        caption=caption,
        max_hashtags=max_hashtags,
        auto_trim=auto_trim,
    )

    # Update file if trimmed and update_file is True
    if result.was_trimmed and update_file:
        try:
            caption_path.write_text(result.caption_after, encoding="utf-8")
            _logger.info(f"Updated {caption_path} with trimmed hashtags")
        except Exception as e:
            _logger.error(f"Failed to update caption file: {e}")
            result.error = f"Trimmed but failed to save: {e}"

    return result


def remove_hashtags_from_caption(caption: str) -> str:
    """Remove all hashtags from a caption.

    Utility function for fallback upload without hashtags.

    Args:
        caption: Caption text with hashtags.

    Returns:
        Caption with all hashtags removed.
    """
    sanitizer = HashtagSanitizer()
    result = sanitizer.remove_all_hashtags(caption)
    return result.sanitized_text
