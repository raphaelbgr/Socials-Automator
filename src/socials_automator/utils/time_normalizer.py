"""Time normalization utilities for converting time mentions in content to UTC.

This module provides functionality to detect and convert time mentions in
news content to UTC format for consistent display.

Usage:
    from socials_automator.utils.time_normalizer import TimeNormalizer

    normalizer = TimeNormalizer()

    # Convert times in text to UTC
    text = "The concert starts at 7pm EST on Friday"
    normalized = normalizer.normalize_times(text, source_tz="EST")
    # -> "The concert starts at 12:00 AM UTC on Friday"

    # Extract time mentions without converting
    times = normalizer.extract_times(text)
    # -> [TimeMatch(hour=19, minute=0, tz="EST", ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .timestamps import (
    TIMEZONE_OFFSETS,
    convert_time_to_utc,
    format_time_utc,
    get_region_timezone,
)


@dataclass
class TimeMatch:
    """A time mention found in text."""

    original: str  # The original matched text (e.g., "7pm EST")
    hour: int  # Hour in 24h format (0-23)
    minute: int  # Minute (0-59)
    timezone: Optional[str]  # Detected timezone abbreviation
    start: int  # Start position in text
    end: int  # End position in text

    @property
    def has_timezone(self) -> bool:
        """Check if timezone was explicitly mentioned."""
        return self.timezone is not None


class TimeNormalizer:
    """Normalizes time mentions in text to UTC.

    Detects common time formats and converts them to UTC with clear labeling.
    This ensures all news content displays times consistently regardless of
    the source's timezone.

    Supported formats:
    - "7pm", "7:30pm", "7:30 pm", "7:30 PM"
    - "19:00", "1900"
    - "7pm EST", "7:30 PM PST", "15:00 KST"
    - "7 o'clock", "7 o'clock pm"

    Example:
        normalizer = TimeNormalizer()

        # With explicit timezone
        result = normalizer.normalize_times("Show starts at 8pm EST")
        # -> "Show starts at 1:00 AM UTC (next day)"

        # With source region fallback
        result = normalizer.normalize_times(
            "Show starts at 8pm",
            source_region="us"
        )
        # -> "Show starts at 1:00 AM UTC (next day)"
    """

    # Build timezone pattern from known abbreviations
    TZ_PATTERN = "|".join(re.escape(tz) for tz in TIMEZONE_OFFSETS.keys())

    # Time patterns (order matters - more specific first)
    PATTERNS = [
        # 24-hour with timezone: "15:30 KST", "23:00 EST"
        re.compile(
            rf"(\d{{1,2}}):(\d{{2}})\s*({TZ_PATTERN})",
            re.IGNORECASE
        ),
        # 12-hour with timezone: "7:30pm EST", "11:00 AM PST"
        re.compile(
            rf"(\d{{1,2}}):(\d{{2}})\s*(am|pm)\s*({TZ_PATTERN})",
            re.IGNORECASE
        ),
        # 12-hour no minutes with timezone: "7pm EST", "11 AM PST"
        re.compile(
            rf"(\d{{1,2}})\s*(am|pm)\s*({TZ_PATTERN})",
            re.IGNORECASE
        ),
        # 24-hour without timezone: "15:30", "23:00"
        re.compile(
            r"(\d{1,2}):(\d{2})(?!\s*(?:am|pm|" + TZ_PATTERN + r"))",
            re.IGNORECASE
        ),
        # 12-hour with minutes: "7:30pm", "11:00 AM"
        re.compile(
            r"(\d{1,2}):(\d{2})\s*(am|pm)(?!\s*(?:" + TZ_PATTERN + r"))",
            re.IGNORECASE
        ),
        # 12-hour no minutes: "7pm", "11 AM"
        re.compile(
            r"(\d{1,2})\s*(am|pm)(?!\s*(?:" + TZ_PATTERN + r"))",
            re.IGNORECASE
        ),
        # O'clock format: "7 o'clock", "7 o'clock pm"
        re.compile(
            r"(\d{1,2})\s+o['\u2019]?clock(?:\s+(am|pm))?",
            re.IGNORECASE
        ),
    ]

    def __init__(self, default_timezone: str = "UTC"):
        """Initialize the time normalizer.

        Args:
            default_timezone: Default timezone to assume when none is specified.
        """
        self.default_timezone = default_timezone

    def extract_times(self, text: str) -> list[TimeMatch]:
        """Extract time mentions from text.

        Args:
            text: Text to search for time mentions.

        Returns:
            List of TimeMatch objects for each time found.
        """
        matches = []
        used_positions = set()

        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                # Skip if this position was already matched by a more specific pattern
                if any(pos in used_positions for pos in range(match.start(), match.end())):
                    continue

                time_match = self._parse_match(match)
                if time_match:
                    matches.append(time_match)
                    used_positions.update(range(match.start(), match.end()))

        # Sort by position in text
        matches.sort(key=lambda m: m.start)
        return matches

    def _parse_match(self, match: re.Match) -> Optional[TimeMatch]:
        """Parse a regex match into a TimeMatch object."""
        groups = match.groups()
        original = match.group(0)

        # Determine hour, minute, am/pm, timezone based on pattern
        hour = None
        minute = 0
        am_pm = None
        timezone = None

        # Try to extract components
        for i, g in enumerate(groups):
            if g is None:
                continue
            g_upper = g.upper()

            # Check if it's a timezone
            if g_upper in TIMEZONE_OFFSETS:
                timezone = g_upper
            # Check if it's am/pm
            elif g_upper in ("AM", "PM"):
                am_pm = g_upper
            # Otherwise it's a number
            elif g.isdigit():
                num = int(g)
                if hour is None:
                    hour = num
                else:
                    minute = num

        if hour is None:
            return None

        # Convert 12-hour to 24-hour
        if am_pm:
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0

        # Validate hour
        if hour < 0 or hour > 23:
            return None
        if minute < 0 or minute > 59:
            return None

        return TimeMatch(
            original=original,
            hour=hour,
            minute=minute,
            timezone=timezone,
            start=match.start(),
            end=match.end(),
        )

    def normalize_times(
        self,
        text: str,
        source_tz: Optional[str] = None,
        source_region: Optional[str] = None,
    ) -> str:
        """Convert all time mentions in text to UTC.

        Args:
            text: Text with time mentions to normalize.
            source_tz: Timezone to assume for times without explicit timezone.
            source_region: Region to derive timezone from (fallback for source_tz).

        Returns:
            Text with time mentions converted to UTC format.

        Example:
            >>> normalizer.normalize_times("Show at 7pm EST")
            "Show at 12:00 AM UTC (next day)"
        """
        # Determine fallback timezone
        fallback_tz = source_tz
        if not fallback_tz and source_region:
            fallback_tz = get_region_timezone(source_region)
        if not fallback_tz:
            fallback_tz = self.default_timezone

        # Extract all time matches
        matches = self.extract_times(text)

        if not matches:
            return text

        # Replace from end to start to preserve positions
        result = text
        for match in reversed(matches):
            # Determine timezone to use
            tz = match.timezone or fallback_tz

            # Convert to UTC
            utc_hour, utc_minute, day_offset = convert_time_to_utc(
                match.hour, match.minute, tz
            )

            # Format the UTC time
            utc_str = format_time_utc(utc_hour, utc_minute, day_offset)

            # Replace in text
            result = result[:match.start] + utc_str + result[match.end:]

        return result

    def normalize_with_annotations(
        self,
        text: str,
        source_tz: Optional[str] = None,
        source_region: Optional[str] = None,
    ) -> tuple[str, list[dict]]:
        """Convert times to UTC and return annotations.

        Similar to normalize_times but also returns metadata about
        what was changed.

        Args:
            text: Text with time mentions to normalize.
            source_tz: Source timezone.
            source_region: Source region for timezone fallback.

        Returns:
            Tuple of (normalized_text, annotations) where annotations
            is a list of dicts with conversion details.
        """
        fallback_tz = source_tz
        if not fallback_tz and source_region:
            fallback_tz = get_region_timezone(source_region)
        if not fallback_tz:
            fallback_tz = self.default_timezone

        matches = self.extract_times(text)

        if not matches:
            return text, []

        annotations = []
        result = text

        for match in reversed(matches):
            tz = match.timezone or fallback_tz
            utc_hour, utc_minute, day_offset = convert_time_to_utc(
                match.hour, match.minute, tz
            )
            utc_str = format_time_utc(utc_hour, utc_minute, day_offset)

            annotations.append({
                "original": match.original,
                "converted": utc_str,
                "source_tz": tz,
                "source_hour": match.hour,
                "source_minute": match.minute,
                "utc_hour": utc_hour,
                "utc_minute": utc_minute,
                "day_offset": day_offset,
            })

            result = result[:match.start] + utc_str + result[match.end:]

        # Reverse annotations to match text order
        annotations.reverse()

        return result, annotations


# Module-level convenience instance
_default_normalizer: Optional[TimeNormalizer] = None


def get_time_normalizer() -> TimeNormalizer:
    """Get or create the default TimeNormalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = TimeNormalizer()
    return _default_normalizer


def normalize_times(
    text: str,
    source_tz: Optional[str] = None,
    source_region: Optional[str] = None,
) -> str:
    """Convenience function to normalize times in text.

    Args:
        text: Text with time mentions.
        source_tz: Source timezone abbreviation.
        source_region: Source region for timezone fallback.

    Returns:
        Text with times converted to UTC.
    """
    return get_time_normalizer().normalize_times(text, source_tz, source_region)
