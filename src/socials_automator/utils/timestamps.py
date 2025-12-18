"""Timestamp utilities for consistent timezone handling.

All internal timestamps should be stored and compared in UTC.
Local time is only used for display purposes.

Usage:
    from socials_automator.utils.timestamps import (
        now_utc,
        now_local,
        to_utc,
        to_local,
        parse_timestamp,
        format_timestamp,
        format_local,
    )

    # Get current time
    utc_now = now_utc()  # For storage/comparison
    local_now = now_local()  # For display

    # Convert between timezones
    utc_dt = to_utc(some_datetime)
    local_dt = to_local(utc_dt)

    # Parse from string (auto-detects format)
    dt = parse_timestamp("2025-12-17T15:20:21+0000")
    dt = parse_timestamp("2025-12-17 12:43:03")  # Assumes local

    # Format for storage or display
    iso_str = format_timestamp(dt)  # ISO format with timezone
    display_str = format_local(dt)  # "Dec 17, 2025 12:43 PM"
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import re


# =============================================================================
# TIMEZONE HELPERS
# =============================================================================

def get_local_timezone() -> timezone:
    """Get the local timezone as a timezone object.

    Returns:
        Local timezone with current UTC offset.
    """
    local_dt = datetime.now().astimezone()
    return local_dt.tzinfo


def get_utc_offset_hours() -> float:
    """Get local timezone offset from UTC in hours.

    Returns:
        Offset in hours (e.g., 1.0 for UTC+1, -5.0 for UTC-5).
    """
    local_dt = datetime.now().astimezone()
    offset = local_dt.utcoffset()
    if offset is None:
        return 0.0
    return offset.total_seconds() / 3600


# =============================================================================
# CURRENT TIME
# =============================================================================

def now_utc() -> datetime:
    """Get current time in UTC with timezone info.

    Use this for:
    - Storing timestamps in metadata
    - Comparing timestamps
    - API calls

    Returns:
        Current UTC datetime with tzinfo.
    """
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    """Get current time in local timezone with timezone info.

    Use this for:
    - Display to user
    - Log messages

    Returns:
        Current local datetime with tzinfo.
    """
    return datetime.now().astimezone()


def now_local_naive() -> datetime:
    """Get current local time without timezone info.

    Use this for:
    - Backwards compatibility with existing code
    - File naming (dates/times in folder names)

    Returns:
        Current local datetime without tzinfo.
    """
    return datetime.now()


# =============================================================================
# TIMEZONE CONVERSION
# =============================================================================

def to_utc(dt: datetime) -> datetime:
    """Convert any datetime to UTC.

    If the datetime has no timezone info, assumes it's local time.

    Args:
        dt: Datetime to convert (with or without tzinfo).

    Returns:
        Datetime in UTC with tzinfo.
    """
    if dt.tzinfo is None:
        # Assume local time
        local_tz = get_local_timezone()
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc)


def to_utc_naive(dt: datetime) -> datetime:
    """Convert any datetime to UTC without timezone info.

    Useful for comparing timestamps from different sources.

    Args:
        dt: Datetime to convert.

    Returns:
        Datetime in UTC without tzinfo (naive).
    """
    utc_dt = to_utc(dt)
    return utc_dt.replace(tzinfo=None)


def to_local(dt: datetime) -> datetime:
    """Convert any datetime to local timezone.

    If the datetime has no timezone info, assumes it's UTC.

    Args:
        dt: Datetime to convert (with or without tzinfo).

    Returns:
        Datetime in local timezone with tzinfo.
    """
    if dt.tzinfo is None:
        # Assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def to_local_naive(dt: datetime) -> datetime:
    """Convert any datetime to local time without timezone info.

    Args:
        dt: Datetime to convert.

    Returns:
        Datetime in local time without tzinfo (naive).
    """
    local_dt = to_local(dt)
    return local_dt.replace(tzinfo=None)


# =============================================================================
# PARSING
# =============================================================================

def parse_timestamp(
    s: str,
    assume_local: bool = True,
) -> datetime:
    """Parse a timestamp string to datetime.

    Handles multiple formats:
    - ISO 8601: "2025-12-17T15:20:21+0000"
    - ISO 8601 with Z: "2025-12-17T15:20:21Z"
    - Naive ISO: "2025-12-17T15:20:21"
    - Space-separated: "2025-12-17 15:20:21"
    - With microseconds: "2025-12-17 15:20:21.123456"

    Args:
        s: Timestamp string to parse.
        assume_local: If True and no timezone in string, assume local time.
                     If False, assume UTC.

    Returns:
        Datetime in UTC with tzinfo.

    Raises:
        ValueError: If string cannot be parsed.
    """
    if not s:
        raise ValueError("Empty timestamp string")

    # Normalize common variations
    s = s.strip()

    # Handle 'Z' suffix (UTC)
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'

    # Handle timezone without colon (e.g., +0000 -> +00:00)
    # Match patterns like +0000, -0500, etc.
    tz_match = re.search(r'([+-])(\d{2})(\d{2})$', s)
    if tz_match:
        sign, hours, minutes = tz_match.groups()
        s = s[:-5] + f'{sign}{hours}:{minutes}'

    # Try parsing with fromisoformat
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Try common alternative formats
        for fmt in [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
        ]:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse timestamp: {s}")

    # Handle timezone
    if dt.tzinfo is None:
        if assume_local:
            dt = dt.replace(tzinfo=get_local_timezone())
        else:
            dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC
    return dt.astimezone(timezone.utc)


def parse_timestamp_lenient(
    s: str,
    default: Optional[datetime] = None,
) -> Optional[datetime]:
    """Parse timestamp, returning default on failure instead of raising.

    Args:
        s: Timestamp string to parse.
        default: Value to return if parsing fails.

    Returns:
        Parsed datetime in UTC, or default if parsing failed.
    """
    try:
        return parse_timestamp(s)
    except (ValueError, TypeError):
        return default


# =============================================================================
# FORMATTING
# =============================================================================

def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO 8601 string with timezone.

    Use this for:
    - Storing in metadata.json
    - API responses
    - Logs

    Args:
        dt: Datetime to format.

    Returns:
        ISO 8601 string like "2025-12-17T15:20:21+00:00".
    """
    # Ensure we have timezone info
    if dt.tzinfo is None:
        dt = to_utc(dt)
    return dt.isoformat()


def format_timestamp_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 string in UTC.

    Args:
        dt: Datetime to format.

    Returns:
        ISO 8601 string in UTC like "2025-12-17T15:20:21+00:00".
    """
    utc_dt = to_utc(dt)
    return utc_dt.isoformat()


def format_local(
    dt: datetime,
    include_seconds: bool = False,
) -> str:
    """Format datetime for human-readable display in local time.

    Args:
        dt: Datetime to format.
        include_seconds: Whether to include seconds.

    Returns:
        Human-readable string like "Dec 17, 2025 3:20 PM".
    """
    local_dt = to_local(dt)
    if include_seconds:
        return local_dt.strftime('%b %d, %Y %I:%M:%S %p')
    return local_dt.strftime('%b %d, %Y %I:%M %p')


def format_relative(dt: datetime) -> str:
    """Format datetime as relative time (e.g., "2 hours ago").

    Args:
        dt: Datetime to format.

    Returns:
        Relative time string.
    """
    utc_dt = to_utc(dt)
    now = now_utc()
    diff = now - utc_dt

    seconds = diff.total_seconds()

    if seconds < 0:
        return "in the future"
    elif seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return format_local(dt)


# =============================================================================
# COMPARISON HELPERS
# =============================================================================

def timestamps_match(
    dt1: datetime,
    dt2: datetime,
    tolerance_seconds: float = 300,
) -> bool:
    """Check if two timestamps are within tolerance of each other.

    Handles timezone conversion automatically.

    Args:
        dt1: First datetime.
        dt2: Second datetime.
        tolerance_seconds: Maximum allowed difference in seconds (default 5 min).

    Returns:
        True if timestamps are within tolerance.
    """
    utc1 = to_utc_naive(dt1)
    utc2 = to_utc_naive(dt2)
    diff = abs((utc1 - utc2).total_seconds())
    return diff <= tolerance_seconds


def time_diff_seconds(dt1: datetime, dt2: datetime) -> float:
    """Get the absolute time difference between two datetimes in seconds.

    Handles timezone conversion automatically.

    Args:
        dt1: First datetime.
        dt2: Second datetime.

    Returns:
        Absolute difference in seconds.
    """
    utc1 = to_utc_naive(dt1)
    utc2 = to_utc_naive(dt2)
    return abs((utc1 - utc2).total_seconds())


def time_diff_human(dt1: datetime, dt2: datetime) -> str:
    """Get human-readable time difference between two datetimes.

    Args:
        dt1: First datetime.
        dt2: Second datetime.

    Returns:
        Human-readable string like "2h 30m" or "45s".
    """
    seconds = time_diff_seconds(dt1, dt2)

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        if secs > 0:
            return f"{minutes}m {secs}s"
        return f"{minutes}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        if hours > 0:
            return f"{days}d {hours}h"
        return f"{days}d"
