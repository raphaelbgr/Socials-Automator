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
        # Timezone conversion for content
        tz_abbrev_to_offset,
        convert_time_to_utc,
        format_time_utc,
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

    # Convert time mentions to UTC (for news content)
    utc_h, utc_m, day_off = convert_time_to_utc(19, 0, "EST")  # 7pm EST
    display = format_time_utc(utc_h, utc_m, day_off)  # "12:00 AM UTC (next day)"
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


# =============================================================================
# TIMEZONE ABBREVIATIONS FOR CONTENT CONVERSION
# =============================================================================

# Common timezone abbreviations to UTC offset in hours
# Used for converting time mentions in news content to UTC
TIMEZONE_OFFSETS: dict[str, float] = {
    # UTC
    "UTC": 0,
    "GMT": 0,
    "Z": 0,

    # North America - Standard
    "EST": -5,   # Eastern Standard
    "CST": -6,   # Central Standard
    "MST": -7,   # Mountain Standard
    "PST": -8,   # Pacific Standard
    "AKST": -9,  # Alaska Standard
    "HST": -10,  # Hawaii Standard

    # North America - Daylight
    "EDT": -4,   # Eastern Daylight
    "CDT": -5,   # Central Daylight
    "MDT": -6,   # Mountain Daylight
    "PDT": -7,   # Pacific Daylight
    "AKDT": -8,  # Alaska Daylight

    # North America - Short forms (assume standard)
    "ET": -5,    # Eastern Time
    "CT": -6,    # Central Time
    "MT": -7,    # Mountain Time
    "PT": -8,    # Pacific Time

    # Europe
    "WET": 0,    # Western European
    "WEST": 1,   # Western European Summer
    "CET": 1,    # Central European
    "CEST": 2,   # Central European Summer
    "EET": 2,    # Eastern European
    "EEST": 3,   # Eastern European Summer
    "BST": 1,    # British Summer Time
    "ISH": 0,    # Ireland (same as GMT)

    # Asia
    "KST": 9,    # Korea Standard
    "JST": 9,    # Japan Standard
    "CST_CN": 8, # China Standard
    "HKT": 8,    # Hong Kong
    "SGT": 8,    # Singapore
    "IST": 5.5,  # India Standard
    "PKT": 5,    # Pakistan
    "ICT": 7,    # Indochina (Thailand, Vietnam)
    "WIB": 7,    # Western Indonesia
    "PHT": 8,    # Philippines

    # South America
    "BRT": -3,   # Brasilia
    "BRST": -2,  # Brasilia Summer
    "ART": -3,   # Argentina
    "CLT": -4,   # Chile Standard
    "CLST": -3,  # Chile Summer
    "COT": -5,   # Colombia
    "PET": -5,   # Peru
    "VET": -4,   # Venezuela

    # Australia/Pacific
    "AEST": 10,  # Australian Eastern Standard
    "AEDT": 11,  # Australian Eastern Daylight
    "ACST": 9.5, # Australian Central Standard
    "ACDT": 10.5,# Australian Central Daylight
    "AWST": 8,   # Australian Western Standard
    "NZST": 12,  # New Zealand Standard
    "NZDT": 13,  # New Zealand Daylight

    # Middle East
    "GST": 4,    # Gulf Standard (UAE, Oman)
    "AST_AR": 3, # Arabia Standard (Saudi, Kuwait, Qatar)
    "IRST": 3.5, # Iran Standard
    "IDT": 3,    # Israel Daylight
    "IST_IL": 2, # Israel Standard
    "TRT": 3,    # Turkey

    # Africa
    "CAT": 2,    # Central Africa
    "EAT": 3,    # East Africa
    "WAT": 1,    # West Africa
    "SAST": 2,   # South Africa Standard
}

# Region to typical timezone mapping (for sources without explicit TZ)
REGION_DEFAULT_TIMEZONES: dict[str, str] = {
    "us": "EST",
    "uk": "GMT",
    "korea": "KST",
    "japan": "JST",
    "india": "IST",
    "latam": "BRT",
    "europe": "CET",
    "australia": "AEST",
    "middle_east": "GST",
    "china": "CST_CN",
}


def tz_abbrev_to_offset(abbrev: str) -> Optional[float]:
    """Convert timezone abbreviation to UTC offset in hours.

    Args:
        abbrev: Timezone abbreviation like "EST", "PST", "KST".

    Returns:
        UTC offset in hours (e.g., -5 for EST, 9 for KST).
        Returns None if abbreviation is unknown.

    Examples:
        >>> tz_abbrev_to_offset("EST")
        -5
        >>> tz_abbrev_to_offset("KST")
        9
        >>> tz_abbrev_to_offset("IST")
        5.5
    """
    return TIMEZONE_OFFSETS.get(abbrev.upper())


def get_region_timezone(region: str) -> str:
    """Get default timezone abbreviation for a region.

    Args:
        region: Region identifier (e.g., "us", "korea", "latam").

    Returns:
        Timezone abbreviation (e.g., "EST", "KST", "BRT").
    """
    return REGION_DEFAULT_TIMEZONES.get(region.lower(), "UTC")


def convert_time_to_utc(
    hour: int,
    minute: int,
    source_tz: str,
) -> tuple[int, int, int]:
    """Convert a time from source timezone to UTC.

    Args:
        hour: Hour in 24-hour format (0-23).
        minute: Minute (0-59).
        source_tz: Timezone abbreviation (e.g., "EST", "PST", "KST").

    Returns:
        Tuple of (utc_hour, utc_minute, day_offset) where:
        - utc_hour: Hour in UTC (0-23)
        - utc_minute: Minute (0-59)
        - day_offset: -1 (previous day), 0 (same day), or 1 (next day)

    Examples:
        >>> convert_time_to_utc(19, 0, "EST")  # 7pm EST
        (0, 0, 1)  # 12:00 AM UTC next day

        >>> convert_time_to_utc(9, 30, "KST")  # 9:30am Korea
        (0, 30, 0)  # 12:30 AM UTC same day
    """
    offset = tz_abbrev_to_offset(source_tz)
    if offset is None:
        # Unknown timezone, assume UTC
        return hour, minute, 0

    # Convert to total minutes, then subtract offset
    total_minutes = hour * 60 + minute
    utc_minutes = total_minutes - int(offset * 60)

    # Handle day rollover
    day_offset = 0
    if utc_minutes < 0:
        utc_minutes += 24 * 60
        day_offset = -1
    elif utc_minutes >= 24 * 60:
        utc_minutes -= 24 * 60
        day_offset = 1

    utc_hour = utc_minutes // 60
    utc_minute = utc_minutes % 60

    return utc_hour, utc_minute, day_offset


def format_time_utc(
    hour: int,
    minute: int = 0,
    day_offset: int = 0,
    use_24h: bool = False,
) -> str:
    """Format time for display with UTC suffix.

    Creates human-readable time strings for news content.
    All times should be displayed in UTC for consistency.

    Args:
        hour: Hour in 24-hour format (0-23).
        minute: Minute (0-59).
        day_offset: -1 (previous day), 0 (same day), or 1 (next day).
        use_24h: If True, use 24-hour format. Default is 12-hour with AM/PM.

    Returns:
        Formatted string like "8:00 PM UTC" or "20:00 UTC".

    Examples:
        >>> format_time_utc(20, 0)
        "8:00 PM UTC"

        >>> format_time_utc(0, 0, day_offset=1)
        "12:00 AM UTC (next day)"

        >>> format_time_utc(14, 30, use_24h=True)
        "14:30 UTC"
    """
    if use_24h:
        time_str = f"{hour:02d}:{minute:02d}"
    else:
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        time_str = f"{display_hour}:{minute:02d} {am_pm}"

    suffix = " UTC"
    if day_offset == -1:
        suffix = " UTC (prev day)"
    elif day_offset == 1:
        suffix = " UTC (next day)"

    return time_str + suffix


def convert_and_format_time(
    hour: int,
    minute: int,
    source_tz: str,
    use_24h: bool = False,
) -> str:
    """Convert time from source timezone to UTC and format for display.

    Convenience function combining convert_time_to_utc and format_time_utc.

    Args:
        hour: Hour in source timezone (0-23).
        minute: Minute (0-59).
        source_tz: Source timezone abbreviation.
        use_24h: Use 24-hour format.

    Returns:
        Formatted UTC time string.

    Examples:
        >>> convert_and_format_time(19, 0, "EST")
        "12:00 AM UTC (next day)"

        >>> convert_and_format_time(15, 30, "KST")
        "6:30 AM UTC"
    """
    utc_hour, utc_minute, day_offset = convert_time_to_utc(hour, minute, source_tz)
    return format_time_utc(utc_hour, utc_minute, day_offset, use_24h)
