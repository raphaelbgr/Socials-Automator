"""Pure parsing functions for CLI arguments."""

from __future__ import annotations

import re
from typing import Tuple


def parse_interval(interval: str) -> int:
    """Parse interval string like '5m', '1h', '30s' to seconds.

    Pure function - no side effects.

    Args:
        interval: String like '5m', '1h', '30s', or just '60'

    Returns:
        Interval in seconds

    Raises:
        ValueError: If format is invalid
    """
    match = re.match(r"^(\d+)(s|m|h)?$", interval.lower().strip())
    if not match:
        raise ValueError(f"Invalid interval format: {interval}. Use format like 5m, 1h, 30s")

    value = int(match.group(1))
    unit = match.group(2) or "s"

    multipliers = {"s": 1, "m": 60, "h": 3600}
    return value * multipliers[unit]


def parse_length(length: str) -> float:
    """Parse length string like '30s', '1m', '90s' to seconds.

    Pure function - no side effects.

    Args:
        length: String like '30s', '1m', '1m30s', or just '60'

    Returns:
        Length in seconds

    Raises:
        ValueError: If format is invalid
    """
    length = length.strip().lower()

    # Handle combined format like '1m30s'
    combined_match = re.match(r"^(\d+)m(\d+)s$", length)
    if combined_match:
        minutes = int(combined_match.group(1))
        seconds = int(combined_match.group(2))
        return minutes * 60 + seconds

    # Handle simple formats
    if length.endswith("m"):
        return float(length[:-1]) * 60
    elif length.endswith("s"):
        return float(length[:-1])
    else:
        # Assume seconds if no suffix
        return float(length)


def parse_voice_preset(
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> Tuple[str, str, str]:
    """Resolve voice preset to (voice, rate, pitch).

    Pure function - no side effects.

    Args:
        voice: Voice name or preset name
        rate: Speech rate adjustment
        pitch: Pitch adjustment

    Returns:
        Tuple of (resolved_voice, rate, pitch)
    """
    # Voice presets with custom rate/pitch
    presets = {
        "adam_excited": ("rvc_adam", "+12%", "+3Hz"),
        "rvc_adam_excited": ("rvc_adam", "+12%", "+3Hz"),
    }

    if voice in presets:
        return presets[voice]

    # Voice aliases
    aliases = {
        "tiktok-adam": "rvc_adam",
        "adam": "rvc_adam",
    }

    resolved_voice = aliases.get(voice, voice)
    return (resolved_voice, rate, pitch)


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Pure function - no side effects.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like '1m30s' or '45s'
    """
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s" if secs else f"{mins}m"
    return f"{int(seconds)}s"


def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable size.

    Pure function - no side effects.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like '1.5 MB' or '256 KB'
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
