"""Utility modules for Socials Automator."""

from .text_fitting import (
    TextFitter,
    FitResult,
    fit_thumbnail_text,
    fit_news_teaser,
)

from .timestamps import (
    # Current time
    now_utc,
    now_local,
    now_local_naive,
    # Timezone conversion
    to_utc,
    to_utc_naive,
    to_local,
    to_local_naive,
    # Parsing
    parse_timestamp,
    parse_timestamp_lenient,
    # Formatting
    format_timestamp,
    format_timestamp_utc,
    format_local,
    format_relative,
    # Comparison
    timestamps_match,
    time_diff_seconds,
    time_diff_human,
    # Helpers
    get_local_timezone,
    get_utc_offset_hours,
)

__all__ = [
    # Text fitting
    "TextFitter",
    "FitResult",
    "fit_thumbnail_text",
    "fit_news_teaser",
    # Timestamps
    "now_utc",
    "now_local",
    "now_local_naive",
    "to_utc",
    "to_utc_naive",
    "to_local",
    "to_local_naive",
    "parse_timestamp",
    "parse_timestamp_lenient",
    "format_timestamp",
    "format_timestamp_utc",
    "format_local",
    "format_relative",
    "timestamps_match",
    "time_diff_seconds",
    "time_diff_human",
    "get_local_timezone",
    "get_utc_offset_hours",
]
