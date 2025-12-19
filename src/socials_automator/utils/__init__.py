"""Utility modules for Socials Automator."""

from .caption_audit import (
    # Audit classes
    CaptionAuditor,
    CaptionIssue,
    CaptionAuditResult,
    LogError,
    # Audit functions
    audit_profile_captions,
    audit_and_verify_captions,
    generate_caption_fix_report,
    # Sync classes
    CaptionSyncer,
    ReelSyncStatus,
    SyncResult,
    # Sync functions
    sync_profile_captions,
    sync_all_captions,
)

from .text_fitting import (
    TextFitter,
    FitResult,
    fit_thumbnail_text,
    fit_news_teaser,
)

from .time_normalizer import (
    TimeNormalizer,
    TimeMatch,
    get_time_normalizer,
    normalize_times,
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
    # Timezone abbreviations for content
    TIMEZONE_OFFSETS,
    REGION_DEFAULT_TIMEZONES,
    tz_abbrev_to_offset,
    get_region_timezone,
    convert_time_to_utc,
    format_time_utc,
    convert_and_format_time,
)

__all__ = [
    # Caption audit
    "CaptionAuditor",
    "CaptionIssue",
    "CaptionAuditResult",
    "LogError",
    "audit_profile_captions",
    "audit_and_verify_captions",
    "generate_caption_fix_report",
    # Text fitting
    "TextFitter",
    "FitResult",
    "fit_thumbnail_text",
    "fit_news_teaser",
    # Time normalization
    "TimeNormalizer",
    "TimeMatch",
    "get_time_normalizer",
    "normalize_times",
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
    # Timezone abbreviations for content
    "TIMEZONE_OFFSETS",
    "REGION_DEFAULT_TIMEZONES",
    "tz_abbrev_to_offset",
    "get_region_timezone",
    "convert_time_to_utc",
    "format_time_utc",
    "convert_and_format_time",
]
