"""Unit tests for CLI core parsers.

Tests parsing functions for intervals, lengths, voice presets, and formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from socials_automator.cli.core.parsers import (
    parse_interval,
    parse_length,
    parse_voice_preset,
    format_duration,
    format_file_size,
)


class TestParseInterval:
    """Tests for parse_interval function."""

    # === Valid interval formats ===

    def test_parse_seconds_with_suffix(self):
        """Test parsing seconds with 's' suffix."""
        assert parse_interval("30s") == 30
        assert parse_interval("1s") == 1
        assert parse_interval("120s") == 120

    def test_parse_minutes_with_suffix(self):
        """Test parsing minutes with 'm' suffix."""
        assert parse_interval("5m") == 300  # 5 * 60
        assert parse_interval("1m") == 60
        assert parse_interval("30m") == 1800  # 30 * 60

    def test_parse_hours_with_suffix(self):
        """Test parsing hours with 'h' suffix."""
        assert parse_interval("1h") == 3600  # 1 * 3600
        assert parse_interval("2h") == 7200  # 2 * 3600
        assert parse_interval("24h") == 86400  # 24 * 3600

    def test_parse_number_without_suffix_defaults_to_seconds(self):
        """Test parsing plain number defaults to seconds."""
        assert parse_interval("60") == 60
        assert parse_interval("300") == 300
        assert parse_interval("1") == 1

    def test_parse_interval_with_whitespace(self):
        """Test parsing interval with leading/trailing whitespace."""
        assert parse_interval("  5m  ") == 300
        assert parse_interval("\t30s\n") == 30
        assert parse_interval(" 1h ") == 3600

    def test_parse_interval_case_insensitive(self):
        """Test parsing interval is case insensitive."""
        assert parse_interval("5M") == 300
        assert parse_interval("30S") == 30
        assert parse_interval("1H") == 3600

    def test_parse_zero_interval(self):
        """Test parsing zero interval."""
        assert parse_interval("0s") == 0
        assert parse_interval("0m") == 0
        assert parse_interval("0h") == 0
        assert parse_interval("0") == 0

    def test_parse_large_intervals(self):
        """Test parsing large interval values."""
        assert parse_interval("1000s") == 1000
        assert parse_interval("1000m") == 60000
        assert parse_interval("100h") == 360000

    # === Invalid interval formats ===

    def test_parse_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_interval("5x")
        assert "Invalid interval format" in str(exc_info.value)

    def test_parse_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_interval("")

    def test_parse_only_whitespace_raises_error(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            parse_interval("   ")

    def test_parse_negative_interval_raises_error(self):
        """Test that negative interval raises ValueError."""
        with pytest.raises(ValueError):
            parse_interval("-5m")

    def test_parse_decimal_interval_raises_error(self):
        """Test that decimal interval raises ValueError."""
        with pytest.raises(ValueError):
            parse_interval("5.5m")

    def test_parse_multiple_units_raises_error(self):
        """Test that multiple units raises ValueError (use parse_length for that)."""
        with pytest.raises(ValueError):
            parse_interval("1h30m")

    def test_parse_invalid_characters_raises_error(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError):
            parse_interval("5m!")
        with pytest.raises(ValueError):
            parse_interval("five minutes")


class TestParseLength:
    """Tests for parse_length function."""

    # === Valid length formats ===

    def test_parse_length_seconds(self):
        """Test parsing length in seconds."""
        assert parse_length("30s") == 30.0
        assert parse_length("90s") == 90.0

    def test_parse_length_minutes(self):
        """Test parsing length in minutes."""
        assert parse_length("1m") == 60.0
        assert parse_length("2m") == 120.0

    def test_parse_length_combined_format(self):
        """Test parsing combined minute+second format."""
        assert parse_length("1m30s") == 90.0
        assert parse_length("2m15s") == 135.0
        assert parse_length("0m45s") == 45.0

    def test_parse_length_plain_number(self):
        """Test parsing plain number as seconds."""
        assert parse_length("60") == 60.0
        assert parse_length("90") == 90.0

    def test_parse_length_with_whitespace(self):
        """Test parsing length with whitespace."""
        assert parse_length("  1m  ") == 60.0
        assert parse_length("\n30s\t") == 30.0

    def test_parse_length_case_insensitive(self):
        """Test parsing length is case insensitive."""
        assert parse_length("1M") == 60.0
        assert parse_length("30S") == 30.0
        assert parse_length("1M30S") == 90.0


class TestParseVoicePreset:
    """Tests for parse_voice_preset function."""

    def test_default_voice_returns_as_is(self):
        """Test that unknown voice returns as-is with default rate/pitch."""
        voice, rate, pitch = parse_voice_preset("rvc_adam")
        assert voice == "rvc_adam"
        assert rate == "+0%"
        assert pitch == "+0Hz"

    def test_adam_excited_preset(self):
        """Test adam_excited preset returns modified rate/pitch."""
        voice, rate, pitch = parse_voice_preset("adam_excited")
        assert voice == "rvc_adam"
        assert rate == "+12%"
        assert pitch == "+3Hz"

    def test_rvc_adam_excited_preset(self):
        """Test rvc_adam_excited preset returns modified rate/pitch."""
        voice, rate, pitch = parse_voice_preset("rvc_adam_excited")
        assert voice == "rvc_adam"
        assert rate == "+12%"
        assert pitch == "+3Hz"

    def test_adam_alias(self):
        """Test 'adam' alias resolves to rvc_adam."""
        voice, rate, pitch = parse_voice_preset("adam")
        assert voice == "rvc_adam"
        assert rate == "+0%"
        assert pitch == "+0Hz"

    def test_tiktok_adam_alias(self):
        """Test 'tiktok-adam' alias resolves to rvc_adam."""
        voice, rate, pitch = parse_voice_preset("tiktok-adam")
        assert voice == "rvc_adam"
        assert rate == "+0%"
        assert pitch == "+0Hz"

    def test_custom_rate_pitch_preserved(self):
        """Test that custom rate/pitch are preserved for non-preset voices."""
        voice, rate, pitch = parse_voice_preset("custom_voice", "+15%", "+5Hz")
        assert voice == "custom_voice"
        assert rate == "+15%"
        assert pitch == "+5Hz"


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_seconds_only(self):
        """Test formatting seconds under 60."""
        assert format_duration(30) == "30s"
        assert format_duration(45) == "45s"
        assert format_duration(59) == "59s"

    def test_format_exact_minutes(self):
        """Test formatting exact minutes."""
        assert format_duration(60) == "1m"
        assert format_duration(120) == "2m"
        assert format_duration(300) == "5m"

    def test_format_minutes_and_seconds(self):
        """Test formatting minutes with remaining seconds."""
        assert format_duration(90) == "1m30s"
        assert format_duration(150) == "2m30s"
        assert format_duration(65) == "1m5s"

    def test_format_zero(self):
        """Test formatting zero seconds."""
        assert format_duration(0) == "0s"

    def test_format_large_duration(self):
        """Test formatting large durations."""
        assert format_duration(3600) == "60m"
        assert format_duration(3661) == "61m1s"


class TestFormatFileSize:
    """Tests for format_file_size function."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_file_size(500) == "500 B"
        assert format_file_size(0) == "0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 50) == "50.0 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1024 * 1024 * 1024 * 2) == "2.0 GB"


class TestParseIntervalEdgeCases:
    """Edge case tests for parse_interval."""

    def test_parse_interval_with_leading_zeros(self):
        """Test parsing interval with leading zeros."""
        assert parse_interval("05m") == 300
        assert parse_interval("007s") == 7

    def test_parse_interval_boundary_values(self):
        """Test parsing interval at boundary values."""
        # Max reasonable values
        assert parse_interval("999h") == 999 * 3600
        assert parse_interval("9999m") == 9999 * 60
        assert parse_interval("99999s") == 99999
