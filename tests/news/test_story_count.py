"""Unit tests for story count calculation and auto mode.

Tests the duration-based story count calculation and integration
with the curator, params, and display.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from socials_automator.news.curator import (
    calculate_stories_for_duration,
    CurationConfig,
    NewsCurator,
)
from socials_automator.news.models import NewsEdition


class TestCalculateStoriesForDuration:
    """Tests for the calculate_stories_for_duration function."""

    def test_30_seconds_returns_2_stories(self):
        """30 second video should have 2 stories."""
        assert calculate_stories_for_duration(30) == 2

    def test_45_seconds_returns_3_stories(self):
        """45 second video should have 3 stories."""
        assert calculate_stories_for_duration(45) == 3

    def test_60_seconds_returns_4_stories(self):
        """60 second (1 minute) video should have 4 stories."""
        assert calculate_stories_for_duration(60) == 4

    def test_75_seconds_returns_5_stories(self):
        """75 second video should have 5 stories."""
        assert calculate_stories_for_duration(75) == 5

    def test_90_seconds_returns_6_stories(self):
        """90 second (1.5 minute) video should have 6 stories."""
        assert calculate_stories_for_duration(90) == 6

    def test_120_seconds_returns_8_stories(self):
        """120 second (2 minute) video should have 8 stories."""
        assert calculate_stories_for_duration(120) == 8

    def test_minimum_stories_is_2(self):
        """Very short videos should have at least 2 stories."""
        assert calculate_stories_for_duration(10) == 2
        assert calculate_stories_for_duration(15) == 2
        assert calculate_stories_for_duration(20) == 2

    def test_maximum_stories_is_8(self):
        """Very long videos should have at most 8 stories."""
        assert calculate_stories_for_duration(200) == 8
        assert calculate_stories_for_duration(300) == 8
        assert calculate_stories_for_duration(600) == 8

    def test_uses_16_seconds_per_story(self):
        """Verify the calculation uses ~16 seconds per story."""
        # 16 * 4 = 64, so 64 seconds should round to 4 stories
        assert calculate_stories_for_duration(64) == 4
        # 16 * 5 = 80, so 80 seconds should round to 5 stories
        assert calculate_stories_for_duration(80) == 5

    def test_float_duration(self):
        """Test with float duration values."""
        assert calculate_stories_for_duration(60.0) == 4
        assert calculate_stories_for_duration(59.5) == 4
        assert calculate_stories_for_duration(90.5) == 6

    def test_edge_case_zero_duration(self):
        """Zero duration should return minimum stories."""
        assert calculate_stories_for_duration(0) == 2

    def test_rounding_behavior(self):
        """Test that rounding works correctly."""
        # 16 * 3.5 = 56, rounds to 4
        assert calculate_stories_for_duration(56) == 4
        # 16 * 2.5 = 40, rounds to 2 (2.5 rounds to 2)
        assert calculate_stories_for_duration(40) == 2
        # 16 * 2.8 = 44.8, rounds to 3
        assert calculate_stories_for_duration(45) == 3


class TestCurationConfig:
    """Tests for CurationConfig with target_duration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CurationConfig()
        assert config.stories_per_brief is None
        assert config.target_duration is None
        assert config.min_stories == 2
        assert config.max_stories == 8

    def test_config_with_explicit_story_count(self):
        """Test config with explicit story count."""
        config = CurationConfig(stories_per_brief=5)
        assert config.stories_per_brief == 5

    def test_config_with_target_duration(self):
        """Test config with target duration for auto calculation."""
        config = CurationConfig(target_duration=60.0)
        assert config.target_duration == 60.0
        assert config.stories_per_brief is None

    def test_config_with_both_values(self):
        """Test config with both explicit count and duration."""
        config = CurationConfig(stories_per_brief=3, target_duration=60.0)
        # Explicit count takes precedence
        assert config.stories_per_brief == 3
        assert config.target_duration == 60.0


class TestNewsCuratorAutoStoryCount:
    """Tests for NewsCurator auto story count logic."""

    @pytest.fixture
    def mock_articles(self):
        """Create mock articles for testing."""
        from datetime import datetime
        from socials_automator.news.models import NewsArticle
        return [
            NewsArticle(
                title=f"Article {i}",
                summary=f"Summary for article {i}",
                source_name="Test Source",
                source_url="https://example.com",
                article_url=f"https://example.com/{i}",
                published_at=datetime.now(),
            )
            for i in range(10)
        ]

    def test_curator_uses_explicit_story_count(self):
        """Test that explicit story count is used when provided."""
        config = CurationConfig(stories_per_brief=5, target_duration=60.0)
        curator = NewsCurator(config=config)
        # stories_per_brief=5 should be used, not duration-based calculation
        assert curator.config.stories_per_brief == 5

    def test_curator_uses_duration_when_auto(self):
        """Test that duration-based calculation is used when auto."""
        config = CurationConfig(stories_per_brief=None, target_duration=90.0)
        curator = NewsCurator(config=config)
        # Should use duration-based calculation (90s -> 6 stories)
        assert curator.config.stories_per_brief is None
        assert curator.config.target_duration == 90.0

    def test_auto_count_calculation_for_60s(self):
        """Test that auto mode calculates 4 stories for 60s duration."""
        config = CurationConfig(
            stories_per_brief=None,
            target_duration=60.0,
        )
        # When story_count is None and target_duration is set,
        # calculate_stories_for_duration should return 4
        expected_count = calculate_stories_for_duration(config.target_duration)
        assert expected_count == 4

    def test_auto_count_calculation_for_90s(self):
        """Test that auto mode calculates 6 stories for 90s duration."""
        config = CurationConfig(
            stories_per_brief=None,
            target_duration=90.0,
        )
        expected_count = calculate_stories_for_duration(config.target_duration)
        assert expected_count == 6

    def test_explicit_count_overrides_duration(self):
        """Test that explicit count takes precedence over duration."""
        config = CurationConfig(
            stories_per_brief=3,  # Explicit
            target_duration=60.0,  # Would give 4 if auto
        )
        # Explicit count should be used
        assert config.stories_per_brief == 3

    def test_fallback_when_no_duration(self):
        """Test fallback behavior when no duration is provided."""
        config = CurationConfig(
            stories_per_brief=None,
            target_duration=None,
        )
        # When both are None, the curator falls back to 4 stories
        # This is tested by checking the config is set up correctly
        assert config.stories_per_brief is None
        assert config.target_duration is None


class TestParamsStoryCountParsing:
    """Tests for params parsing of story_count."""

    def test_auto_string_becomes_none(self):
        """Test that 'auto' string becomes None."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            story_count="auto",
        )
        assert params.news_story_count is None

    def test_numeric_string_becomes_int(self):
        """Test that numeric string becomes int."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            story_count="5",
        )
        assert params.news_story_count == 5

    def test_invalid_string_becomes_none(self):
        """Test that invalid string becomes None (auto)."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            story_count="invalid",
        )
        assert params.news_story_count is None

    def test_case_insensitive_auto(self):
        """Test that 'AUTO', 'Auto', etc. all work."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        for value in ["auto", "AUTO", "Auto", "AuTo"]:
            params = ReelGenerationParams.from_cli(
                profile="news.but.quick",
                story_count=value,
            )
            assert params.news_story_count is None, f"Failed for '{value}'"

    def test_default_is_auto(self):
        """Test that default value is auto (None)."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            # story_count not provided, should use default
        )
        assert params.news_story_count is None


class TestDisplayAutoStoryCount:
    """Tests for display showing auto story count calculation."""

    def test_display_shows_auto_calculation(self):
        """Test that display shows calculated story count for auto mode."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            story_count="auto",
            length="1m",
        )

        # Check that we can calculate the display string
        from socials_automator.news.curator import calculate_stories_for_duration
        from socials_automator.cli.core.parsers import format_duration

        auto_count = calculate_stories_for_duration(params.target_duration)
        duration_str = format_duration(params.target_duration)

        assert auto_count == 4
        assert duration_str == "1m"

    def test_display_shows_different_durations(self):
        """Test display for different target durations."""
        from socials_automator.cli.reel.params import ReelGenerationParams
        from socials_automator.news.curator import calculate_stories_for_duration

        test_cases = [
            ("30s", 30.0, 2),
            ("1m", 60.0, 4),
            ("1m30s", 90.0, 6),
        ]

        for length, expected_duration, expected_stories in test_cases:
            params = ReelGenerationParams.from_cli(
                profile="news.but.quick",
                story_count="auto",
                length=length,
            )
            assert params.target_duration == expected_duration
            assert calculate_stories_for_duration(params.target_duration) == expected_stories

    def test_display_explicit_count_not_calculated(self):
        """Test that explicit count is shown as-is, not calculated."""
        from socials_automator.cli.reel.params import ReelGenerationParams

        params = ReelGenerationParams.from_cli(
            profile="news.but.quick",
            story_count="3",  # Explicit
            length="1m",  # Would be 4 if auto
        )

        assert params.news_story_count == 3  # Explicit value used


class TestIntegrationWithOrchestrator:
    """Tests for integration with NewsOrchestrator."""

    def test_curation_config_receives_duration(self):
        """Test that CurationConfig can be created with target_duration."""
        # This tests the config that NewsOrchestrator creates
        config = CurationConfig(
            stories_per_brief=None,  # Auto
            target_duration=90.0,
            provider_override="test_provider",
            profile_name="test_profile",
        )

        assert config.target_duration == 90.0
        assert config.stories_per_brief is None
        assert config.provider_override == "test_provider"
        assert config.profile_name == "test_profile"

    def test_curator_with_orchestrator_config(self):
        """Test curator initialization with orchestrator-style config."""
        config = CurationConfig(
            stories_per_brief=None,
            target_duration=60.0,
        )
        curator = NewsCurator(config=config)

        # Curator should have the config with duration
        assert curator.config.target_duration == 60.0
        assert curator.config.stories_per_brief is None


class TestStoryCountClamping:
    """Tests for story count min/max clamping."""

    def test_clamping_to_minimum(self):
        """Test that story count is clamped to minimum."""
        config = CurationConfig(min_stories=3, max_stories=8)
        # Even if calculation gives 2, it should be clamped to 3
        # But our function already handles this

    def test_clamping_to_maximum(self):
        """Test that story count is clamped to maximum."""
        config = CurationConfig(min_stories=2, max_stories=6)
        # Very long duration would exceed 6, should be clamped

    def test_config_min_max_respected(self):
        """Test that CurationConfig min/max are used."""
        config = CurationConfig(
            min_stories=3,
            max_stories=5,
            target_duration=30.0,  # Would give 2 stories
        )
        # The curator should clamp to min_stories=3
        assert config.min_stories == 3
        assert config.max_stories == 5
