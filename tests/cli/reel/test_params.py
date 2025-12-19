"""Unit tests for reel params.

Tests ReelGenerationParams and ReelUploadParams creation and parsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from socials_automator.cli.reel.params import (
    ReelGenerationParams,
    ReelUploadParams,
    _is_news_profile,
    resolve_platforms,
)


class TestReelGenerationParamsLoopEach:
    """Tests for ReelGenerationParams loop_each functionality."""

    @pytest.fixture
    def temp_profile(self, tmp_path: Path) -> Path:
        """Create a temporary profile directory with metadata."""
        profile_dir = tmp_path / "profiles" / "test-profile"
        profile_dir.mkdir(parents=True)

        metadata = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
            "hashtags": ["#test"],
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))
        return profile_dir

    def test_loop_each_parses_minutes(self, temp_profile: Path):
        """Test that loop_each parses minute intervals correctly."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="5m",
            )

            assert params.loop_each == 300  # 5 minutes = 300 seconds
            assert params.loop is True  # loop_each enables loop mode

    def test_loop_each_parses_seconds(self, temp_profile: Path):
        """Test that loop_each parses second intervals correctly."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="30s",
            )

            assert params.loop_each == 30
            assert params.loop is True

    def test_loop_each_parses_hours(self, temp_profile: Path):
        """Test that loop_each parses hour intervals correctly."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="1h",
            )

            assert params.loop_each == 3600  # 1 hour = 3600 seconds
            assert params.loop is True

    def test_loop_each_enables_loop_mode_automatically(self, temp_profile: Path):
        """Test that providing loop_each automatically enables loop mode."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop=False,  # Explicitly set to False
                loop_each="10m",
            )

            assert params.loop is True  # Should be True because loop_each is set
            assert params.loop_each == 600

    def test_loop_each_none_when_not_provided(self, temp_profile: Path):
        """Test that loop_each is None when not provided."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
            )

            assert params.loop_each is None
            assert params.loop is False

    def test_loop_each_with_loop_count(self, temp_profile: Path):
        """Test that loop_each works with loop_count."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_count=10,
                loop_each="30m",
            )

            assert params.loop_each == 1800  # 30 minutes
            assert params.loop_count == 10
            assert params.loop is True

    def test_loop_each_with_upload(self, temp_profile: Path):
        """Test that loop_each works with upload flag."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="5m",
                upload=True,
            )

            assert params.loop_each == 300
            assert params.upload_after is True
            assert params.loop is True

    def test_loop_each_plain_number_defaults_to_seconds(self, temp_profile: Path):
        """Test that plain number in loop_each defaults to seconds."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="60",  # Plain 60 = 60 seconds
            )

            assert params.loop_each == 60
            assert params.loop is True


class TestReelGenerationParamsFromCli:
    """Tests for ReelGenerationParams.from_cli factory method."""

    @pytest.fixture
    def temp_profile(self, tmp_path: Path) -> Path:
        """Create a temporary profile directory with metadata."""
        profile_dir = tmp_path / "profiles" / "test-profile"
        profile_dir.mkdir(parents=True)

        metadata = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))
        return profile_dir

    def test_from_cli_with_defaults(self, temp_profile: Path):
        """Test from_cli with all defaults."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(profile="test-profile")

            assert params.profile == "test-profile"
            assert params.video_matcher == "pexels"
            assert params.voice == "rvc_adam"
            assert params.voice_rate == "+0%"
            assert params.voice_pitch == "+0Hz"
            assert params.subtitle_size == 80
            assert params.target_duration == 60.0  # 1m default
            assert params.dry_run is False
            assert params.upload_after is False
            assert params.loop is False
            assert params.loop_count is None
            assert params.loop_each is None
            assert params.gpu_accelerate is False

    def test_from_cli_parses_length(self, temp_profile: Path):
        """Test from_cli correctly parses length parameter."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                length="1m30s",
            )

            assert params.target_duration == 90.0

    def test_from_cli_resolves_voice_preset(self, temp_profile: Path):
        """Test from_cli correctly resolves voice presets."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                voice="adam_excited",
            )

            assert params.voice == "rvc_adam"
            assert params.voice_rate == "+12%"
            assert params.voice_pitch == "+3Hz"

    def test_from_cli_custom_voice_settings(self, temp_profile: Path):
        """Test from_cli with custom voice rate and pitch."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                voice="rvc_adam",
                voice_rate="+15%",
                voice_pitch="+10Hz",
            )

            assert params.voice == "rvc_adam"
            assert params.voice_rate == "+15%"
            assert params.voice_pitch == "+10Hz"

    def test_from_cli_loop_mode_with_count(self, temp_profile: Path):
        """Test from_cli enables loop mode when loop_count is set."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_count=5,
            )

            assert params.loop is True
            assert params.loop_count == 5

    def test_from_cli_explicit_loop_flag(self, temp_profile: Path):
        """Test from_cli with explicit loop flag."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop=True,
            )

            assert params.loop is True
            assert params.loop_count is None

    def test_from_cli_gpu_options(self, temp_profile: Path):
        """Test from_cli with GPU options."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                gpu_accelerate=True,
                gpu=1,
            )

            assert params.gpu_accelerate is True
            assert params.gpu_index == 1


class TestReelGenerationParamsImmutability:
    """Tests for ReelGenerationParams immutability."""

    @pytest.fixture
    def params(self, tmp_path: Path) -> ReelGenerationParams:
        """Create ReelGenerationParams for testing."""
        profile_dir = tmp_path / "test-profile"
        profile_dir.mkdir()
        (profile_dir / "metadata.json").write_text('{"profile": {"id": "test"}}')

        return ReelGenerationParams(
            profile="test-profile",
            profile_path=profile_dir,
            topic=None,
            text_ai=None,
            video_matcher="pexels",
            voice="rvc_adam",
            voice_rate="+0%",
            voice_pitch="+0Hz",
            subtitle_size=80,
            font="Montserrat-Bold.ttf",
            target_duration=60.0,
            output_dir=None,
            dry_run=False,
            upload_after=False,
            loop=False,
            loop_count=None,
            loop_each=None,
            gpu_accelerate=False,
            gpu_index=None,
        )

    def test_params_are_frozen(self, params: ReelGenerationParams):
        """Test that params cannot be modified after creation."""
        with pytest.raises(AttributeError):
            params.loop_each = 300  # type: ignore

    def test_params_are_hashable(self, params: ReelGenerationParams):
        """Test that frozen params are hashable."""
        # Should not raise
        hash(params)


class TestIsNewsProfile:
    """Tests for _is_news_profile helper function."""

    def test_is_news_profile_with_news_sources(self, tmp_path: Path):
        """Test detection of news profile with news_sources key."""
        profile_dir = tmp_path / "news-profile"
        profile_dir.mkdir()

        metadata = {
            "profile": {"id": "news"},
            "news_sources": ["https://rss.example.com"],
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))

        assert _is_news_profile(profile_dir) is True

    def test_is_news_profile_without_news_sources(self, tmp_path: Path):
        """Test detection of non-news profile."""
        profile_dir = tmp_path / "regular-profile"
        profile_dir.mkdir()

        metadata = {
            "profile": {"id": "regular"},
            "content_strategy": {},
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))

        assert _is_news_profile(profile_dir) is False

    def test_is_news_profile_missing_metadata(self, tmp_path: Path):
        """Test detection when metadata.json doesn't exist."""
        profile_dir = tmp_path / "no-metadata"
        profile_dir.mkdir()

        assert _is_news_profile(profile_dir) is False

    def test_is_news_profile_invalid_json(self, tmp_path: Path):
        """Test detection with invalid JSON in metadata."""
        profile_dir = tmp_path / "invalid-profile"
        profile_dir.mkdir()
        (profile_dir / "metadata.json").write_text("not valid json")

        assert _is_news_profile(profile_dir) is False


class TestResolvePlatforms:
    """Tests for resolve_platforms function."""

    def test_default_returns_instagram(self, tmp_path: Path):
        """Test that default returns Instagram only."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        platforms = resolve_platforms(profile_dir)

        assert platforms == ["instagram"]

    def test_instagram_flag_returns_instagram(self, tmp_path: Path):
        """Test that --instagram flag returns Instagram."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        platforms = resolve_platforms(profile_dir, instagram=True)

        assert platforms == ["instagram"]

    def test_tiktok_flag_returns_tiktok(self, tmp_path: Path):
        """Test that --tiktok flag returns TikTok."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        platforms = resolve_platforms(profile_dir, tiktok=True)

        assert platforms == ["tiktok"]

    def test_both_flags_returns_both(self, tmp_path: Path):
        """Test that both flags return both platforms."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        platforms = resolve_platforms(profile_dir, instagram=True, tiktok=True)

        assert "instagram" in platforms
        assert "tiktok" in platforms
        assert len(platforms) == 2
