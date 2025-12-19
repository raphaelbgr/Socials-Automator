"""Unit tests for reel display functions.

Tests display functions for Rich output with various configurations.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rich.console import Console

from socials_automator.cli.reel.display import (
    show_reel_config,
    show_reel_result,
    show_reel_error,
    show_upload_config,
    show_upload_result,
    show_pending_reels_table,
    show_loop_progress,
    show_loop_complete,
)
from socials_automator.cli.reel.params import ReelGenerationParams, ReelUploadParams


@pytest.fixture
def mock_console() -> Console:
    """Create a Console that captures output without ANSI codes."""
    return Console(file=StringIO(), force_terminal=False, no_color=True, width=120)


@pytest.fixture
def temp_profile_path(tmp_path: Path) -> Path:
    """Create a temporary profile path."""
    profile_dir = tmp_path / "test-profile"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def basic_params(temp_profile_path: Path) -> ReelGenerationParams:
    """Create basic ReelGenerationParams for testing."""
    return ReelGenerationParams(
        profile="test-profile",
        profile_path=temp_profile_path,
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


class TestShowReelConfigWithLoopEach:
    """Tests for show_reel_config with loop_each parameter."""

    def test_shows_loop_each_interval_in_seconds(self, mock_console: Console, temp_profile_path: Path):
        """Test that loop_each interval is displayed in seconds when < 60."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=None,
            loop_each=30,  # 30 seconds
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "Interval" in output
        assert "30s" in output

    def test_shows_loop_each_interval_in_minutes(self, mock_console: Console, temp_profile_path: Path):
        """Test that loop_each interval is displayed in minutes when >= 60."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=None,
            loop_each=300,  # 5 minutes
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "Interval" in output
        assert "5m" in output

    def test_shows_loop_each_with_minutes_and_seconds(self, mock_console: Console, temp_profile_path: Path):
        """Test that loop_each interval shows minutes and seconds."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=None,
            loop_each=90,  # 1m30s
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "Interval" in output
        assert "1m30s" in output

    def test_no_interval_shown_when_loop_each_none(self, mock_console: Console, temp_profile_path: Path):
        """Test that no interval is shown when loop_each is None."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=5,
            loop_each=None,  # No interval
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        # Should show loop info but not interval
        assert "Loop" in output
        assert "5 videos" in output
        assert "Interval" not in output

    def test_shows_loop_each_with_loop_count(self, mock_console: Console, temp_profile_path: Path):
        """Test that both loop_count and loop_each are shown together."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=10,
            loop_each=1800,  # 30 minutes
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "10 videos" in output
        assert "30m" in output


class TestShowLoopProgress:
    """Tests for show_loop_progress function."""

    def test_shows_default_3_second_wait(self, mock_console: Console):
        """Test default 3 second wait display."""
        show_loop_progress(mock_console, video_count=1, loop_limit=5)
        output = mock_console.file.getvalue()

        assert "3 seconds" in output
        assert "4 remaining" in output

    def test_shows_custom_wait_seconds(self, mock_console: Console):
        """Test custom wait time in seconds."""
        show_loop_progress(mock_console, video_count=1, loop_limit=5, wait_seconds=30)
        output = mock_console.file.getvalue()

        assert "30 seconds" in output

    def test_shows_wait_formatted_as_minutes(self, mock_console: Console):
        """Test wait time formatted as minutes when >= 60."""
        show_loop_progress(mock_console, video_count=1, loop_limit=5, wait_seconds=300)
        output = mock_console.file.getvalue()

        assert "5m" in output

    def test_shows_wait_formatted_as_minutes_and_seconds(self, mock_console: Console):
        """Test wait time formatted as minutes+seconds."""
        show_loop_progress(mock_console, video_count=1, loop_limit=5, wait_seconds=90)
        output = mock_console.file.getvalue()

        assert "1m30s" in output

    def test_shows_infinite_loop_without_limit(self, mock_console: Console):
        """Test display without loop limit."""
        show_loop_progress(mock_console, video_count=5, loop_limit=None, wait_seconds=60)
        output = mock_console.file.getvalue()

        assert "1m" in output
        assert "remaining" not in output
        assert "Ctrl+C" in output

    def test_shows_correct_remaining_count(self, mock_console: Console):
        """Test correct remaining count calculation."""
        show_loop_progress(mock_console, video_count=7, loop_limit=10, wait_seconds=3)
        output = mock_console.file.getvalue()

        assert "3 remaining" in output

    def test_shows_1_remaining_on_second_to_last(self, mock_console: Console):
        """Test showing 1 remaining on second-to-last video."""
        show_loop_progress(mock_console, video_count=9, loop_limit=10, wait_seconds=3)
        output = mock_console.file.getvalue()

        assert "1 remaining" in output


class TestShowReelConfig:
    """Tests for show_reel_config function."""

    def test_shows_basic_config(self, mock_console: Console, basic_params: ReelGenerationParams):
        """Test basic config display."""
        show_reel_config(mock_console, basic_params)
        output = mock_console.file.getvalue()

        assert "test-profile" in output
        assert "pexels" in output
        assert "rvc_adam" in output
        assert "1m" in output

    def test_shows_custom_topic(self, mock_console: Console, temp_profile_path: Path):
        """Test config with custom topic."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
            topic="5 AI Tools for Productivity",
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

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "5 AI Tools for Productivity" in output

    def test_shows_gpu_enabled(self, mock_console: Console, temp_profile_path: Path):
        """Test config with GPU enabled."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            gpu_accelerate=True,
            gpu_index=1,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "GPU 1" in output
        assert "Enabled" in output

    def test_shows_voice_rate_pitch(self, mock_console: Console, temp_profile_path: Path):
        """Test config with custom voice rate and pitch."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
            topic=None,
            text_ai=None,
            video_matcher="pexels",
            voice="rvc_adam",
            voice_rate="+15%",
            voice_pitch="+10Hz",
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

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "+15%" in output
        assert "+10Hz" in output

    def test_shows_infinite_loop_mode(self, mock_console: Console, temp_profile_path: Path):
        """Test config with infinite loop mode."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile_path,
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
            loop=True,
            loop_count=None,
            loop_each=None,
            gpu_accelerate=False,
            gpu_index=None,
        )

        show_reel_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "Infinite" in output
        assert "Ctrl+C" in output


class TestShowReelResult:
    """Tests for show_reel_result function."""

    def test_shows_basic_result(self, mock_console: Console, tmp_path: Path):
        """Test basic result display."""
        video_path = tmp_path / "final.mp4"

        show_reel_result(mock_console, video_path, duration=60)
        output = mock_console.file.getvalue()

        assert "generated successfully" in output.lower()
        assert "60 seconds" in output

    def test_shows_loop_progress(self, mock_console: Console, tmp_path: Path):
        """Test result with loop progress."""
        video_path = tmp_path / "final.mp4"

        show_reel_result(mock_console, video_path, duration=45, video_count=3, loop_limit=10)
        output = mock_console.file.getvalue()

        assert "Video #3/10" in output

    def test_shows_loop_without_limit(self, mock_console: Console, tmp_path: Path):
        """Test result with loop progress but no limit."""
        video_path = tmp_path / "final.mp4"

        show_reel_result(mock_console, video_path, duration=45, video_count=7, loop_limit=None)
        output = mock_console.file.getvalue()

        assert "Video #7" in output


class TestShowLoopComplete:
    """Tests for show_loop_complete function."""

    def test_shows_completion_message(self, mock_console: Console):
        """Test loop completion message."""
        show_loop_complete(mock_console, total=10)
        output = mock_console.file.getvalue()

        assert "Completed" in output
        assert "10 videos" in output


class TestShowReelError:
    """Tests for show_reel_error function."""

    def test_shows_error_message(self, mock_console: Console):
        """Test error message display."""
        show_reel_error(mock_console, "Something went wrong")
        output = mock_console.file.getvalue()

        assert "Something went wrong" in output

    def test_shows_error_details(self, mock_console: Console):
        """Test error with details."""
        show_reel_error(mock_console, "API Error", {"code": "429", "message": "Rate limited"})
        output = mock_console.file.getvalue()

        assert "API Error" in output
        assert "429" in output
        assert "Rate limited" in output


class TestShowUploadConfig:
    """Tests for show_upload_config function."""

    def test_shows_basic_upload_config(self, mock_console: Console, temp_profile_path: Path):
        """Test basic upload config display."""
        params = ReelUploadParams(
            profile="test-profile",
            profile_path=temp_profile_path,
            reel_id=None,
            platforms=("instagram",),
            post_one=False,
            dry_run=False,
        )

        show_upload_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "test-profile" in output
        assert "instagram" in output

    def test_shows_multiple_platforms(self, mock_console: Console, temp_profile_path: Path):
        """Test upload config with multiple platforms."""
        params = ReelUploadParams(
            profile="test-profile",
            profile_path=temp_profile_path,
            reel_id=None,
            platforms=("instagram", "tiktok"),
            post_one=False,
            dry_run=False,
        )

        show_upload_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "instagram" in output
        assert "tiktok" in output

    def test_shows_specific_reel_id(self, mock_console: Console, temp_profile_path: Path):
        """Test upload config with specific reel ID."""
        params = ReelUploadParams(
            profile="test-profile",
            profile_path=temp_profile_path,
            reel_id="18-001-my-reel",
            platforms=("instagram",),
            post_one=False,
            dry_run=False,
        )

        show_upload_config(mock_console, params)
        output = mock_console.file.getvalue()

        assert "18-001-my-reel" in output


class TestShowUploadResult:
    """Tests for show_upload_result function."""

    def test_shows_success_result(self, mock_console: Console):
        """Test successful upload result."""
        show_upload_result(mock_console, success_count=5, failed_count=0, results=[])
        output = mock_console.file.getvalue()

        assert "5 reel(s)" in output
        assert "[OK]" in output

    def test_shows_partial_result(self, mock_console: Console):
        """Test partial upload result."""
        results = [
            {"success": False, "error": "Rate limited", "path": MagicMock(name="18-001")}
        ]
        show_upload_result(mock_console, success_count=3, failed_count=1, results=results)
        output = mock_console.file.getvalue()

        assert "3 reel(s)" in output
        assert "1 reel(s)" in output

    def test_shows_all_failed_result(self, mock_console: Console):
        """Test all failed upload result."""
        show_upload_result(mock_console, success_count=0, failed_count=5, results=[])
        output = mock_console.file.getvalue()

        assert "Failed" in output


class TestShowPendingReelsTable:
    """Tests for show_pending_reels_table function."""

    def test_shows_empty_message_when_no_reels(self, mock_console: Console):
        """Test message when no pending reels."""
        show_pending_reels_table(mock_console, [])
        output = mock_console.file.getvalue()

        assert "No pending reels" in output

    def test_shows_reels_table(self, mock_console: Console):
        """Test reels table display."""
        reels = [
            {"folder": "18-001", "topic": "AI Tools", "duration": 60, "created": "2025-12-18"},
            {"folder": "18-002", "topic": "Productivity Tips", "duration": 45, "created": "2025-12-18"},
        ]

        show_pending_reels_table(mock_console, reels)
        output = mock_console.file.getvalue()

        assert "18-001" in output
        assert "AI Tools" in output
        assert "60s" in output
        assert "Total: 2 reel(s)" in output
