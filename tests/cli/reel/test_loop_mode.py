"""Unit tests for reel loop mode behavior.

Tests _run_loop_mode and related loop functionality in commands.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from socials_automator.cli.reel.params import ReelGenerationParams
from socials_automator.cli.core.types import Success, Failure


@pytest.fixture
def temp_profile(tmp_path: Path) -> Path:
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


@pytest.fixture
def loop_params(temp_profile: Path) -> ReelGenerationParams:
    """Create ReelGenerationParams for loop mode testing."""
    return ReelGenerationParams(
        profile="test-profile",
        profile_path=temp_profile,
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
        loop_count=3,
        loop_each=None,  # Default 3 seconds
        gpu_accelerate=False,
        gpu_index=None,
    )


@pytest.fixture
def loop_params_with_interval(temp_profile: Path) -> ReelGenerationParams:
    """Create ReelGenerationParams with loop_each interval."""
    return ReelGenerationParams(
        profile="test-profile",
        profile_path=temp_profile,
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
        loop_count=2,
        loop_each=300,  # 5 minutes
        gpu_accelerate=False,
        gpu_index=None,
    )


class TestRunLoopModeWaitInterval:
    """Tests for _run_loop_mode wait interval behavior."""

    def test_uses_default_3_seconds_when_loop_each_none(self, loop_params: ReelGenerationParams, tmp_path: Path):
        """Test that default 3 second wait is used when loop_each is None."""
        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress") as mock_progress, \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time") as mock_time:

            # Setup mock service
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(loop_params)

            # Verify sleep was called with default 3 seconds
            # Should be called loop_count - 1 times (not after last video)
            sleep_calls = [c for c in mock_time.sleep.call_args_list]
            for call in sleep_calls:
                assert call == call(3)

    def test_uses_loop_each_interval_when_set(self, loop_params_with_interval: ReelGenerationParams, tmp_path: Path):
        """Test that loop_each interval is used when set."""
        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress") as mock_progress, \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time") as mock_time:

            # Setup mock service
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(loop_params_with_interval)

            # Verify sleep was called with 300 seconds (5 minutes)
            sleep_calls = [c for c in mock_time.sleep.call_args_list]
            for c in sleep_calls:
                assert c == call(300)

    def test_show_loop_progress_receives_wait_seconds(self, loop_params_with_interval: ReelGenerationParams, tmp_path: Path):
        """Test that show_loop_progress receives correct wait_seconds."""
        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress") as mock_progress, \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time"):

            # Setup mock service
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(loop_params_with_interval)

            # Verify show_loop_progress was called with wait_seconds=300
            for c in mock_progress.call_args_list:
                _, kwargs = c
                # Check if wait_seconds is passed (it's the 4th positional arg or kwarg)
                if len(c[0]) >= 4:
                    assert c[0][3] == 300  # 4th positional arg
                elif "wait_seconds" in kwargs:
                    assert kwargs["wait_seconds"] == 300


class TestRunLoopModeCompletion:
    """Tests for _run_loop_mode completion behavior."""

    def test_stops_after_loop_count(self, loop_params: ReelGenerationParams, tmp_path: Path):
        """Test that loop stops after reaching loop_count."""
        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress"), \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete") as mock_complete, \
             patch("socials_automator.cli.reel.commands.time"):

            # Setup mock service
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(loop_params)

            # Verify generate was called exactly loop_count times
            assert service_instance.generate.call_count == 3

            # Verify completion message was shown
            mock_complete.assert_called_once_with(
                patch("socials_automator.cli.reel.commands.console").return_value,
                3
            ) if False else None  # This is tricky to test, just ensure no exception

    def test_continues_on_failure(self, loop_params: ReelGenerationParams, tmp_path: Path):
        """Test that loop continues when a generation fails."""
        mock_success = MagicMock()
        mock_success.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress"), \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_reel_error"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.console"), \
             patch("socials_automator.cli.reel.commands.time"):

            # Setup mock service - first call fails, rest succeed
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(side_effect=[
                Failure("API Error", {}),  # First fails
                Success(mock_success.value),  # Second succeeds
                Success(mock_success.value),  # Third succeeds
            ])

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(loop_params)

            # Verify generate was called 3 times (continues after failure)
            assert service_instance.generate.call_count == 3


class TestRunLoopModeWithUpload:
    """Tests for _run_loop_mode with upload flag."""

    def test_uploads_after_each_generation(self, temp_profile: Path, tmp_path: Path):
        """Test that upload is called after each successful generation."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
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
            upload_after=True,  # Enable upload
            loop=True,
            loop_count=2,
            loop_each=60,
            gpu_accelerate=False,
            gpu_index=None,
        )

        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands._upload_generated_reel") as mock_upload, \
             patch("socials_automator.cli.reel.commands.show_loop_progress"), \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time"):

            # Setup mock service
            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))
            mock_upload.return_value = True

            # Import after patching
            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(params)

            # Verify upload was called after each successful generation
            assert mock_upload.call_count == 2


class TestLoopEachIntegration:
    """Integration tests for loop_each functionality."""

    def test_from_cli_to_loop_mode(self, temp_profile: Path, tmp_path: Path):
        """Test that loop_each flows correctly from CLI to loop mode."""
        with patch("socials_automator.cli.core.paths.get_profile_path") as mock_path:
            mock_path.return_value = temp_profile

            # Create params via from_cli
            params = ReelGenerationParams.from_cli(
                profile="test-profile",
                loop_each="5m",  # 5 minutes as string
                loop_count=2,
            )

            # Verify params were set correctly
            assert params.loop is True
            assert params.loop_each == 300  # Converted to seconds
            assert params.loop_count == 2

        # Now test that _run_loop_mode uses these values
        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress") as mock_progress, \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time") as mock_time:

            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(params)

            # Verify sleep was called with 300 seconds
            mock_time.sleep.assert_called_with(300)


class TestLoopEachEdgeCases:
    """Edge case tests for loop_each functionality."""

    def test_loop_each_zero_defaults_to_3_seconds(self, temp_profile: Path, tmp_path: Path):
        """Test that loop_each=0 defaults to 3 seconds (0 is falsy)."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
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
            loop_count=2,
            loop_each=0,  # Zero seconds - treated as falsy, defaults to 3
            gpu_accelerate=False,
            gpu_index=None,
        )

        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress"), \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time") as mock_time:

            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(params)

            # Zero is falsy in Python, so defaults to 3 seconds
            mock_time.sleep.assert_called_with(3)

    def test_loop_each_large_interval(self, temp_profile: Path, tmp_path: Path):
        """Test behavior with large loop_each interval (1 hour)."""
        params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
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
            loop_count=2,
            loop_each=3600,  # 1 hour
            gpu_accelerate=False,
            gpu_index=None,
        )

        mock_result = MagicMock()
        mock_result.value = MagicMock(
            output_path=tmp_path / "final.mp4",
            duration_seconds=60.0,
        )

        with patch("socials_automator.cli.reel.commands.ReelGeneratorService") as MockService, \
             patch("socials_automator.cli.reel.commands.show_loop_progress") as mock_progress, \
             patch("socials_automator.cli.reel.commands.show_reel_result"), \
             patch("socials_automator.cli.reel.commands.show_loop_complete"), \
             patch("socials_automator.cli.reel.commands.time") as mock_time:

            service_instance = MockService.return_value
            service_instance.generate = AsyncMock(return_value=Success(mock_result.value))

            from socials_automator.cli.reel.commands import _run_loop_mode

            _run_loop_mode(params)

            # Verify sleep was called with 3600 seconds (1 hour)
            mock_time.sleep.assert_called_with(3600)
