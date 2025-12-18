"""Unit tests for reel validators.

Tests validation logic for ReelGenerationParams and ReelUploadParams.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from socials_automator.cli.reel.validators import (
    validate_reel_generation_params,
    validate_reel_upload_params,
    VALID_VOICES,
    VALID_VIDEO_MATCHERS,
    VALID_TEXT_AI_PROVIDERS,
)
from socials_automator.cli.reel.params import ReelGenerationParams, ReelUploadParams
from socials_automator.cli.core.types import Success, Failure


class TestValidateReelGenerationParams:
    """Tests for validate_reel_generation_params."""

    @pytest.fixture
    def temp_profile(self, tmp_path: Path) -> Path:
        """Create a temporary profile directory with metadata."""
        profile_dir = tmp_path / "profiles" / "test-profile"
        profile_dir.mkdir(parents=True)

        metadata = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
            "hashtags": ["#test", "#automation"],
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))
        return profile_dir

    @pytest.fixture
    def valid_params(self, temp_profile: Path) -> ReelGenerationParams:
        """Create valid ReelGenerationParams for testing."""
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
            loop=False,
            loop_count=None,
            gpu_accelerate=False,
            gpu_index=None,
        )

    def test_valid_params_pass_validation(self, valid_params: ReelGenerationParams):
        """Test that valid parameters pass validation."""
        # Patch at the module where functions are imported/used
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(valid_params.profile)
            mock_api.return_value = Success("fake_key")

            result = validate_reel_generation_params(valid_params)

            assert isinstance(result, Success)
            assert result.value == valid_params

    def test_invalid_profile_fails_validation(self, valid_params: ReelGenerationParams):
        """Test that invalid profile fails validation."""
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile:
            mock_profile.return_value = Failure("Profile not found", {})

            result = validate_reel_generation_params(valid_params)

            assert isinstance(result, Failure)
            assert "not found" in result.error.lower()

    def test_invalid_voice_fails_validation(self, valid_params: ReelGenerationParams, temp_profile: Path):
        """Test that invalid voice fails validation."""
        # Create params with invalid voice
        invalid_params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
            topic=None,
            text_ai=None,
            video_matcher="pexels",
            voice="invalid_voice_that_does_not_exist",
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
            gpu_accelerate=False,
            gpu_index=None,
        )

        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(invalid_params.profile)
            mock_api.return_value = Success("fake_key")

            result = validate_reel_generation_params(invalid_params)

            assert isinstance(result, Failure)
            assert "voice" in result.error.lower()

    def test_invalid_video_matcher_fails_validation(self, temp_profile: Path):
        """Test that invalid video matcher fails validation."""
        invalid_params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
            topic=None,
            text_ai=None,
            video_matcher="invalid_matcher",
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
            gpu_accelerate=False,
            gpu_index=None,
        )

        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile:
            mock_profile.return_value = Success(invalid_params.profile)

            result = validate_reel_generation_params(invalid_params)

            assert isinstance(result, Failure)
            assert "video matcher" in result.error.lower()

    def test_invalid_text_ai_fails_validation(self, temp_profile: Path):
        """Test that invalid text AI provider fails validation."""
        invalid_params = ReelGenerationParams(
            profile="test-profile",
            profile_path=temp_profile,
            topic=None,
            text_ai="invalid_ai_provider",
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
            gpu_accelerate=False,
            gpu_index=None,
        )

        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_length") as mock_length:
            mock_profile.return_value = Success(invalid_params.profile)
            mock_length.return_value = Success(60.0)

            result = validate_reel_generation_params(invalid_params)

            assert isinstance(result, Failure)
            assert "text ai" in result.error.lower()

    def test_missing_pexels_api_key_fails_validation(self, valid_params: ReelGenerationParams):
        """Test that missing Pexels API key fails validation."""
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(valid_params.profile)
            mock_api.return_value = Failure("PEXELS_API_KEY not found", {})

            result = validate_reel_generation_params(valid_params)

            assert isinstance(result, Failure)
            assert "pexels" in result.error.lower()


class TestValidateReelUploadParams:
    """Tests for validate_reel_upload_params."""

    @pytest.fixture
    def temp_profile(self, tmp_path: Path) -> Path:
        """Create a temporary profile directory with metadata."""
        profile_dir = tmp_path / "profiles" / "test-profile"
        profile_dir.mkdir(parents=True)

        metadata = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
            "hashtags": ["#test", "#automation"],
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))
        return profile_dir

    @pytest.fixture
    def valid_upload_params(self, temp_profile: Path) -> ReelUploadParams:
        """Create valid ReelUploadParams for testing."""
        return ReelUploadParams(
            profile="test-profile",
            profile_path=temp_profile,
            reel_id=None,
            platforms=("instagram",),
            post_one=False,
            dry_run=False,
        )

    def test_valid_upload_params_pass_validation(self, valid_upload_params: ReelUploadParams):
        """Test that valid upload parameters pass validation."""
        # Patch at the module where functions are imported/used
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(valid_upload_params.profile)
            mock_api.return_value = Success("fake_key")

            result = validate_reel_upload_params(valid_upload_params)

            assert isinstance(result, Success)
            assert result.value == valid_upload_params

    def test_missing_instagram_credentials_fails_validation(self, valid_upload_params: ReelUploadParams):
        """Test that missing Instagram credentials fails validation."""
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(valid_upload_params.profile)
            # First call for Instagram, second for Cloudinary
            mock_api.side_effect = [
                Failure("INSTAGRAM_ACCESS_TOKEN not found", {}),
                Success("fake_key"),
            ]

            result = validate_reel_upload_params(valid_upload_params)

            assert isinstance(result, Failure)
            assert "instagram" in result.error.lower()

    def test_missing_cloudinary_credentials_fails_validation(self, valid_upload_params: ReelUploadParams):
        """Test that missing Cloudinary credentials fails validation."""
        with patch("socials_automator.cli.reel.validators.validate_profile") as mock_profile, \
             patch("socials_automator.cli.reel.validators.validate_api_key") as mock_api:
            mock_profile.return_value = Success(valid_upload_params.profile)
            # First call for Instagram, second for Cloudinary
            mock_api.side_effect = [
                Success("fake_token"),
                Failure("CLOUDINARY_CLOUD_NAME not found", {}),
            ]

            result = validate_reel_upload_params(valid_upload_params)

            assert isinstance(result, Failure)
            assert "cloudinary" in result.error.lower()


class TestValidVoices:
    """Tests for VALID_VOICES constant."""

    def test_valid_voices_contains_expected_values(self):
        """Test that VALID_VOICES contains expected voice options."""
        expected_voices = ["rvc_adam", "adam", "professional_female", "professional_male"]
        for voice in expected_voices:
            assert voice in VALID_VOICES, f"Expected voice '{voice}' not in VALID_VOICES"

    def test_valid_video_matchers_contains_pexels(self):
        """Test that VALID_VIDEO_MATCHERS contains pexels."""
        assert "pexels" in VALID_VIDEO_MATCHERS

    def test_valid_text_ai_providers_contains_expected(self):
        """Test that VALID_TEXT_AI_PROVIDERS contains expected providers."""
        expected_providers = ["zai", "groq", "gemini", "openai"]
        for provider in expected_providers:
            assert provider in VALID_TEXT_AI_PROVIDERS, f"Expected provider '{provider}' not found"
