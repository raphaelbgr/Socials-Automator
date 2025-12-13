"""Tests for video configuration."""

import os

import pytest

from socials_automator.video import (
    KEYWORD_FALLBACKS,
    VOICE_PRESETS,
    OutputConfig,
    PexelsConfig,
    TTSConfig,
    VideoGeneratorConfig,
    get_keyword_fallbacks,
)


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_defaults(self):
        """Test default values."""
        config = TTSConfig()
        assert config.provider == "edge-tts"
        assert config.voice == "en-US-AriaNeural"
        assert config.rate == "+0%"
        assert config.pitch == "+0Hz"
        assert config.volume == "+0%"

    def test_from_preset(self):
        """Test creating config from preset."""
        config = TTSConfig.from_preset("professional_female")
        assert config.voice == "en-US-AriaNeural"

        config = TTSConfig.from_preset("british_male")
        assert config.voice == "en-GB-RyanNeural"

    def test_from_preset_unknown(self):
        """Test preset fallback for unknown names."""
        config = TTSConfig.from_preset("unknown_voice_name")
        # Should use the name directly if not a preset
        assert config.voice == "unknown_voice_name"

    def test_voice_presets_exist(self):
        """Test that all expected presets exist."""
        expected = [
            "professional_male",
            "professional_female",
            "friendly_male",
            "friendly_female",
            "british_male",
            "british_female",
        ]
        for preset in expected:
            assert preset in VOICE_PRESETS


class TestPexelsConfig:
    """Tests for PexelsConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PexelsConfig()
        assert config.api_key_env == "PEXELS_API_KEY"
        assert config.prefer_orientation == "portrait"
        assert config.fallback_orientation == "landscape"
        assert config.quality == "hd"
        assert config.per_page == 15

    def test_api_key_from_env(self, monkeypatch):
        """Test API key retrieval from environment."""
        monkeypatch.setenv("PEXELS_API_KEY", "test_key_123")
        config = PexelsConfig()
        assert config.api_key == "test_key_123"

    def test_api_key_missing(self, monkeypatch):
        """Test API key when not set."""
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        config = PexelsConfig()
        assert config.api_key is None

    def test_custom_env_var(self, monkeypatch):
        """Test custom environment variable name."""
        monkeypatch.setenv("MY_PEXELS_KEY", "custom_key")
        config = PexelsConfig(api_key_env="MY_PEXELS_KEY")
        assert config.api_key == "custom_key"


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_defaults(self):
        """Test default values."""
        config = OutputConfig()
        assert config.width == 1080
        assert config.height == 1920
        assert config.fps == 30
        assert config.duration == 60
        assert config.codec == "libx264"
        assert config.audio_codec == "aac"

    def test_resolution_property(self):
        """Test resolution tuple property."""
        config = OutputConfig(width=720, height=1280)
        assert config.resolution == (720, 1280)

    def test_aspect_ratio_property(self):
        """Test aspect ratio calculation."""
        config = OutputConfig()
        assert config.aspect_ratio == pytest.approx(9 / 16, rel=0.01)


class TestVideoGeneratorConfig:
    """Tests for VideoGeneratorConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = VideoGeneratorConfig.default()
        assert config.target_duration == 60
        assert config.words_per_minute == 150
        assert config.min_scene_duration == 3.0
        assert config.max_scene_duration == 15.0

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = VideoGeneratorConfig.default()
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.pexels, PexelsConfig)
        assert isinstance(config.output, OutputConfig)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "target_duration": 30,
            "words_per_minute": 120,
            "tts": {"voice": "en-GB-RyanNeural"},
            "output": {"width": 720, "height": 1280},
        }
        config = VideoGeneratorConfig.from_dict(data)
        assert config.target_duration == 30
        assert config.words_per_minute == 120
        assert config.tts.voice == "en-GB-RyanNeural"
        assert config.output.width == 720

    def test_validate_pexels_key(self, monkeypatch):
        """Test Pexels key validation."""
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
        config = VideoGeneratorConfig.default()
        assert config.validate_pexels_key() is False

        monkeypatch.setenv("PEXELS_API_KEY", "test_key")
        assert config.validate_pexels_key() is True


class TestKeywordFallbacks:
    """Tests for keyword fallback functionality."""

    def test_fallbacks_exist(self):
        """Test that fallbacks dictionary exists."""
        assert len(KEYWORD_FALLBACKS) > 0

    def test_get_fallbacks_known(self):
        """Test getting fallbacks for known categories."""
        fallbacks = get_keyword_fallbacks("ai technology")
        assert len(fallbacks) > 0

        fallbacks = get_keyword_fallbacks("productivity tips")
        assert len(fallbacks) > 0

    def test_get_fallbacks_unknown(self):
        """Test fallback for unknown keywords."""
        fallbacks = get_keyword_fallbacks("xyz123unknown")
        assert fallbacks == ["technology abstract"]

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        fallbacks1 = get_keyword_fallbacks("AI")
        fallbacks2 = get_keyword_fallbacks("ai")
        assert fallbacks1 == fallbacks2
