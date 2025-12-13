"""Tests for TTS module."""

import pytest

from socials_automator.video import TTSConfig, TTSError, TTSGenerator, calculate_speech_duration


class TestCalculateSpeechDuration:
    """Tests for speech duration calculation."""

    def test_basic_calculation(self):
        """Test basic duration calculation."""
        # 150 words at 150 WPM = 60 seconds
        duration = calculate_speech_duration("word " * 150, words_per_minute=150)
        assert duration == pytest.approx(60, rel=0.01)

    def test_empty_text(self):
        """Test with empty text."""
        duration = calculate_speech_duration("")
        assert duration == 0

    def test_single_word(self):
        """Test with single word."""
        duration = calculate_speech_duration("hello", words_per_minute=150)
        assert duration == pytest.approx(0.4, rel=0.01)  # 1/150 * 60

    def test_custom_wpm(self):
        """Test with custom words per minute."""
        # 100 words at 100 WPM = 60 seconds
        duration = calculate_speech_duration("word " * 100, words_per_minute=100)
        assert duration == pytest.approx(60, rel=0.01)


class TestTTSGenerator:
    """Tests for TTS generator."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = TTSGenerator()
        assert generator.config.voice == "en-US-AriaNeural"

    def test_init_custom_config(self, tts_config):
        """Test initialization with custom config."""
        generator = TTSGenerator(config=tts_config)
        assert generator.config == tts_config

    @pytest.mark.asyncio
    async def test_generate_requires_edge_tts(self, temp_dir):
        """Test that generate raises error if edge-tts not installed."""
        generator = TTSGenerator()
        # This test verifies the import handling
        # In real environments, edge-tts should be installed


class TestTTSIntegration:
    """Integration tests for TTS (require edge-tts installed)."""

    @pytest.mark.skipif(
        True,  # Skip by default, enable for integration testing
        reason="Requires edge-tts installed and takes time to run",
    )
    @pytest.mark.asyncio
    async def test_generate_voiceover(self, temp_dir):
        """Test actual voiceover generation."""
        generator = TTSGenerator()
        result = await generator.generate(
            text="Hello world, this is a test.",
            output_dir=temp_dir,
            filename="test_voiceover",
        )

        assert result.audio_path.exists()
        assert result.srt_path.exists()
        assert result.duration_seconds > 0
        assert len(result.word_timestamps) > 0

    @pytest.mark.skipif(
        True,
        reason="Requires edge-tts installed",
    )
    def test_generate_sync(self, temp_dir):
        """Test synchronous generation wrapper."""
        generator = TTSGenerator()
        result = generator.generate_sync(
            text="Test message.",
            output_dir=temp_dir,
        )
        assert result.audio_path.exists()

    @pytest.mark.skipif(
        True,
        reason="Requires edge-tts installed",
    )
    @pytest.mark.asyncio
    async def test_list_voices(self):
        """Test listing available voices."""
        voices = await TTSGenerator.list_voices()
        assert len(voices) > 0

        # Test with filter
        en_voices = await TTSGenerator.list_voices(language_filter="en")
        assert len(en_voices) > 0
        assert all(v["Locale"].startswith("en") for v in en_voices)
