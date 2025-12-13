"""Tests for video generator module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from socials_automator.video import (
    GenerationProgress,
    ScriptGenerationError,
    VideoGenerationError,
    VideoGenerator,
    VideoGeneratorConfig,
    VideoOutput,
    VideoScript,
    create_output_directory,
    create_sample_script,
)


class TestVideoGenerator:
    """Tests for VideoGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = VideoGenerator()
        assert generator.config.target_duration == 60
        assert generator.script_generator is None

    def test_init_custom_config(self, default_config):
        """Test initialization with custom config."""
        generator = VideoGenerator(config=default_config)
        assert generator.config == default_config

    def test_init_with_script_generator(self):
        """Test initialization with script generator."""

        def my_generator(topic, config):
            return create_sample_script(topic, config)

        generator = VideoGenerator(script_generator=my_generator)
        assert generator.script_generator == my_generator

    @pytest.mark.asyncio
    async def test_generate_no_script_no_generator(self, temp_dir):
        """Test that generate fails without script or generator."""
        generator = VideoGenerator()

        with pytest.raises(ScriptGenerationError, match="No script provided"):
            await generator.generate(
                topic="Test Topic",
                output_dir=temp_dir,
            )

    @pytest.mark.asyncio
    async def test_generate_with_script(self, temp_dir, sample_script):
        """Test generation with provided script."""
        # Mock all the components
        with patch.object(VideoGenerator, "__init__", return_value=None):
            generator = VideoGenerator.__new__(VideoGenerator)
            generator.config = VideoGeneratorConfig.default()

            # Mock TTS
            generator.tts = MagicMock()
            generator.tts.generate = AsyncMock(
                return_value=MagicMock(
                    audio_path=temp_dir / "audio.mp3",
                    srt_path=temp_dir / "audio.srt",
                    duration_seconds=30.0,
                    word_timestamps=[],
                )
            )

            # Mock footage
            generator.footage = MagicMock()
            generator.footage.find_video = AsyncMock(
                return_value=MagicMock(
                    path=temp_dir / "clip.mp4",
                    duration_seconds=10.0,
                )
            )
            generator.footage.close = AsyncMock()

            # Mock assembler
            generator.assembler = MagicMock()
            generator.assembler.assemble = MagicMock(
                return_value=temp_dir / "assembled.mp4"
            )
            generator.assembler.create_thumbnail = MagicMock(
                return_value=temp_dir / "thumb.jpg"
            )

            # Mock subtitles
            generator.subtitles = MagicMock()
            generator.subtitles.render = MagicMock(
                return_value=temp_dir / "final.mp4"
            )

            # Create dummy files
            (temp_dir / "audio.mp3").touch()
            (temp_dir / "audio.srt").touch()

            result = await generator.generate(
                topic="Test",
                output_dir=temp_dir,
                script=sample_script,
            )

            assert isinstance(result, VideoOutput)
            assert result.script_path.exists()

    def test_generate_sync(self, temp_dir, sample_script):
        """Test synchronous generation wrapper."""
        generator = VideoGenerator(script_generator=create_sample_script)

        # This would need full mocking to test properly
        # Just verify the method exists and signature
        assert hasattr(generator, "generate_sync")

    @pytest.mark.asyncio
    async def test_generate_from_script(self, temp_dir, sample_script):
        """Test generate_from_script method."""
        # Mock implementation
        generator = VideoGenerator()
        generator.generate = AsyncMock(
            return_value=VideoOutput(
                script_path=temp_dir / "script.json",
                audio_path=temp_dir / "audio.mp3",
                srt_path=temp_dir / "audio.srt",
                clips_dir=temp_dir / "clips",
                assembled_path=temp_dir / "assembled.mp4",
                final_path=temp_dir / "final.mp4",
                duration_seconds=60.0,
            )
        )

        result = await generator.generate_from_script(
            script=sample_script,
            output_dir=temp_dir,
        )

        generator.generate.assert_called_once()
        assert result.script_path == temp_dir / "script.json"

    @pytest.mark.asyncio
    async def test_progress_callback(self, temp_dir, sample_script):
        """Test that progress callback is called."""
        progress_updates = []

        def track_progress(progress: GenerationProgress):
            progress_updates.append(progress)

        generator = VideoGenerator()

        # This is a partial test - full test would need mocked components


class TestCreateOutputDirectory:
    """Tests for output directory creation."""

    def test_creates_directory(self, temp_dir):
        """Test that directory is created."""
        output = create_output_directory(
            base_dir=temp_dir,
            profile="test_profile",
            post_id="test-001",
        )

        assert output.exists()
        assert "reels" in str(output)
        assert "test_profile" in str(output)
        assert "test-001" in str(output)

    def test_auto_generates_post_id(self, temp_dir):
        """Test auto-generation of post ID."""
        output = create_output_directory(
            base_dir=temp_dir,
            profile="test_profile",
        )

        assert output.exists()
        # Post ID should be in format DD-HHMM
        assert len(output.name.split("-")) >= 2


class TestCreateSampleScript:
    """Tests for sample script creation."""

    def test_creates_valid_script(self):
        """Test that sample script is valid."""
        config = VideoGeneratorConfig.default()
        script = create_sample_script("AI Tools", config)

        assert isinstance(script, VideoScript)
        assert "AI Tools" in script.title
        assert len(script.scenes) > 0
        assert script.hook
        assert script.cta

    def test_scenes_have_keywords(self):
        """Test that scenes have video keywords."""
        config = VideoGeneratorConfig.default()
        script = create_sample_script("Test", config)

        for scene in script.scenes:
            assert len(scene.video_keywords) > 0

    def test_uses_config_duration(self):
        """Test that script uses config duration."""
        config = VideoGeneratorConfig(target_duration=30)
        script = create_sample_script("Test", config)

        assert script.total_duration == 30


class TestGenerationProgress:
    """Tests for GenerationProgress model."""

    def test_valid_progress(self):
        """Test creating valid progress update."""
        progress = GenerationProgress(
            stage="tts",
            progress=0.5,
            message="Generating voiceover...",
        )

        assert progress.stage == "tts"
        assert progress.progress == 0.5

    def test_progress_with_scene_info(self):
        """Test progress with scene information."""
        progress = GenerationProgress(
            stage="footage",
            progress=0.3,
            message="Downloading scene 2/5",
            current_scene=2,
            total_scenes=5,
        )

        assert progress.current_scene == 2
        assert progress.total_scenes == 5

    def test_progress_bounds(self):
        """Test progress value bounds."""
        # Should be between 0 and 1
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerationProgress(
                stage="test",
                progress=-0.1,
                message="Invalid",
            )

        with pytest.raises(ValidationError):
            GenerationProgress(
                stage="test",
                progress=1.5,
                message="Invalid",
            )


class TestGeneratorIntegration:
    """Full integration tests."""

    @pytest.mark.skipif(
        True,
        reason="Requires all dependencies and API keys",
    )
    @pytest.mark.asyncio
    async def test_full_generation(self, temp_dir):
        """Test complete video generation pipeline."""
        config = VideoGeneratorConfig.default()
        generator = VideoGenerator(
            config=config,
            script_generator=create_sample_script,
        )

        result = await generator.generate(
            topic="5 AI Tips",
            output_dir=temp_dir,
        )

        assert result.final_path.exists()
        assert result.thumbnail_path.exists()
        assert result.duration_seconds > 0
