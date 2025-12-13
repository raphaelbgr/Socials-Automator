"""Tests for video models."""

import pytest
from pydantic import ValidationError

from socials_automator.video import (
    SubtitleAnimation,
    SubtitlePosition,
    SubtitleStyle,
    VideoClip,
    VideoScene,
    VideoScript,
    VoiceoverResult,
    WordTimestamp,
)


class TestVideoScene:
    """Tests for VideoScene model."""

    def test_valid_scene(self):
        """Test creating a valid scene."""
        scene = VideoScene(
            text="Test narration.",
            duration_seconds=10.0,
            video_keywords=["tech", "abstract"],
        )
        assert scene.text == "Test narration."
        assert scene.duration_seconds == 10.0
        assert scene.video_keywords == ["tech", "abstract"]

    def test_duration_must_be_positive(self):
        """Test that duration must be positive."""
        with pytest.raises(ValidationError):
            VideoScene(
                text="Test",
                duration_seconds=0,
                video_keywords=["test"],
            )

        with pytest.raises(ValidationError):
            VideoScene(
                text="Test",
                duration_seconds=-5,
                video_keywords=["test"],
            )

    def test_keywords_required(self):
        """Test that keywords are required."""
        with pytest.raises(ValidationError):
            VideoScene(
                text="Test",
                duration_seconds=10.0,
                video_keywords=[],
            )

    def test_keywords_stripped(self):
        """Test that keywords are stripped of whitespace."""
        scene = VideoScene(
            text="Test",
            duration_seconds=10.0,
            video_keywords=["  tech  ", "abstract  ", ""],
        )
        assert scene.video_keywords == ["tech", "abstract"]


class TestVideoScript:
    """Tests for VideoScript model."""

    def test_valid_script(self, sample_script):
        """Test creating a valid script."""
        assert sample_script.title == "Test Video"
        assert sample_script.hook == "This is the hook."
        assert len(sample_script.scenes) == 3
        assert sample_script.cta == "Follow for more!"

    def test_full_narration(self, sample_script):
        """Test full narration property."""
        narration = sample_script.full_narration
        assert "This is the hook." in narration
        assert "First scene content." in narration
        assert "Follow for more!" in narration

    def test_word_count(self, sample_script):
        """Test word count property."""
        word_count = sample_script.word_count
        assert word_count > 0
        assert word_count == len(sample_script.full_narration.split())

    def test_estimated_duration(self, sample_script):
        """Test estimated duration calculation."""
        duration = sample_script.estimated_duration(words_per_minute=150)
        assert duration > 0
        # Word count / 150 * 60 = seconds
        expected = (sample_script.word_count / 150) * 60
        assert duration == expected

    def test_duration_range(self):
        """Test total_duration validation."""
        # Valid range
        script = VideoScript(
            title="Test",
            hook="Hook",
            scenes=[
                VideoScene(
                    text="Scene",
                    duration_seconds=10.0,
                    video_keywords=["test"],
                )
            ],
            cta="CTA",
            total_duration=60,
        )
        assert script.total_duration == 60

        # Too short
        with pytest.raises(ValidationError):
            VideoScript(
                title="Test",
                hook="Hook",
                scenes=[
                    VideoScene(
                        text="Scene",
                        duration_seconds=10.0,
                        video_keywords=["test"],
                    )
                ],
                cta="CTA",
                total_duration=5,  # Below minimum of 15
            )


class TestWordTimestamp:
    """Tests for WordTimestamp model."""

    def test_valid_timestamp(self):
        """Test creating a valid timestamp."""
        ts = WordTimestamp(word="test", start_ms=100, end_ms=500)
        assert ts.word == "test"
        assert ts.start_ms == 100
        assert ts.end_ms == 500

    def test_seconds_conversion(self):
        """Test milliseconds to seconds conversion."""
        ts = WordTimestamp(word="test", start_ms=1500, end_ms=2000)
        assert ts.start_seconds == 1.5
        assert ts.end_seconds == 2.0

    def test_duration_ms(self):
        """Test duration calculation."""
        ts = WordTimestamp(word="test", start_ms=100, end_ms=500)
        assert ts.duration_ms == 400


class TestVoiceoverResult:
    """Tests for VoiceoverResult model."""

    def test_valid_result(self, sample_voiceover_result):
        """Test creating a valid voiceover result."""
        assert sample_voiceover_result.audio_path.suffix == ".mp3"
        assert sample_voiceover_result.srt_path.suffix == ".srt"
        assert sample_voiceover_result.duration_seconds == 0.8
        assert len(sample_voiceover_result.word_timestamps) == 4


class TestVideoClip:
    """Tests for VideoClip model."""

    def test_valid_clip(self, sample_video_clip):
        """Test creating a valid video clip."""
        assert sample_video_clip.path.suffix == ".mp4"
        assert sample_video_clip.duration_seconds == 10.0
        assert sample_video_clip.width == 1080
        assert sample_video_clip.height == 1920

    def test_aspect_ratio(self, sample_video_clip):
        """Test aspect ratio calculation."""
        ratio = sample_video_clip.aspect_ratio
        assert ratio == pytest.approx(9 / 16, rel=0.01)

    def test_is_portrait(self):
        """Test portrait detection."""
        portrait = VideoClip(
            path="/test.mp4",
            source_url="",
            duration_seconds=10.0,
            width=1080,
            height=1920,
            scene_index=1,
            keywords_used=[],
        )
        assert portrait.is_portrait is True

        landscape = VideoClip(
            path="/test.mp4",
            source_url="",
            duration_seconds=10.0,
            width=1920,
            height=1080,
            scene_index=1,
            keywords_used=[],
        )
        assert landscape.is_portrait is False

    def test_is_9_16(self):
        """Test 9:16 ratio detection."""
        perfect = VideoClip(
            path="/test.mp4",
            source_url="",
            duration_seconds=10.0,
            width=1080,
            height=1920,
            scene_index=1,
            keywords_used=[],
        )
        assert perfect.is_9_16 is True

        not_9_16 = VideoClip(
            path="/test.mp4",
            source_url="",
            duration_seconds=10.0,
            width=1000,
            height=1000,
            scene_index=1,
            keywords_used=[],
        )
        assert not_9_16.is_9_16 is False


class TestSubtitleStyle:
    """Tests for SubtitleStyle model."""

    def test_defaults(self):
        """Test default values."""
        style = SubtitleStyle()
        assert style.font == "Montserrat-Bold"
        assert style.font_size == 60
        assert style.color == "white"
        assert style.highlight_color == "#FFD700"
        assert style.position == SubtitlePosition.CENTER
        assert style.animation == SubtitleAnimation.POP

    def test_custom_style(self):
        """Test custom style values."""
        style = SubtitleStyle(
            font="Arial",
            font_size=80,
            color="yellow",
            position=SubtitlePosition.BOTTOM,
            animation=SubtitleAnimation.FADE,
        )
        assert style.font == "Arial"
        assert style.font_size == 80
        assert style.color == "yellow"

    def test_font_size_range(self):
        """Test font size validation."""
        with pytest.raises(ValidationError):
            SubtitleStyle(font_size=10)  # Below minimum

        with pytest.raises(ValidationError):
            SubtitleStyle(font_size=200)  # Above maximum
