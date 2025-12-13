"""Tests for video assembler module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socials_automator.video import OutputConfig, VideoAssemblyError, VideoClip
from socials_automator.video.assembler import VideoAssembler, get_video_info, select_clip_segment


# Check if moviepy is available
try:
    import moviepy.editor
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False


class TestVideoAssembler:
    """Tests for VideoAssembler class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        assembler = VideoAssembler()
        assert assembler.config.width == 1080
        assert assembler.config.height == 1920

    def test_init_custom_config(self, output_config):
        """Test initialization with custom config."""
        assembler = VideoAssembler(config=output_config)
        assert assembler.config == output_config

    def test_crop_to_9_16_already_correct(self):
        """Test cropping when already 9:16."""
        assembler = VideoAssembler()

        mock_clip = MagicMock()
        mock_clip.size = (1080, 1920)

        result = assembler._crop_to_9_16(mock_clip)
        # Should return unchanged if already correct ratio
        assert result == mock_clip

    def test_crop_to_9_16_landscape(self):
        """Test cropping landscape to 9:16."""
        assembler = VideoAssembler()

        mock_clip = MagicMock()
        mock_clip.size = (1920, 1080)  # 16:9 landscape
        mock_clip.crop = MagicMock(return_value=mock_clip)

        result = assembler._crop_to_9_16(mock_clip)
        mock_clip.crop.assert_called_once()

    def test_adjust_duration_longer_clip(self):
        """Test adjusting duration when clip is longer."""
        assembler = VideoAssembler()

        mock_clip = MagicMock()
        mock_clip.duration = 20.0
        mock_clip.subclip = MagicMock(return_value=mock_clip)

        result = assembler._adjust_duration(mock_clip, target_duration=10.0)
        mock_clip.subclip.assert_called_once()

    def test_adjust_duration_shorter_clip(self):
        """Test adjusting duration when clip is shorter."""
        assembler = VideoAssembler()

        mock_clip = MagicMock()
        mock_clip.duration = 5.0
        mock_clip.loop = MagicMock(return_value=mock_clip)

        result = assembler._adjust_duration(mock_clip, target_duration=15.0)
        # Should loop because much shorter
        mock_clip.loop.assert_called_once()

    def test_assemble_no_clips(self, temp_dir):
        """Test assembly with no clips raises error."""
        assembler = VideoAssembler()

        with pytest.raises(VideoAssemblyError, match="No clips provided"):
            assembler.assemble(
                clips=[],
                audio_path=temp_dir / "audio.mp3",
                output_path=temp_dir / "output.mp4",
            )


class TestSelectClipSegment:
    """Tests for clip segment selection."""

    @pytest.mark.skipif(not HAS_MOVIEPY, reason="Requires moviepy")
    def test_clip_shorter_than_needed(self):
        """Test when clip is shorter than needed duration."""
        with patch("moviepy.editor.VideoFileClip") as mock_vfc:
            mock_clip = MagicMock()
            mock_clip.duration = 5.0
            mock_vfc.return_value = mock_clip

            start, end = select_clip_segment(Path("/test.mp4"), needed_duration=10.0)
            assert start == 0
            assert end == 5.0

    @pytest.mark.skipif(not HAS_MOVIEPY, reason="Requires moviepy")
    def test_clip_longer_than_needed(self):
        """Test when clip is longer than needed duration."""
        with patch("moviepy.editor.VideoFileClip") as mock_vfc:
            mock_clip = MagicMock()
            mock_clip.duration = 30.0
            mock_vfc.return_value = mock_clip

            start, end = select_clip_segment(Path("/test.mp4"), needed_duration=10.0)
            # Should select middle segment
            assert start == 10.0  # (30 - 10) / 2
            assert end == 20.0


class TestGetVideoInfo:
    """Tests for video info extraction."""

    @pytest.mark.skipif(not HAS_MOVIEPY, reason="Requires moviepy")
    def test_get_video_info(self):
        """Test getting video information."""
        with patch("moviepy.editor.VideoFileClip") as mock_vfc:
            mock_clip = MagicMock()
            mock_clip.duration = 10.0
            mock_clip.size = (1080, 1920)
            mock_clip.fps = 30
            mock_vfc.return_value = mock_clip

            info = get_video_info(Path("/test.mp4"))
            assert info["duration"] == 10.0
            assert info["width"] == 1080
            assert info["height"] == 1920
            assert info["fps"] == 30
            assert info["aspect_ratio"] == pytest.approx(9 / 16, rel=0.01)


class TestAssemblerIntegration:
    """Integration tests (require moviepy and actual files)."""

    @pytest.mark.skipif(
        not HAS_MOVIEPY,
        reason="Requires moviepy and actual video/audio files",
    )
    def test_full_assembly(self, temp_dir, sample_video_clip, sample_voiceover_result):
        """Test full video assembly."""
        assembler = VideoAssembler()

        output_path = temp_dir / "assembled.mp4"
        result = assembler.assemble(
            clips=[sample_video_clip],
            audio_path=sample_voiceover_result.audio_path,
            output_path=output_path,
        )

        assert result.exists()

    @pytest.mark.skipif(
        not HAS_MOVIEPY,
        reason="Requires moviepy and actual video files",
    )
    def test_create_thumbnail(self, temp_dir):
        """Test thumbnail creation."""
        assembler = VideoAssembler()

        # Would need actual video file
        video_path = temp_dir / "video.mp4"
        thumbnail_path = temp_dir / "thumbnail.jpg"

        result = assembler.create_thumbnail(
            video_path=video_path,
            output_path=thumbnail_path,
            time=2.0,
        )
        assert result.exists()
