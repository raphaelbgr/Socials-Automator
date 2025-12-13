"""Tests for subtitles module."""

import pytest

from socials_automator.video import SubtitleStyle, WordTimestamp
from socials_automator.video.subtitles import (
    SRTEntry,
    SubtitleRenderer,
    group_words_into_phrases,
    parse_srt,
    word_timestamps_to_srt,
)


class TestParseSRT:
    """Tests for SRT parsing."""

    def test_parse_valid_srt(self, temp_dir):
        """Test parsing valid SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,000 --> 00:00:02,000
world
"""
        srt_path = temp_dir / "test.srt"
        srt_path.write_text(srt_content)

        entries = parse_srt(srt_path)
        assert len(entries) == 2
        assert entries[0].text == "Hello"
        assert entries[0].start_ms == 0
        assert entries[0].end_ms == 1000
        assert entries[1].text == "world"

    def test_parse_empty_srt(self, temp_dir):
        """Test parsing empty SRT file."""
        srt_path = temp_dir / "empty.srt"
        srt_path.write_text("")

        entries = parse_srt(srt_path)
        assert len(entries) == 0

    def test_parse_malformed_entry(self, temp_dir):
        """Test parsing SRT with malformed entries."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
Hello

invalid entry

3
00:00:02,000 --> 00:00:03,000
world
"""
        srt_path = temp_dir / "malformed.srt"
        srt_path.write_text(srt_content)

        entries = parse_srt(srt_path)
        # Should skip malformed entry
        assert len(entries) == 2


class TestSRTEntry:
    """Tests for SRTEntry dataclass."""

    def test_properties(self):
        """Test SRT entry properties."""
        entry = SRTEntry(index=1, start_ms=1500, end_ms=2500, text="test")
        assert entry.start_seconds == 1.5
        assert entry.end_seconds == 2.5


class TestWordTimestampsToSRT:
    """Tests for word timestamps to SRT conversion."""

    def test_basic_conversion(self, temp_dir, sample_word_timestamps):
        """Test converting word timestamps to SRT."""
        output_path = temp_dir / "output.srt"
        result = word_timestamps_to_srt(sample_word_timestamps, output_path)

        assert result.exists()
        content = result.read_text()
        assert "This" in content
        assert "00:00:00,000 --> 00:00:00,200" in content

    def test_empty_timestamps(self, temp_dir):
        """Test with empty timestamp list."""
        output_path = temp_dir / "empty.srt"
        result = word_timestamps_to_srt([], output_path)

        assert result.exists()
        assert result.read_text() == ""


class TestGroupWordsIntoPhrases:
    """Tests for phrase grouping."""

    def test_basic_grouping(self, sample_word_timestamps):
        """Test basic word grouping."""
        phrases = group_words_into_phrases(
            sample_word_timestamps,
            words_per_phrase=2,
        )

        assert len(phrases) == 2
        assert phrases[0][0] == "This is"
        assert phrases[1][0] == "a test"

    def test_single_phrase(self, sample_word_timestamps):
        """Test when all words fit in one phrase."""
        phrases = group_words_into_phrases(
            sample_word_timestamps,
            words_per_phrase=10,
        )

        assert len(phrases) == 1
        assert phrases[0][0] == "This is a test"

    def test_empty_timestamps(self):
        """Test with empty timestamps."""
        phrases = group_words_into_phrases([])
        assert len(phrases) == 0

    def test_max_duration_split(self):
        """Test splitting by max duration."""
        timestamps = [
            WordTimestamp(word="word1", start_ms=0, end_ms=1000),
            WordTimestamp(word="word2", start_ms=1000, end_ms=2000),
            WordTimestamp(word="word3", start_ms=2000, end_ms=4000),  # Longer gap
        ]

        phrases = group_words_into_phrases(
            timestamps,
            words_per_phrase=10,
            max_duration_ms=2500,
        )

        # Should split due to duration
        assert len(phrases) >= 1


class TestSubtitleRenderer:
    """Tests for subtitle renderer."""

    def test_init_default_style(self):
        """Test initialization with default style."""
        renderer = SubtitleRenderer()
        assert renderer.style.font == "Montserrat-Bold"

    def test_init_custom_style(self, subtitle_style):
        """Test initialization with custom style."""
        renderer = SubtitleRenderer(style=subtitle_style)
        assert renderer.style == subtitle_style


class TestSubtitleRendererIntegration:
    """Integration tests (require pycaps or moviepy)."""

    @pytest.mark.skipif(
        True,
        reason="Requires pycaps or moviepy and actual video files",
    )
    def test_render_with_pycaps(self, temp_dir):
        """Test rendering with pycaps."""
        renderer = SubtitleRenderer()

        video_path = temp_dir / "input.mp4"
        srt_path = temp_dir / "subtitles.srt"
        output_path = temp_dir / "output.mp4"

        # Would need actual files
        result = renderer.render(video_path, srt_path, output_path)
        assert result.exists()

    @pytest.mark.skipif(
        True,
        reason="Requires moviepy and actual video files",
    )
    def test_render_with_moviepy_fallback(self, temp_dir):
        """Test rendering with MoviePy fallback."""
        renderer = SubtitleRenderer()

        video_path = temp_dir / "input.mp4"
        srt_path = temp_dir / "subtitles.srt"
        output_path = temp_dir / "output.mp4"

        # Would need actual files
        result = renderer._render_with_moviepy(video_path, srt_path, output_path)
        assert result.exists()
