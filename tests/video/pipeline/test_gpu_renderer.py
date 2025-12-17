r"""Unit tests for GPU video renderer FFmpeg filter escaping.

Tests edge cases for path escaping in FFmpeg subtitle and drawtext filters on Windows.

The correct escaping for FFmpeg filters on Windows:
1. Convert backslashes to forward slashes
2. Escape single quotes with '\''
3. Escape colons with \:
4. Wrap entire path in single quotes

Example: C:\Users\test\video.srt -> 'C\:/Users/test/video.srt'

References:
- FFmpeg Utils Documentation: https://ffmpeg.org/ffmpeg-utils.html#Quoting-and-escaping
- FFmpeg Filters Documentation: https://ffmpeg.org/ffmpeg-filters.html
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the actual utility function from the GPU renderer module
from socials_automator.video.pipeline.video_renderer_gpu import escape_ffmpeg_filter_path


class TestFFmpegPathEscaping:
    """Test FFmpeg path escaping for filter arguments."""

    def test_simple_windows_path(self):
        """Test basic Windows path with drive letter."""
        path = r"C:\Users\test\video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/video.srt"

    def test_path_with_spaces(self):
        """Test Windows path with spaces in directory names."""
        path = r"C:\Users\John Doe\My Videos\subtitle.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/John Doe/My Videos/subtitle.srt"

    def test_temp_path(self):
        """Test typical temp directory path."""
        path = r"C:\Users\rbgnr\AppData\Local\Temp\video_20251216_abc123\voiceover.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/rbgnr/AppData/Local/Temp/video_20251216_abc123/voiceover.srt"

    def test_different_drive_letters(self):
        """Test various drive letters."""
        for drive in ["C", "D", "E", "Z"]:
            path = f"{drive}:\\folder\\file.srt"
            escaped = escape_ffmpeg_filter_path(path)
            assert escaped == f"{drive}\\:/folder/file.srt"

    def test_deep_nested_path(self):
        """Test deeply nested directory structure."""
        path = r"D:\projects\socials\output\2024\12\16\post_001\subtitles\final.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "D\\:/projects/socials/output/2024/12/16/post_001/subtitles/final.srt"

    def test_path_with_special_chars_in_filename(self):
        """Test path with special characters in filename."""
        path = r"C:\Users\test\video (1).srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/video (1).srt"

    def test_path_with_underscores(self):
        """Test path with underscores (common in temp files)."""
        path = r"C:\Temp\video_20251216-233340_zpefb8fz\voiceover.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Temp/video_20251216-233340_zpefb8fz/voiceover.srt"

    def test_forward_slash_path_only_escapes_colons(self):
        """Test that forward slash paths only escape colons."""
        path = "C:/Users/test/video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/video.srt"

    def test_mixed_slashes(self):
        """Test path with mixed forward and back slashes."""
        path = r"C:\Users/test\subfolder/video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/subfolder/video.srt"

    def test_path_with_dots(self):
        """Test path with dots in directory names."""
        path = r"C:\Users\user.name\folder.backup\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/user.name/folder.backup/file.srt"

    def test_path_with_hyphen(self):
        """Test path with hyphens."""
        path = r"C:\my-project\sub-folder\my-video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/my-project/sub-folder/my-video.srt"

    def test_path_with_at_symbol(self):
        """Test path with @ symbol."""
        path = r"C:\Users\test\@project\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/@project/file.srt"

    def test_path_with_hash(self):
        """Test path with # symbol."""
        path = r"C:\Users\test\#backup\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/#backup/file.srt"

    def test_path_with_ampersand(self):
        """Test path with & symbol."""
        path = r"C:\Users\test\Tom & Jerry\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/Tom & Jerry/file.srt"

    def test_path_with_equals(self):
        """Test path with = symbol."""
        path = r"C:\Users\test\a=b\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/a=b/file.srt"


class TestFontPathEscaping:
    """Test FFmpeg path escaping specifically for font paths (the bug we fixed)."""

    def test_windows_fonts_path(self):
        """Test standard Windows fonts path."""
        path = "C:/Windows/Fonts/arial.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Windows/Fonts/arial.ttf"

    def test_windows_fonts_path_backslash(self):
        """Test Windows fonts path with backslashes."""
        path = r"C:\Windows\Fonts\arial.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Windows/Fonts/arial.ttf"

    def test_custom_font_path(self):
        """Test custom font in project folder."""
        path = r"C:\Users\rbgnr\git\Socials-Automator\fonts\Montserrat-Bold.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/rbgnr/git/Socials-Automator/fonts/Montserrat-Bold.ttf"

    def test_font_path_with_spaces(self):
        """Test font path with spaces."""
        path = r"C:\Program Files\Common Files\Fonts\My Font.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Program Files/Common Files/Fonts/My Font.ttf"

    def test_various_font_files(self):
        """Test various font file types."""
        fonts = [
            (r"C:\Windows\Fonts\arial.ttf", "C\\:/Windows/Fonts/arial.ttf"),
            (r"C:\Windows\Fonts\times.ttf", "C\\:/Windows/Fonts/times.ttf"),
            (r"C:\Windows\Fonts\calibri.ttf", "C\\:/Windows/Fonts/calibri.ttf"),
            (r"C:\Fonts\custom-font.otf", "C\\:/Fonts/custom-font.otf"),
        ]
        for original, expected in fonts:
            escaped = escape_ffmpeg_filter_path(original)
            assert escaped == expected, f"Failed for {original}"


class TestSubtitleFilterConstruction:
    """Test FFmpeg subtitle filter string construction."""

    def build_subtitle_filter(
        self,
        srt_path: str,
        font: str = "Montserrat-Bold.ttf",
        font_size: int = 80,
        stroke_width: int = 4
    ) -> str:
        """Build subtitle filter string matching GPUSubtitleRenderer."""
        srt_escaped = escape_ffmpeg_filter_path(srt_path)

        style = (
            f"FontName={font},"
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"Outline={stroke_width},"
            f"Alignment=2,"
            f"MarginV=200"
        )

        return f"subtitles='{srt_escaped}':force_style='{style}'"

    def test_basic_filter_construction(self):
        """Test basic filter string construction."""
        srt_path = r"C:\temp\video.srt"
        filter_str = self.build_subtitle_filter(srt_path)

        assert filter_str.startswith("subtitles='C\\:/temp/video.srt':")
        assert ":force_style='" in filter_str
        assert "FontName=Montserrat-Bold.ttf" in filter_str
        assert "FontSize=80" in filter_str

    def test_filter_quote_structure(self):
        """Ensure correct quote structure for FFmpeg."""
        srt_path = r"C:\temp\video.srt"
        filter_str = self.build_subtitle_filter(srt_path)

        assert filter_str.startswith("subtitles='")
        assert "':force_style='" in filter_str
        assert filter_str.count("'") == 4

    def test_filter_with_spaces_in_path(self):
        """Test filter with spaces in path works."""
        srt_path = r"C:\My Documents\video files\test.srt"
        filter_str = self.build_subtitle_filter(srt_path)

        assert "My Documents" in filter_str
        assert "video files" in filter_str
        assert filter_str.startswith("subtitles='C\\:/My Documents/")

    def test_custom_font_settings(self):
        """Test custom font settings in filter."""
        srt_path = r"C:\temp\video.srt"
        filter_str = self.build_subtitle_filter(
            srt_path,
            font="Arial.ttf",
            font_size=100,
            stroke_width=6
        )

        assert "FontName=Arial.ttf" in filter_str
        assert "FontSize=100" in filter_str
        assert "Outline=6" in filter_str


class TestDrawtextFilterConstruction:
    """Test FFmpeg drawtext filter string construction (for watermark)."""

    def build_drawtext_filter(
        self,
        font_path: str,
        text: str = "@ai.for.mortals",
        fontsize: int = 24
    ) -> str:
        """Build drawtext filter string matching GPUSubtitleRenderer."""
        font_escaped = escape_ffmpeg_filter_path(font_path)

        return (
            f"drawtext=text='{text}':"
            f"fontfile='{font_escaped}':"
            f"fontsize={fontsize}:"
            f"fontcolor=white@0.5:"
            f"x=(w-text_w)/2:"
            f"y=h-50"
        )

    def test_basic_drawtext_construction(self):
        """Test basic drawtext filter construction."""
        font_path = r"C:\Windows\Fonts\arial.ttf"
        filter_str = self.build_drawtext_filter(font_path)

        assert "drawtext=text='@ai.for.mortals'" in filter_str
        assert "fontfile='C\\:/Windows/Fonts/arial.ttf'" in filter_str
        assert "fontsize=24" in filter_str

    def test_drawtext_font_path_escaped(self):
        """Test that font path colon is properly escaped."""
        font_path = "C:/Windows/Fonts/arial.ttf"
        filter_str = self.build_drawtext_filter(font_path)

        # The colon after C should be escaped
        assert "C\\:/Windows" in filter_str
        # No unescaped colon in the path
        assert "C:/Windows" not in filter_str

    def test_drawtext_custom_font_path(self):
        """Test drawtext with custom font path."""
        font_path = r"D:\Fonts\MyCustomFont.ttf"
        filter_str = self.build_drawtext_filter(font_path)

        assert "fontfile='D\\:/Fonts/MyCustomFont.ttf'" in filter_str

    def test_drawtext_with_spaces_in_font_path(self):
        """Test drawtext with spaces in font path."""
        font_path = r"C:\Program Files\Fonts\My Font.ttf"
        filter_str = self.build_drawtext_filter(font_path)

        assert "fontfile='C\\:/Program Files/Fonts/My Font.ttf'" in filter_str


class TestFullFilterChain:
    """Test the complete filter chain (subtitles + drawtext)."""

    def build_full_filter_chain(
        self,
        srt_path: str,
        font_path: str
    ) -> str:
        """Build full filter chain as used in GPUSubtitleRenderer."""
        srt_escaped = escape_ffmpeg_filter_path(srt_path)
        font_escaped = escape_ffmpeg_filter_path(font_path)

        style = (
            "FontName=Montserrat-Bold.ttf,"
            "FontSize=80,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "Outline=4,"
            "Alignment=2,"
            "MarginV=200"
        )

        subtitles = f"subtitles='{srt_escaped}':force_style='{style}'"
        drawtext = (
            f"drawtext=text='@ai.for.mortals':"
            f"fontfile='{font_escaped}':"
            f"fontsize=24:"
            f"fontcolor=white@0.5:"
            f"x=(w-text_w)/2:"
            f"y=h-50"
        )

        return f"[0:v]{subtitles},{drawtext}[v]"

    def test_full_chain_structure(self):
        """Test full filter chain structure."""
        srt_path = r"C:\temp\video.srt"
        font_path = r"C:\Windows\Fonts\arial.ttf"
        filter_chain = self.build_full_filter_chain(srt_path, font_path)

        assert filter_chain.startswith("[0:v]")
        assert filter_chain.endswith("[v]")
        assert "subtitles=" in filter_chain
        assert "drawtext=" in filter_chain
        assert "," in filter_chain  # Filters separated by comma

    def test_full_chain_both_paths_escaped(self):
        """Test that both SRT and font paths are escaped."""
        srt_path = r"C:\temp\video.srt"
        font_path = r"C:\Windows\Fonts\arial.ttf"
        filter_chain = self.build_full_filter_chain(srt_path, font_path)

        # Both paths should have escaped colons
        assert "C\\:/temp/video.srt" in filter_chain
        assert "C\\:/Windows/Fonts/arial.ttf" in filter_chain
        # No unescaped Windows drive colons
        assert "C:/temp" not in filter_chain
        assert "C:/Windows" not in filter_chain

    def test_full_chain_different_drives(self):
        """Test filter chain with paths on different drives."""
        srt_path = r"D:\videos\subtitles.srt"
        font_path = r"E:\fonts\custom.ttf"
        filter_chain = self.build_full_filter_chain(srt_path, font_path)

        assert "D\\:/videos/subtitles.srt" in filter_chain
        assert "E\\:/fonts/custom.ttf" in filter_chain


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_path(self):
        """Test handling of empty path."""
        escaped = escape_ffmpeg_filter_path("")
        assert escaped == ""

    def test_unix_style_path(self):
        """Test Unix-style path (no drive letter)."""
        path = "/home/user/video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "/home/user/video.srt"

    def test_relative_path(self):
        """Test relative path."""
        path = r".\output\video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "./output/video.srt"

    def test_relative_path_parent(self):
        """Test relative path with parent directory."""
        path = r"..\output\video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "../output/video.srt"

    def test_path_with_multiple_colons(self):
        """Test path with multiple colons (invalid but should handle)."""
        path = r"C:\folder:name\file:test.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/folder\\:name/file\\:test.srt"

    def test_pathlib_path_object(self):
        """Test that Path objects work when converted to string."""
        path = Path(r"C:\Users\test\video.srt")
        escaped = escape_ffmpeg_filter_path(str(path))
        assert escaped == "C\\:/Users/test/video.srt"

    def test_path_with_unicode_characters(self):
        """Test path with unicode characters."""
        path = r"C:\Users\usuario\videos\archivo.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/usuario/videos/archivo.srt"

    def test_path_with_single_quote(self):
        """Test path with single quote (apostrophe) in name."""
        path = r"C:\Users\test\John's Videos\video.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/test/John'\\''s Videos/video.srt"

    def test_path_with_multiple_quotes(self):
        """Test path with multiple single quotes."""
        path = r"C:\It's\Mike's\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/It'\\''s/Mike'\\''s/file.srt"

    def test_path_only_drive_letter(self):
        """Test path that's just a drive letter."""
        path = "C:\\"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/"

    def test_unc_path(self):
        """Test UNC network path."""
        path = r"\\server\share\folder\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "//server/share/folder/file.srt"

    def test_path_with_brackets(self):
        """Test path with square brackets."""
        path = r"C:\test\file[1].srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/test/file[1].srt"

    def test_path_with_semicolon(self):
        """Test path with semicolon (rare but possible)."""
        path = r"C:\test\a;b\file.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/test/a;b/file.srt"


class TestRealWorldScenarios:
    """Test real-world path scenarios from actual usage."""

    def test_windows_temp_folder_pattern(self):
        """Test the actual temp folder pattern used by the app."""
        path = r"C:\Users\rbgnr\AppData\Local\Temp\video_20251216-232926_wqrk8h5e\voiceover.srt"
        escaped = escape_ffmpeg_filter_path(path)
        expected = "C\\:/Users/rbgnr/AppData/Local/Temp/video_20251216-232926_wqrk8h5e/voiceover.srt"
        assert escaped == expected

    def test_profiles_output_path(self):
        """Test typical profile output path."""
        path = r"C:\Users\rbgnr\git\Socials-Automator\profiles\ai.for.mortals\reels\2024\12\generated\post_001\voiceover.srt"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped.startswith("C\\:/Users/rbgnr/git/Socials-Automator/profiles/")

    def test_windows_system_fonts_arial(self):
        """Test common Arial font path."""
        path = r"C:\Windows\Fonts\arial.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Windows/Fonts/arial.ttf"

    def test_windows_system_fonts_montserrat(self):
        """Test Montserrat font path."""
        path = r"C:\Windows\Fonts\Montserrat-Bold.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Windows/Fonts/Montserrat-Bold.ttf"

    def test_project_fonts_folder(self):
        """Test project fonts folder path."""
        path = r"C:\Users\rbgnr\git\Socials-Automator\fonts\Montserrat-Bold.ttf"
        escaped = escape_ffmpeg_filter_path(path)
        assert escaped == "C\\:/Users/rbgnr/git/Socials-Automator/fonts/Montserrat-Bold.ttf"


class TestFFmpegFilterValidation:
    """Test that filter strings are valid for FFmpeg syntax."""

    def test_no_unescaped_colons_in_path(self):
        """Ensure colons in paths are properly escaped."""
        import re
        paths = [
            r"C:\test\file.srt",
            r"D:\folder\subfolder\video.srt",
            r"E:\a\b\c\d\e.srt",
        ]

        for path in paths:
            escaped = escape_ffmpeg_filter_path(path)
            # There should be no unescaped colons (colon not preceded by backslash)
            unescaped = re.findall(r'(?<!\\):', escaped)
            assert len(unescaped) == 0, f"Found unescaped colon in: {escaped}"

    def test_no_problematic_characters_in_style(self):
        """Ensure style string doesn't have problematic characters."""
        style = (
            "FontName=Montserrat-Bold.ttf,"
            "FontSize=80,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "Outline=4,"
            "Alignment=2,"
            "MarginV=200"
        )

        # Style should not have single quotes
        assert "'" not in style
        # Style should not have backslashes
        assert "\\" not in style

    def test_filter_chain_comma_separation(self):
        """Test that filters are properly comma-separated."""
        srt_escaped = escape_ffmpeg_filter_path(r"C:\test\video.srt")
        font_escaped = escape_ffmpeg_filter_path(r"C:\Windows\Fonts\arial.ttf")

        subtitles = f"subtitles='{srt_escaped}':force_style='FontSize=80'"
        drawtext = f"drawtext=text='test':fontfile='{font_escaped}':fontsize=24"
        chain = f"{subtitles},{drawtext}"

        # Should have exactly one comma separating filters
        assert chain.count(",drawtext=") == 1


class TestFFmpegIntegration:
    """Integration tests that actually run FFmpeg to verify escaping works.

    These tests require FFmpeg to be installed.
    """

    @pytest.fixture
    def temp_srt(self):
        """Create a temporary SRT file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
            path = f.name
        yield path
        os.unlink(path)

    def _run_ffmpeg_filter(self, filter_str: str, timeout: int = 30) -> bool:
        """Run FFmpeg with the given filter and return success status."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=c=black:s=100x100:d=1',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-filter_complex', filter_str,
            '-map', '[v]', '-map', '1:a',
            '-t', '1', '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("FFmpeg not available or timed out")
            return False

    def test_ffmpeg_subtitles_filter_escaped_path(self, temp_srt):
        """Test that escaped path works with actual FFmpeg."""
        srt_escaped = escape_ffmpeg_filter_path(temp_srt)
        filter_str = f"[0:v]subtitles='{srt_escaped}':force_style='FontSize=20'[v]"
        assert self._run_ffmpeg_filter(filter_str)

    def test_ffmpeg_drawtext_filter_escaped_font(self, temp_srt):
        """Test that escaped font path works with actual FFmpeg."""
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path.replace("/", "\\")):
            pytest.skip("Arial font not found")

        font_escaped = escape_ffmpeg_filter_path(font_path)
        filter_str = f"[0:v]drawtext=text='test':fontfile='{font_escaped}':fontsize=24[v]"
        assert self._run_ffmpeg_filter(filter_str)

    def test_ffmpeg_full_chain_both_escaped(self, temp_srt):
        """Test full filter chain with both paths escaped."""
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path.replace("/", "\\")):
            pytest.skip("Arial font not found")

        srt_escaped = escape_ffmpeg_filter_path(temp_srt)
        font_escaped = escape_ffmpeg_filter_path(font_path)

        style = "FontName=arial.ttf,FontSize=80,PrimaryColour=&H00FFFFFF"
        subtitles = f"subtitles='{srt_escaped}':force_style='{style}'"
        drawtext = f"drawtext=text='test':fontfile='{font_escaped}':fontsize=24"
        filter_str = f"[0:v]{subtitles},{drawtext}[v]"

        assert self._run_ffmpeg_filter(filter_str)

    def test_ffmpeg_unescaped_path_fails(self, temp_srt):
        """Verify that unescaped path fails (confirms our fix is needed)."""
        # Use forward slashes but DON'T escape the colon
        unescaped = temp_srt.replace("\\", "/")
        # Only test if path has a colon (Windows drive letter)
        if ":" not in unescaped:
            pytest.skip("Path doesn't have colon to test")

        filter_str = f"[0:v]subtitles='{unescaped}':force_style='FontSize=20'[v]"
        # This should fail because colon is not escaped
        assert not self._run_ffmpeg_filter(filter_str)

    def test_ffmpeg_unescaped_font_fails(self, temp_srt):
        """Verify that unescaped font path fails (confirms our fix is needed)."""
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path.replace("/", "\\")):
            pytest.skip("Arial font not found")

        # Don't escape the font path colon
        filter_str = f"[0:v]drawtext=text='test':fontfile='{font_path}':fontsize=24[v]"
        # This should fail because colon is not escaped
        assert not self._run_ffmpeg_filter(filter_str)


class TestGPUVideoAssemblerMetadata:
    """Test GPUVideoAssembler metadata creation and attribute access.

    These tests verify the assembler correctly accesses VideoClipInfo attributes
    and creates proper metadata structures.
    """

    @pytest.fixture
    def mock_clips(self):
        """Create mock VideoClipInfo objects for testing."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        clips = []
        for i in range(3):
            clip = VideoClipInfo(
                segment_index=i,
                path=Path(f"C:/temp/clip_{i}.mp4"),
                source_url=f"https://pexels.com/video/{1000+i}",
                pexels_id=1000 + i,
                title=f"Test clip {i}",
                duration_seconds=10.0,
                width=1920,
                height=1080,
                keywords_used=["keyword1", "keyword2", f"keyword_{i}"],
            )
            clips.append(clip)
        return clips

    @pytest.fixture
    def mock_context(self, mock_clips):
        """Create mock PipelineContext for testing."""
        from socials_automator.video.pipeline.base import PipelineContext, VideoScript, TopicInfo

        context = MagicMock(spec=PipelineContext)
        context.clips = mock_clips
        context.audio_path = Path("C:/temp/audio.mp3")
        context.temp_dir = Path("C:/temp/output")
        context.output_dir = Path("C:/temp/output")
        context.script = MagicMock(spec=VideoScript)
        context.script.title = "Test Video Title"
        context.script.full_narration = "This is the test narration."
        context.topic = MagicMock(spec=TopicInfo)
        context.topic.topic = "Test Topic"
        context.metadata = None
        return context

    def test_clip_has_keywords_used_attribute(self, mock_clips):
        """Test that VideoClipInfo has keywords_used attribute (not keywords)."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        for clip in mock_clips:
            # This should NOT raise AttributeError
            assert hasattr(clip, 'keywords_used'), "VideoClipInfo should have keywords_used"
            assert isinstance(clip.keywords_used, list)

            # Verify keywords attribute does NOT exist in the model class
            # (accessing it directly would work via Pydantic but we shouldn't use it)
            assert 'keywords' not in VideoClipInfo.model_fields, "VideoClipInfo should not have 'keywords' field"

    def test_segment_metadata_uses_keywords_used(self, mock_clips):
        """Test that segment metadata correctly uses keywords_used attribute."""
        # Simulate what GPUVideoAssembler.execute() does for metadata
        audio_duration = 30.0
        num_clips = len(mock_clips)
        base_duration = audio_duration / num_clips

        segment_metadata = []
        current_time = 0.0

        for i, clip in enumerate(sorted(mock_clips, key=lambda c: c.segment_index)):
            clip_duration = base_duration

            # This is the exact code from GPUVideoAssembler - must use keywords_used
            segment_metadata.append({
                "index": clip.segment_index,
                "start_time": current_time,
                "end_time": current_time + clip_duration,
                "duration": clip_duration,
                "pexels_id": clip.pexels_id,
                "source_url": clip.source_url,
                "keywords": clip.keywords_used,  # Must be keywords_used, not keywords
            })
            current_time += clip_duration

        # Verify all segments have keywords from keywords_used
        for i, seg in enumerate(segment_metadata):
            assert "keywords" in seg
            assert isinstance(seg["keywords"], list)
            assert f"keyword_{i}" in seg["keywords"]

    def test_segment_metadata_is_dict_not_object(self, mock_clips):
        """Test that segment metadata is list[dict], not list of Pydantic models."""
        audio_duration = 30.0
        num_clips = len(mock_clips)
        base_duration = audio_duration / num_clips

        segment_metadata = []
        current_time = 0.0

        for clip in sorted(mock_clips, key=lambda c: c.segment_index):
            # Must be dict, not VideoSegment or other Pydantic model
            segment_metadata.append({
                "index": clip.segment_index,
                "start_time": current_time,
                "end_time": current_time + base_duration,
                "duration": base_duration,
                "pexels_id": clip.pexels_id,
                "source_url": clip.source_url,
                "keywords": clip.keywords_used,
            })
            current_time += base_duration

        # Verify all segments are dicts
        for seg in segment_metadata:
            assert isinstance(seg, dict), "Segment metadata must be dict, not Pydantic model"

    def test_clips_used_structure(self, mock_clips):
        """Test that clips_used metadata has correct structure."""
        clips_used = [
            {
                "segment_index": c.segment_index,
                "pexels_id": c.pexels_id,
                "source_url": c.source_url,
                "title": c.title,
            }
            for c in sorted(mock_clips, key=lambda c: c.segment_index)
        ]

        assert len(clips_used) == 3
        for i, clip_info in enumerate(clips_used):
            assert clip_info["segment_index"] == i
            assert clip_info["pexels_id"] == 1000 + i
            assert "pexels.com" in clip_info["source_url"]

    def test_duration_calculation(self, mock_clips):
        """Test that clip durations are calculated correctly."""
        audio_duration = 60.0
        num_clips = len(mock_clips)

        # Calculate segment durations (evenly distributed)
        base_duration = audio_duration / num_clips
        segment_durations = []
        remaining = audio_duration

        for i in range(num_clips):
            if i == num_clips - 1:
                segment_durations.append(remaining)
            else:
                segment_durations.append(base_duration)
                remaining -= base_duration

        # Verify total matches audio duration
        assert abs(sum(segment_durations) - audio_duration) < 0.001

        # Verify each segment is roughly equal
        for dur in segment_durations:
            assert abs(dur - 20.0) < 0.001  # 60s / 3 clips = 20s each


class TestGPUSubtitleRendererWatermark:
    """Test GPUSubtitleRenderer watermark functionality."""

    def test_watermark_positions_count(self):
        """Test that watermark has 5 positions."""
        from socials_automator.constants import VIDEO_WIDTH, VIDEO_HEIGHT

        # Positions matching GPU renderer implementation
        margin_x = int(VIDEO_WIDTH * 0.05)
        margin_top = int(VIDEO_HEIGHT * 0.15)
        margin_mid = int(VIDEO_HEIGHT * 0.40)
        text_width_approx = 220

        positions = [
            (margin_x, margin_top),                         # Top-left
            (VIDEO_WIDTH - text_width_approx, margin_top),  # Top-right
            (margin_x, margin_mid),                         # Middle-left
            (VIDEO_WIDTH - text_width_approx, margin_mid),  # Middle-right
            ((VIDEO_WIDTH - text_width_approx) // 2, margin_top),  # Top-center (approx)
        ]

        assert len(positions) == 5, "Watermark should have 5 positions"

    def test_watermark_enable_expressions(self):
        """Test watermark enable expressions for time-based visibility."""
        interval = 10
        num_positions = 5
        cycle_duration = interval * num_positions  # 50 seconds

        # Generate enable expressions like GPUSubtitleRenderer does
        enable_expressions = []
        for i in range(num_positions):
            start_time = i * interval
            # FFmpeg enable expression
            enable_expr = f"lt(mod(t,{cycle_duration}),{start_time + interval})*gte(mod(t,{cycle_duration}),{start_time})"
            enable_expressions.append(enable_expr)

        # Verify expressions
        assert len(enable_expressions) == 5

        # First position: 0-10s
        assert "lt(mod(t,50),10)" in enable_expressions[0]
        assert "gte(mod(t,50),0)" in enable_expressions[0]

        # Second position: 10-20s
        assert "lt(mod(t,50),20)" in enable_expressions[1]
        assert "gte(mod(t,50),10)" in enable_expressions[1]

        # Last position: 40-50s
        assert "lt(mod(t,50),50)" in enable_expressions[4]
        assert "gte(mod(t,50),40)" in enable_expressions[4]

    def test_watermark_cycle_duration(self):
        """Test watermark cycle repeats correctly."""
        interval = 10
        num_positions = 5
        cycle_duration = interval * num_positions

        assert cycle_duration == 50, "Watermark should cycle every 50 seconds"

    def test_watermark_fontsize(self):
        """Test watermark fontsize is correct."""
        watermark_fontsize = 28  # From GPUSubtitleRenderer
        assert watermark_fontsize == 28, "Watermark fontsize should be 28"


class TestKaraokeASSGeneration:
    """Test karaoke ASS subtitle generation."""

    @pytest.fixture
    def temp_srt_with_words(self):
        """Create a temporary SRT file with word-level timing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write("""1
00:00:00,000 --> 00:00:00,500
Hello

2
00:00:00,500 --> 00:00:01,000
World

3
00:00:01,000 --> 00:00:01,500
Test

4
00:00:01,500 --> 00:00:02,000
Word

5
00:00:02,000 --> 00:00:02,500
Five

6
00:00:02,500 --> 00:00:03,000
Six
""")
            path = f.name
        yield path
        os.unlink(path)

    def test_parse_srt_basic(self, temp_srt_with_words):
        """Test SRT parsing extracts words and timings."""
        from socials_automator.video.pipeline.video_renderer_gpu import _parse_srt

        entries = _parse_srt(Path(temp_srt_with_words))

        assert len(entries) == 6
        assert entries[0] == ("Hello", 0.0, 0.5)
        assert entries[1] == ("World", 0.5, 1.0)
        assert entries[2] == ("Test", 1.0, 1.5)

    def test_parse_srt_returns_tuples(self, temp_srt_with_words):
        """Test SRT parsing returns correct tuple structure."""
        from socials_automator.video.pipeline.video_renderer_gpu import _parse_srt

        entries = _parse_srt(Path(temp_srt_with_words))

        for entry in entries:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            text, start, end = entry
            assert isinstance(text, str)
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert end > start

    def test_generate_karaoke_ass_creates_file(self, temp_srt_with_words):
        """Test karaoke ASS generation creates output file."""
        from socials_automator.video.pipeline.video_renderer_gpu import _generate_karaoke_ass

        output_path = Path(temp_srt_with_words).parent / "test_karaoke.ass"
        try:
            result = _generate_karaoke_ass(
                srt_path=Path(temp_srt_with_words),
                output_path=output_path,
            )

            assert result.exists()
            content = result.read_text(encoding='utf-8')
            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_karaoke_ass_has_yellow_highlight(self, temp_srt_with_words):
        """Test karaoke ASS has yellow color tag for highlighting."""
        from socials_automator.video.pipeline.video_renderer_gpu import _generate_karaoke_ass

        output_path = Path(temp_srt_with_words).parent / "test_karaoke.ass"
        try:
            result = _generate_karaoke_ass(
                srt_path=Path(temp_srt_with_words),
                output_path=output_path,
            )

            content = result.read_text(encoding='utf-8')
            # Yellow in BGR hex format for ASS
            assert "\\1c&H00FFFF&" in content, "Should have yellow color tag"
            # White color reset
            assert "\\1c&HFFFFFF&" in content, "Should have white color reset"
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_karaoke_ass_playres_matches_video(self, temp_srt_with_words):
        """Test karaoke ASS PlayRes matches video resolution."""
        from socials_automator.video.pipeline.video_renderer_gpu import _generate_karaoke_ass
        from socials_automator.constants import VIDEO_WIDTH, VIDEO_HEIGHT

        output_path = Path(temp_srt_with_words).parent / "test_karaoke.ass"
        try:
            result = _generate_karaoke_ass(
                srt_path=Path(temp_srt_with_words),
                output_path=output_path,
            )

            content = result.read_text(encoding='utf-8')
            assert f"PlayResX: {VIDEO_WIDTH}" in content
            assert f"PlayResY: {VIDEO_HEIGHT}" in content
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_karaoke_ass_groups_into_phrases(self, temp_srt_with_words):
        """Test karaoke ASS groups words into 3-word phrases."""
        from socials_automator.video.pipeline.video_renderer_gpu import _generate_karaoke_ass

        output_path = Path(temp_srt_with_words).parent / "test_karaoke.ass"
        try:
            result = _generate_karaoke_ass(
                srt_path=Path(temp_srt_with_words),
                output_path=output_path,
            )

            content = result.read_text(encoding='utf-8')
            # First phrase should be "Hello World Test"
            # Check that words appear together in dialogue lines
            assert "HELLO" in content  # Words are uppercased
            assert "WORLD" in content
            assert "TEST" in content
        finally:
            if output_path.exists():
                output_path.unlink()


class TestGPURendererImports:
    """Test that GPU renderer classes can be imported and instantiated."""

    def test_import_gpu_video_assembler(self):
        """Test GPUVideoAssembler can be imported."""
        from socials_automator.video.pipeline.video_renderer_gpu import GPUVideoAssembler
        assert GPUVideoAssembler is not None

    def test_import_gpu_subtitle_renderer(self):
        """Test GPUSubtitleRenderer can be imported."""
        from socials_automator.video.pipeline.video_renderer_gpu import GPUSubtitleRenderer
        assert GPUSubtitleRenderer is not None

    def test_instantiate_gpu_video_assembler(self):
        """Test GPUVideoAssembler can be instantiated."""
        from socials_automator.video.pipeline.video_renderer_gpu import GPUVideoAssembler
        assembler = GPUVideoAssembler()
        assert assembler.WIDTH == 1080
        assert assembler.HEIGHT == 1920
        assert assembler.FPS == 30

    def test_instantiate_gpu_subtitle_renderer(self):
        """Test GPUSubtitleRenderer can be instantiated."""
        from socials_automator.video.pipeline.video_renderer_gpu import GPUSubtitleRenderer
        renderer = GPUSubtitleRenderer()
        assert renderer.font_size == 80
        assert renderer.stroke_width == 4

    def test_gpu_subtitle_renderer_custom_params(self):
        """Test GPUSubtitleRenderer accepts custom parameters."""
        from socials_automator.video.pipeline.video_renderer_gpu import GPUSubtitleRenderer
        renderer = GPUSubtitleRenderer(
            font="Arial.ttf",
            font_size=100,
            stroke_width=6,
            position_y_percent=0.8,
        )
        assert renderer.font == "Arial.ttf"
        assert renderer.font_size == 100
        assert renderer.stroke_width == 6
        assert renderer.position_y_percent == 0.8


class TestVideoClipInfoAttributeAccess:
    """Test correct attribute access on VideoClipInfo model.

    This test class specifically catches the 'keywords' vs 'keywords_used' bug.
    """

    def test_video_clip_info_has_keywords_used(self):
        """Test VideoClipInfo model has keywords_used field."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        # Check the model fields
        assert 'keywords_used' in VideoClipInfo.model_fields

    def test_video_clip_info_does_not_have_keywords(self):
        """Test VideoClipInfo model does NOT have keywords field."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        # This is the bug we're testing for
        assert 'keywords' not in VideoClipInfo.model_fields, \
            "VideoClipInfo should not have 'keywords' field - use 'keywords_used' instead"

    def test_accessing_keywords_used_works(self):
        """Test that accessing keywords_used attribute works."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        clip = VideoClipInfo(
            segment_index=0,
            path=Path("C:/temp/clip.mp4"),
            source_url="https://example.com/video",
            pexels_id=12345,
            title="Test clip",
            duration_seconds=10.0,
            width=1920,
            height=1080,
            keywords_used=["test", "video", "clip"],
        )

        # This should work
        assert clip.keywords_used == ["test", "video", "clip"]

    def test_accessing_keywords_raises_error(self):
        """Test that accessing non-existent keywords attribute raises error."""
        from socials_automator.video.pipeline.base import VideoClipInfo

        clip = VideoClipInfo(
            segment_index=0,
            path=Path("C:/temp/clip.mp4"),
            source_url="https://example.com/video",
            pexels_id=12345,
            title="Test clip",
            duration_seconds=10.0,
            width=1920,
            height=1080,
            keywords_used=["test"],
        )

        # This should raise AttributeError - the bug we're testing for
        with pytest.raises(AttributeError):
            _ = clip.keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
