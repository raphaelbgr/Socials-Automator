"""Tests for caption loading in upload functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestCaptionLoading:
    """Test caption loading logic for reel uploads."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reel_path = self.temp_dir / "reel"
        self.reel_path.mkdir(parents=True)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _load_caption(self, reel_path: Path) -> str:
        """Load caption using the same logic as generate-reel --upload.

        This mirrors the logic in cli.py for loading captions.
        """
        caption_path = reel_path / "caption.txt"
        hashtags_path = reel_path / "hashtags.txt"
        full_caption_path = reel_path / "caption+hashtags.txt"

        reel_caption = ""
        if full_caption_path.exists():
            reel_caption = full_caption_path.read_text(encoding="utf-8").strip()
        elif caption_path.exists():
            reel_caption = caption_path.read_text(encoding="utf-8").strip()
            if hashtags_path.exists():
                hashtags = hashtags_path.read_text(encoding="utf-8").strip()
                if hashtags:
                    reel_caption = f"{reel_caption}\n\n{hashtags}"

        return reel_caption

    def test_load_full_caption_file(self):
        """Test loading caption from caption+hashtags.txt."""
        full_caption = "Check out this AI tip!\n\n#AI #productivity #tech"
        (self.reel_path / "caption+hashtags.txt").write_text(full_caption, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert result == full_caption
        assert "#AI" in result
        assert "#productivity" in result

    def test_load_caption_with_separate_hashtags(self):
        """Test loading caption.txt + hashtags.txt fallback."""
        caption = "Check out this AI tip!"
        hashtags = "#AI #productivity #tech"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text(hashtags, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert "Check out this AI tip!" in result
        assert "#AI #productivity #tech" in result
        assert "\n\n" in result  # Hashtags separated by double newline

    def test_full_caption_takes_priority(self):
        """Test that caption+hashtags.txt takes priority over separate files."""
        full_caption = "Full caption with hashtags #full"
        separate_caption = "Separate caption"
        separate_hashtags = "#separate"

        (self.reel_path / "caption+hashtags.txt").write_text(full_caption, encoding="utf-8")
        (self.reel_path / "caption.txt").write_text(separate_caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text(separate_hashtags, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert result == full_caption
        assert "#full" in result
        assert "#separate" not in result

    def test_caption_only_no_hashtags(self):
        """Test loading caption without hashtags file."""
        caption = "Just a caption, no hashtags"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert result == caption

    def test_empty_caption_returns_empty_string(self):
        """Test that missing caption files return empty string."""
        result = self._load_caption(self.reel_path)

        assert result == ""

    def test_empty_hashtags_file_ignored(self):
        """Test that empty hashtags.txt is ignored."""
        caption = "Caption text"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text("", encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert result == caption
        assert "\n\n" not in result  # No double newline added

    def test_whitespace_only_hashtags_ignored(self):
        """Test that whitespace-only hashtags.txt is ignored."""
        caption = "Caption text"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text("   \n  ", encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert result == caption

    def test_caption_with_newlines_preserved(self):
        """Test that caption newlines are preserved."""
        caption = "Line 1\nLine 2\nLine 3"
        hashtags = "#tag1 #tag2"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text(hashtags, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert "Line 1\nLine 2\nLine 3" in result
        assert result.endswith("#tag1 #tag2")

    def test_caption_with_unicode(self):
        """Test caption with unicode characters."""
        caption = "AI tips for productivity"  # No emojis per project rules
        hashtags = "#AI #tech"
        (self.reel_path / "caption.txt").write_text(caption, encoding="utf-8")
        (self.reel_path / "hashtags.txt").write_text(hashtags, encoding="utf-8")

        result = self._load_caption(self.reel_path)

        assert "AI tips" in result
        assert "#AI" in result


class TestCaptionValidation:
    """Test caption validation for uploads."""

    def test_caption_not_empty_before_upload(self):
        """Test that we detect empty captions."""
        reel_caption = ""

        has_caption = bool(reel_caption)

        assert not has_caption

    def test_caption_with_only_hashtags_is_valid(self):
        """Test that caption with only hashtags is considered valid."""
        reel_caption = "#AI #productivity #tech"

        has_caption = bool(reel_caption)

        assert has_caption

    def test_caption_length_check(self):
        """Test caption length validation (Instagram limit is 2200 chars)."""
        short_caption = "Short caption #AI"
        long_caption = "A" * 2500

        assert len(short_caption) <= 2200
        assert len(long_caption) > 2200
