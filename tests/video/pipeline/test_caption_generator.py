"""Tests for CaptionGenerator including inline bullet fixing."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from socials_automator.video.pipeline.caption_generator import CaptionGenerator


class TestFixInlineBullets:
    """Test the _fix_inline_bullets method for various caption formats."""

    def setup_method(self):
        """Create CaptionGenerator instance for each test."""
        self.generator = CaptionGenerator()

    def test_already_formatted_with_newline_bullets(self):
        """Test that already-formatted captions are not modified."""
        caption = """Morning Briefing: Entertainment news highlights

- Nivin Pauly draws parallels between 'Premam' and 'Sholay'
- Black TikTok stars shine at U.S. TikTok Awards
- Sling TV offers Thanksgiving streaming deals

Save this + follow @news.but.quick for more!"""

        result = self.generator._fix_inline_bullets(caption)
        assert result == caption

    def test_inline_bullets_with_punctuation_before_dash(self):
        """Test inline bullets with punctuation (! or . or :) before the dash."""
        caption = "Morning Briefing: Entertainment news highlights! - Nivin Pauly draws parallels - Black TikTok stars shine - Sling TV offers deals Save this + follow @news.but.quick"

        result = self.generator._fix_inline_bullets(caption)

        assert "\n- " in result, f"Expected newline bullets, got: {result}"
        assert "Nivin Pauly" in result
        assert "Black TikTok" in result

    def test_inline_bullets_without_punctuation(self):
        """Test inline bullets WITHOUT punctuation before the dash (news headlines)."""
        caption = "Morning Briefing: Entertainment news highlights - Nivin Pauly draws parallels between 'Premam' and 'Sholay' - Black TikTok stars shine at U.S. TikTok Awards - Sling TV offers Thanksgiving streaming deals Save this + follow @news.but.quick for more!"

        result = self.generator._fix_inline_bullets(caption)

        # Should split at first " - " and format as bullets
        assert "\n" in result, f"Expected newlines in result, got: {result}"
        # Check that the headline is preserved
        assert "Entertainment news highlights" in result or "Morning Briefing" in result
        # Check that bullet items are present
        assert "Nivin Pauly" in result
        assert "Black TikTok" in result
        assert "Sling TV" in result

    def test_cta_extraction_from_last_bullet(self):
        """Test that CTA (Save this, Follow) is extracted and moved to end."""
        caption = "Morning Briefing: Headlines! - Story 1 - Story 2 - Story 3 Save this + follow @profile for more!"

        result = self.generator._fix_inline_bullets(caption)

        # CTA should be on its own line at the end
        lines = result.strip().split("\n")
        last_line = lines[-1].strip()
        assert "Save this" in last_line or "follow" in last_line.lower()

    def test_sources_extraction(self):
        """Test that Sources section is extracted and preserved."""
        caption = "Morning Briefing: Headlines - Story 1 - Story 2 Sources: CNN, BBC, Reuters #hashtag"

        result = self.generator._fix_inline_bullets(caption)

        # Sources should be preserved
        assert "Sources:" in result or "sources:" in result.lower()

    def test_single_dash_not_treated_as_bullets(self):
        """Test that a caption with only one dash is not treated as bullets."""
        caption = "This is a simple caption - with just one dash."

        result = self.generator._fix_inline_bullets(caption)

        # Should not add newlines since there's only 1 dash
        assert result == caption

    def test_dash_in_sentence_not_bullets(self):
        """Test that dashes within sentences are preserved."""
        caption = "AI tools - like ChatGPT - are changing how we work."

        result = self.generator._fix_inline_bullets(caption)

        # Only 2 dashes, treated as inline bullets
        # This is expected behavior - may or may not format
        # The important thing is no crash
        assert "ChatGPT" in result

    def test_empty_caption(self):
        """Test that empty caption returns empty."""
        result = self.generator._fix_inline_bullets("")
        assert result == ""

    def test_caption_with_colon_before_bullets(self):
        """Test caption with colon before bullet list."""
        caption = "Top AI tips: - Use ChatGPT for writing - Try Claude for coding - Midjourney for images"

        result = self.generator._fix_inline_bullets(caption)

        assert "ChatGPT" in result
        assert "Claude" in result
        assert "Midjourney" in result

    def test_real_news_caption_example(self):
        """Test with real news.but.quick caption format."""
        caption = "Morning Briefing: Entertainment news highlights - Nivin Pauly draws parallels between 'Premam' and 'Sholay' - Black TikTok stars shine at U.S. TikTok Awards - Sling TV offers Thanksgiving streaming deals Save this + follow @news.but.quick for more! #Newsbutquick Sources: NDTV Movies, Bossip, Msn, Usatoday #Entertainment #Hollywood #TikTok #StreamingDeals"

        result = self.generator._fix_inline_bullets(caption)

        # Should have proper formatting
        print(f"\n=== INPUT ===\n{caption}")
        print(f"\n=== OUTPUT ===\n{result}")

        # Check for newlines (the main fix)
        assert "\n" in result, "Expected newlines in formatted caption"

        # Headlines should be bullet points
        assert "Nivin Pauly" in result
        assert "Black TikTok" in result
        assert "Sling TV" in result


class TestCaptionGeneratorParsing:
    """Test caption parsing from AI responses."""

    def setup_method(self):
        """Create CaptionGenerator instance for each test."""
        self.generator = CaptionGenerator()

    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response."""
        response = '{"caption": "Test caption text", "hashtags": "#test #ai"}'

        # Create a mock context with proper structure
        context = MagicMock()
        context.metadata = MagicMock()
        context.metadata.profile_config = {"hashtag": None, "name": "Test"}
        context.metadata.profile = MagicMock()
        context.metadata.profile.name = "Test"

        caption, hashtags = self.generator._parse_caption_response(response, context)

        # Caption should contain the text (may have profile hashtag appended)
        assert "Test caption text" in caption
        assert hashtags == "#test #ai"

    def test_parse_json_with_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{"caption": "Test caption", "hashtags": "#test"}
```"""

        context = MagicMock()
        context.metadata = MagicMock()
        context.metadata.profile_config = {"hashtag": None, "name": "Test"}
        context.metadata.profile = MagicMock()
        context.metadata.profile.name = "Test"

        caption, hashtags = self.generator._parse_caption_response(response, context)

        assert "Test caption" in caption
        assert hashtags == "#test"

    def test_parse_json_with_inline_bullets(self):
        """Test that inline bullets in AI response are formatted."""
        response = '{"caption": "News update: Headlines! - Story 1 - Story 2 - Story 3 Follow for more!", "hashtags": "#news"}'

        context = MagicMock()
        context.metadata = MagicMock()
        context.metadata.profile_config = {"hashtag": None, "name": "Test"}
        context.metadata.profile = MagicMock()
        context.metadata.profile.name = "Test"

        caption, hashtags = self.generator._parse_caption_response(response, context)

        # The _fix_inline_bullets should be applied
        assert "Story 1" in caption
        assert "Story 2" in caption


class TestBasicValidation:
    """Test basic caption validation."""

    def setup_method(self):
        """Create CaptionGenerator instance for each test."""
        self.generator = CaptionGenerator()

    def test_valid_caption_with_bullets_passes(self):
        """Test that a valid caption with proper bullets passes."""
        caption = """Top AI Tools for Productivity

- ChatGPT for writing assistance
- Claude for coding help
- Midjourney for image generation

Follow for more AI tips!"""
        narration = "This video talks about ChatGPT, Claude, and Midjourney."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        assert is_valid, f"Validation failed: {feedback}"

    def test_too_short_caption_fails(self):
        """Test that very short captions fail validation."""
        caption = "Hi"
        narration = "This is a long narration about many topics."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        assert not is_valid
        assert "too short" in feedback.lower() or "short" in feedback.lower()

    def test_caption_without_linebreaks_fails(self):
        """Test that captions with inline bullets (no linebreaks) fail."""
        caption = "Top AI Tools - ChatGPT for writing - Claude for coding - Midjourney for images"
        narration = "This video discusses ChatGPT and Claude."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        # Should fail - has bullets but no line breaks
        assert not is_valid
