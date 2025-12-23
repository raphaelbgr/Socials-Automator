"""Tests for caption generation with actual AI providers.

These tests require LMStudio to be running on localhost:1234.
Run with: pytest tests/ai_test/test_caption_generation.py -v
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from socials_automator.video.pipeline.caption_generator import CaptionGenerator
from socials_automator.services.llm_fallback import LLMFallbackManager


class TestCaptionGeneratorUnit:
    """Unit tests for CaptionGenerator (no AI required)."""

    def setup_method(self):
        """Create CaptionGenerator instance."""
        self.generator = CaptionGenerator()

    def test_fix_inline_bullets_news_format(self):
        """Test fixing inline bullets in news-style caption."""
        caption = (
            "Morning Briefing: Entertainment news highlights - "
            "Stranger Things Season 5 hits Netflix Dec 26 - "
            "Chiranjeevi and Mohanlal star in new film - "
            "Rihanna teams up with GloRilla "
            "Save this + follow @news.but.quick for more!"
        )

        result = self.generator._fix_inline_bullets(caption)

        # Should have proper newlines
        assert "\n" in result, f"No newlines in: {result}"
        assert "Stranger Things" in result
        assert "Chiranjeevi" in result
        assert "Rihanna" in result

    def test_fix_inline_bullets_with_sources(self):
        """Test that Sources section is properly extracted."""
        caption = (
            "News Update: Headlines - Story 1 - Story 2 - Story 3 "
            "Sources: CNN, BBC, Reuters #news #headlines"
        )

        result = self.generator._fix_inline_bullets(caption)

        # Sources should be on its own line
        assert "Sources:" in result
        # Should still have bullet points
        assert "Story 1" in result

    def test_fix_inline_bullets_already_formatted(self):
        """Test that already-formatted captions are not modified."""
        caption = """Morning Briefing: Headlines

- Story 1 about important topic
- Story 2 about another topic
- Story 3 about final topic

Follow for more! #news"""

        result = self.generator._fix_inline_bullets(caption)

        # Should be unchanged
        assert result == caption

    def test_parse_caption_response_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"caption": "Test caption text", "hashtags": "#test #ai"}'

        context = MagicMock()
        context.metadata = MagicMock()
        context.metadata.profile_config = {"hashtag": None, "name": "Test"}
        context.metadata.profile = MagicMock()
        context.metadata.profile.name = "Test"

        caption, hashtags = self.generator._parse_caption_response(response, context)

        assert "Test caption text" in caption
        assert hashtags == "#test #ai"

    def test_parse_caption_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = '''```json
{"caption": "Test caption", "hashtags": "#test"}
```'''

        context = MagicMock()
        context.metadata = MagicMock()
        context.metadata.profile_config = {"hashtag": None, "name": "Test"}
        context.metadata.profile = MagicMock()
        context.metadata.profile.name = "Test"

        caption, hashtags = self.generator._parse_caption_response(response, context)

        assert "Test caption" in caption


class TestCaptionBasicValidation:
    """Test basic caption validation rules."""

    def setup_method(self):
        """Create CaptionGenerator instance."""
        self.generator = CaptionGenerator()

    def test_valid_caption_with_bullets_passes(self):
        """Test that a valid caption with bullets passes validation."""
        caption = """Top AI Tools for Productivity

- ChatGPT for writing assistance
- Claude for coding help
- Midjourney for image generation

Follow for more AI tips!"""
        narration = "This video discusses ChatGPT, Claude, and Midjourney for productivity."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        # Should pass - has bullets on separate lines
        assert is_valid, f"Validation failed: {feedback}"

    def test_empty_caption_fails(self):
        """Test that empty caption fails validation."""
        caption = ""
        narration = "This is the narration text."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        assert not is_valid

    def test_very_short_caption_fails(self):
        """Test that very short captions fail validation."""
        caption = "Hi"
        narration = "This is a long narration about many topics."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        assert not is_valid

    def test_caption_without_linebreaks_fails(self):
        """Test that caption with inline bullets (no linebreaks) fails."""
        caption = "Top AI Tools - ChatGPT for writing - Claude for coding - Midjourney for images"
        narration = "This video discusses ChatGPT and Claude."

        is_valid, feedback = self.generator._basic_validation(caption, narration)

        # Should fail - has bullets but no line breaks
        assert not is_valid
        assert "line" in feedback.lower() or "separate" in feedback.lower()


@pytest.mark.asyncio
class TestCaptionGeneratorWithAI:
    """Integration tests requiring LMStudio to be running."""

    @pytest.fixture(autouse=True)
    def check_lmstudio(self, lmstudio_available):
        """Skip tests if LMStudio is not available."""
        if not lmstudio_available:
            pytest.skip("LMStudio not available at localhost:1234")

    async def test_caption_generation_with_lmstudio(self, mock_context):
        """Test full caption generation with LMStudio."""
        generator = CaptionGenerator(preferred_provider="lmstudio")

        # Execute caption generation
        result_context = await generator.execute(mock_context)

        # Check caption was saved
        caption_path = result_context.output_dir / "caption.txt"
        assert caption_path.exists(), "caption.txt not created"

        caption_text = caption_path.read_text(encoding="utf-8")
        assert len(caption_text) > 10, f"Caption too short: {caption_text}"

        # Check hashtags were saved
        hashtags_path = result_context.output_dir / "caption+hashtags.txt"
        assert hashtags_path.exists(), "caption+hashtags.txt not created"

    async def test_caption_has_proper_formatting(self, mock_context):
        """Test that generated caption has proper formatting."""
        generator = CaptionGenerator(preferred_provider="lmstudio")

        # Override script with news-style content
        mock_context.script.full_narration = (
            "Tonight's top stories. "
            "Stranger Things Season 5 releases next week. "
            "OpenAI announces new GPT model. "
            "Apple unveils new iPhone features. "
            "Follow AI For Mortals for more tech news!"
        )
        mock_context.topic.category = "technology"

        result_context = await generator.execute(mock_context)

        caption_path = result_context.output_dir / "caption.txt"
        caption_text = caption_path.read_text(encoding="utf-8")

        print(f"\n=== Generated Caption ===\n{caption_text}")

        # Basic checks
        assert len(caption_text) > 20, "Caption too short"


class TestImageOverlayTiming:
    """Test image overlay timing calculations."""

    def test_min_overlay_start_time_enforced(self):
        """Test that overlays don't start before MIN_OVERLAY_START_TIME."""
        from socials_automator.video.pipeline.image_overlay_planner import (
            MIN_OVERLAY_START_TIME,
            SEGMENT_START_PADDING,
        )

        # First overlay should start at least at MIN_OVERLAY_START_TIME
        assert MIN_OVERLAY_START_TIME >= 3.0, "MIN_OVERLAY_START_TIME should be >= 3s"
        assert SEGMENT_START_PADDING >= 0.3, "SEGMENT_START_PADDING should be >= 0.3s"

    def test_overlay_timing_boundaries(self):
        """Test overlay timing boundary calculations."""
        from socials_automator.video.pipeline.image_overlay_planner import (
            MIN_OVERLAY_START_TIME,
            CTA_BUFFER_TIME,
        )

        total_duration = 60.0
        hook_end = 3.0
        cta_start = 56.0

        # Calculate effective boundaries
        min_start = max(MIN_OVERLAY_START_TIME, hook_end)
        max_end = min(cta_start, total_duration - CTA_BUFFER_TIME)

        assert min_start >= 3.0, f"min_start should be >= 3.0, got {min_start}"
        assert max_end <= 56.0, f"max_end should be <= 56.0, got {max_end}"
        assert min_start < max_end, "min_start should be less than max_end"


class TestScriptPlannerCTA:
    """Test script planner CTA generation."""

    def test_script_planner_builds_empty_cta(self):
        """Test that script planner creates scripts with empty CTA field."""
        from socials_automator.video.pipeline.script_planner import ScriptPlanner
        from socials_automator.video.pipeline.base import VideoSegment

        planner = ScriptPlanner(target_duration=60)

        # Build full narration with empty CTA
        hook = "This is an attention-grabbing hook!"
        segments = [
            VideoSegment(index=1, text="First segment.", duration_seconds=10.0),
            VideoSegment(index=2, text="Second segment.", duration_seconds=10.0),
            VideoSegment(index=3, text="Final segment. Follow Test Profile for more!", duration_seconds=10.0),
        ]
        cta = ""  # Empty CTA - now in last segment

        full_narration = planner._build_full_narration(hook, segments, cta)

        # Should not have duplicate "Follow"
        follow_count = full_narration.lower().count("follow")
        assert follow_count == 1, f"Expected 1 'follow', got {follow_count}: {full_narration}"


class TestNewsOrchestrator:
    """Test news orchestrator components."""

    def test_caption_uses_fix_inline_bullets(self):
        """Verify caption generation flow uses _fix_inline_bullets."""
        # This is a structural test - verify the code path exists
        from socials_automator.video.pipeline.caption_generator import CaptionGenerator

        gen = CaptionGenerator()

        # The method should exist and be callable
        assert hasattr(gen, '_fix_inline_bullets')
        assert callable(gen._fix_inline_bullets)

        # Test with news-style input
        news_caption = "Breaking: Headlines - Story 1 - Story 2 - Story 3"
        result = gen._fix_inline_bullets(news_caption)

        # Should add newlines
        assert "\n" in result
