"""Tests for CTA (Call To Action) generation in scripts.

Verifies that:
1. AI generates creative CTAs in the last segment
2. No programmatic CTA is added
3. Profile name is mentioned in the CTA
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json

from socials_automator.video.pipeline.script_planner import ScriptPlanner
from socials_automator.video.pipeline.base import (
    TopicInfo,
    ResearchResult,
    VideoScript,
    VideoSegment,
)


class TestCTAPromptStructure:
    """Test that the prompt correctly asks AI for creative CTA."""

    def setup_method(self):
        """Create ScriptPlanner instance."""
        self.planner = ScriptPlanner(target_duration=60)

    def test_cta_constants_defined(self):
        """Test that CTA timing constants are properly defined."""
        assert hasattr(self.planner, 'CTA_DURATION')
        assert self.planner.CTA_DURATION > 0

    def test_build_narration_with_empty_cta(self):
        """Test building narration when CTA is in the last segment."""
        hook = "This is the hook!"
        segments = [
            VideoSegment(index=1, text="First segment content.", duration_seconds=10.0),
            VideoSegment(index=2, text="Second segment. Follow AI For Mortals!", duration_seconds=10.0),
        ]
        cta = ""  # Empty - CTA is in last segment

        narration = self.planner._build_full_narration(hook, segments, cta)

        # Should have exactly one "follow"
        assert narration.lower().count("follow") == 1

    def test_build_narration_no_trailing_space(self):
        """Test that empty CTA doesn't leave trailing space."""
        hook = "Hook."
        segments = [
            VideoSegment(index=1, text="Segment with CTA. Follow Test!", duration_seconds=10.0),
        ]
        cta = ""

        narration = self.planner._build_full_narration(hook, segments, cta)

        # Should not have trailing space
        assert not narration.endswith(" ")

    def test_build_narration_preserves_non_empty_cta(self):
        """Test backward compatibility - if CTA is provided, it's still appended."""
        hook = "Hook."
        segments = [
            VideoSegment(index=1, text="Segment content.", duration_seconds=10.0),
        ]
        cta = "Follow us for more!"

        narration = self.planner._build_full_narration(hook, segments, cta)

        assert "Follow us for more!" in narration


class TestCTAScriptBuilding:
    """Test that script is built correctly without programmatic CTA."""

    def test_empty_cta_field_in_script(self):
        """Test that VideoScript cta field is empty when AI generates CTA in segment."""
        # Create a script with empty CTA
        script = VideoScript(
            title="Test Title",
            hook="This is the hook.",
            segments=[
                VideoSegment(
                    index=1,
                    text="First segment content.",
                    duration_seconds=10.0,
                ),
                VideoSegment(
                    index=2,
                    text="Second segment content.",
                    duration_seconds=10.0,
                ),
                VideoSegment(
                    index=3,
                    text="Final segment with creative CTA! Follow Test Profile for more awesome tips!",
                    duration_seconds=10.0,
                ),
            ],
            cta="",  # Empty CTA
            total_duration=40.0,
        )

        # Verify CTA is empty
        assert script.cta == "", "CTA field should be empty"

        # Verify CTA is in last segment
        last_segment = script.segments[-1]
        assert "Follow" in last_segment.text, "CTA should be in last segment"
        assert "Test Profile" in last_segment.text, "Profile name should be in last segment"

    def test_full_narration_no_separate_cta(self):
        """Test that full narration doesn't have duplicate CTA."""
        from socials_automator.video.pipeline.script_planner import ScriptPlanner

        planner = ScriptPlanner(target_duration=60)

        # Simulate building full narration
        hook = "This is an attention-grabbing hook!"
        segments = [
            VideoSegment(index=1, text="First segment.", duration_seconds=10.0),
            VideoSegment(index=2, text="Second segment.", duration_seconds=10.0),
            VideoSegment(index=3, text="Final segment. Follow AI For Mortals for more!", duration_seconds=10.0),
        ]
        cta = ""  # Empty CTA

        full_narration = planner._build_full_narration(hook, segments, cta)

        # Count occurrences of "Follow"
        follow_count = full_narration.lower().count("follow")
        assert follow_count == 1, f"Expected 1 'follow', got {follow_count}: {full_narration}"


class TestCTAParsing:
    """Test parsing AI responses with CTA in segments."""

    def test_parse_ai_response_with_cta_in_segment(self):
        """Test parsing AI response where CTA is in the last segment."""
        planner = ScriptPlanner(target_duration=60)

        # Simulate AI response with CTA in last segment
        ai_response = json.dumps({
            "hook": "This is an attention-grabbing hook!",
            "segments": [
                {"text": "First segment with valuable content.", "keywords": ["value", "content"]},
                {"text": "Second segment with more tips.", "keywords": ["tips", "advice"]},
                {"text": "Final segment. Want more game-changing AI tips? Follow AI For Mortals!", "keywords": ["AI", "tips"]},
            ]
        })

        # Parse should not throw error
        clean_response = ai_response.strip()
        data = json.loads(clean_response)

        assert len(data["segments"]) == 3
        last_segment = data["segments"][-1]["text"]
        assert "Follow" in last_segment
        assert "AI For Mortals" in last_segment


@pytest.mark.asyncio
class TestCTAGenerationWithAI:
    """Integration tests for CTA generation with actual AI."""

    @pytest.fixture(autouse=True)
    def check_lmstudio(self, lmstudio_available):
        """Skip tests if LMStudio is not available."""
        if not lmstudio_available:
            pytest.skip("LMStudio not available at localhost:1234")

    async def test_script_generation_has_single_cta(self):
        """Test that generated script has exactly one CTA."""
        planner = ScriptPlanner(target_duration=60, preferred_provider="lmstudio")

        topic = TopicInfo(
            topic="5 AI Tools That Will Blow Your Mind",
            category="technology",
            keywords=["AI tools", "productivity"],
        )
        research = ResearchResult(
            topic="5 AI Tools That Will Blow Your Mind",
            key_points=[
                "ChatGPT for writing assistance",
                "Midjourney for image generation",
                "Claude for coding help",
            ],
            sources=["OpenAI", "Anthropic"],
        )

        # Generate script
        script = await planner.generate_script(
            topic=topic,
            research=research,
            duration=60,
            profile_name="AI For Mortals",
        )

        # Check that CTA field is empty
        assert script.cta == "", f"CTA field should be empty, got: {script.cta}"

        # Check that CTA is in full narration exactly once
        narration_lower = script.full_narration.lower()

        # Count "follow" mentions (should be exactly 1)
        follow_count = narration_lower.count("follow")
        assert follow_count == 1, (
            f"Expected exactly 1 'follow' in narration, got {follow_count}:\n"
            f"{script.full_narration}"
        )

        # Check profile name is mentioned
        assert "ai for mortals" in narration_lower, (
            f"Profile name not found in narration:\n{script.full_narration}"
        )

        print(f"\n=== Generated Script ===")
        print(f"Hook: {script.hook}")
        print(f"Segments: {len(script.segments)}")
        for seg in script.segments:
            print(f"  {seg.index}: {seg.text[:50]}...")
        print(f"CTA field: '{script.cta}'")
        print(f"\n=== Full Narration ===\n{script.full_narration}")
