"""Integration tests for the complete generation flow.

Tests the end-to-end workflow of generating carousel posts,
with mocked external APIs but real internal components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from socials_automator.content.orchestrator import ContentOrchestrator
from socials_automator.content.models import SlideType, HookType


class TestFullGenerationFlow:
    """Integration tests for complete post generation."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock text and image providers."""
        text_provider = AsyncMock()
        text_provider.generate.return_value = "Generated text response"
        text_provider.current_provider = "mock-text"

        image_provider = AsyncMock()
        # Return minimal valid JPEG
        image_provider.generate.return_value = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
        ])
        image_provider.current_provider = "mock-image"

        return text_provider, image_provider

    @pytest.fixture
    def integration_profile_config(self) -> dict[str, Any]:
        """Create a comprehensive profile config for integration tests."""
        return {
            "profile": {
                "id": "integration-test",
                "instagram_handle": "@integration.test",
                "niche_id": "ai-tools",
            },
            "content_strategy": {
                "content_pillars": [
                    {"id": "tools", "name": "AI Tools", "description": "AI tool reviews"},
                    {"id": "tips", "name": "Tips", "description": "Productivity tips"},
                ],
                "carousel_settings": {
                    "min_slides": 3,
                    "max_slides": 10,
                },
            },
            "design": {
                "image_generation": {
                    "style_prompt_suffix": "minimal, clean, modern",
                },
                "cta_image": {
                    "enabled": True,
                },
            },
            "hashtag_strategy": {
                "hashtag_sets": {
                    "primary": ["#AI", "#Tech", "#Automation"],
                    "secondary": ["#Productivity", "#Tools"],
                    "niche": ["#ChatGPT", "#Claude"],
                    "branded": ["#IntegrationTest"],
                },
            },
            "output_settings": {
                "folder_structure": "posts/{year}/{month}/{status}/{day}-{post_number}-{slug}",
                "file_naming": {
                    "slides": "slide_{number:02d}.jpg",
                    "caption": "caption.txt",
                    "hashtags": "hashtags.txt",
                    "combined": "caption+hashtags.txt",
                    "alt_texts": "alt_texts.json",
                    "metadata": "metadata.json",
                },
            },
            "ai_config": {
                "prompts": {
                    "system_context": "You are a social media content creator.",
                },
            },
        }

    @pytest.fixture
    def mock_planner(self) -> AsyncMock:
        """Create a realistic mock planner."""
        from socials_automator.content.models import PostPlan, HookType

        planner = AsyncMock()

        plan = PostPlan(
            topic="5 AI Tools for Productivity",
            content_pillar="tools",
            hook_type=HookType.NUMBER_BENEFIT,
            hook_text="5 AI Tools That Will 10x Your Productivity",
            hook_subtext="Save this for later!",
            target_slides=7,
            slides=[
                {
                    "slide_type": "hook",
                    "number": 1,
                    "heading": "5 AI Tools That Will 10x Your Productivity",
                    "body": "Save this for later!",
                    "needs_image": True,
                    "image_description": "Modern tech workspace with AI elements",
                },
                {
                    "slide_type": "content",
                    "number": 2,
                    "heading": "1. ChatGPT",
                    "body": "Best for writing, brainstorming, and code assistance. Use the GPT-4 model for best results.",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 3,
                    "heading": "2. Claude",
                    "body": "Excellent for analysis, reasoning, and long-form content. Great at following complex instructions.",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 4,
                    "heading": "3. Perplexity",
                    "body": "Perfect for research with citations. Always provides sources for its answers.",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 5,
                    "heading": "4. Midjourney",
                    "body": "Create stunning images from text prompts. V6 model produces photorealistic results.",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 6,
                    "heading": "5. Notion AI",
                    "body": "Built into Notion for seamless workflow. Great for summarizing notes and generating content.",
                    "needs_image": False,
                },
                {
                    "slide_type": "cta",
                    "number": 7,
                    "heading": "Follow for more AI tips!",
                    "body": "Save & share this post",
                    "needs_image": True,
                },
            ],
            keywords=["AI", "productivity", "tools", "ChatGPT", "automation"],
        )

        planner.plan_post.return_value = plan
        planner.generate_caption.return_value = (
            "Want to 10x your productivity? These 5 AI tools have changed how I work!\n\n"
            "Which one are you going to try first? Let me know in the comments!"
        )

        return planner

    @pytest.mark.asyncio
    async def test_full_generation_creates_valid_post(
        self,
        temp_profile_dir: Path,
        integration_profile_config: dict[str, Any],
        mock_providers: tuple[AsyncMock, AsyncMock],
        mock_planner: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test complete generation flow produces valid post."""
        text_provider, image_provider = mock_providers

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=integration_profile_config,
            text_provider=text_provider,
            image_provider=image_provider,
            composer=mock_composer,
            planner=mock_planner,
        )

        post = await orchestrator.generate_post(
            topic="5 AI Tools for Productivity",
            content_pillar="tools",
        )

        # Verify post structure
        assert post.id is not None
        assert post.topic == "5 AI Tools for Productivity"
        assert post.content_pillar == "tools"
        assert post.hook_type == HookType.NUMBER_BENEFIT
        assert post.status == "generated"

        # Verify slides
        assert len(post.slides) == 7
        assert post.slides[0].slide_type == SlideType.HOOK
        assert post.slides[-1].slide_type == SlideType.CTA

        # Verify caption
        assert len(post.caption) > 0
        assert "productivity" in post.caption.lower()

        # Verify hashtags
        assert len(post.hashtags) > 0
        assert len(post.hashtags) <= 10

    @pytest.mark.asyncio
    async def test_full_generation_and_save(
        self,
        temp_profile_dir: Path,
        integration_profile_config: dict[str, Any],
        mock_providers: tuple[AsyncMock, AsyncMock],
        mock_planner: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test generation and saving produces complete output."""
        text_provider, image_provider = mock_providers

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=integration_profile_config,
            text_provider=text_provider,
            image_provider=image_provider,
            composer=mock_composer,
            planner=mock_planner,
        )

        post = await orchestrator.generate_post(
            topic="5 AI Tools for Productivity",
            content_pillar="tools",
        )

        output_path = await orchestrator.save_post(post)

        # Verify output directory
        assert output_path.exists()

        # Verify all expected files exist
        assert (output_path / "slide_01.jpg").exists()
        assert (output_path / "slide_07.jpg").exists()
        assert (output_path / "caption.txt").exists()
        assert (output_path / "hashtags.txt").exists()
        assert (output_path / "caption+hashtags.txt").exists()
        assert (output_path / "alt_texts.json").exists()
        assert (output_path / "metadata.json").exists()

        # Verify metadata content
        with open(output_path / "metadata.json") as f:
            metadata = json.load(f)

        assert metadata["post"]["topic"] == "5 AI Tools for Productivity"
        assert metadata["post"]["slides_count"] == 7
        assert len(metadata["content"]["slides"]) == 7

    @pytest.mark.asyncio
    async def test_generation_with_research_context(
        self,
        temp_profile_dir: Path,
        integration_profile_config: dict[str, Any],
        mock_providers: tuple[AsyncMock, AsyncMock],
        mock_planner: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test generation with pre-provided research context."""
        text_provider, image_provider = mock_providers

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=integration_profile_config,
            text_provider=text_provider,
            image_provider=image_provider,
            composer=mock_composer,
            planner=mock_planner,
        )

        research_context = """
        Recent studies show that AI tools can increase productivity by up to 40%.
        ChatGPT has over 100 million users.
        Claude is known for its strong reasoning capabilities.
        """

        post = await orchestrator.generate_post(
            topic="5 AI Tools for Productivity",
            content_pillar="tools",
            research_context=research_context,
        )

        # Verify research was passed to planner
        call_args = mock_planner.plan_post.call_args
        assert call_args.kwargs.get("research_context") == research_context

        # Post should still be valid
        assert post.status == "generated"
        assert len(post.slides) == 7

    @pytest.mark.asyncio
    async def test_generation_slide_order(
        self,
        temp_profile_dir: Path,
        integration_profile_config: dict[str, Any],
        mock_providers: tuple[AsyncMock, AsyncMock],
        mock_planner: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test that slides are generated in correct order."""
        text_provider, image_provider = mock_providers

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=integration_profile_config,
            text_provider=text_provider,
            image_provider=image_provider,
            composer=mock_composer,
            planner=mock_planner,
        )

        post = await orchestrator.generate_post(
            topic="5 AI Tools for Productivity",
            content_pillar="tools",
        )

        # Verify slide order
        for i, slide in enumerate(post.slides):
            assert slide.number == i + 1

        # First should be hook
        assert post.slides[0].slide_type == SlideType.HOOK

        # Last should be CTA
        assert post.slides[-1].slide_type == SlideType.CTA

        # Middle should all be content
        for slide in post.slides[1:-1]:
            assert slide.slide_type == SlideType.CONTENT


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_image_generation_failure_continues(
        self,
        temp_profile_dir: Path,
        sample_profile_config: dict[str, Any],
        mock_composer: AsyncMock,
    ):
        """Test that image generation failure doesn't stop generation."""
        from socials_automator.content.models import PostPlan, HookType

        text_provider = AsyncMock()
        text_provider.generate.return_value = "Generated text"
        text_provider.current_provider = "mock"

        # Image provider that fails
        image_provider = AsyncMock()
        image_provider.generate.side_effect = Exception("Image API error")
        image_provider.current_provider = "mock"

        planner = AsyncMock()
        planner.plan_post.return_value = PostPlan(
            topic="Test",
            content_pillar="test",
            hook_type=HookType.NUMBER_BENEFIT,
            hook_text="Test Hook",
            slides=[
                {
                    "slide_type": "hook",
                    "number": 1,
                    "heading": "Test",
                    "needs_image": True,
                    "image_description": "Test image",
                },
                {
                    "slide_type": "cta",
                    "number": 2,
                    "heading": "Follow!",
                    "needs_image": True,
                },
            ],
        )
        planner.generate_caption.return_value = "Test caption"

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=sample_profile_config,
            text_provider=text_provider,
            image_provider=image_provider,
            composer=mock_composer,
            planner=planner,
        )

        # Should complete despite image failures
        post = await orchestrator.generate_post(
            topic="Test",
            content_pillar="test",
        )

        assert post.status == "generated"
        assert len(post.slides) == 2

        # Slides should not have background images due to failure
        for slide in post.slides:
            assert slide.has_background_image is False
