"""Unit tests for ContentOrchestrator.

Tests the orchestrator's coordination of components without hitting real APIs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from socials_automator.content.orchestrator import ContentOrchestrator
from socials_automator.content.models import (
    CarouselPost,
    SlideContent,
    SlideType,
    HookType,
    PostPlan,
)


class TestContentOrchestrator:
    """Tests for ContentOrchestrator."""

    @pytest.fixture
    def mock_planner(self) -> AsyncMock:
        """Create a mock ContentPlanner."""
        planner = AsyncMock()

        # Create a realistic PostPlan
        plan = PostPlan(
            topic="5 AI Tools",
            content_pillar="tools",
            hook_type=HookType.NUMBER_BENEFIT,
            hook_text="5 AI Tools You NEED",
            hook_subtext="Save this!",
            target_slides=5,
            slides=[
                {
                    "slide_type": "hook",
                    "number": 1,
                    "heading": "5 AI Tools You NEED",
                    "body": "Save this!",
                    "needs_image": True,
                    "image_description": "Tech background",
                },
                {
                    "slide_type": "content",
                    "number": 2,
                    "heading": "Tool #1: ChatGPT",
                    "body": "Best for writing",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 3,
                    "heading": "Tool #2: Claude",
                    "body": "Best for reasoning",
                    "needs_image": False,
                },
                {
                    "slide_type": "content",
                    "number": 4,
                    "heading": "Tool #3: Midjourney",
                    "body": "Best for images",
                    "needs_image": False,
                },
                {
                    "slide_type": "cta",
                    "number": 5,
                    "heading": "Follow for more!",
                    "body": None,
                    "needs_image": True,
                },
            ],
            keywords=["AI", "tools", "productivity"],
        )

        planner.plan_post.return_value = plan
        planner.generate_caption.return_value = "Check out these amazing AI tools! #AI #Tools"

        return planner

    @pytest.fixture
    def orchestrator(
        self,
        temp_profile_dir: Path,
        sample_profile_config: dict[str, Any],
        mock_text_provider: AsyncMock,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        mock_planner: AsyncMock,
    ) -> ContentOrchestrator:
        """Create a ContentOrchestrator with all mocks."""
        return ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=sample_profile_config,
            text_provider=mock_text_provider,
            image_provider=mock_image_provider,
            composer=mock_composer,
            planner=mock_planner,
        )

    def test_generate_post_id_format(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that post ID has correct format."""
        post_id = orchestrator._generate_post_id()

        # Should be YYYYMMDD-NNN format
        parts = post_id.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 3  # NNN

    def test_create_slug(self, orchestrator: ContentOrchestrator):
        """Test slug creation from topic."""
        slug = orchestrator._create_slug("5 ChatGPT Tricks for Email!")

        assert slug == "5-chatgpt-tricks-for-email"
        assert " " not in slug
        assert "!" not in slug

    def test_create_slug_truncates_long_topics(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that slug is truncated for very long topics."""
        long_topic = "This is a very long topic that should be truncated because it exceeds fifty characters"
        slug = orchestrator._create_slug(long_topic)

        assert len(slug) <= 50

    @pytest.mark.asyncio
    async def test_generate_post_returns_carousel_post(
        self,
        orchestrator: ContentOrchestrator,
        mock_planner: AsyncMock,
    ):
        """Test that generate_post returns a CarouselPost."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        assert isinstance(post, CarouselPost)
        assert post.topic == "5 AI Tools"
        assert post.content_pillar == "tools"

    @pytest.mark.asyncio
    async def test_generate_post_calls_planner(
        self,
        orchestrator: ContentOrchestrator,
        mock_planner: AsyncMock,
    ):
        """Test that generate_post calls the planner."""
        await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        mock_planner.plan_post.assert_called_once()
        mock_planner.generate_caption.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_post_creates_correct_slides(
        self,
        orchestrator: ContentOrchestrator,
        mock_planner: AsyncMock,
    ):
        """Test that generate_post creates slides for each outline."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        # Should have 5 slides (1 hook + 3 content + 1 cta)
        assert len(post.slides) == 5

        # First slide should be hook
        assert post.slides[0].slide_type == SlideType.HOOK

        # Middle slides should be content
        assert post.slides[1].slide_type == SlideType.CONTENT
        assert post.slides[2].slide_type == SlideType.CONTENT
        assert post.slides[3].slide_type == SlideType.CONTENT

        # Last slide should be CTA
        assert post.slides[4].slide_type == SlideType.CTA

    @pytest.mark.asyncio
    async def test_generate_post_enables_cta_image(
        self,
        orchestrator: ContentOrchestrator,
        mock_planner: AsyncMock,
    ):
        """Test that CTA image is enabled when configured."""
        # The fixture config has cta_image.enabled = True
        await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        # Check that CTA outline was modified
        plan = mock_planner.plan_post.return_value
        cta_outline = next(s for s in plan.slides if s["slide_type"] == "cta")
        assert cta_outline["needs_image"] is True

    @pytest.mark.asyncio
    async def test_generate_post_sets_timing(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that generation_time_seconds is set."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        assert post.generation_time_seconds is not None
        assert post.generation_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_generate_post_sets_providers(
        self,
        orchestrator: ContentOrchestrator,
        mock_text_provider: AsyncMock,
        mock_image_provider: AsyncMock,
    ):
        """Test that providers are recorded on post."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        assert post.text_provider == "mock"
        assert post.image_provider == "mock"

    @pytest.mark.asyncio
    async def test_generate_post_sets_hashtags(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that hashtags are configured from profile."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        # Should have hashtags from profile config
        assert len(post.hashtags) > 0
        assert "#AI" in post.hashtags or "AI" in str(post.hashtags)

    @pytest.mark.asyncio
    async def test_generate_post_passes_research_context(
        self,
        orchestrator: ContentOrchestrator,
        mock_planner: AsyncMock,
    ):
        """Test that research_context is passed to planner."""
        research = "Some research data about AI tools"

        await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
            research_context=research,
        )

        # Check planner received research context
        call_args = mock_planner.plan_post.call_args
        assert call_args.kwargs.get("research_context") == research

    @pytest.mark.asyncio
    async def test_save_post_uses_output_service(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that save_post delegates to OutputService."""
        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        path = await orchestrator.save_post(post)

        assert path.exists()
        assert (path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_generate_post_with_progress_callback(
        self,
        temp_profile_dir: Path,
        sample_profile_config: dict[str, Any],
        mock_text_provider: AsyncMock,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        mock_planner: AsyncMock,
        mock_progress_callback: AsyncMock,
    ):
        """Test that progress callback receives updates."""
        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=sample_profile_config,
            text_provider=mock_text_provider,
            image_provider=mock_image_provider,
            composer=mock_composer,
            planner=mock_planner,
            progress_callback=mock_progress_callback,
        )

        await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        # Progress callback should have been called multiple times
        assert mock_progress_callback.call_count > 0


class TestConfigure:
    """Tests for configuration helper methods."""

    @pytest.fixture
    def orchestrator(
        self,
        temp_profile_dir: Path,
        sample_profile_config: dict[str, Any],
        mock_text_provider: AsyncMock,
        mock_image_provider: AsyncMock,
    ) -> ContentOrchestrator:
        """Create a minimal orchestrator for config tests."""
        return ContentOrchestrator(
            profile_path=temp_profile_dir,
            profile_config=sample_profile_config,
            text_provider=mock_text_provider,
            image_provider=mock_image_provider,
        )

    def test_configure_cta_image_sets_needs_image(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that _configure_cta_image sets needs_image on CTA."""
        plan = MagicMock()
        plan.slides = [
            {"slide_type": "hook", "needs_image": True},
            {"slide_type": "content", "needs_image": False},
            {"slide_type": "cta", "needs_image": False},
        ]

        orchestrator._configure_cta_image(plan, "Test Topic")

        cta_slide = plan.slides[2]
        assert cta_slide["needs_image"] is True
        assert "image_description" in cta_slide

    def test_configure_cta_image_includes_topic(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that CTA image description includes the topic."""
        plan = MagicMock()
        plan.slides = [{"slide_type": "cta", "needs_image": False}]

        orchestrator._configure_cta_image(plan, "Amazing AI Tools")

        cta_slide = plan.slides[0]
        assert "Amazing AI Tools" in cta_slide["image_description"]

    def test_configure_hashtags_from_profile(
        self,
        orchestrator: ContentOrchestrator,
    ):
        """Test that _configure_hashtags pulls from profile config."""
        post = MagicMock()
        post.hashtags = []

        orchestrator._configure_hashtags(post)

        # Should set hashtags from config
        assert len(post.hashtags) > 0
        # Should limit to 10
        assert len(post.hashtags) <= 10
