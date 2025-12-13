"""Unit tests for SlideJob classes.

Tests each SlideJob (Hook, Content, CTA) independently using mocks
to verify they correctly:
- Generate images when needed
- Compose slides correctly
- Handle errors gracefully
- Return proper results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import from src since tests is outside the package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from socials_automator.content.slides import (
    SlideJob,
    SlideJobContext,
    SlideJobResult,
    SlideJobFactory,
    HookSlideJob,
    ContentSlideJob,
    CTASlideJob,
)
from socials_automator.content.models import SlideType, SlideContent


class TestSlideJobContext:
    """Tests for SlideJobContext data class."""

    def test_get_handle_returns_handle(self, sample_profile_config: dict[str, Any]):
        """Test that get_handle returns the Instagram handle."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="Test Topic",
            outline={},
            profile_config=sample_profile_config,
            design_config={},
        )

        assert context.get_handle() == "@test.account"

    def test_get_handle_returns_empty_when_missing(self):
        """Test that get_handle returns empty string when handle is missing."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="Test Topic",
            outline={},
            profile_config={},
            design_config={},
        )

        assert context.get_handle() == ""

    def test_get_image_style_returns_style(self, sample_design_config: dict[str, Any]):
        """Test that get_image_style returns the style suffix."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="Test Topic",
            outline={},
            profile_config={},
            design_config=sample_design_config,
        )

        assert "minimal, clean" in context.get_image_style()

    def test_get_image_style_returns_default_when_missing(self):
        """Test that get_image_style returns default when not configured."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="Test Topic",
            outline={},
            profile_config={},
            design_config={},
        )

        # Should return default style
        assert "tech aesthetic" in context.get_image_style()


class TestHookSlideJob:
    """Tests for HookSlideJob."""

    @pytest.fixture
    def hook_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ) -> HookSlideJob:
        """Create a HookSlideJob with mocks."""
        return HookSlideJob(
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

    def test_get_slide_type_returns_hook(self, hook_job: HookSlideJob):
        """Test that get_slide_type returns HOOK."""
        assert hook_job.get_slide_type() == SlideType.HOOK

    @pytest.mark.asyncio
    async def test_execute_with_image(
        self,
        hook_job: HookSlideJob,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test hook slide generation with background image."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="5 AI Tools",
            outline={
                "heading": "5 AI Tools You NEED",
                "body": "Save this!",
                "needs_image": True,
                "image_description": "Tech background",
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await hook_job.execute(context)

        # Verify result
        assert result.success
        assert result.slide_content.slide_type == SlideType.HOOK
        assert result.slide_content.heading == "5 AI Tools You NEED"
        assert result.slide_content.number == 1

        # Verify image was generated
        mock_image_provider.generate.assert_called_once()

        # Verify composer was called
        mock_composer.create_hook_slide.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_without_image(
        self,
        hook_job: HookSlideJob,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test hook slide generation without background image."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="5 AI Tools",
            outline={
                "heading": "5 AI Tools You NEED",
                "needs_image": False,  # No image needed
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await hook_job.execute(context)

        # Verify result
        assert result.success
        assert result.slide_content.has_background_image is False

        # Verify image was NOT generated
        mock_image_provider.generate.assert_not_called()

        # Verify composer was still called (with None image)
        mock_composer.create_hook_slide.assert_called_once()
        call_args = mock_composer.create_hook_slide.call_args
        assert call_args.kwargs.get("background_image") is None

    @pytest.mark.asyncio
    async def test_execute_image_failure_graceful(
        self,
        hook_job: HookSlideJob,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test graceful handling when image generation fails."""
        # Configure image provider to fail
        mock_image_provider.generate.side_effect = Exception("API error")

        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="5 AI Tools",
            outline={
                "heading": "5 AI Tools You NEED",
                "needs_image": True,
                "image_description": "Tech background",
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await hook_job.execute(context)

        # Should still succeed, just without image
        assert result.success
        assert result.slide_content.has_background_image is False

        # Composer should be called with None image
        call_args = mock_composer.create_hook_slide.call_args
        assert call_args.kwargs.get("background_image") is None

    @pytest.mark.asyncio
    async def test_execute_uses_topic_as_fallback_heading(
        self,
        hook_job: HookSlideJob,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test that topic is used as fallback when heading is missing."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="Amazing AI Tools",
            outline={
                "needs_image": False,
                # No heading provided
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await hook_job.execute(context)

        # Should use topic as heading
        assert result.slide_content.heading == "Amazing AI Tools"


class TestContentSlideJob:
    """Tests for ContentSlideJob."""

    @pytest.fixture
    def content_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ) -> ContentSlideJob:
        """Create a ContentSlideJob with mocks."""
        return ContentSlideJob(
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

    def test_get_slide_type_returns_content(self, content_job: ContentSlideJob):
        """Test that get_slide_type returns CONTENT."""
        assert content_job.get_slide_type() == SlideType.CONTENT

    @pytest.mark.asyncio
    async def test_execute_creates_content_slide(
        self,
        content_job: ContentSlideJob,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test content slide generation."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=3,
            topic="5 AI Tools",
            outline={
                "heading": "Tool #2: ChatGPT",
                "body": "Best for writing and brainstorming",
                "needs_image": False,
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await content_job.execute(context)

        # Verify result
        assert result.success
        assert result.slide_content.slide_type == SlideType.CONTENT
        assert result.slide_content.heading == "Tool #2: ChatGPT"
        assert result.slide_content.body == "Best for writing and brainstorming"
        assert result.slide_content.number == 3

        # Verify composer was called with correct number
        mock_composer.create_content_slide.assert_called_once()
        call_args = mock_composer.create_content_slide.call_args
        # Display number should be slide_number - 1 (exclude hook)
        assert call_args.kwargs.get("number") == 2

    @pytest.mark.asyncio
    async def test_execute_with_image(
        self,
        content_job: ContentSlideJob,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test content slide with background image."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=2,
            topic="5 AI Tools",
            outline={
                "heading": "Tool #1: Claude",
                "body": "Best for reasoning",
                "needs_image": True,
                "image_description": "AI assistant illustration",
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await content_job.execute(context)

        # Verify image was generated
        mock_image_provider.generate.assert_called_once()

        # Verify slide has image
        assert result.slide_content.has_background_image is True


class TestCTASlideJob:
    """Tests for CTASlideJob."""

    @pytest.fixture
    def cta_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ) -> CTASlideJob:
        """Create a CTASlideJob with mocks."""
        return CTASlideJob(
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

    def test_get_slide_type_returns_cta(self, cta_job: CTASlideJob):
        """Test that get_slide_type returns CTA."""
        assert cta_job.get_slide_type() == SlideType.CTA

    @pytest.mark.asyncio
    async def test_execute_creates_cta_slide(
        self,
        cta_job: CTASlideJob,
        mock_composer: AsyncMock,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test CTA slide generation."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=7,
            topic="5 AI Tools",
            outline={
                "heading": "Follow for more AI tips!",
                "body": "Save & share this post",
                "needs_image": True,
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await cta_job.execute(context)

        # Verify result
        assert result.success
        assert result.slide_content.slide_type == SlideType.CTA
        assert result.slide_content.heading == "Follow for more AI tips!"
        assert result.slide_content.number == 7

        # Verify handle is passed to composer
        mock_composer.create_cta_slide.assert_called_once()
        call_args = mock_composer.create_cta_slide.call_args
        assert call_args.kwargs.get("handle") == "@test.account"

    @pytest.mark.asyncio
    async def test_execute_uses_default_heading(
        self,
        cta_job: CTASlideJob,
        sample_profile_config: dict[str, Any],
        sample_design_config: dict[str, Any],
    ):
        """Test that default CTA heading is used when not provided."""
        context = SlideJobContext(
            post_id="test-001",
            slide_number=5,
            topic="AI Tools",
            outline={
                "needs_image": False,
                # No heading provided
            },
            profile_config=sample_profile_config,
            design_config=sample_design_config,
        )

        result = await cta_job.execute(context)

        # Should use default heading
        assert result.slide_content.heading == "Follow for more!"


class TestSlideJobFactory:
    """Tests for SlideJobFactory."""

    def test_create_hook_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test creating a hook slide job."""
        job = SlideJobFactory.create(
            slide_type=SlideType.HOOK,
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

        assert isinstance(job, HookSlideJob)
        assert job.get_slide_type() == SlideType.HOOK

    def test_create_content_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test creating a content slide job."""
        job = SlideJobFactory.create(
            slide_type=SlideType.CONTENT,
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

        assert isinstance(job, ContentSlideJob)
        assert job.get_slide_type() == SlideType.CONTENT

    def test_create_cta_job(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test creating a CTA slide job."""
        job = SlideJobFactory.create(
            slide_type=SlideType.CTA,
            image_provider=mock_image_provider,
            composer=mock_composer,
        )

        assert isinstance(job, CTASlideJob)
        assert job.get_slide_type() == SlideType.CTA

    def test_create_unknown_type_raises_error(
        self,
        mock_image_provider: AsyncMock,
        mock_composer: AsyncMock,
    ):
        """Test that creating an unknown type raises ValueError."""
        # Create a fake slide type
        class FakeSlideType:
            pass

        with pytest.raises(ValueError, match="No job registered"):
            SlideJobFactory.create(
                slide_type=FakeSlideType(),  # type: ignore
                image_provider=mock_image_provider,
                composer=mock_composer,
            )

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        types = SlideJobFactory.get_registered_types()

        assert SlideType.HOOK in types
        assert SlideType.CONTENT in types
        assert SlideType.CTA in types


class TestSlideJobResult:
    """Tests for SlideJobResult data class."""

    def test_success_when_has_bytes(self):
        """Test that success is True when image_bytes is present."""
        result = SlideJobResult(
            slide_content=MagicMock(spec=SlideContent),
            image_bytes=b"test_bytes",
        )

        assert result.success is True

    def test_not_success_when_empty_bytes(self):
        """Test that success is False when image_bytes is empty."""
        result = SlideJobResult(
            slide_content=MagicMock(spec=SlideContent),
            image_bytes=b"",
        )

        assert result.success is False

    def test_not_success_when_none_bytes(self):
        """Test that success is False when image_bytes is None."""
        result = SlideJobResult(
            slide_content=MagicMock(spec=SlideContent),
            image_bytes=None,  # type: ignore
        )

        assert result.success is False
