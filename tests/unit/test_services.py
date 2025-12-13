"""Unit tests for services module.

Tests the ProgressManager and OutputService classes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from socials_automator.services import ProgressManager, OutputService
from socials_automator.content.models import GenerationProgress


class TestProgressManager:
    """Tests for ProgressManager."""

    @pytest.fixture
    def progress_manager(self, mock_progress_callback: AsyncMock) -> ProgressManager:
        """Create a ProgressManager with mock callback."""
        return ProgressManager(
            post_id="test-001",
            callback=mock_progress_callback,
            total_steps=5,
        )

    def test_init_creates_progress(self, progress_manager: ProgressManager):
        """Test that initialization creates progress state."""
        assert progress_manager.progress is not None
        assert progress_manager.progress.post_id == "test-001"
        assert progress_manager.progress.total_steps == 5
        assert progress_manager.progress.status == "starting"

    @pytest.mark.asyncio
    async def test_emit_calls_callback(
        self,
        progress_manager: ProgressManager,
        mock_progress_callback: AsyncMock,
    ):
        """Test that emit calls the callback."""
        await progress_manager.emit()

        mock_progress_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_without_callback_no_error(self):
        """Test that emit without callback doesn't raise error."""
        manager = ProgressManager(post_id="test-001", callback=None)

        # Should not raise
        await manager.emit()

    @pytest.mark.asyncio
    async def test_update_modifies_progress(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that update modifies progress state."""
        await progress_manager.update(
            status="planning",
            current_step="Step 1",
        )

        assert progress_manager.progress.status == "planning"
        assert progress_manager.progress.current_step == "Step 1"

    @pytest.mark.asyncio
    async def test_start_phase_updates_state(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that start_phase updates progress correctly."""
        await progress_manager.start_phase("Research", phase_num=0)

        assert progress_manager.progress.phase_name == "Research"
        assert progress_manager.progress.current_phase == 0
        assert progress_manager.progress.status == "research"

    @pytest.mark.asyncio
    async def test_complete_phase_increments_steps(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that complete_phase increments completed_steps."""
        initial_steps = progress_manager.progress.completed_steps

        await progress_manager.complete_phase("Research")

        assert progress_manager.progress.completed_steps == initial_steps + 1

    @pytest.mark.asyncio
    async def test_set_total_slides_recalculates_steps(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that set_total_slides updates total_steps."""
        await progress_manager.set_total_slides(7)

        assert progress_manager.progress.total_slides == 7
        # total_steps = slides + 3 (research, planning, caption, save)
        assert progress_manager.progress.total_steps == 10

    @pytest.mark.asyncio
    async def test_start_slide_updates_current_slide(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that start_slide updates current_slide."""
        await progress_manager.set_total_slides(5)
        await progress_manager.start_slide(3)

        assert progress_manager.progress.current_slide == 3
        assert "slide 3" in progress_manager.progress.current_step.lower()

    @pytest.mark.asyncio
    async def test_complete_slide_increments_steps(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that complete_slide increments completed_steps."""
        initial_steps = progress_manager.progress.completed_steps

        await progress_manager.complete_slide(1)

        assert progress_manager.progress.completed_steps == initial_steps + 1

    @pytest.mark.asyncio
    async def test_handle_ai_event_updates_stats(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that handle_ai_event updates stats correctly."""
        # Text response event
        await progress_manager.handle_ai_event({
            "type": "text_response",
            "provider": "openai",
            "model": "gpt-4",
            "cost_usd": 0.05,
        }, "text")

        assert progress_manager.total_text_calls == 1
        assert progress_manager.total_cost == 0.05

        # Image response event
        await progress_manager.handle_ai_event({
            "type": "image_response",
            "provider": "dalle",
            "model": "dall-e-3",
            "cost_usd": 0.02,
        }, "image")

        assert progress_manager.total_image_calls == 1
        assert progress_manager.total_cost == 0.07

    @pytest.mark.asyncio
    async def test_complete_marks_done(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that complete marks status as completed."""
        await progress_manager.complete()

        assert progress_manager.progress.status == "completed"
        assert progress_manager.progress.current_step == "Done"

    @pytest.mark.asyncio
    async def test_fail_marks_failed(
        self,
        progress_manager: ProgressManager,
    ):
        """Test that fail marks status as failed."""
        await progress_manager.fail("Something went wrong")

        assert progress_manager.progress.status == "failed"
        assert "Something went wrong" in progress_manager.progress.errors


class TestOutputService:
    """Tests for OutputService."""

    @pytest.fixture
    def output_service(self, temp_profile_dir: Path) -> OutputService:
        """Create an OutputService with temp directory."""
        output_config = {
            "folder_structure": "posts/{year}/{month}/{status}/{day}-{post_number}-{slug}",
            "file_naming": {
                "slides": "slide_{number:02d}.jpg",
                "caption": "caption.txt",
                "hashtags": "hashtags.txt",
                "combined": "caption+hashtags.txt",
                "alt_texts": "alt_texts.json",
                "metadata": "metadata.json",
            },
        }
        return OutputService(temp_profile_dir, output_config)

    @pytest.fixture
    def sample_post(self) -> MagicMock:
        """Create a sample post for testing."""
        from socials_automator.content.models import CarouselPost, SlideContent, SlideType, HookType
        from datetime import datetime

        post = MagicMock(spec=CarouselPost)
        post.id = "20241212-001"
        post.date = "2024-12-12"
        post.slug = "5-ai-tools"
        post.topic = "5 AI Tools"
        post.content_pillar = "tools"
        post.hook_type = HookType.NUMBER_BENEFIT
        post.hook_text = "5 AI Tools"
        post.slides_count = 3
        post.status = "generated"
        post.created_at = datetime.now()
        post.generation_time_seconds = 45.5
        post.total_cost_usd = 0.15
        post.text_provider = "openai"
        post.image_provider = "dalle"
        post.caption = "Check out these AI tools!"
        post.hashtags = ["#AI", "#Tools", "#Tech"]

        # Create mock slides
        slide1 = MagicMock(spec=SlideContent)
        slide1.number = 1
        slide1.slide_type = SlideType.HOOK
        slide1.heading = "5 AI Tools"
        slide1.body = None
        slide1.image_path = None
        slide1.image_prompt = "Tech background"
        slide1.image_bytes = b"slide1_bytes"

        slide2 = MagicMock(spec=SlideContent)
        slide2.number = 2
        slide2.slide_type = SlideType.CONTENT
        slide2.heading = "Tool #1"
        slide2.body = "Description"
        slide2.image_path = None
        slide2.image_prompt = None
        slide2.image_bytes = b"slide2_bytes"

        slide3 = MagicMock(spec=SlideContent)
        slide3.number = 3
        slide3.slide_type = SlideType.CTA
        slide3.heading = "Follow!"
        slide3.body = None
        slide3.image_path = None
        slide3.image_prompt = None
        slide3.image_bytes = b"slide3_bytes"

        post.slides = [slide1, slide2, slide3]

        return post

    def test_get_output_path_creates_correct_path(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that get_output_path creates correct path structure."""
        with patch('socials_automator.services.output.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime = lambda x: {
                "%Y": "2024",
                "%m": "12",
                "%d": "12",
            }[x]

            path = output_service.get_output_path(sample_post)

            # Should contain year/month structure
            assert "2024" in str(path)
            assert "12" in str(path)
            assert "generated" in str(path)
            assert "5-ai-tools" in str(path)

    @pytest.mark.asyncio
    async def test_save_creates_all_files(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that save creates all expected files."""
        output_path = await output_service.save(sample_post)

        # Check slides exist
        assert (output_path / "slide_01.jpg").exists()
        assert (output_path / "slide_02.jpg").exists()
        assert (output_path / "slide_03.jpg").exists()

        # Check text files exist
        assert (output_path / "caption.txt").exists()
        assert (output_path / "hashtags.txt").exists()
        assert (output_path / "caption+hashtags.txt").exists()

        # Check JSON files exist
        assert (output_path / "alt_texts.json").exists()
        assert (output_path / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_save_caption_content(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that caption file contains correct content."""
        output_path = await output_service.save(sample_post)

        caption_content = (output_path / "caption.txt").read_text(encoding="utf-8")
        assert caption_content == "Check out these AI tools!"

    @pytest.mark.asyncio
    async def test_save_hashtags_content(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that hashtags file contains correct content."""
        output_path = await output_service.save(sample_post)

        hashtags_content = (output_path / "hashtags.txt").read_text(encoding="utf-8")
        assert "#AI" in hashtags_content
        assert "#Tools" in hashtags_content
        assert "#Tech" in hashtags_content

    @pytest.mark.asyncio
    async def test_save_metadata_structure(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that metadata JSON has correct structure."""
        output_path = await output_service.save(sample_post)

        with open(output_path / "metadata.json") as f:
            metadata = json.load(f)

        # Check top-level keys
        assert "post" in metadata
        assert "generation" in metadata
        assert "content" in metadata

        # Check post info
        assert metadata["post"]["id"] == "20241212-001"
        assert metadata["post"]["topic"] == "5 AI Tools"

        # Check generation info
        assert metadata["generation"]["text_provider"] == "openai"
        assert metadata["generation"]["image_provider"] == "dalle"

        # Check content info
        assert len(metadata["content"]["slides"]) == 3

    @pytest.mark.asyncio
    async def test_save_alt_texts(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that alt texts are saved correctly."""
        output_path = await output_service.save(sample_post)

        with open(output_path / "alt_texts.json") as f:
            alt_texts = json.load(f)

        assert len(alt_texts) == 3
        assert "Slide 1: 5 AI Tools" in alt_texts[0]

    @pytest.mark.asyncio
    async def test_save_updates_slide_paths(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test that save updates slide.image_path."""
        await output_service.save(sample_post)

        # Slides should have their image_path set
        for slide in sample_post.slides:
            assert slide.image_path is not None

    @pytest.mark.asyncio
    async def test_move_to_status(
        self,
        output_service: OutputService,
        sample_post: MagicMock,
    ):
        """Test moving post between status folders."""
        # First save to generated
        old_path = await output_service.save(sample_post, "generated")
        assert old_path.exists()

        # Move to pending-post
        new_path = await output_service.move_to_status(
            sample_post,
            from_status="generated",
            to_status="pending-post",
        )

        # Old path should not exist
        assert not old_path.exists()

        # New path should exist with files
        assert new_path.exists()
        assert (new_path / "metadata.json").exists()
