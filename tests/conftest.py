"""Shared test fixtures and configuration.

Provides mocks and fixtures for testing the Socials-Automator components.
All fixtures follow the pattern of returning async-compatible mocks that
can be used with the async/await syntax.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test output directory
TEST_OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create a test output directory.

    Returns:
        Path to temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def save_test_output(test_output_dir: Path):
    """Helper fixture to save test outputs for visual inspection.

    Usage:
        def test_something(save_test_output):
            slide_bytes = await job.execute(context)
            save_test_output("hook-001", slide_bytes.image_bytes)
    """
    def _save(name: str, data: bytes, extension: str = "jpg") -> Path:
        post_dir = test_output_dir / f"post-{name}"
        post_dir.mkdir(exist_ok=True)
        file_path = post_dir / f"slide.{extension}"
        file_path.write_bytes(data)
        return file_path

    return _save


@pytest.fixture
def mock_image_provider() -> AsyncMock:
    """Create a mock ImageProvider.

    Returns:
        AsyncMock configured as ImageProvider.
    """
    provider = AsyncMock()
    provider.generate.return_value = b"fake_image_bytes_12345"
    provider.current_provider = "mock"
    return provider


@pytest.fixture
def mock_text_provider() -> AsyncMock:
    """Create a mock TextProvider.

    Returns:
        AsyncMock configured as TextProvider.
    """
    provider = AsyncMock()
    provider.generate.return_value = "Mock generated text response"
    provider.current_provider = "mock"
    return provider


@pytest.fixture
def mock_composer() -> AsyncMock:
    """Create a mock SlideComposer.

    Returns:
        AsyncMock configured as SlideComposer.
    """
    composer = AsyncMock()

    # Create realistic JPEG bytes (minimal valid JPEG header)
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
    ])

    composer.create_hook_slide.return_value = jpeg_bytes
    composer.create_content_slide.return_value = jpeg_bytes
    composer.create_cta_slide.return_value = jpeg_bytes

    return composer


@pytest.fixture
def sample_outline() -> dict[str, Any]:
    """Create a sample slide outline.

    Returns:
        Dictionary with typical slide outline structure.
    """
    return {
        "slide_type": "hook",
        "number": 1,
        "heading": "5 AI Tools That Changed My Life",
        "body": "Save this for later!",
        "needs_image": True,
        "image_description": "Abstract tech background with AI elements",
    }


@pytest.fixture
def sample_profile_config() -> dict[str, Any]:
    """Create a sample profile configuration.

    Returns:
        Dictionary with typical profile config structure.
    """
    return {
        "profile": {
            "id": "test-profile",
            "instagram_handle": "@test.account",
            "niche_id": "ai-tools",
        },
        "content_strategy": {
            "content_pillars": [
                {"id": "tools", "name": "AI Tools", "description": "Reviews of AI tools"},
                {"id": "tips", "name": "Tips", "description": "AI productivity tips"},
            ],
            "carousel_settings": {
                "min_slides": 3,
                "max_slides": 10,
            },
        },
        "design": {
            "image_generation": {
                "style_prompt_suffix": "minimal, clean, tech aesthetic",
            },
            "cta_image": {
                "enabled": True,
            },
        },
        "hashtag_strategy": {
            "hashtag_sets": {
                "primary": ["#AI", "#Tech"],
                "secondary": ["#Productivity"],
                "branded": ["#TestBrand"],
            },
        },
        "output_settings": {
            "folder_structure": "posts/{year}/{month}/{status}/{day}-{post_number}-{slug}",
            "file_naming": {
                "slides": "slide_{number:02d}.jpg",
                "caption": "caption.txt",
                "hashtags": "hashtags.txt",
                "metadata": "metadata.json",
            },
        },
    }


@pytest.fixture
def sample_design_config() -> dict[str, Any]:
    """Create a sample design configuration.

    Returns:
        Dictionary with typical design config structure.
    """
    return {
        "image_generation": {
            "style_prompt_suffix": "minimal, clean, tech aesthetic, dark mode",
        },
        "cta_image": {
            "enabled": True,
        },
    }


@pytest.fixture
def sample_slide_job_context(
    sample_profile_config: dict[str, Any],
    sample_design_config: dict[str, Any],
) -> "SlideJobContext":
    """Create a sample SlideJobContext.

    Returns:
        SlideJobContext for testing.
    """
    from src.socials_automator.content.slides import SlideJobContext

    return SlideJobContext(
        post_id="test-001",
        slide_number=1,
        topic="5 AI Tools for Productivity",
        outline={
            "heading": "5 AI Tools You NEED",
            "body": "Save this for later!",
            "needs_image": True,
            "image_description": "Abstract AI tech background",
        },
        profile_config=sample_profile_config,
        design_config=sample_design_config,
        logo_path=None,
    )


@pytest.fixture
def mock_progress_callback() -> AsyncMock:
    """Create a mock progress callback.

    Returns:
        AsyncMock that records all progress updates.
    """
    callback = AsyncMock()
    callback.updates = []

    async def record_update(progress):
        callback.updates.append(progress)

    callback.side_effect = record_update
    return callback


@pytest.fixture
def temp_profile_dir(tmp_path: Path) -> Path:
    """Create a temporary profile directory structure.

    Returns:
        Path to temporary profile directory.
    """
    profile_dir = tmp_path / "test-profile"
    profile_dir.mkdir()

    # Create required subdirectories
    (profile_dir / "brand" / "fonts").mkdir(parents=True)
    (profile_dir / "posts").mkdir()

    # Create a minimal metadata.json
    import json
    metadata = {
        "profile": {"id": "test", "instagram_handle": "@test"},
        "content_strategy": {"content_pillars": []},
    }
    (profile_dir / "metadata.json").write_text(json.dumps(metadata))

    return profile_dir
