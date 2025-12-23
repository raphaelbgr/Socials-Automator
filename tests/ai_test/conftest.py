"""Pytest fixtures for AI testing module."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_profile_path(tmp_path):
    """Create a mock profile directory structure."""
    profile = tmp_path / "test-profile"
    profile.mkdir(parents=True)

    # Create metadata.json
    metadata = {
        "name": "Test Profile",
        "handle": "@testprofile",
        "hashtag": "#testprofile",
        "niche": "technology"
    }
    import json
    (profile / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    return profile


@pytest.fixture
def mock_script():
    """Create a mock VideoScript for testing."""
    from socials_automator.video.pipeline.base import VideoScript, VideoSegment

    return VideoScript(
        title="Test Video Title",
        hook="This is an attention-grabbing hook!",
        segments=[
            VideoSegment(
                index=1,
                text="First segment with useful information about AI tools.",
                duration_seconds=10.0,
                start_time=3.0,
                end_time=13.0,
            ),
            VideoSegment(
                index=2,
                text="Second segment with more valuable content.",
                duration_seconds=10.0,
                start_time=13.0,
                end_time=23.0,
            ),
            VideoSegment(
                index=3,
                text="Final segment with creative CTA. Follow Test Profile for more!",
                duration_seconds=10.0,
                start_time=23.0,
                end_time=33.0,
            ),
        ],
        cta="",  # Empty CTA - now in last segment
        total_duration=40.0,
        full_narration="This is an attention-grabbing hook! First segment with useful information about AI tools. Second segment with more valuable content. Final segment with creative CTA. Follow Test Profile for more!",
        hook_end_time=3.0,
        cta_start_time=33.0,
    )


@pytest.fixture
def mock_topic():
    """Create a mock TopicInfo for testing."""
    from socials_automator.video.pipeline.base import TopicInfo

    return TopicInfo(
        topic="Top 5 AI Tools for Productivity",
        category="technology",
        keywords=["AI tools", "productivity", "automation"],
    )


@pytest.fixture
def mock_context(mock_script, mock_topic, tmp_path):
    """Create a mock PipelineContext for testing."""
    from socials_automator.video.pipeline.base import PipelineContext, VideoMetadata, ArtifactStatus

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    metadata = VideoMetadata(
        post_id="test_video_001",
        title="Test Video Title",
        topic="Top 5 AI Tools for Productivity",
        profile="test-profile",
        profile_config={"hashtag": "#testprofile", "name": "Test Profile"},
    )

    return PipelineContext(
        profile_name="test-profile",
        target_duration=60.0,
        output_dir=output_dir,
        script=mock_script,
        topic=mock_topic,
        metadata=metadata,
    )


@pytest.fixture
def lmstudio_available():
    """Check if LMStudio is running and available."""
    import httpx
    try:
        response = httpx.get("http://localhost:1234/v1/models", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False
