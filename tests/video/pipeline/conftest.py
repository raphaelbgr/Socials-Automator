"""Pytest fixtures for pipeline tests."""

import json
import tempfile
from pathlib import Path

import pytest

from socials_automator.video.pipeline import (
    PipelineContext,
    ProfileMetadata,
    ResearchResult,
    TopicInfo,
    VideoScript,
    VideoSegment,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_profile_data():
    """Sample profile metadata dictionary."""
    return {
        "profile": {
            "id": "test-profile",
            "name": "test.profile",
            "display_name": "Test Profile",
            "niche_id": "ai-productivity",
            "tagline": "Test tagline",
            "description": "Test description",
            "language": "en",
        },
        "content_strategy": {
            "content_pillars": [
                {
                    "id": "tool_tutorials",
                    "name": "Tool Tutorials",
                    "description": "How to use AI tools",
                    "frequency_percent": 30,
                    "examples": ["How to use ChatGPT", "Getting started with AI"],
                },
                {
                    "id": "productivity_hacks",
                    "name": "Productivity Hacks",
                    "description": "Tips to save time",
                    "frequency_percent": 30,
                    "examples": ["5 AI tips", "Automate your work"],
                },
            ],
        },
        "research_sources": {
            "trending_keywords": ["ChatGPT", "AI tools", "productivity"],
            "sources": {},
        },
        "ai_generation": {},
    }


@pytest.fixture
def sample_profile_dir(temp_dir, sample_profile_data):
    """Create a profile directory with metadata.json."""
    profile_dir = temp_dir / "profiles" / "test-profile"
    profile_dir.mkdir(parents=True)

    metadata_path = profile_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(sample_profile_data, f)

    return profile_dir


@pytest.fixture
def sample_profile_metadata(sample_profile_dir):
    """Load sample profile metadata."""
    return ProfileMetadata.from_file(sample_profile_dir / "metadata.json")


@pytest.fixture
def sample_topic_info():
    """Sample topic information."""
    return TopicInfo(
        topic="How to use ChatGPT for productivity",
        pillar_id="tool_tutorials",
        pillar_name="Tool Tutorials",
        keywords=["chatgpt", "productivity", "ai", "tools"],
        search_queries=[
            "How to use ChatGPT for productivity",
            "ChatGPT tips 2025",
        ],
    )


@pytest.fixture
def sample_research_result(sample_topic_info):
    """Sample research result."""
    return ResearchResult(
        topic=sample_topic_info.topic,
        summary="ChatGPT is a powerful AI tool for productivity.",
        key_points=[
            "ChatGPT can help automate repetitive tasks",
            "Use specific prompts for better results",
            "Save time with templates",
            "Integrate with other tools",
            "Learn the best practices",
        ],
        sources=[
            {"title": "ChatGPT Guide", "url": "https://example.com/guide"},
        ],
        raw_content="Raw content from web search...",
    )


@pytest.fixture
def sample_video_script(sample_topic_info):
    """Sample video script."""
    return VideoScript(
        title="How to use ChatGPT for productivity",
        hook="Stop wasting time! Here's how to 10x your productivity with ChatGPT.",
        segments=[
            VideoSegment(
                index=1,
                text="First, understand what ChatGPT can do for you.",
                duration_seconds=10.0,
                keywords=["chatgpt", "ai", "introduction"],
            ),
            VideoSegment(
                index=2,
                text="Tip one: Use specific prompts for better results.",
                duration_seconds=10.0,
                keywords=["prompts", "tips", "productivity"],
            ),
            VideoSegment(
                index=3,
                text="Tip two: Save your best prompts as templates.",
                duration_seconds=10.0,
                keywords=["templates", "efficiency", "workflow"],
            ),
            VideoSegment(
                index=4,
                text="Tip three: Integrate ChatGPT with your daily workflow.",
                duration_seconds=10.0,
                keywords=["integration", "workflow", "automation"],
            ),
            VideoSegment(
                index=5,
                text="Now you're ready to boost your productivity!",
                duration_seconds=10.0,
                keywords=["success", "productivity", "achievement"],
            ),
        ],
        cta="Follow for more AI tips!",
        total_duration=60.0,
        full_narration="Stop wasting time! First... templates... Follow for more!",
    )


@pytest.fixture
def sample_pipeline_context(sample_profile_metadata, temp_dir):
    """Sample pipeline context."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()

    return PipelineContext(
        profile=sample_profile_metadata,
        post_id="test-001",
        output_dir=output_dir,
        temp_dir=temp_dir,
    )
