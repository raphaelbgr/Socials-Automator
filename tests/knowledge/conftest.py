"""Fixtures for knowledge module tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from socials_automator.knowledge.models import (
    AIToolRecord,
    AIToolsConfig,
    AIToolCategory,
    AIToolUsageRecord,
)


@pytest.fixture
def sample_ai_tool() -> AIToolRecord:
    """Create a sample AI tool record."""
    return AIToolRecord(
        id="chatgpt",
        name="ChatGPT",
        company="OpenAI",
        category="text",
        current_version="GPT-4o",
        version_date="2024-05",
        url="https://chat.openai.com",
        pricing="freemium",
        hidden_gem=False,
        content_potential="high",
        features=["Text generation", "Code assistance", "Image analysis"],
        best_for=["Writing", "Coding", "Research"],
        video_ideas=["5 ChatGPT prompts for productivity", "ChatGPT vs Claude comparison"],
    )


@pytest.fixture
def sample_hidden_gem_tool() -> AIToolRecord:
    """Create a hidden gem AI tool record."""
    return AIToolRecord(
        id="notebooklm",
        name="NotebookLM",
        company="Google",
        category="research",
        current_version="2.0",
        version_date="2024-11",
        url="https://notebooklm.google.com",
        pricing="free",
        hidden_gem=True,
        content_potential="high",
        features=["Document analysis", "Audio summaries", "Citation tracking"],
        best_for=["Research", "Learning", "Content analysis"],
        video_ideas=["NotebookLM tutorial", "How to use NotebookLM for research"],
    )


@pytest.fixture
def sample_free_tool() -> AIToolRecord:
    """Create a free AI tool record."""
    return AIToolRecord(
        id="perplexity",
        name="Perplexity",
        company="Perplexity AI",
        category="research",
        current_version="Pro",
        version_date="2024-12",
        url="https://perplexity.ai",
        pricing="free",
        hidden_gem=False,
        content_potential="medium",
        features=["AI search", "Source citations", "Pro search mode"],
        best_for=["Research", "Fact-checking"],
        video_ideas=[],
    )


@pytest.fixture
def sample_paid_tool() -> AIToolRecord:
    """Create a paid AI tool record."""
    return AIToolRecord(
        id="midjourney",
        name="Midjourney",
        company="Midjourney Inc",
        category="image",
        current_version="V7",
        version_date="2024-12",
        url="https://midjourney.com",
        pricing="paid",
        hidden_gem=False,
        content_potential="high",
        features=["Image generation", "Style control", "Variations"],
        best_for=["Art", "Design", "Marketing"],
        video_ideas=["Midjourney V7 tips", "Best Midjourney prompts"],
    )


@pytest.fixture
def sample_low_potential_tool() -> AIToolRecord:
    """Create a low content potential tool."""
    return AIToolRecord(
        id="obscure-tool",
        name="Obscure Tool",
        company="Unknown Corp",
        category="other",
        current_version="1.0",
        version_date="2024-01",
        url="https://example.com",
        pricing="enterprise",
        hidden_gem=True,
        content_potential="low",
        features=["Some feature"],
        best_for=["Niche use case"],
        video_ideas=[],
    )


@pytest.fixture
def sample_category() -> AIToolCategory:
    """Create a sample category."""
    return AIToolCategory(
        id="text",
        name="Text AI",
        description="Large language models for text generation",
        icon="[T]",
    )


@pytest.fixture
def sample_categories() -> list[AIToolCategory]:
    """Create sample categories."""
    return [
        AIToolCategory(id="text", name="Text AI", description="LLMs", icon="[T]"),
        AIToolCategory(id="image", name="Image AI", description="Image generation", icon="[I]"),
        AIToolCategory(id="video", name="Video AI", description="Video generation", icon="[V]"),
        AIToolCategory(id="research", name="Research AI", description="Research tools", icon="[R]"),
    ]


@pytest.fixture
def sample_tools(
    sample_ai_tool: AIToolRecord,
    sample_hidden_gem_tool: AIToolRecord,
    sample_free_tool: AIToolRecord,
    sample_paid_tool: AIToolRecord,
    sample_low_potential_tool: AIToolRecord,
) -> list[AIToolRecord]:
    """Create a collection of sample tools."""
    return [
        sample_ai_tool,
        sample_hidden_gem_tool,
        sample_free_tool,
        sample_paid_tool,
        sample_low_potential_tool,
    ]


@pytest.fixture
def sample_ai_tools_config(
    sample_categories: list[AIToolCategory],
    sample_tools: list[AIToolRecord],
) -> AIToolsConfig:
    """Create a sample AIToolsConfig."""
    return AIToolsConfig(
        version="1.0.0",
        last_updated="2024-12-19",
        categories=sample_categories,
        tools=sample_tools,
        version_check_notes="Always verify versions before publishing",
        content_hints={"prioritize_hidden_gems": True},
    )


@pytest.fixture
def sample_yaml_config(
    sample_categories: list[AIToolCategory],
    sample_tools: list[AIToolRecord],
) -> dict[str, Any]:
    """Create sample YAML config data."""
    return {
        "version": "1.0.0",
        "last_updated": "2024-12-19",
        "categories": [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "icon": cat.icon,
            }
            for cat in sample_categories
        ],
        "tools": [
            {
                "id": tool.id,
                "name": tool.name,
                "company": tool.company,
                "category": tool.category,
                "current_version": tool.current_version,
                "version_date": tool.version_date,
                "url": tool.url,
                "pricing": tool.pricing,
                "hidden_gem": tool.hidden_gem,
                "content_potential": tool.content_potential,
                "features": tool.features,
                "best_for": tool.best_for,
                "video_ideas": tool.video_ideas,
            }
            for tool in sample_tools
        ],
        "version_check_notes": "Always verify versions before publishing",
        "content_hints": {"prioritize_hidden_gems": True},
    }


@pytest.fixture
def temp_yaml_config(sample_yaml_config: dict[str, Any], tmp_path: Path) -> Path:
    """Create a temporary YAML config file."""
    config_path = tmp_path / "ai_tools.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_yaml_config, f)
    return config_path


@pytest.fixture
def temp_profile_path(tmp_path: Path) -> Path:
    """Create a temporary profile directory."""
    profile_path = tmp_path / "test-profile"
    profile_path.mkdir(parents=True, exist_ok=True)

    # Create knowledge subdirectory
    (profile_path / "knowledge").mkdir(exist_ok=True)

    return profile_path


@pytest.fixture
def sample_usage_records() -> list[AIToolUsageRecord]:
    """Create sample usage records with recent dates."""
    from datetime import datetime, timedelta

    today = datetime.now()
    # Use relative dates to ensure tests work regardless of current date
    return [
        AIToolUsageRecord(
            tool_id="chatgpt",
            post_id="recent-001",
            date_used=(today - timedelta(days=5)).strftime("%Y-%m-%d"),  # 5 days ago
            context="main_topic",
        ),
        AIToolUsageRecord(
            tool_id="claude",
            post_id="recent-002",
            date_used=(today - timedelta(days=10)).strftime("%Y-%m-%d"),  # 10 days ago
            context="comparison",
        ),
        AIToolUsageRecord(
            tool_id="midjourney",
            post_id="old-001",
            date_used=(today - timedelta(days=45)).strftime("%Y-%m-%d"),  # 45 days ago
            context="main_topic",
        ),
    ]


@pytest.fixture
def temp_profile_with_usage(
    temp_profile_path: Path,
    sample_usage_records: list[AIToolUsageRecord],
) -> Path:
    """Create a profile with existing usage data."""
    usage_path = temp_profile_path / "knowledge" / "ai_tools_usage.json"
    data = [r.model_dump() for r in sample_usage_records]
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return temp_profile_path


