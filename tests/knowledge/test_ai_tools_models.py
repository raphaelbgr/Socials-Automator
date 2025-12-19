"""Unit tests for AI Tools models.

Tests for:
- AIToolRecord: properties, methods, validation
- AIToolsConfig: aggregation methods, filtering
- AIToolUsageRecord: usage tracking model
- AIToolCategory: category model
"""

from __future__ import annotations

import pytest

from socials_automator.knowledge.models import (
    AIToolRecord,
    AIToolsConfig,
    AIToolCategory,
    AIToolUsageRecord,
)


class TestAIToolRecord:
    """Tests for AIToolRecord model."""

    def test_create_minimal_tool(self):
        """Test creating a tool with minimal required fields."""
        tool = AIToolRecord(
            id="test-tool",
            name="Test Tool",
            company="Test Co",
            category="text",
            current_version="1.0",
            version_date="2024-12",
            url="https://example.com",
            pricing="free",
        )

        assert tool.id == "test-tool"
        assert tool.name == "Test Tool"
        assert tool.hidden_gem is False  # Default
        assert tool.content_potential == "medium"  # Default
        assert tool.features == []  # Default
        assert tool.video_ideas == []  # Default

    def test_is_free_property_free(self, sample_free_tool: AIToolRecord):
        """Test is_free returns True for free tools."""
        assert sample_free_tool.is_free is True

    def test_is_free_property_freemium(self, sample_ai_tool: AIToolRecord):
        """Test is_free returns True for freemium tools."""
        # sample_ai_tool (ChatGPT) is freemium
        assert sample_ai_tool.pricing == "freemium"
        assert sample_ai_tool.is_free is True

    def test_is_free_property_paid(self, sample_paid_tool: AIToolRecord):
        """Test is_free returns False for paid tools."""
        assert sample_paid_tool.pricing == "paid"
        assert sample_paid_tool.is_free is False

    def test_is_free_property_enterprise(self, sample_low_potential_tool: AIToolRecord):
        """Test is_free returns False for enterprise tools."""
        assert sample_low_potential_tool.pricing == "enterprise"
        assert sample_low_potential_tool.is_free is False

    def test_has_video_ideas_true(self, sample_ai_tool: AIToolRecord):
        """Test has_video_ideas returns True when ideas exist."""
        assert len(sample_ai_tool.video_ideas) > 0
        assert sample_ai_tool.has_video_ideas is True

    def test_has_video_ideas_false(self, sample_free_tool: AIToolRecord):
        """Test has_video_ideas returns False when no ideas."""
        assert len(sample_free_tool.video_ideas) == 0
        assert sample_free_tool.has_video_ideas is False

    def test_content_score_high(self, sample_ai_tool: AIToolRecord):
        """Test content_score returns 3 for high potential."""
        assert sample_ai_tool.content_potential == "high"
        assert sample_ai_tool.content_score == 3

    def test_content_score_medium(self, sample_free_tool: AIToolRecord):
        """Test content_score returns 2 for medium potential."""
        assert sample_free_tool.content_potential == "medium"
        assert sample_free_tool.content_score == 2

    def test_content_score_low(self, sample_low_potential_tool: AIToolRecord):
        """Test content_score returns 1 for low potential."""
        assert sample_low_potential_tool.content_potential == "low"
        assert sample_low_potential_tool.content_score == 1

    def test_content_score_unknown_defaults_to_medium(self):
        """Test content_score returns 2 for unknown values."""
        tool = AIToolRecord(
            id="test",
            name="Test",
            company="Test",
            category="text",
            current_version="1.0",
            version_date="2024-12",
            url="https://example.com",
            pricing="free",
            content_potential="unknown",  # Invalid value
        )
        assert tool.content_score == 2

    def test_to_prompt_context_format(self, sample_ai_tool: AIToolRecord):
        """Test to_prompt_context returns properly formatted string."""
        context = sample_ai_tool.to_prompt_context()

        # Check required elements are present
        assert "ChatGPT" in context
        assert "OpenAI" in context
        assert "GPT-4o" in context
        assert "2024-05" in context
        assert "text" in context
        assert "freemium" in context

        # Check features are included (up to 5)
        assert "Text generation" in context

        # Check best_for is included (up to 3)
        assert "Writing" in context

    def test_to_prompt_context_truncates_features(self):
        """Test to_prompt_context only includes first 5 features."""
        tool = AIToolRecord(
            id="test",
            name="Test",
            company="Test Co",
            category="text",
            current_version="1.0",
            version_date="2024-12",
            url="https://example.com",
            pricing="free",
            features=["F1", "F2", "F3", "F4", "F5", "F6", "F7"],
            best_for=["U1", "U2", "U3", "U4", "U5"],
        )

        context = tool.to_prompt_context()

        # First 5 features should be there
        assert "F1" in context
        assert "F5" in context

        # 6th and 7th should NOT be there
        assert "F6" not in context
        assert "F7" not in context

        # Only first 3 best_for should be there
        assert "U1" in context
        assert "U3" in context
        assert "U4" not in context


class TestAIToolsConfig:
    """Tests for AIToolsConfig model."""

    def test_total_tools_property(self, sample_ai_tools_config: AIToolsConfig):
        """Test total_tools counts all tools."""
        assert sample_ai_tools_config.total_tools == 5

    def test_hidden_gems_count_property(self, sample_ai_tools_config: AIToolsConfig):
        """Test hidden_gems_count filters correctly."""
        # We have 2 hidden gems: notebooklm and obscure-tool
        assert sample_ai_tools_config.hidden_gems_count == 2

    def test_high_potential_count_property(self, sample_ai_tools_config: AIToolsConfig):
        """Test high_potential_count filters correctly."""
        # High potential: chatgpt, notebooklm, midjourney = 3
        assert sample_ai_tools_config.high_potential_count == 3

    def test_get_tools_by_category_found(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_tools_by_category returns matching tools."""
        research_tools = sample_ai_tools_config.get_tools_by_category("research")

        assert len(research_tools) == 2  # notebooklm and perplexity
        assert all(t.category == "research" for t in research_tools)

    def test_get_tools_by_category_not_found(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_tools_by_category returns empty list for unknown category."""
        tools = sample_ai_tools_config.get_tools_by_category("nonexistent")
        assert tools == []

    def test_get_hidden_gems(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_hidden_gems returns only hidden gem tools."""
        gems = sample_ai_tools_config.get_hidden_gems()

        assert len(gems) == 2
        assert all(t.hidden_gem is True for t in gems)

    def test_get_high_potential_tools(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_high_potential_tools returns only high potential tools."""
        tools = sample_ai_tools_config.get_high_potential_tools()

        assert len(tools) == 3
        assert all(t.content_potential == "high" for t in tools)

    def test_get_tool_by_id_found(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_tool_by_id returns the correct tool."""
        tool = sample_ai_tools_config.get_tool_by_id("chatgpt")

        assert tool is not None
        assert tool.id == "chatgpt"
        assert tool.name == "ChatGPT"

    def test_get_tool_by_id_not_found(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_tool_by_id returns None for unknown ID."""
        tool = sample_ai_tools_config.get_tool_by_id("nonexistent")
        assert tool is None

    def test_get_tools_with_video_ideas(self, sample_ai_tools_config: AIToolsConfig):
        """Test get_tools_with_video_ideas filters correctly."""
        tools = sample_ai_tools_config.get_tools_with_video_ideas()

        # Should only include tools with video_ideas
        assert all(len(t.video_ideas) > 0 for t in tools)

        # Check expected tools are included
        tool_ids = [t.id for t in tools]
        assert "chatgpt" in tool_ids
        assert "notebooklm" in tool_ids
        assert "midjourney" in tool_ids

        # perplexity and obscure-tool have no video ideas
        assert "perplexity" not in tool_ids
        assert "obscure-tool" not in tool_ids


class TestAIToolCategory:
    """Tests for AIToolCategory model."""

    def test_create_category(self):
        """Test creating a category."""
        category = AIToolCategory(
            id="text",
            name="Text AI",
            description="Large language models",
            icon="[T]",
        )

        assert category.id == "text"
        assert category.name == "Text AI"
        assert category.description == "Large language models"
        assert category.icon == "[T]"

    def test_category_default_icon(self):
        """Test category with default empty icon."""
        category = AIToolCategory(
            id="test",
            name="Test",
            description="Test category",
        )

        assert category.icon == ""


class TestAIToolUsageRecord:
    """Tests for AIToolUsageRecord model."""

    def test_create_usage_record(self):
        """Test creating a usage record."""
        record = AIToolUsageRecord(
            tool_id="chatgpt",
            post_id="2024-12-19-001",
            date_used="2024-12-19",
            context="main_topic",
        )

        assert record.tool_id == "chatgpt"
        assert record.post_id == "2024-12-19-001"
        assert record.date_used == "2024-12-19"
        assert record.context == "main_topic"

    def test_usage_record_default_context(self):
        """Test usage record with default empty context."""
        record = AIToolUsageRecord(
            tool_id="test",
            post_id="test-001",
            date_used="2024-12-19",
        )

        assert record.context == ""


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_tools_list(self):
        """Test config with no tools."""
        config = AIToolsConfig(
            version="1.0",
            last_updated="2024-12-19",
            categories=[],
            tools=[],
        )

        assert config.total_tools == 0
        assert config.hidden_gems_count == 0
        assert config.high_potential_count == 0
        assert config.get_hidden_gems() == []

    def test_tool_with_empty_features(self):
        """Test tool with no features."""
        tool = AIToolRecord(
            id="empty",
            name="Empty Tool",
            company="Test",
            category="text",
            current_version="1.0",
            version_date="2024-12",
            url="https://example.com",
            pricing="free",
            features=[],
            best_for=[],
        )

        context = tool.to_prompt_context()
        assert "Empty Tool" in context
        # Features line should be mostly empty
        assert "Features:" in context

    def test_tool_with_special_characters(self):
        """Test tool with special characters in name."""
        tool = AIToolRecord(
            id="dall-e",
            name="DALL-E 3",
            company="OpenAI",
            category="image",
            current_version="3.0",
            version_date="2024-12",
            url="https://openai.com/dall-e",
            pricing="paid",
        )

        assert tool.name == "DALL-E 3"
        context = tool.to_prompt_context()
        assert "DALL-E 3" in context
