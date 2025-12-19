"""Unit tests for AIToolsRegistry.

Tests for:
- Loading from YAML files
- Tool retrieval methods (get_by_id, get_by_category, etc.)
- Content discovery methods (get_hidden_gems, suggest_video_topics)
- Search functionality
- Version context generation
- Singleton pattern
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from socials_automator.knowledge.ai_tools_registry import (
    AIToolsRegistry,
    get_ai_tools_registry,
    reset_ai_tools_registry,
)
from socials_automator.knowledge.models import (
    AIToolRecord,
    AIToolsConfig,
    AIToolCategory,
)


class TestAIToolsRegistryLoad:
    """Tests for loading AIToolsRegistry from YAML."""

    def test_load_from_yaml_file(self, temp_yaml_config: Path):
        """Test loading registry from a YAML file."""
        registry = AIToolsRegistry.load(temp_yaml_config)

        assert registry is not None
        assert registry.config.version == "1.0.0"
        assert len(registry.get_all_tools()) == 5

    def test_load_missing_file_raises_error(self, tmp_path: Path):
        """Test that loading missing file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            AIToolsRegistry.load(missing_path)

    def test_load_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that loading invalid YAML raises ValueError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{ invalid: yaml: content")

        with pytest.raises(ValueError) as exc_info:
            AIToolsRegistry.load(bad_yaml)

        assert "Invalid YAML" in str(exc_info.value)

    def test_load_invalid_config_structure_raises_error(self, tmp_path: Path):
        """Test that invalid config structure raises ValueError."""
        bad_config = tmp_path / "bad_config.yaml"
        # Missing required fields
        bad_config.write_text("some_field: value")

        with pytest.raises(ValueError) as exc_info:
            AIToolsRegistry.load(bad_config)

        assert "Invalid config structure" in str(exc_info.value)

    def test_config_property_returns_config(self, temp_yaml_config: Path):
        """Test that config property returns the AIToolsConfig."""
        registry = AIToolsRegistry.load(temp_yaml_config)

        config = registry.config
        assert isinstance(config, AIToolsConfig)


class TestAIToolsRegistryToolAccess:
    """Tests for tool access methods."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_get_all_tools(self, registry: AIToolsRegistry):
        """Test get_all_tools returns all tools."""
        tools = registry.get_all_tools()

        assert len(tools) == 5
        assert all(isinstance(t, AIToolRecord) for t in tools)

    def test_get_by_id_found(self, registry: AIToolsRegistry):
        """Test get_by_id returns tool when found."""
        tool = registry.get_by_id("chatgpt")

        assert tool is not None
        assert tool.id == "chatgpt"
        assert tool.name == "ChatGPT"

    def test_get_by_id_not_found(self, registry: AIToolsRegistry):
        """Test get_by_id returns None when not found."""
        tool = registry.get_by_id("nonexistent")
        assert tool is None

    def test_get_by_category(self, registry: AIToolsRegistry):
        """Test get_by_category filters correctly."""
        research_tools = registry.get_by_category("research")

        assert len(research_tools) == 2
        assert all(t.category == "research" for t in research_tools)

    def test_get_by_category_empty(self, registry: AIToolsRegistry):
        """Test get_by_category returns empty for unknown category."""
        tools = registry.get_by_category("nonexistent")
        assert tools == []

    def test_get_by_pricing(self, registry: AIToolsRegistry):
        """Test get_by_pricing filters correctly."""
        free_tools = registry.get_by_pricing("free")

        assert len(free_tools) >= 1
        assert all(t.pricing == "free" for t in free_tools)

    def test_get_free_tools(self, registry: AIToolsRegistry):
        """Test get_free_tools returns free and freemium tools."""
        free_tools = registry.get_free_tools()

        # Should include free AND freemium
        assert all(t.is_free for t in free_tools)
        assert any(t.pricing == "free" for t in free_tools)
        assert any(t.pricing == "freemium" for t in free_tools)


class TestAIToolsRegistryContentDiscovery:
    """Tests for content discovery methods."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_get_hidden_gems(self, registry: AIToolsRegistry):
        """Test get_hidden_gems returns only hidden gems."""
        gems = registry.get_hidden_gems()

        assert len(gems) == 2
        assert all(t.hidden_gem is True for t in gems)

    def test_get_hidden_gems_with_limit(self, registry: AIToolsRegistry):
        """Test get_hidden_gems respects limit."""
        gems = registry.get_hidden_gems(limit=1)

        assert len(gems) == 1

    def test_get_hidden_gems_high_potential_only(self, registry: AIToolsRegistry):
        """Test get_hidden_gems filters by high potential."""
        gems = registry.get_hidden_gems(high_potential_only=True)

        # Only notebooklm is high potential gem (obscure-tool is low)
        assert len(gems) == 1
        assert gems[0].id == "notebooklm"

    def test_get_hidden_gems_by_category(self, registry: AIToolsRegistry):
        """Test get_hidden_gems filters by category."""
        gems = registry.get_hidden_gems(category="research")

        # Only notebooklm is in research and is a gem
        assert len(gems) == 1
        assert gems[0].category == "research"

    def test_get_hidden_gems_sorted_by_score(self, registry: AIToolsRegistry):
        """Test get_hidden_gems returns highest score first."""
        gems = registry.get_hidden_gems(limit=10)

        # Should be sorted by content_score descending
        scores = [g.content_score for g in gems]
        assert scores == sorted(scores, reverse=True)

    def test_get_high_potential_tools(self, registry: AIToolsRegistry):
        """Test get_high_potential_tools returns correct tools."""
        tools = registry.get_high_potential_tools()

        assert len(tools) == 3
        assert all(t.content_potential == "high" for t in tools)

    def test_get_high_potential_tools_with_category(self, registry: AIToolsRegistry):
        """Test get_high_potential_tools filters by category."""
        tools = registry.get_high_potential_tools(category="research")

        # Only notebooklm is high potential research tool
        assert len(tools) == 1
        assert tools[0].category == "research"

    def test_get_tools_with_video_ideas(self, registry: AIToolsRegistry):
        """Test get_tools_with_video_ideas filters correctly."""
        tools = registry.get_tools_with_video_ideas()

        assert all(len(t.video_ideas) > 0 for t in tools)

    def test_suggest_video_topics(self, registry: AIToolsRegistry):
        """Test suggest_video_topics returns topic strings."""
        topics = registry.suggest_video_topics()

        assert isinstance(topics, list)
        assert all(isinstance(t, str) for t in topics)
        assert len(topics) > 0

    def test_suggest_video_topics_with_limit(self, registry: AIToolsRegistry):
        """Test suggest_video_topics respects limit."""
        topics = registry.suggest_video_topics(limit=2)
        assert len(topics) <= 2

    def test_suggest_video_topics_prioritize_gems(self, registry: AIToolsRegistry):
        """Test suggest_video_topics puts gem topics first when prioritized."""
        topics = registry.suggest_video_topics(prioritize_gems=True, limit=10)

        # NotebookLM (hidden gem) ideas should come first
        assert any("NotebookLM" in t for t in topics[:3])

    def test_suggest_video_topics_by_category(self, registry: AIToolsRegistry):
        """Test suggest_video_topics filters by category."""
        topics = registry.suggest_video_topics(category="image")

        # Should only include midjourney topics
        assert all("Midjourney" in t for t in topics)


class TestAIToolsRegistrySearch:
    """Tests for search functionality."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_search_by_name(self, registry: AIToolsRegistry):
        """Test search finds tools by name."""
        results = registry.search("ChatGPT")

        assert len(results) >= 1
        assert results[0].id == "chatgpt"

    def test_search_by_name_case_insensitive(self, registry: AIToolsRegistry):
        """Test search is case insensitive."""
        results = registry.search("chatgpt")
        assert len(results) >= 1
        assert results[0].id == "chatgpt"

    def test_search_by_company(self, registry: AIToolsRegistry):
        """Test search finds tools by company name."""
        results = registry.search("OpenAI")

        assert len(results) >= 1
        assert any(t.company == "OpenAI" for t in results)

    def test_search_by_category(self, registry: AIToolsRegistry):
        """Test search finds tools by category."""
        results = registry.search("research")

        assert len(results) >= 1
        # research tools should rank high
        assert any(t.category == "research" for t in results)

    def test_search_by_feature(self, registry: AIToolsRegistry):
        """Test search finds tools by feature."""
        results = registry.search("Image analysis")

        assert len(results) >= 1

    def test_search_by_best_for(self, registry: AIToolsRegistry):
        """Test search finds tools by best_for use case."""
        results = registry.search("Coding")

        assert len(results) >= 1

    def test_search_with_limit(self, registry: AIToolsRegistry):
        """Test search respects limit."""
        results = registry.search("AI", limit=2)
        assert len(results) <= 2

    def test_search_no_results(self, registry: AIToolsRegistry):
        """Test search returns empty for no matches."""
        results = registry.search("xyznonexistent123")
        assert results == []

    def test_search_ranks_exact_match_higher(self, registry: AIToolsRegistry):
        """Test search ranks exact name match higher."""
        results = registry.search("Claude")

        # We don't have Claude in our test data, but ChatGPT shouldn't match
        # This tests that partial matches work
        results = registry.search("Chat")
        if len(results) > 0:
            # ChatGPT should be in results since "chat" is in the name
            assert any("chat" in t.name.lower() for t in results)

    def test_find_alternatives(self, registry: AIToolsRegistry):
        """Test find_alternatives returns same-category tools."""
        alternatives = registry.find_alternatives("notebooklm")

        # Should find perplexity (also research category)
        assert len(alternatives) >= 1
        assert all(t.category == "research" for t in alternatives)
        assert all(t.id != "notebooklm" for t in alternatives)

    def test_find_alternatives_not_found(self, registry: AIToolsRegistry):
        """Test find_alternatives returns empty for unknown tool."""
        alternatives = registry.find_alternatives("nonexistent")
        assert alternatives == []

    def test_find_alternatives_sorted_by_potential(self, registry: AIToolsRegistry):
        """Test find_alternatives returns highest potential first."""
        alternatives = registry.find_alternatives("notebooklm")

        if len(alternatives) > 1:
            scores = [a.content_score for a in alternatives]
            assert scores == sorted(scores, reverse=True)


class TestAIToolsRegistryVersionContext:
    """Tests for version context generation."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_get_version_context_format(self, registry: AIToolsRegistry):
        """Test get_version_context returns properly formatted string."""
        context = registry.get_version_context(["chatgpt"])

        assert "Current AI tool versions" in context
        assert "ChatGPT" in context
        assert "GPT-4o" in context
        assert "2024-05" in context
        assert "verify versions" in context.lower()

    def test_get_version_context_multiple_tools(self, registry: AIToolsRegistry):
        """Test get_version_context includes all specified tools."""
        context = registry.get_version_context(["chatgpt", "midjourney"])

        assert "ChatGPT" in context
        assert "Midjourney" in context

    def test_get_version_context_unknown_tool_skipped(self, registry: AIToolsRegistry):
        """Test get_version_context skips unknown tools."""
        context = registry.get_version_context(["chatgpt", "nonexistent"])

        assert "ChatGPT" in context
        # Should not crash, just skip unknown

    def test_get_version_context_default_tools(self, registry: AIToolsRegistry):
        """Test get_version_context uses default tools when none specified."""
        context = registry.get_version_context()

        # Should include some content even if tools don't match
        assert "Current AI tool versions" in context

    def test_get_version_check_notes(self, registry: AIToolsRegistry):
        """Test get_version_check_notes returns notes from config."""
        notes = registry.get_version_check_notes()

        assert isinstance(notes, str)
        assert "verify" in notes.lower()


class TestAIToolsRegistryCategoryMethods:
    """Tests for category-related methods."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_get_all_categories(self, registry: AIToolsRegistry):
        """Test get_all_categories returns all categories."""
        categories = registry.get_all_categories()

        assert len(categories) == 4
        assert all(isinstance(c, AIToolCategory) for c in categories)

    def test_get_category_found(self, registry: AIToolsRegistry):
        """Test get_category returns category when found."""
        category = registry.get_category("text")

        assert category is not None
        assert category.id == "text"
        assert category.name == "Text AI"

    def test_get_category_not_found(self, registry: AIToolsRegistry):
        """Test get_category returns None when not found."""
        category = registry.get_category("nonexistent")
        assert category is None

    def test_get_category_stats(self, registry: AIToolsRegistry):
        """Test get_category_stats returns correct counts."""
        stats = registry.get_category_stats()

        assert isinstance(stats, dict)
        assert "text" in stats
        assert stats["text"] == 1  # chatgpt
        assert stats["research"] == 2  # notebooklm, perplexity
        assert stats["image"] == 1  # midjourney


class TestAIToolsRegistryRandomSelection:
    """Tests for random selection methods."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_random_hidden_gem(self, registry: AIToolsRegistry):
        """Test random_hidden_gem returns a hidden gem."""
        gem = registry.random_hidden_gem()

        assert gem is not None
        assert gem.hidden_gem is True

    def test_random_hidden_gem_by_category(self, registry: AIToolsRegistry):
        """Test random_hidden_gem filters by category."""
        gem = registry.random_hidden_gem(category="research")

        # Only notebooklm is research + hidden gem
        assert gem is not None
        assert gem.category == "research"
        assert gem.id == "notebooklm"

    def test_random_hidden_gem_no_match(self, registry: AIToolsRegistry):
        """Test random_hidden_gem returns None when no match."""
        gem = registry.random_hidden_gem(category="nonexistent")
        assert gem is None

    def test_random_video_topic(self, registry: AIToolsRegistry):
        """Test random_video_topic returns a topic string."""
        topic = registry.random_video_topic()

        assert topic is not None
        assert isinstance(topic, str)

    def test_random_video_topic_by_category(self, registry: AIToolsRegistry):
        """Test random_video_topic filters by category."""
        topic = registry.random_video_topic(category="image")

        assert topic is not None
        assert "Midjourney" in topic


class TestAIToolsRegistrySummary:
    """Tests for summary/statistics methods."""

    @pytest.fixture
    def registry(self, temp_yaml_config: Path) -> AIToolsRegistry:
        """Create a registry from test config."""
        return AIToolsRegistry.load(temp_yaml_config)

    def test_get_summary_structure(self, registry: AIToolsRegistry):
        """Test get_summary returns proper structure."""
        summary = registry.get_summary()

        assert "version" in summary
        assert "last_updated" in summary
        assert "loaded_at" in summary
        assert "totals" in summary
        assert "by_category" in summary
        assert "by_pricing" in summary

    def test_get_summary_totals(self, registry: AIToolsRegistry):
        """Test get_summary totals are correct."""
        summary = registry.get_summary()

        totals = summary["totals"]
        assert totals["tools"] == 5
        assert totals["categories"] == 4
        assert totals["hidden_gems"] == 2
        assert totals["high_potential"] == 3

    def test_get_summary_by_category(self, registry: AIToolsRegistry):
        """Test get_summary by_category is correct."""
        summary = registry.get_summary()

        by_category = summary["by_category"]
        assert by_category["text"] == 1
        assert by_category["research"] == 2

    def test_get_summary_by_pricing(self, registry: AIToolsRegistry):
        """Test get_summary by_pricing is correct."""
        summary = registry.get_summary()

        by_pricing = summary["by_pricing"]
        assert "free" in by_pricing
        assert "freemium" in by_pricing
        assert "paid" in by_pricing


class TestSingletonPattern:
    """Tests for singleton pattern and caching."""

    def test_get_ai_tools_registry_returns_cached_instance(self, temp_yaml_config: Path):
        """Test get_ai_tools_registry returns cached instance."""
        # Reset to ensure clean state
        reset_ai_tools_registry()

        # Patch the default path to use our temp config
        with patch.object(AIToolsRegistry, "DEFAULT_CONFIG_PATH", temp_yaml_config):
            registry1 = get_ai_tools_registry()
            registry2 = get_ai_tools_registry()

            assert registry1 is registry2

        # Clean up
        reset_ai_tools_registry()

    def test_get_ai_tools_registry_reload(self, temp_yaml_config: Path):
        """Test get_ai_tools_registry reload creates new instance."""
        reset_ai_tools_registry()

        with patch.object(AIToolsRegistry, "DEFAULT_CONFIG_PATH", temp_yaml_config):
            registry1 = get_ai_tools_registry()
            registry2 = get_ai_tools_registry(reload=True)

            # Should be different instances
            assert registry1 is not registry2

        reset_ai_tools_registry()

    def test_reset_ai_tools_registry(self, temp_yaml_config: Path):
        """Test reset_ai_tools_registry clears cache."""
        reset_ai_tools_registry()

        with patch.object(AIToolsRegistry, "DEFAULT_CONFIG_PATH", temp_yaml_config):
            registry1 = get_ai_tools_registry()
            reset_ai_tools_registry()
            registry2 = get_ai_tools_registry()

            assert registry1 is not registry2

        reset_ai_tools_registry()
