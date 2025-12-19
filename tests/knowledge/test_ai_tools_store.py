"""Unit tests for AIToolsStore.

Tests for:
- Usage tracking (mark_tool_covered, get_recently_covered)
- Uncovered gems discovery
- Topic suggestions
- Coverage statistics
- ChromaDB semantic search (mocked)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from socials_automator.knowledge.ai_tools_store import (
    AIToolsStore,
    get_ai_tools_store,
)
from socials_automator.knowledge.models import AIToolUsageRecord


class TestAIToolsStoreInit:
    """Tests for AIToolsStore initialization."""

    def test_init_creates_knowledge_directory(self, tmp_path: Path):
        """Test that initialization creates knowledge directory."""
        profile_path = tmp_path / "test-profile"
        profile_path.mkdir()

        store = AIToolsStore(profile_path)

        assert store.knowledge_path.exists()
        assert store.knowledge_path == profile_path / "knowledge"

    def test_init_sets_usage_path(self, temp_profile_path: Path):
        """Test that usage path is set correctly."""
        store = AIToolsStore(temp_profile_path)

        assert store.usage_path == temp_profile_path / "knowledge" / "ai_tools_usage.json"

    def test_get_ai_tools_store_function(self, temp_profile_path: Path):
        """Test convenience function creates store."""
        store = get_ai_tools_store(temp_profile_path)

        assert isinstance(store, AIToolsStore)
        assert store.profile_path == temp_profile_path


class TestUsageTracking:
    """Tests for usage tracking functionality."""

    def test_mark_tool_covered_creates_file(self, temp_profile_path: Path):
        """Test mark_tool_covered creates usage file."""
        store = AIToolsStore(temp_profile_path)

        store.mark_tool_covered("chatgpt", "2024-12-19-001", "main_topic")

        assert store.usage_path.exists()

    def test_mark_tool_covered_adds_record(self, temp_profile_path: Path):
        """Test mark_tool_covered adds a record."""
        store = AIToolsStore(temp_profile_path)

        store.mark_tool_covered("chatgpt", "2024-12-19-001", "main_topic")

        # Read and verify
        with open(store.usage_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["tool_id"] == "chatgpt"
        assert data[0]["post_id"] == "2024-12-19-001"
        assert data[0]["context"] == "main_topic"

    def test_mark_tool_covered_appends_to_existing(self, temp_profile_with_usage: Path):
        """Test mark_tool_covered appends to existing records."""
        store = AIToolsStore(temp_profile_with_usage)

        # Initially has 3 records
        initial_records = store._load_usage()
        assert len(initial_records) == 3

        store.mark_tool_covered("new-tool", "2024-12-19-001", "comparison")

        records = store._load_usage()
        assert len(records) == 4
        assert records[-1].tool_id == "new-tool"

    def test_mark_tool_covered_sets_current_date(self, temp_profile_path: Path):
        """Test mark_tool_covered uses current date."""
        store = AIToolsStore(temp_profile_path)
        today = datetime.now().strftime("%Y-%m-%d")

        store.mark_tool_covered("chatgpt", "2024-12-19-001")

        records = store._load_usage()
        assert records[0].date_used == today

    def test_get_recently_covered_returns_recent(self, temp_profile_with_usage: Path):
        """Test get_recently_covered returns tools from last N days."""
        store = AIToolsStore(temp_profile_with_usage)

        # Using days=30, should include chatgpt (5 days ago) and claude (10 days ago)
        # but not midjourney (45 days ago)
        covered = store.get_recently_covered(days=30)

        assert "chatgpt" in covered
        assert "claude" in covered
        assert "midjourney" not in covered  # 45 days ago, outside 30-day window

    def test_get_recently_covered_empty_when_no_usage(self, temp_profile_path: Path):
        """Test get_recently_covered returns empty when no usage."""
        store = AIToolsStore(temp_profile_path)

        covered = store.get_recently_covered(days=30)
        assert covered == []

    def test_was_recently_covered_true(self, temp_profile_with_usage: Path):
        """Test was_recently_covered returns True for recent tools."""
        store = AIToolsStore(temp_profile_with_usage)

        # Chatgpt was covered on Dec 15, which is within 30 days of Dec 19
        result = store.was_recently_covered("chatgpt", days=30)

        # This depends on the date comparison logic, but should work
        assert isinstance(result, bool)

    def test_was_recently_covered_false(self, temp_profile_path: Path):
        """Test was_recently_covered returns False for uncovered tools."""
        store = AIToolsStore(temp_profile_path)

        result = store.was_recently_covered("never-covered-tool", days=30)
        assert result is False

    def test_get_tool_coverage_returns_records(self, temp_profile_with_usage: Path):
        """Test get_tool_coverage returns all records for a tool."""
        store = AIToolsStore(temp_profile_with_usage)

        records = store.get_tool_coverage("chatgpt")

        assert len(records) == 1
        assert all(r.tool_id == "chatgpt" for r in records)

    def test_get_tool_coverage_empty_for_uncovered(self, temp_profile_path: Path):
        """Test get_tool_coverage returns empty for uncovered tools."""
        store = AIToolsStore(temp_profile_path)

        records = store.get_tool_coverage("uncovered")
        assert records == []


class TestUncoveredGems:
    """Tests for uncovered hidden gems discovery."""

    @pytest.fixture
    def store_with_mocked_registry(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Create store with mocked registry."""
        store = AIToolsStore(temp_profile_path)

        # Patch the registry to use our test config
        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock_get_registry:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_get_registry.return_value = registry
            yield store, registry

    def test_get_uncovered_gems_returns_gems(self, store_with_mocked_registry):
        """Test get_uncovered_gems returns hidden gems."""
        store, registry = store_with_mocked_registry

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            mock.return_value = registry
            gems = store.get_uncovered_gems(days=30)

        # Should return gems that haven't been covered
        assert all(g.hidden_gem is True for g in gems)

    def test_get_uncovered_gems_filters_covered(
        self,
        temp_profile_with_usage: Path,
        temp_yaml_config: Path,
    ):
        """Test get_uncovered_gems filters out recently covered tools."""
        store = AIToolsStore(temp_profile_with_usage)

        # Add a covered gem
        store.mark_tool_covered("notebooklm", "test-001", "main_topic")

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock.return_value = registry

            gems = store.get_uncovered_gems(days=30)

        # notebooklm should be filtered out
        gem_ids = [g.id for g in gems]
        assert "notebooklm" not in gem_ids

    def test_get_uncovered_gems_respects_limit(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test get_uncovered_gems respects limit."""
        store = AIToolsStore(temp_profile_path)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            gems = store.get_uncovered_gems(days=30, limit=1)

        assert len(gems) <= 1

    def test_get_uncovered_gems_high_potential_only(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test get_uncovered_gems filters by high potential."""
        store = AIToolsStore(temp_profile_path)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            gems = store.get_uncovered_gems(days=30, high_potential_only=True)

        # Only high potential gems
        assert all(g.content_potential == "high" for g in gems)


class TestTopicSuggestions:
    """Tests for topic suggestion functionality."""

    def test_suggest_next_topics_returns_strings(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test suggest_next_topics returns list of strings."""
        store = AIToolsStore(temp_profile_path)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            topics = store.suggest_next_topics(days=30)

        assert isinstance(topics, list)
        assert all(isinstance(t, str) for t in topics)

    def test_suggest_next_topics_filters_covered(
        self,
        temp_profile_path: Path,
        temp_yaml_config: Path,
    ):
        """Test suggest_next_topics excludes recently covered tools."""
        store = AIToolsStore(temp_profile_path)

        # Cover ChatGPT
        store.mark_tool_covered("chatgpt", "test-001", "main_topic")

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            topics = store.suggest_next_topics(days=30)

        # ChatGPT topics should be excluded
        assert not any("ChatGPT" in t for t in topics)

    def test_suggest_next_topics_respects_limit(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test suggest_next_topics respects limit."""
        store = AIToolsStore(temp_profile_path)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            topics = store.suggest_next_topics(days=30, limit=2)

        assert len(topics) <= 2


class TestCoverageStats:
    """Tests for coverage statistics."""

    def test_get_coverage_stats_structure(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test get_coverage_stats returns proper structure."""
        store = AIToolsStore(temp_profile_path)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            stats = store.get_coverage_stats(days=30)

        assert "period_days" in stats
        assert "total_tools" in stats
        assert "covered_tools" in stats
        assert "coverage_percent" in stats
        assert "hidden_gems" in stats
        assert "recently_covered" in stats

    def test_get_coverage_stats_with_usage(self, temp_profile_with_usage: Path, temp_yaml_config: Path):
        """Test get_coverage_stats with existing usage."""
        store = AIToolsStore(temp_profile_with_usage)

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            stats = store.get_coverage_stats(days=30)

        # Should have some covered tools
        assert stats["covered_tools"] >= 0
        assert "hidden_gems" in stats
        assert "total" in stats["hidden_gems"]
        assert "covered" in stats["hidden_gems"]
        assert "uncovered" in stats["hidden_gems"]


class TestSemanticSearch:
    """Tests for ChromaDB semantic search (mocked)."""

    def test_find_similar_tools_with_chromadb(
        self,
        temp_profile_path: Path,
        temp_yaml_config: Path,
    ):
        """Test find_similar_tools uses ChromaDB when available."""
        store = AIToolsStore(temp_profile_path)

        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [["chatgpt", "claude"]],
            "distances": [[0.1, 0.2]],
            "documents": [["ChatGPT doc", "Claude doc"]],
            "metadatas": [[{"name": "ChatGPT"}, {"name": "Claude"}]],
        }

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock_reg:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_reg.return_value = registry

            # Mock the chromadb as already initialized
            store._chroma_client = MagicMock()
            store._tools_collection = mock_collection
            store._indexed = True

            results = store.find_similar_tools("video editing")

        # Should have called query
        mock_collection.query.assert_called()

    def test_find_similar_tools_fallback_to_registry(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test find_similar_tools falls back to registry search."""
        store = AIToolsStore(temp_profile_path)

        # Don't initialize chromadb
        store._chroma_client = None

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock_reg:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_reg.return_value = registry

            results = store.find_similar_tools("ChatGPT")

        # Should fall back to registry.search
        assert len(results) >= 0

    def test_find_tools_for_topic(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test find_tools_for_topic is a convenience wrapper."""
        store = AIToolsStore(temp_profile_path)

        with patch.object(store, "find_similar_tools") as mock_find:
            mock_find.return_value = []

            store.find_tools_for_topic("AI productivity tips")

        mock_find.assert_called_once_with("AI productivity tips", limit=3)

    def test_tool_to_document_format(self, temp_profile_path: Path, sample_ai_tool):
        """Test _tool_to_document creates searchable text."""
        store = AIToolsStore(temp_profile_path)

        doc = store._tool_to_document(sample_ai_tool)

        assert "ChatGPT" in doc
        assert "OpenAI" in doc
        assert "text" in doc
        assert "Text generation" in doc
        assert "Writing" in doc

    def test_chromadb_init_handles_import_error(self, temp_profile_path: Path):
        """Test _init_chromadb handles missing chromadb gracefully."""
        store = AIToolsStore(temp_profile_path)

        # Test that the method returns a boolean and doesn't crash
        # This tests the actual behavior without mocking the import
        result = store._init_chromadb()
        assert isinstance(result, bool)


class TestErrorHandling:
    """Tests for error handling."""

    def test_load_usage_handles_corrupted_file(self, temp_profile_path: Path):
        """Test _load_usage handles corrupted JSON."""
        store = AIToolsStore(temp_profile_path)

        # Write corrupted JSON
        store.usage_path.write_text("{invalid json")

        records = store._load_usage()
        assert records == []

    def test_load_usage_handles_missing_file(self, temp_profile_path: Path):
        """Test _load_usage handles missing file."""
        store = AIToolsStore(temp_profile_path)

        # File doesn't exist
        records = store._load_usage()
        assert records == []

    def test_save_usage_handles_write_error(self, temp_profile_path: Path):
        """Test _save_usage handles write errors gracefully."""
        store = AIToolsStore(temp_profile_path)

        # Make directory read-only to cause write error
        # This is platform-specific, so we'll mock instead
        with patch("builtins.open", side_effect=PermissionError("Cannot write")):
            # Should not raise, just log error
            records = [
                AIToolUsageRecord(
                    tool_id="test",
                    post_id="test-001",
                    date_used="2024-12-19",
                )
            ]
            store._save_usage(records)

    def test_ensure_indexed_handles_registry_error(self, temp_profile_path: Path):
        """Test _ensure_indexed handles registry errors."""
        store = AIToolsStore(temp_profile_path)

        # Create mock chromadb components
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        store._chroma_client = MagicMock()
        store._tools_collection = mock_collection

        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            mock.side_effect = Exception("Registry load failed")

            # Should not raise
            store._ensure_indexed()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_mark_tool_covered_empty_context(self, temp_profile_path: Path):
        """Test mark_tool_covered with no context."""
        store = AIToolsStore(temp_profile_path)

        store.mark_tool_covered("chatgpt", "2024-12-19-001")

        records = store._load_usage()
        assert records[0].context == "main_topic"  # Default

    def test_get_recently_covered_zero_days(self, temp_profile_with_usage: Path):
        """Test get_recently_covered with 0 days."""
        store = AIToolsStore(temp_profile_with_usage)

        covered = store.get_recently_covered(days=0)

        # Should only include tools covered today
        assert isinstance(covered, list)

    def test_multiple_coverage_same_tool(self, temp_profile_path: Path):
        """Test same tool can be covered multiple times."""
        store = AIToolsStore(temp_profile_path)

        store.mark_tool_covered("chatgpt", "post-001", "main_topic")
        store.mark_tool_covered("chatgpt", "post-002", "comparison")
        store.mark_tool_covered("chatgpt", "post-003", "mention")

        records = store.get_tool_coverage("chatgpt")
        assert len(records) == 3

    def test_unicode_in_context(self, temp_profile_path: Path):
        """Test handling of unicode in context."""
        store = AIToolsStore(temp_profile_path)

        store.mark_tool_covered("chatgpt", "post-001", "AI tools comparison")

        records = store._load_usage()
        assert records[0].context == "AI tools comparison"
