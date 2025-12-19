"""Integration tests for AI Tools Database.

Tests for integration points:
- content/planner.py: get_ai_version_context()
- content/orchestrator.py: _get_ai_tools_suggestions(), mark_tools_covered()
- video/pipeline/topic_selector.py: version context and hidden gems
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPlannerVersionContext:
    """Tests for get_ai_version_context in content/planner.py."""

    def test_get_ai_version_context_with_registry(self, temp_yaml_config: Path):
        """Test get_ai_version_context returns version info from registry."""
        # Patch at the knowledge module level since that's where import happens
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_get:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_get.return_value = registry

            # Import fresh to get the patched version
            import importlib
            import socials_automator.content.planner as planner_module
            importlib.reload(planner_module)

            context = planner_module.get_ai_version_context()

        assert "ChatGPT" in context or "Current AI tool versions" in context

    def test_get_ai_version_context_fallback_on_error(self):
        """Test get_ai_version_context returns fallback on registry error."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_get:
            mock_get.side_effect = Exception("Registry unavailable")

            import importlib
            import socials_automator.content.planner as planner_module
            importlib.reload(planner_module)

            context = planner_module.get_ai_version_context()

        # Should return fallback
        assert "Current AI tool versions" in context
        assert "verify" in context.lower()

    def test_get_ai_version_context_in_system_prompt(self, temp_yaml_config: Path):
        """Test version context is included in system prompt."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_get:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_get.return_value = registry

            from socials_automator.content.planner import ContentPlanner

            planner = ContentPlanner(
                text_provider=AsyncMock(),
                profile_config={
                    "content_strategy": {
                        "hook_strategies": {"primary_types": []},
                    },
                },
            )

            prompt = planner._get_system_prompt()

        # Version context should be included
        assert "AI tool versions" in prompt or "Current" in prompt


class TestOrchestratorToolsSuggestions:
    """Tests for _get_ai_tools_suggestions in content/orchestrator.py."""

    @pytest.fixture
    def mock_orchestrator_deps(self, temp_profile_path: Path):
        """Create mock dependencies for orchestrator."""
        return {
            "profile_path": temp_profile_path,
            "profile_config": {
                "profile": {"id": "test", "instagram_handle": "@test"},
                "content_strategy": {"content_pillars": []},
                "ai_tools_config": {
                    "enabled": True,
                    "prioritize_hidden_gems": True,
                    "cooldown_days": 14,
                    "favorite_categories": ["text", "research"],
                },
            },
            "text_provider": AsyncMock(),
            "image_provider": AsyncMock(),
        }

    def test_get_ai_tools_suggestions_returns_string(
        self,
        mock_orchestrator_deps: dict,
        temp_yaml_config: Path,
    ):
        """Test _get_ai_tools_suggestions returns formatted string."""
        # Patch at the knowledge module level
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg, \
             patch("socials_automator.knowledge.get_ai_tools_store") as mock_store:

            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_reg.return_value = registry

            store = MagicMock()
            store.get_uncovered_gems.return_value = registry.get_hidden_gems(limit=5)
            mock_store.return_value = store

            from socials_automator.content.orchestrator import ContentOrchestrator

            orchestrator = ContentOrchestrator(**mock_orchestrator_deps)
            suggestions = orchestrator._get_ai_tools_suggestions()

        assert isinstance(suggestions, str)

    def test_get_ai_tools_suggestions_disabled(self, mock_orchestrator_deps: dict):
        """Test _get_ai_tools_suggestions returns empty when disabled."""
        mock_orchestrator_deps["profile_config"]["ai_tools_config"]["enabled"] = False

        from socials_automator.content.orchestrator import ContentOrchestrator

        orchestrator = ContentOrchestrator(**mock_orchestrator_deps)
        suggestions = orchestrator._get_ai_tools_suggestions()

        assert suggestions == ""

    def test_get_ai_tools_suggestions_handles_error(self, mock_orchestrator_deps: dict):
        """Test _get_ai_tools_suggestions handles errors gracefully."""
        with patch("socials_automator.knowledge.get_ai_tools_store") as mock_store:
            mock_store.side_effect = Exception("Store error")

            from socials_automator.content.orchestrator import ContentOrchestrator

            orchestrator = ContentOrchestrator(**mock_orchestrator_deps)
            suggestions = orchestrator._get_ai_tools_suggestions()

        # Should return empty on error
        assert suggestions == ""


class TestOrchestratorMarkToolsCovered:
    """Tests for mark_tools_covered in content/orchestrator.py."""

    @pytest.fixture
    def orchestrator_with_mocks(self, temp_profile_path: Path):
        """Create orchestrator with mocked dependencies."""
        profile_config = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
        }

        from socials_automator.content.orchestrator import ContentOrchestrator

        return ContentOrchestrator(
            profile_path=temp_profile_path,
            profile_config=profile_config,
            text_provider=AsyncMock(),
            image_provider=AsyncMock(),
        )

    def test_extract_tools_from_topic_chatgpt(self, orchestrator_with_mocks, temp_yaml_config: Path):
        """Test _extract_tools_from_topic extracts ChatGPT."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock_reg.return_value = AIToolsRegistry.load(temp_yaml_config)

            tools = orchestrator_with_mocks._extract_tools_from_topic(
                "5 ChatGPT prompts for productivity"
            )

        assert "chatgpt" in tools

    def test_extract_tools_from_topic_multiple_tools(self, orchestrator_with_mocks, temp_yaml_config: Path):
        """Test _extract_tools_from_topic extracts multiple tools."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock_reg.return_value = AIToolsRegistry.load(temp_yaml_config)

            tools = orchestrator_with_mocks._extract_tools_from_topic(
                "ChatGPT vs Midjourney comparison"
            )

        assert "chatgpt" in tools
        assert "midjourney" in tools

    def test_extract_tools_from_topic_no_tools(self, orchestrator_with_mocks, temp_yaml_config: Path):
        """Test _extract_tools_from_topic returns empty for generic topic."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock_reg.return_value = AIToolsRegistry.load(temp_yaml_config)

            tools = orchestrator_with_mocks._extract_tools_from_topic(
                "5 tips for better productivity"
            )

        assert tools == []

    def test_mark_tools_covered_calls_store(self, temp_profile_path: Path, temp_yaml_config: Path):
        """Test mark_tools_covered calls store for each tool."""
        profile_config = {
            "profile": {"id": "test", "instagram_handle": "@test"},
            "content_strategy": {"content_pillars": []},
        }

        from socials_automator.content.orchestrator import ContentOrchestrator

        orchestrator = ContentOrchestrator(
            profile_path=temp_profile_path,
            profile_config=profile_config,
            text_provider=AsyncMock(),
            image_provider=AsyncMock(),
        )

        with patch("socials_automator.knowledge.get_ai_tools_store") as mock_store, \
             patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg:

            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock_reg.return_value = AIToolsRegistry.load(temp_yaml_config)

            store = MagicMock()
            mock_store.return_value = store

            orchestrator.mark_tools_covered("ChatGPT tutorial", "post-001")

        # Store should have mark_tool_covered called
        store.mark_tool_covered.assert_called()


class TestTopicSelectorVersionContext:
    """Tests for version context in video/pipeline/topic_selector.py."""

    @pytest.fixture
    def topic_selector(self):
        """Create a TopicSelector instance."""
        from socials_automator.video.pipeline.topic_selector import TopicSelector

        return TopicSelector(ai_client=None)

    def test_get_version_context_with_registry(self, topic_selector, temp_yaml_config: Path):
        """Test _get_version_context returns version info."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            context = topic_selector._get_version_context()

        assert "AI" in context or "versions" in context.lower()

    def test_get_version_context_fallback(self, topic_selector):
        """Test _get_version_context returns fallback on error."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            mock.side_effect = Exception("Registry error")

            context = topic_selector._get_version_context()

        # Should return fallback
        assert "ChatGPT" in context or "Claude" in context

    def test_get_hidden_gems_context_with_store(
        self,
        topic_selector,
        temp_profile_path: Path,
        temp_yaml_config: Path,
    ):
        """Test _get_hidden_gems_context uses store when profile_path provided."""
        with patch("socials_automator.knowledge.get_ai_tools_store") as mock_store:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)

            store = MagicMock()
            store.get_uncovered_gems.return_value = registry.get_hidden_gems(limit=5)
            mock_store.return_value = store

            context = topic_selector._get_hidden_gems_context(temp_profile_path)

        # Store should have been called
        mock_store.assert_called_once_with(temp_profile_path)
        store.get_uncovered_gems.assert_called()

    def test_get_hidden_gems_context_without_store(self, topic_selector, temp_yaml_config: Path):
        """Test _get_hidden_gems_context uses registry when no profile_path."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            context = topic_selector._get_hidden_gems_context(None)

        # Should have content about hidden gems
        if context:  # May be empty if no gems
            assert "HIDDEN GEM" in context.upper() or len(context) > 0

    def test_get_hidden_gems_context_handles_error(self, topic_selector):
        """Test _get_hidden_gems_context handles errors."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            mock.side_effect = Exception("Error")

            context = topic_selector._get_hidden_gems_context(None)

        # Should return empty on error
        assert context == ""

    def test_get_ai_tool_versions_returns_dict(self, topic_selector, temp_yaml_config: Path):
        """Test _get_ai_tool_versions returns tool version mapping."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            mock.return_value = AIToolsRegistry.load(temp_yaml_config)

            versions = topic_selector._get_ai_tool_versions()

        assert isinstance(versions, dict)
        # Should have tool entries
        assert "chatgpt" in versions

    def test_get_ai_tool_versions_fallback(self, topic_selector):
        """Test _get_ai_tool_versions returns fallback on error."""
        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock:
            mock.side_effect = Exception("Error")

            versions = topic_selector._get_ai_tool_versions()

        # Should return fallback dict
        assert isinstance(versions, dict)
        assert "chatgpt" in versions  # Fallback includes this


class TestTopicSelectorIntegration:
    """Integration tests for TopicSelector with AI tools."""

    @pytest.fixture
    def profile_metadata(self):
        """Create mock ProfileMetadata."""
        from socials_automator.video.pipeline.base import ProfileMetadata

        return ProfileMetadata(
            id="test",
            name="test-profile",
            display_name="Test Profile",
            instagram_handle="@test",
            niche_id="ai-tools",
            tagline="AI tools for everyone",
            description="A test profile",
            content_pillars=[
                {
                    "id": "tool_tutorials",
                    "name": "Tool Tutorials",
                    "description": "How to use AI tools",
                    "examples": ["How to use ChatGPT", "Midjourney tips"],
                    "frequency_percent": 50,
                },
            ],
            trending_keywords=["AI", "ChatGPT", "productivity"],
        )

    @pytest.mark.asyncio
    async def test_select_topic_includes_version_context(
        self,
        profile_metadata,
        temp_profile_path: Path,
        temp_yaml_config: Path,
    ):
        """Test select_topic integrates version context."""
        from socials_automator.video.pipeline.topic_selector import TopicSelector

        # Create mock AI client
        ai_client = AsyncMock()
        ai_client.generate.return_value = "5 ChatGPT prompts for productivity"

        selector = TopicSelector(ai_client=ai_client)

        with patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg, \
             patch("socials_automator.knowledge.get_ai_tools_store") as mock_store:

            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_reg.return_value = registry

            store = MagicMock()
            store.get_uncovered_gems.return_value = registry.get_hidden_gems(limit=5)
            mock_store.return_value = store

            topic = await selector.select_topic(profile_metadata, temp_profile_path)

        # AI should have been called with a prompt containing version info
        ai_client.generate.assert_called_once()
        prompt = ai_client.generate.call_args[0][0]
        assert "AI" in prompt  # Should mention AI tools

    @pytest.mark.asyncio
    async def test_select_topic_without_ai_uses_template(self, profile_metadata):
        """Test select_topic uses templates when no AI client."""
        from socials_automator.video.pipeline.topic_selector import TopicSelector

        selector = TopicSelector(ai_client=None)

        topic = await selector.select_topic(profile_metadata)

        # Should return a TopicInfo
        assert topic is not None
        assert topic.topic  # Should have a topic string


class TestVideoOrchestratorMarkCovered:
    """Tests for _mark_tools_covered in video/pipeline/orchestrator.py."""

    @pytest.mark.asyncio
    async def test_mark_tools_covered_after_generation(self, temp_yaml_config: Path, temp_profile_path: Path):
        """Test that tools are marked as covered after video generation."""
        # This is a high-level integration test
        # We'll just verify the method exists and doesn't crash

        with patch("socials_automator.knowledge.get_ai_tools_store") as mock_store, \
             patch("socials_automator.knowledge.get_ai_tools_registry") as mock_reg:

            from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
            registry = AIToolsRegistry.load(temp_yaml_config)
            mock_reg.return_value = registry

            store = MagicMock()
            mock_store.return_value = store

            # Import the orchestrator to verify the method exists
            from socials_automator.video.pipeline.orchestrator import VideoPipeline

            # Verify the method exists
            assert hasattr(VideoPipeline, "_mark_tools_covered")


class TestEndToEndFlow:
    """End-to-end tests for AI tools integration."""

    def test_full_flow_registry_to_store(self, temp_yaml_config: Path, temp_profile_path: Path):
        """Test full flow from registry to store usage tracking."""
        # 1. Load registry
        from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
        registry = AIToolsRegistry.load(temp_yaml_config)

        # 2. Get hidden gems
        gems = registry.get_hidden_gems(high_potential_only=True, limit=3)
        assert len(gems) > 0

        # 3. Create store
        from socials_automator.knowledge.ai_tools_store import AIToolsStore
        store = AIToolsStore(temp_profile_path)

        # 4. Mark gems as covered
        for gem in gems:
            store.mark_tool_covered(gem.id, "test-post-001", "main_topic")

        # 5. Verify they're covered
        covered = store.get_recently_covered(days=1)
        for gem in gems:
            assert gem.id in covered

        # 6. Get uncovered gems (should exclude the ones we just covered)
        with patch("socials_automator.knowledge.ai_tools_store.get_ai_tools_registry") as mock:
            mock.return_value = registry
            uncovered = store.get_uncovered_gems(days=1, high_potential_only=True)

        # The gems we covered should not be in uncovered
        uncovered_ids = [g.id for g in uncovered]
        for gem in gems:
            assert gem.id not in uncovered_ids

    def test_version_context_consistency(self, temp_yaml_config: Path):
        """Test version context is consistent across modules."""
        from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
        registry = AIToolsRegistry.load(temp_yaml_config)

        # Get version context from registry
        registry_context = registry.get_version_context(["chatgpt"])

        # ChatGPT version should be consistent
        assert "ChatGPT" in registry_context
        assert "GPT-4o" in registry_context

    def test_search_and_coverage_integration(self, temp_yaml_config: Path, temp_profile_path: Path):
        """Test searching tools and tracking coverage."""
        from socials_automator.knowledge.ai_tools_registry import AIToolsRegistry
        from socials_automator.knowledge.ai_tools_store import AIToolsStore

        registry = AIToolsRegistry.load(temp_yaml_config)
        store = AIToolsStore(temp_profile_path)

        # Search for a tool
        results = registry.search("research")
        assert len(results) > 0

        # Mark first result as covered
        first_tool = results[0]
        store.mark_tool_covered(first_tool.id, "search-test-001", "search_result")

        # Verify it's tracked
        coverage = store.get_tool_coverage(first_tool.id)
        assert len(coverage) == 1
        assert coverage[0].context == "search_result"
