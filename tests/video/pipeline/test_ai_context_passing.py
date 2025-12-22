"""Tests for AI context passing across pipeline components.

Verifies that version context, uniqueness constraints, and profile data
are correctly passed to each AI call in the video generation pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
from datetime import datetime

from socials_automator.video.pipeline.base import (
    PipelineContext,
    ProfileMetadata,
    TopicInfo,
    ResearchResult,
    VideoScript,
    VideoSegment,
)


class TestVersionContextPassing:
    """Tests that version context is passed to all AI components."""

    def test_research_queries_prompt_includes_version_context(self):
        """Test that QUERY_GENERATION_PROMPT has version_context placeholder."""
        from socials_automator.video.pipeline.research_queries import (
            QUERY_GENERATION_PROMPT,
        )
        assert "{version_context}" in QUERY_GENERATION_PROMPT

    def test_research_queries_get_version_context_exists(self):
        """Test that get_ai_version_context function exists and works."""
        from socials_automator.video.pipeline.research_queries import (
            get_ai_version_context,
        )
        context = get_ai_version_context()
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain AI tool names
        assert "ChatGPT" in context or "Claude" in context or "Gemini" in context

    def test_script_planner_get_version_context_exists(self):
        """Test that script_planner has get_ai_version_context function."""
        from socials_automator.video.pipeline.script_planner import (
            get_ai_version_context,
        )
        context = get_ai_version_context()
        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_research_query_generator_includes_version_in_prompt(self):
        """Test that ResearchQueryGenerator includes version context in AI call."""
        from socials_automator.video.pipeline.research_queries import (
            ResearchQueryGenerator,
        )

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value='[{"query": "test", "language": "en", "category": "tutorial", "priority": 1}]')

        generator = ResearchQueryGenerator(text_provider=mock_provider)
        await generator.generate_queries(topic="ChatGPT tips", pillar="tutorials")

        # Verify generate was called
        mock_provider.generate.assert_called_once()

        # Get the prompt that was passed
        call_kwargs = mock_provider.generate.call_args
        prompt = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]

        # Verify version context is in the prompt
        assert "ChatGPT" in prompt or "GPT" in prompt or "version" in prompt.lower()


class TestTopicSelectorContext:
    """Tests for TopicSelector context passing."""

    @pytest.fixture
    def mock_profile(self):
        """Create a mock profile for testing."""
        return ProfileMetadata(
            id="test",
            name="test_profile",
            display_name="Test Profile",
            instagram_handle="@test",
            niche_id="ai_tools",
            tagline="Test tagline",
            description="Test description",
            content_pillars=[
                {"id": "tutorials", "name": "Tool Tutorials", "examples": ["Example 1"]}
            ],
            trending_keywords=["AI", "ChatGPT", "productivity"],
        )

    @pytest.mark.asyncio
    async def test_topic_selector_includes_version_context(self, mock_profile, tmp_path):
        """Test that TopicSelector includes version context in AI prompt."""
        from socials_automator.video.pipeline.topic_selector import TopicSelector

        mock_ai_client = MagicMock()
        mock_ai_client.generate = AsyncMock(return_value="Test AI topic for 2025")

        selector = TopicSelector(ai_client=mock_ai_client)
        topic = await selector.select_topic(mock_profile, tmp_path)

        # Verify AI was called
        mock_ai_client.generate.assert_called_once()

        # Get the prompt
        call_kwargs = mock_ai_client.generate.call_args.kwargs
        prompt = call_kwargs.get("prompt", "")

        # Verify version context elements are present
        assert "ChatGPT" in prompt or "Claude" in prompt or "version" in prompt.lower()

    @pytest.mark.asyncio
    async def test_topic_selector_includes_topic_history(self, mock_profile, tmp_path):
        """Test that TopicSelector includes topic history for uniqueness."""
        from socials_automator.video.pipeline.topic_selector import (
            TopicSelector,
            save_topic_to_history,
        )

        # Save some topics to history
        save_topic_to_history(tmp_path, "Previous topic about ChatGPT")
        save_topic_to_history(tmp_path, "Another topic about Claude")

        mock_ai_client = MagicMock()
        mock_ai_client.generate = AsyncMock(return_value="New unique topic")

        selector = TopicSelector(ai_client=mock_ai_client)
        await selector.select_topic(mock_profile, tmp_path)

        # Get the prompt
        call_kwargs = mock_ai_client.generate.call_args.kwargs
        prompt = call_kwargs.get("prompt", "")

        # Verify history context mentions avoiding repetition
        assert "RECENTLY USED" in prompt or "avoid" in prompt.lower() or "repeat" in prompt.lower()


class TestScriptPlannerContext:
    """Tests for ScriptPlanner context passing."""

    @pytest.fixture
    def mock_topic(self):
        """Create a mock topic."""
        return TopicInfo(
            topic="ChatGPT GPT-5.2 productivity tips",
            pillar_id="tutorials",
            pillar_name="Tool Tutorials",
            keywords=["ChatGPT", "productivity"],
            search_queries=["ChatGPT tips 2025"],
        )

    @pytest.fixture
    def mock_research(self):
        """Create mock research results."""
        return ResearchResult(
            topic="ChatGPT GPT-5.2 productivity tips",
            summary="Summary of research",
            key_points=["Point 1", "Point 2", "Point 3"],
            sources=[],
            raw_content="Raw content here",
        )

    @pytest.fixture
    def mock_profile(self):
        """Create a mock profile."""
        return ProfileMetadata(
            id="test",
            name="test_profile",
            display_name="Test Profile",
            instagram_handle="@test",
            niche_id="ai_tools",
            tagline="Test tagline",
            description="Test description",
        )

    def test_script_planner_has_session_constraints_method(self):
        """Test that ScriptPlanner has _get_session_constraints method."""
        from socials_automator.video.pipeline.script_planner import ScriptPlanner

        planner = ScriptPlanner()
        assert hasattr(planner, "_get_session_constraints")

        # Call without session should return empty string
        constraints = planner._get_session_constraints()
        assert constraints == ""

    def test_script_planner_session_constraints_with_session(self):
        """Test that ScriptPlanner returns constraints when session is provided."""
        from socials_automator.video.pipeline.script_planner import ScriptPlanner

        # Create mock session
        mock_session = MagicMock()
        mock_session.get_constraints.return_value = [
            "Avoid statement hooks (used 8 times recently)",
            "Avoid topic: ChatGPT tips",
        ]

        planner = ScriptPlanner(session=mock_session)
        constraints = planner._get_session_constraints()

        assert "UNIQUENESS CONSTRAINTS" in constraints
        assert "statement hooks" in constraints or "Avoid" in constraints


class TestOrchestratorContextPassing:
    """Tests that orchestrator passes context to all components."""

    def test_orchestrator_passes_ai_client_to_topic_researcher(self):
        """Test that orchestrator passes ai_client to TopicResearcher."""
        from socials_automator.video.pipeline.orchestrator import VideoPipeline

        with patch("socials_automator.video.pipeline.orchestrator.TopicResearcher") as MockResearcher:
            with patch("socials_automator.video.pipeline.orchestrator.TopicSelector"):
                with patch("socials_automator.video.pipeline.orchestrator.ScriptPlanner"):
                    with patch("socials_automator.video.pipeline.orchestrator.VoiceGenerator"):
                        with patch("socials_automator.video.pipeline.orchestrator.VideoSearcher"):
                            with patch("socials_automator.video.pipeline.orchestrator.VideoDownloader"):
                                with patch("socials_automator.video.pipeline.orchestrator.VideoAssembler"):
                                    with patch("socials_automator.video.pipeline.orchestrator.ThumbnailGenerator"):
                                        with patch("socials_automator.video.pipeline.orchestrator.SubtitleRenderer"):
                                            with patch("socials_automator.video.pipeline.orchestrator.CaptionGenerator"):
                                                orchestrator = VideoPipeline(text_ai="lmstudio")

            # Verify TopicResearcher was called with ai_client
            MockResearcher.assert_called_once()
            call_kwargs = MockResearcher.call_args.kwargs
            assert "ai_client" in call_kwargs
            assert call_kwargs["ai_client"] is not None

    def test_orchestrator_passes_preferred_provider_to_script_planner(self):
        """Test that orchestrator passes preferred_provider to ScriptPlanner."""
        from socials_automator.video.pipeline.orchestrator import VideoPipeline

        with patch("socials_automator.video.pipeline.orchestrator.TopicResearcher"):
            with patch("socials_automator.video.pipeline.orchestrator.TopicSelector"):
                with patch("socials_automator.video.pipeline.orchestrator.ScriptPlanner") as MockPlanner:
                    with patch("socials_automator.video.pipeline.orchestrator.VoiceGenerator"):
                        with patch("socials_automator.video.pipeline.orchestrator.VideoSearcher"):
                            with patch("socials_automator.video.pipeline.orchestrator.VideoDownloader"):
                                with patch("socials_automator.video.pipeline.orchestrator.VideoAssembler"):
                                    with patch("socials_automator.video.pipeline.orchestrator.ThumbnailGenerator"):
                                        with patch("socials_automator.video.pipeline.orchestrator.SubtitleRenderer"):
                                            with patch("socials_automator.video.pipeline.orchestrator.CaptionGenerator"):
                                                orchestrator = VideoPipeline(text_ai="lmstudio")

            # Verify ScriptPlanner was called with preferred_provider
            MockPlanner.assert_called_once()
            call_kwargs = MockPlanner.call_args.kwargs
            assert "preferred_provider" in call_kwargs
            assert call_kwargs["preferred_provider"] == "lmstudio"

    def test_orchestrator_passes_preferred_provider_to_caption_generator(self):
        """Test that orchestrator passes preferred_provider to CaptionGenerator."""
        from socials_automator.video.pipeline.orchestrator import VideoPipeline

        with patch("socials_automator.video.pipeline.orchestrator.TopicResearcher"):
            with patch("socials_automator.video.pipeline.orchestrator.TopicSelector"):
                with patch("socials_automator.video.pipeline.orchestrator.ScriptPlanner"):
                    with patch("socials_automator.video.pipeline.orchestrator.VoiceGenerator"):
                        with patch("socials_automator.video.pipeline.orchestrator.VideoSearcher"):
                            with patch("socials_automator.video.pipeline.orchestrator.VideoDownloader"):
                                with patch("socials_automator.video.pipeline.orchestrator.VideoAssembler"):
                                    with patch("socials_automator.video.pipeline.orchestrator.ThumbnailGenerator"):
                                        with patch("socials_automator.video.pipeline.orchestrator.SubtitleRenderer"):
                                            with patch("socials_automator.video.pipeline.orchestrator.CaptionGenerator") as MockCaption:
                                                orchestrator = VideoPipeline(text_ai="lmstudio")

            # Verify CaptionGenerator was called with preferred_provider
            MockCaption.assert_called_once()
            call_kwargs = MockCaption.call_args.kwargs
            assert "preferred_provider" in call_kwargs
            assert call_kwargs["preferred_provider"] == "lmstudio"


class TestTopicResearcherContextPassing:
    """Tests that TopicResearcher passes ai_client to ResearchQueryGenerator."""

    def test_topic_researcher_passes_ai_client_to_query_generator(self):
        """Test that TopicResearcher passes ai_client to ResearchQueryGenerator."""
        from socials_automator.video.pipeline.topic_researcher import TopicResearcher

        mock_ai_client = MagicMock()
        researcher = TopicResearcher(ai_client=mock_ai_client)

        # Access the query_generator property (lazy-loaded)
        query_gen = researcher.query_generator

        # Verify the query generator received the ai_client as text_provider
        assert query_gen._text_provider == mock_ai_client


class TestCaptionGeneratorContext:
    """Tests for CaptionGenerator context."""

    def test_caption_generator_has_preferred_provider(self):
        """Test that CaptionGenerator accepts preferred_provider."""
        from socials_automator.video.pipeline.caption_generator import CaptionGenerator

        generator = CaptionGenerator(preferred_provider="lmstudio")
        assert generator._preferred_provider == "lmstudio"

    def test_caption_generator_prompt_includes_profile_context(self):
        """Test that CaptionGenerator prompt includes profile information."""
        from socials_automator.video.pipeline.caption_generator import CaptionGenerator
        from socials_automator.video.pipeline.base import (
            PipelineContext,
            ProfileMetadata,
            TopicInfo,
            VideoScript,
        )

        generator = CaptionGenerator()

        # Create minimal context
        profile = ProfileMetadata(
            id="test",
            name="ai_for_mortals",
            display_name="AI For Mortals",
            instagram_handle="@ai.for.mortals",
            niche_id="ai_tools",
            tagline="Test",
            description="Test",
        )

        topic = TopicInfo(
            topic="ChatGPT productivity tips",
            pillar_id="tutorials",
            pillar_name="Tutorials",
        )

        script = VideoScript(
            title="Test",
            hook="Test hook",
            cta="Follow us",
            full_narration="Test narration about ChatGPT and Claude AI tools.",
        )

        context = MagicMock()
        context.profile = profile
        context.topic = topic
        context.script = script

        prompt = generator._build_caption_prompt(context)

        # Verify profile information is included
        assert "@ai.for.mortals" in prompt
        assert "AI For Mortals" in prompt or "ai_for_mortals" in prompt


class TestPromptTemplateValidation:
    """Tests that all prompt templates have required placeholders."""

    def test_research_query_prompt_has_all_placeholders(self):
        """Test QUERY_GENERATION_PROMPT has all required placeholders."""
        from socials_automator.video.pipeline.research_queries import (
            QUERY_GENERATION_PROMPT,
        )

        required_placeholders = ["{topic}", "{pillar}", "{current_date}", "{version_context}"]
        for placeholder in required_placeholders:
            assert placeholder in QUERY_GENERATION_PROMPT, f"Missing {placeholder}"

    def test_research_query_prompt_mentions_version_usage(self):
        """Test that prompt instructs to use correct version numbers."""
        from socials_automator.video.pipeline.research_queries import (
            QUERY_GENERATION_PROMPT,
        )

        # Should mention using correct versions
        assert "version" in QUERY_GENERATION_PROMPT.lower()


class TestIntegrationVersionContext:
    """Integration tests for version context flow."""

    @pytest.mark.asyncio
    async def test_full_version_context_flow(self):
        """Test that version context flows through the pipeline correctly."""
        from socials_automator.video.pipeline.research_queries import (
            get_ai_version_context,
            ResearchQueryGenerator,
        )

        # Get version context
        version_context = get_ai_version_context()

        # Verify it contains expected tools
        expected_tools = ["ChatGPT", "Claude", "Gemini"]
        found_tools = [tool for tool in expected_tools if tool in version_context]
        assert len(found_tools) >= 2, f"Version context should mention AI tools: {version_context}"

        # Verify it contains version-like patterns (e.g., "GPT-5.2", "Opus 4.5")
        import re
        version_pattern = r'\d+\.\d+|\d{4}'  # Matches "5.2" or "2025"
        versions_found = re.findall(version_pattern, version_context)
        assert len(versions_found) > 0, "Version context should contain version numbers"
