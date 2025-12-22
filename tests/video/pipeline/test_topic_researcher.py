"""Unit tests for topic_researcher.py.

Tests the TopicResearcher with enhanced multi-source search.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from socials_automator.video.pipeline.topic_researcher import (
    TopicResearcher,
    SEARCH_ENGINES,
)
from socials_automator.video.pipeline.base import (
    TopicInfo,
    ResearchResult,
    PipelineContext,
    ProfileMetadata,
)
from socials_automator.video.pipeline.research_queries import ResearchQuery


class TestTopicResearcherInit:
    """Tests for TopicResearcher initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        researcher = TopicResearcher()
        assert researcher.search_client is None
        assert researcher.ai_client is None
        assert researcher.max_results == 10
        assert researcher.use_enhanced_search is True
        assert researcher._content_pillar == "general"

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        mock_search = MagicMock()
        mock_ai = MagicMock()

        researcher = TopicResearcher(
            search_client=mock_search,
            ai_client=mock_ai,
            max_results=20,
            use_enhanced_search=False,
        )

        assert researcher.search_client == mock_search
        assert researcher.ai_client == mock_ai
        assert researcher.max_results == 20
        assert researcher.use_enhanced_search is False

    def test_query_generator_lazy_loading(self):
        """Test that query_generator is lazy-loaded."""
        researcher = TopicResearcher()
        assert researcher._query_generator is None

        # Accessing property should create it
        generator = researcher.query_generator
        assert generator is not None
        assert researcher._query_generator is not None


class TestSearchEngineConfig:
    """Tests for search engine configuration."""

    def test_search_engines_defined(self):
        """Test that search engines are defined."""
        assert "duckduckgo" in SEARCH_ENGINES
        assert "bing" in SEARCH_ENGINES
        assert "yahoo" in SEARCH_ENGINES

    def test_search_engines_have_priority(self):
        """Test that search engines have priority field."""
        for engine, config in SEARCH_ENGINES.items():
            assert "priority" in config
            assert isinstance(config["priority"], int)

    def test_search_engines_have_enabled(self):
        """Test that search engines have enabled field."""
        for engine, config in SEARCH_ENGINES.items():
            assert "enabled" in config
            assert isinstance(config["enabled"], bool)


class TestWebSearch:
    """Tests for web search functionality."""

    @pytest.fixture
    def researcher(self):
        """Create a researcher for testing."""
        return TopicResearcher()

    @pytest.mark.asyncio
    async def test_web_search_with_ddgs(self, researcher):
        """Test web search using DuckDuckGo."""
        with patch("ddgs.DDGS") as MockDDGS:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.text.return_value = [
                {"title": "Test Result", "body": "Test body", "href": "https://test.com"}
            ]
            MockDDGS.return_value = mock_instance

            results = await researcher._web_search("test query")

            assert len(results) == 1
            assert results[0]["title"] == "Test Result"

    @pytest.mark.asyncio
    async def test_web_search_fallback_on_error(self, researcher):
        """Test fallback when DuckDuckGo fails."""
        with patch(
            "ddgs.DDGS",
            side_effect=Exception("API error"),
        ):
            results = await researcher._web_search("test query")

            # Should return fallback results
            assert len(results) == 1
            assert "test query" in results[0]["title"]

    def test_fallback_search(self, researcher):
        """Test fallback search returns placeholder results."""
        results = researcher._fallback_search("AI tools")

        assert len(results) == 1
        assert "AI tools" in results[0]["title"]
        assert "AI tools" in results[0]["body"]


class TestContentExtraction:
    """Tests for content extraction from search results."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    def test_extract_content_basic(self, researcher):
        """Test basic content extraction."""
        results = [
            {"title": "Title 1", "body": "Body 1"},
            {"title": "Title 2", "body": "Body 2"},
        ]

        content = researcher._extract_content(results)

        assert "Title 1" in content
        assert "Body 1" in content
        assert "Title 2" in content
        assert "Body 2" in content

    def test_extract_content_empty_results(self, researcher):
        """Test content extraction with empty results."""
        content = researcher._extract_content([])
        assert content == ""

    def test_extract_content_missing_fields(self, researcher):
        """Test content extraction with missing fields."""
        results = [
            {"title": "Only Title"},
            {"body": "Only Body"},
            {},
        ]

        content = researcher._extract_content(results)

        assert "Only Title" in content
        assert "Only Body" in content


class TestResultDeduplication:
    """Tests for result deduplication."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    def test_deduplicate_by_url(self, researcher):
        """Test deduplication removes duplicate URLs."""
        results = [
            {"title": "First", "href": "https://example.com/page1"},
            {"title": "Duplicate", "href": "https://example.com/page1"},
            {"title": "Second", "href": "https://example.com/page2"},
        ]

        unique = researcher._deduplicate_results(results)

        assert len(unique) == 2
        assert unique[0]["title"] == "First"
        assert unique[1]["title"] == "Second"

    def test_deduplicate_keeps_no_url_results(self, researcher):
        """Test that results without URLs are kept."""
        results = [
            {"title": "With URL", "href": "https://example.com"},
            {"title": "No URL 1"},
            {"title": "No URL 2"},
        ]

        unique = researcher._deduplicate_results(results)

        assert len(unique) == 3

    def test_deduplicate_empty_urls(self, researcher):
        """Test handling of empty URL strings."""
        results = [
            {"title": "Empty URL 1", "href": ""},
            {"title": "Empty URL 2", "href": ""},
            {"title": "Valid URL", "href": "https://example.com"},
        ]

        unique = researcher._deduplicate_results(results)

        # Empty URLs should all be kept (treated as different)
        assert len(unique) == 3


class TestParallelSearch:
    """Tests for parallel search functionality."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    @pytest.mark.asyncio
    async def test_parallel_search_combines_results(self, researcher):
        """Test that parallel search combines results from all queries."""
        queries = [
            ResearchQuery("query1", "en", "tutorial", 1),
            ResearchQuery("query2", "es", "review", 2),
        ]

        with patch.object(
            researcher, "_web_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = [
                [{"title": "Result 1", "href": "url1"}],
                [{"title": "Result 2", "href": "url2"}],
            ]

            results = await researcher._parallel_search(queries)

            assert len(results) == 2
            assert mock_search.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_search_handles_failures(self, researcher):
        """Test that parallel search handles individual query failures."""
        queries = [
            ResearchQuery("query1", "en", "tutorial", 1),
            ResearchQuery("query2", "es", "review", 2),
        ]

        with patch.object(
            researcher, "_web_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = [
                [{"title": "Result 1", "href": "url1"}],
                Exception("Search failed"),
            ]

            results = await researcher._parallel_search(queries)

            # Should still return results from successful query
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_parallel_search_tags_results(self, researcher):
        """Test that parallel search tags results with query metadata."""
        queries = [
            ResearchQuery("query1", "en", "tutorial", 1),
        ]

        with patch.object(
            researcher, "_web_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = [{"title": "Result", "href": "url"}]

            results = await researcher._parallel_search(queries)

            assert results[0]["_query_language"] == "en"
            assert results[0]["_query_category"] == "tutorial"


class TestEnhancedResearch:
    """Tests for enhanced research mode."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher(use_enhanced_search=True)

    @pytest.fixture
    def topic_info(self):
        return TopicInfo(
            topic="AI meeting notes tools",
            pillar_id="productivity_hacks",
            pillar_name="Productivity Hacks",
            search_queries=["AI meeting tools"],
        )

    @pytest.mark.asyncio
    async def test_enhanced_research_uses_ai_queries(self, researcher, topic_info):
        """Test that enhanced research generates AI queries."""
        with patch.object(
            researcher.query_generator,
            "generate_queries",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = [
                ResearchQuery("AI query", "en", "tutorial", 1)
            ]

            with patch.object(
                researcher, "_parallel_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = []

                with patch.object(
                    researcher, "_web_search", new_callable=AsyncMock
                ) as mock_version_search:
                    mock_version_search.return_value = []

                    with patch.object(
                        researcher, "_process_results", new_callable=AsyncMock
                    ) as mock_process:
                        mock_process.return_value = ResearchResult(
                            topic="test",
                            summary="test",
                            key_points=[],
                            sources=[],
                        )

                        await researcher._enhanced_research(topic_info)

                        mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_research_includes_version_context(
        self, researcher, topic_info
    ):
        """Test that enhanced research adds AI version context search."""
        with patch.object(
            researcher.query_generator,
            "generate_queries",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = []

            with patch.object(
                researcher, "_parallel_search", new_callable=AsyncMock
            ) as mock_parallel:
                mock_parallel.return_value = []

                with patch.object(
                    researcher, "_web_search", new_callable=AsyncMock
                ) as mock_search:
                    mock_search.return_value = []

                    with patch.object(
                        researcher, "_process_results", new_callable=AsyncMock
                    ) as mock_process:
                        mock_process.return_value = ResearchResult(
                            topic="test",
                            summary="test",
                            key_points=[],
                            sources=[],
                        )

                        await researcher._enhanced_research(topic_info)

                        # Should have called _web_search for version context
                        mock_search.assert_called_once()
                        query = mock_search.call_args[0][0]
                        assert "ChatGPT" in query or "Claude" in query


class TestBasicResearch:
    """Tests for basic research mode."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher(use_enhanced_search=False)

    @pytest.fixture
    def topic_info(self):
        return TopicInfo(
            topic="AI tools",
            pillar_id="general",
            pillar_name="General",
            search_queries=["query1", "query2", "query3", "query4"],
        )

    @pytest.mark.asyncio
    async def test_basic_research_uses_topic_queries(self, researcher, topic_info):
        """Test that basic research uses topic's search queries."""
        with patch.object(
            researcher, "_web_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            with patch.object(
                researcher, "_process_results", new_callable=AsyncMock
            ) as mock_process:
                mock_process.return_value = ResearchResult(
                    topic="test",
                    summary="test",
                    key_points=[],
                    sources=[],
                )

                await researcher._basic_research(topic_info)

                # Should use first 3 queries from topic + version query
                assert mock_search.call_count == 4

    @pytest.mark.asyncio
    async def test_research_delegates_based_on_mode(self, topic_info):
        """Test that research() delegates based on use_enhanced_search."""
        # Enhanced mode
        enhanced_researcher = TopicResearcher(use_enhanced_search=True)
        with patch.object(
            enhanced_researcher, "_enhanced_research", new_callable=AsyncMock
        ) as mock_enhanced:
            mock_enhanced.return_value = ResearchResult(
                topic="test", summary="", key_points=[], sources=[]
            )
            await enhanced_researcher.research(topic_info)
            mock_enhanced.assert_called_once()

        # Basic mode
        basic_researcher = TopicResearcher(use_enhanced_search=False)
        with patch.object(
            basic_researcher, "_basic_research", new_callable=AsyncMock
        ) as mock_basic:
            mock_basic.return_value = ResearchResult(
                topic="test", summary="", key_points=[], sources=[]
            )
            await basic_researcher.research(topic_info)
            mock_basic.assert_called_once()


class TestSummarization:
    """Tests for content summarization."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    def test_simple_summarize_extracts_sentences(self, researcher):
        """Test simple summarization extracts sentences."""
        content = "This is sentence one. This is sentence two. This is sentence three."
        summary, key_points = researcher._simple_summarize("test topic", content)

        assert len(summary) > 0
        assert "sentence" in summary.lower()

    def test_simple_summarize_extracts_key_points(self, researcher):
        """Test simple summarization extracts key points with keywords."""
        content = (
            "This is a tip for productivity. "
            "Here is why you should use AI. "
            "The best approach is automation. "
            "You can improve workflow. "
            "This will save time."
        )
        summary, key_points = researcher._simple_summarize("test topic", content)

        assert len(key_points) > 0

    def test_simple_summarize_fallback_key_points(self, researcher):
        """Test that simple summarization provides fallback key points."""
        content = "Short content without keywords"
        summary, key_points = researcher._simple_summarize("AI tools", content)

        # Should have fallback key points
        assert len(key_points) == 5
        assert any("AI tools" in point for point in key_points)

    def test_simple_summarize_empty_content(self, researcher):
        """Test simple summarization with empty content."""
        summary, key_points = researcher._simple_summarize("test topic", "")

        assert "test topic" in summary.lower()
        assert len(key_points) == 5


class TestProcessResults:
    """Tests for result processing."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    @pytest.mark.asyncio
    async def test_process_results_creates_research_result(self, researcher):
        """Test that _process_results creates ResearchResult."""
        results = [
            {"title": "Title 1", "body": "Body 1", "href": "https://example.com/1"},
            {"title": "Title 2", "body": "Body 2", "href": "https://example.com/2"},
        ]

        result = await researcher._process_results("test topic", results)

        assert isinstance(result, ResearchResult)
        assert result.topic == "test topic"
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_process_results_limits_sources(self, researcher):
        """Test that _process_results limits sources to 5."""
        results = [
            {"title": f"Title {i}", "body": f"Body {i}", "href": f"url{i}"}
            for i in range(10)
        ]

        result = await researcher._process_results("test", results)

        assert len(result.sources) <= 5

    @pytest.mark.asyncio
    async def test_process_results_limits_raw_content(self, researcher):
        """Test that _process_results limits raw_content size."""
        results = [
            {"title": "T", "body": "B" * 10000}  # Large body
        ]

        result = await researcher._process_results("test", results)

        assert len(result.raw_content) <= 5000


class TestPipelineExecution:
    """Tests for pipeline execution integration."""

    @pytest.fixture
    def researcher(self):
        return TopicResearcher()

    @pytest.fixture
    def context(self, tmp_path):
        """Create a pipeline context for testing."""
        profile = ProfileMetadata(
            id="test",
            name="test",
            display_name="Test Profile",
            instagram_handle="@test",
            niche_id="test",
            tagline="Test tagline",
            description="Test description",
            language="en",
            content_pillars=[],
            hashtags=[],
        )
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        output_dir.mkdir()
        temp_dir.mkdir()
        return PipelineContext(
            profile=profile,
            post_id="test-post",
            output_dir=output_dir,
            temp_dir=temp_dir,
            topic=TopicInfo(
                topic="Test Topic",
                pillar_id="test_pillar",
                pillar_name="Test Pillar",
                search_queries=["test query"],
            ),
        )

    @pytest.mark.asyncio
    async def test_execute_captures_pillar_from_topic(self, researcher, context):
        """Test that execute captures pillar_id from topic."""
        with patch.object(
            researcher, "research", new_callable=AsyncMock
        ) as mock_research:
            mock_research.return_value = ResearchResult(
                topic="test",
                summary="test",
                key_points=["point1"],
                sources=[],
            )

            await researcher.execute(context)

            assert researcher._content_pillar == "test_pillar"

    @pytest.mark.asyncio
    async def test_execute_updates_context_with_research(self, researcher, context):
        """Test that execute updates context with research results."""
        expected_result = ResearchResult(
            topic="test",
            summary="test summary",
            key_points=["point1", "point2"],
            sources=[{"title": "Source", "url": "https://example.com"}],
        )

        with patch.object(
            researcher, "research", new_callable=AsyncMock
        ) as mock_research:
            mock_research.return_value = expected_result

            result_context = await researcher.execute(context)

            assert result_context.research == expected_result
            assert len(result_context.research.key_points) == 2
