"""Unit tests for research_queries.py.

Tests AI-powered research query generation for topic research.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from socials_automator.video.pipeline.research_queries import (
    ResearchQueryGenerator,
    ResearchQuery,
    generate_research_queries,
    QUERY_GENERATION_PROMPT,
    QUERY_GENERATION_SYSTEM,
)


class TestResearchQuery:
    """Tests for ResearchQuery dataclass."""

    def test_research_query_creation(self):
        """Test creating a ResearchQuery with all fields."""
        query = ResearchQuery(
            query="AI meeting tools tutorial 2025",
            language="en",
            category="tutorial",
            priority=1,
        )
        assert query.query == "AI meeting tools tutorial 2025"
        assert query.language == "en"
        assert query.category == "tutorial"
        assert query.priority == 1

    def test_research_query_languages(self):
        """Test queries with different languages."""
        en_query = ResearchQuery("test query", "en", "tutorial", 1)
        es_query = ResearchQuery("consulta de prueba", "es", "review", 2)
        pt_query = ResearchQuery("consulta de teste", "pt", "howto", 3)

        assert en_query.language == "en"
        assert es_query.language == "es"
        assert pt_query.language == "pt"

    def test_research_query_categories(self):
        """Test all valid category types."""
        categories = ["tutorial", "review", "comparison", "news", "howto"]
        for category in categories:
            query = ResearchQuery("test", "en", category, 1)
            assert query.category == category

    def test_research_query_priority_range(self):
        """Test priority values 1-3."""
        for priority in [1, 2, 3]:
            query = ResearchQuery("test", "en", "tutorial", priority)
            assert query.priority == priority


class TestResearchQueryGenerator:
    """Tests for ResearchQueryGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance for testing."""
        return ResearchQueryGenerator()

    @pytest.fixture
    def mock_text_provider(self):
        """Create a mock text provider."""
        provider = MagicMock()
        provider.generate = AsyncMock()
        return provider

    def test_generator_initialization_default(self, generator):
        """Test default initialization."""
        assert generator._text_provider is None

    def test_generator_initialization_with_provider(self, mock_text_provider):
        """Test initialization with custom provider."""
        generator = ResearchQueryGenerator(text_provider=mock_text_provider)
        assert generator._text_provider == mock_text_provider

    def test_text_provider_lazy_loading(self, generator):
        """Test that text_provider is lazy-loaded."""
        # Accessing the property should create it
        with patch(
            "socials_automator.video.pipeline.research_queries.TextProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance
            provider = generator.text_provider
            assert provider == mock_instance
            MockProvider.assert_called_once()

    def test_extract_key_terms_basic(self, generator):
        """Test key term extraction from topic."""
        topic = "This FREE AI tool creates perfect meeting notes"
        terms = generator._extract_key_terms(topic)

        # Should remove filler words like "this", "free"
        assert "this" not in terms.lower()
        assert "free" not in terms.lower()
        # Should keep key terms
        assert "ai" in terms.lower() or "tool" in terms.lower()

    def test_extract_key_terms_empty(self, generator):
        """Test key term extraction with empty input."""
        terms = generator._extract_key_terms("")
        assert terms == ""

    def test_extract_key_terms_only_fillers(self, generator):
        """Test key term extraction when input is all filler words."""
        topic = "this is the best amazing"
        terms = generator._extract_key_terms(topic)
        # All are fillers, should result in empty or minimal
        assert len(terms.split()) <= 1

    def test_extract_key_terms_max_words(self, generator):
        """Test that key terms are limited to 5 words."""
        topic = "word1 word2 word3 word4 word5 word6 word7 word8"
        terms = generator._extract_key_terms(topic)
        assert len(terms.split()) <= 5

    @pytest.mark.asyncio
    async def test_generate_queries_with_ai_success(self, mock_text_provider):
        """Test successful AI query generation."""
        mock_text_provider.generate.return_value = """[
            {"query": "AI tools tutorial", "language": "en", "category": "tutorial", "priority": 1},
            {"query": "mejores herramientas IA", "language": "es", "category": "review", "priority": 2}
        ]"""

        generator = ResearchQueryGenerator(text_provider=mock_text_provider)
        queries = await generator.generate_queries(
            topic="AI productivity tools",
            pillar="productivity_hacks",
            count=5,
        )

        assert len(queries) == 2
        assert queries[0].query == "AI tools tutorial"
        assert queries[0].language == "en"
        assert queries[1].query == "mejores herramientas IA"
        assert queries[1].language == "es"

    @pytest.mark.asyncio
    async def test_generate_queries_fallback_on_error(self, mock_text_provider):
        """Test fallback to template queries when AI fails."""
        mock_text_provider.generate.side_effect = Exception("API error")

        generator = ResearchQueryGenerator(text_provider=mock_text_provider)
        queries = await generator.generate_queries(
            topic="AI meeting tools",
            pillar="productivity_hacks",
            count=10,
        )

        # Should get fallback queries
        assert len(queries) > 0
        # Fallback should include English queries
        en_queries = [q for q in queries if q.language == "en"]
        assert len(en_queries) > 0

    @pytest.mark.asyncio
    async def test_generate_queries_fallback_on_invalid_json(self, mock_text_provider):
        """Test fallback when AI returns invalid JSON."""
        mock_text_provider.generate.return_value = "This is not JSON"

        generator = ResearchQueryGenerator(text_provider=mock_text_provider)
        queries = await generator.generate_queries(
            topic="ChatGPT tips",
            pillar="tool_tutorials",
        )

        # Should fall back to template queries
        assert len(queries) > 0

    def test_parse_response_valid_json(self, generator):
        """Test parsing valid JSON response."""
        response = """[
            {"query": "test query", "language": "en", "category": "tutorial", "priority": 1}
        ]"""
        queries = generator._parse_response(response)

        assert len(queries) == 1
        assert queries[0].query == "test query"

    def test_parse_response_with_markdown_wrapper(self, generator):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
[
    {"query": "test query", "language": "en", "category": "tutorial", "priority": 1}
]
```"""
        queries = generator._parse_response(response)

        assert len(queries) == 1

    def test_parse_response_empty_array(self, generator):
        """Test parsing empty JSON array."""
        response = "[]"
        queries = generator._parse_response(response)
        assert queries == []

    def test_parse_response_invalid_json(self, generator):
        """Test parsing invalid JSON returns empty list."""
        response = "not valid json"
        queries = generator._parse_response(response)
        assert queries == []

    def test_parse_response_missing_fields(self, generator):
        """Test parsing JSON with missing optional fields."""
        response = """[{"query": "test"}]"""
        queries = generator._parse_response(response)

        assert len(queries) == 1
        # Should use defaults for missing fields
        assert queries[0].language == "en"
        assert queries[0].category == "general"
        assert queries[0].priority == 2

    def test_generate_fallback_queries_includes_all_categories(self, generator):
        """Test that fallback queries include all categories."""
        queries = generator._generate_fallback_queries("AI tools", count=20)

        categories = set(q.category for q in queries)
        assert "tutorial" in categories
        assert "review" in categories
        assert "comparison" in categories
        assert "news" in categories
        assert "howto" in categories

    def test_generate_fallback_queries_includes_multiple_languages(self, generator):
        """Test that fallback queries include multiple languages."""
        queries = generator._generate_fallback_queries("AI tools", count=20)

        languages = set(q.language for q in queries)
        assert "en" in languages
        assert "es" in languages
        assert "pt" in languages

    def test_generate_fallback_queries_respects_count(self, generator):
        """Test that fallback respects the count limit."""
        queries = generator._generate_fallback_queries("AI tools", count=5)
        assert len(queries) <= 5

    def test_generate_fallback_queries_current_year(self, generator):
        """Test that fallback queries include current year."""
        queries = generator._generate_fallback_queries("AI tools", count=20)
        current_year = str(datetime.now().year)

        # At least some queries should mention current year
        year_queries = [q for q in queries if current_year in q.query]
        assert len(year_queries) > 0


class TestConvenienceFunction:
    """Tests for the generate_research_queries convenience function."""

    @pytest.mark.asyncio
    async def test_generate_research_queries_returns_strings(self):
        """Test that convenience function returns list of strings."""
        with patch.object(
            ResearchQueryGenerator,
            "generate_queries",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = [
                ResearchQuery("query1", "en", "tutorial", 1),
                ResearchQuery("query2", "es", "review", 2),
            ]

            result = await generate_research_queries(
                topic="test topic",
                pillar="test",
            )

            assert result == ["query1", "query2"]

    @pytest.mark.asyncio
    async def test_generate_research_queries_passes_params(self):
        """Test that convenience function passes all parameters."""
        with patch.object(
            ResearchQueryGenerator,
            "generate_queries",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_generate.return_value = []

            await generate_research_queries(
                topic="my topic",
                pillar="my_pillar",
                count=20,
            )

            mock_generate.assert_called_once_with("my topic", "my_pillar", 20)


class TestPromptTemplates:
    """Tests for prompt template content."""

    def test_query_generation_prompt_has_placeholders(self):
        """Test that prompt template has required placeholders."""
        assert "{topic}" in QUERY_GENERATION_PROMPT
        assert "{pillar}" in QUERY_GENERATION_PROMPT
        assert "{current_date}" in QUERY_GENERATION_PROMPT

    def test_query_generation_system_mentions_languages(self):
        """Test that system prompt mentions multi-language support."""
        assert "English" in QUERY_GENERATION_SYSTEM or "en" in QUERY_GENERATION_SYSTEM
        assert "Spanish" in QUERY_GENERATION_SYSTEM or "es" in QUERY_GENERATION_SYSTEM
        assert (
            "Portuguese" in QUERY_GENERATION_SYSTEM or "pt" in QUERY_GENERATION_SYSTEM
        )

    def test_query_generation_system_mentions_categories(self):
        """Test that system prompt mentions query categories."""
        assert "tutorial" in QUERY_GENERATION_SYSTEM.lower()
        assert "review" in QUERY_GENERATION_SYSTEM.lower()
        assert "comparison" in QUERY_GENERATION_SYSTEM.lower()
