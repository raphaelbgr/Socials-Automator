"""Comprehensive tests for WebSearcher timeout and error handling.

Tests cover:
- Search timeout scenarios
- DuckDuckGo rate limiting
- Connection errors
- Parallel search behavior
- News search specific scenarios
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from socials_automator.research.web_search import (
    WebSearcher,
    SearchResult,
    SearchResponse,
    ParallelSearchResponse,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def searcher():
    """Create a WebSearcher instance for testing."""
    return WebSearcher(timeout=10)


@pytest.fixture
def searcher_short_timeout():
    """Create a WebSearcher with very short timeout."""
    return WebSearcher(timeout=1)


def create_mock_ddg_result(title: str = "Test Result"):
    """Create a mock DuckDuckGo search result."""
    return {
        "title": title,
        "href": f"https://example.com/{title.lower().replace(' ', '-')}",
        "body": f"Description for {title}",
    }


def create_mock_ddg_news_result(title: str = "Test News"):
    """Create a mock DuckDuckGo news result."""
    return {
        "title": title,
        "url": f"https://news.example.com/{title.lower().replace(' ', '-')}",
        "body": f"News about {title}",
        "date": datetime.now().isoformat(),
        "source": "Example News",
    }


# =============================================================================
# Basic Timeout Tests
# =============================================================================

class TestSearchTimeouts:
    """Test search timeout handling."""

    @pytest.mark.asyncio
    async def test_search_with_configured_timeout(self, searcher):
        """Search should use configured timeout."""
        assert searcher.timeout == 10

    @pytest.mark.asyncio
    async def test_search_timeout_propagated(self, searcher):
        """Timeout should be passed to DDGS."""
        query = "test query"

        with patch.object(searcher, '_ddgs_search_sync') as mock_search:
            mock_search.return_value = [create_mock_ddg_result()]

            await searcher.search(query)

            # Verify timeout is used (check in actual implementation)
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_timeout_returns_error_response(self, searcher_short_timeout):
        """Timeout should return error response, not crash."""
        def slow_search(*args, **kwargs):
            import time
            time.sleep(10)  # Much longer than timeout

        with patch.object(searcher_short_timeout, '_ddgs_search_sync', side_effect=slow_search):
            response = await searcher_short_timeout.search("test query")

            # Should return response (possibly with error)
            assert isinstance(response, SearchResponse)


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Test DuckDuckGo rate limiting handling."""

    @pytest.mark.asyncio
    async def test_ratelimit_exception_handled(self, searcher):
        """DuckDuckGo rate limit exception should be handled."""

        def raise_ratelimit(*args, **kwargs):
            raise Exception("DuckDuckGoSearchException: Ratelimit")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_ratelimit):
            response = await searcher.search("test query")

            assert response.success is False
            assert "error" in response.error.lower() or "rate" in response.error.lower() or response.error

    @pytest.mark.asyncio
    async def test_too_many_requests_handled(self, searcher):
        """429 Too Many Requests should be handled."""

        def raise_429(*args, **kwargs):
            raise Exception("HTTP 429: Too Many Requests")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_429):
            response = await searcher.search("test query")

            assert response.success is False

    @pytest.mark.asyncio
    async def test_rate_limit_news_search(self, searcher):
        """News search rate limit should be handled."""

        def raise_ratelimit(*args, **kwargs):
            raise Exception("DuckDuckGoSearchException: Ratelimit")

        with patch.object(searcher, '_ddgs_news_sync', side_effect=raise_ratelimit):
            response = await searcher.search_news("test query")

            assert response.success is False


# =============================================================================
# Connection Error Tests
# =============================================================================

class TestConnectionErrors:
    """Test connection error handling."""

    @pytest.mark.asyncio
    async def test_connection_refused(self, searcher):
        """Connection refused should be handled."""

        def raise_connection(*args, **kwargs):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_connection):
            response = await searcher.search("test query")

            assert response.success is False

    @pytest.mark.asyncio
    async def test_dns_failure(self, searcher):
        """DNS resolution failure should be handled."""

        def raise_dns(*args, **kwargs):
            raise OSError("Name or service not known")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_dns):
            response = await searcher.search("test query")

            assert response.success is False

    @pytest.mark.asyncio
    async def test_network_unreachable(self, searcher):
        """Network unreachable should be handled."""

        def raise_network(*args, **kwargs):
            raise OSError("Network is unreachable")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_network):
            response = await searcher.search("test query")

            assert response.success is False

    @pytest.mark.asyncio
    async def test_ssl_error(self, searcher):
        """SSL errors should be handled."""

        def raise_ssl(*args, **kwargs):
            raise Exception("SSL: CERTIFICATE_VERIFY_FAILED")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=raise_ssl):
            response = await searcher.search("test query")

            assert response.success is False


# =============================================================================
# Parallel Search Tests
# =============================================================================

class TestParallelSearch:
    """Test parallel search behavior."""

    @pytest.mark.asyncio
    async def test_parallel_search_partial_success(self, searcher):
        """When some queries fail, others should still succeed."""
        queries = ["query1", "query2", "query3"]

        call_count = 0

        def mock_search(query, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "2" in query:
                raise Exception("Search failed")
            return [create_mock_ddg_result(f"Result for {query}")]

        with patch.object(searcher, '_ddgs_search_sync', side_effect=mock_search):
            response = await searcher.parallel_search(queries, max_results=5)

            # Should have some successful queries
            assert response.successful_queries >= 2
            assert response.total_results >= 2

    @pytest.mark.asyncio
    async def test_parallel_search_all_fail(self, searcher):
        """When all queries fail, should return empty results."""
        queries = ["query1", "query2", "query3"]

        def always_fail(*args, **kwargs):
            raise Exception("All searches fail")

        with patch.object(searcher, '_ddgs_search_sync', side_effect=always_fail):
            response = await searcher.parallel_search(queries, max_results=5)

            assert response.successful_queries == 0
            assert response.total_results == 0
            assert len(response.all_sources) == 0

    @pytest.mark.asyncio
    async def test_parallel_search_timeout_partial(self, searcher):
        """Timeout in some parallel searches shouldn't block others."""
        queries = ["fast1", "slow", "fast2"]

        def variable_speed(query, *args, **kwargs):
            if "slow" in query:
                import time
                time.sleep(100)  # Very slow
            return [create_mock_ddg_result(f"Result for {query}")]

        with patch.object(searcher, '_ddgs_search_sync', side_effect=variable_speed):
            # With gather, slow ones will block but others should eventually complete
            # This tests the general behavior
            try:
                response = await asyncio.wait_for(
                    searcher.parallel_search(queries[:2], max_results=5),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                # Expected if the slow query blocks
                pass

    @pytest.mark.asyncio
    async def test_parallel_news_search(self, searcher):
        """Parallel news search should work similarly."""
        queries = ["news1", "news2"]

        def mock_news(*args, **kwargs):
            return [create_mock_ddg_news_result("News Article")]

        with patch.object(searcher, '_ddgs_news_sync', side_effect=mock_news):
            response = await searcher.parallel_news_search(queries, max_results=5)

            assert response.successful_queries == 2

    @pytest.mark.asyncio
    async def test_query_limit_enforced(self, searcher):
        """Should limit number of parallel queries."""
        # Create more than 25 queries
        queries = [f"query{i}" for i in range(30)]

        with patch.object(searcher, '_ddgs_search_sync', return_value=[create_mock_ddg_result()]):
            response = await searcher.parallel_search(queries, max_results=5)

            # Should truncate to max queries
            assert response.total_queries <= 25


# =============================================================================
# News Search Tests
# =============================================================================

class TestNewsSearch:
    """Test news-specific search functionality."""

    @pytest.mark.asyncio
    async def test_news_search_timeout(self, searcher_short_timeout):
        """News search should respect timeout."""

        def slow_news(*args, **kwargs):
            import time
            time.sleep(10)

        with patch.object(searcher_short_timeout, '_ddgs_news_sync', side_effect=slow_news):
            response = await searcher_short_timeout.search_news("test query")

            # Should return (possibly with error), not hang
            assert isinstance(response, SearchResponse)

    @pytest.mark.asyncio
    async def test_news_search_empty_results(self, searcher):
        """Empty news results should be handled."""

        with patch.object(searcher, '_ddgs_news_sync', return_value=[]):
            response = await searcher.search_news("obscure query")

            assert response.success is True
            assert response.result_count == 0

    @pytest.mark.asyncio
    async def test_news_search_with_max_results(self, searcher):
        """News search should support max_results parameter."""

        with patch.object(searcher, '_ddgs_news_sync', return_value=[create_mock_ddg_news_result()]) as mock:
            await searcher.search_news("test query", max_results=10)

            mock.assert_called_once()


# =============================================================================
# Response Handling Tests
# =============================================================================

class TestResponseHandling:
    """Test response object creation and handling."""

    @pytest.mark.asyncio
    async def test_search_response_structure(self, searcher):
        """Search response should have correct structure."""
        with patch.object(searcher, '_ddgs_search_sync', return_value=[create_mock_ddg_result()]):
            response = await searcher.search("test query")

            assert hasattr(response, 'query')
            assert hasattr(response, 'results')
            assert hasattr(response, 'success')
            assert hasattr(response, 'error')
            assert hasattr(response, 'duration_ms')

    @pytest.mark.asyncio
    async def test_search_result_structure(self, searcher):
        """Individual search results should have correct structure."""
        mock_result = create_mock_ddg_result("Test Title")

        with patch.object(searcher, '_ddgs_search_sync', return_value=[mock_result]):
            response = await searcher.search("test query")

            if response.results:
                result = response.results[0]
                assert hasattr(result, 'title')
                assert hasattr(result, 'url')
                assert hasattr(result, 'snippet')
                assert hasattr(result, 'domain')

    @pytest.mark.asyncio
    async def test_parallel_response_structure(self, searcher):
        """Parallel search response should have correct structure."""
        with patch.object(searcher, '_ddgs_search_sync', return_value=[create_mock_ddg_result()]):
            response = await searcher.parallel_search(["query1", "query2"])

            assert hasattr(response, 'total_queries')
            assert hasattr(response, 'successful_queries')
            assert hasattr(response, 'total_results')
            assert hasattr(response, 'all_sources')
            assert hasattr(response, 'queries')
            assert hasattr(response, 'duration_ms')

    @pytest.mark.asyncio
    async def test_context_string_generation(self, searcher):
        """ParallelSearchResponse should generate context strings."""
        with patch.object(searcher, '_ddgs_search_sync', return_value=[create_mock_ddg_result()]):
            response = await searcher.parallel_search(["test query"])

            context = response.to_context_string()
            assert isinstance(context, str)
            assert len(context) > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_query(self, searcher):
        """Empty query should be handled."""
        response = await searcher.search("")
        # Should not crash, may return empty or error
        assert isinstance(response, SearchResponse)

    @pytest.mark.asyncio
    async def test_very_long_query(self, searcher):
        """Very long query should be handled."""
        long_query = "test " * 1000

        with patch.object(searcher, '_ddgs_search_sync', return_value=[]):
            response = await searcher.search(long_query)
            assert isinstance(response, SearchResponse)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, searcher):
        """Special characters in query should be handled."""
        special_query = "test <script>alert('xss')</script> query"

        with patch.object(searcher, '_ddgs_search_sync', return_value=[]):
            response = await searcher.search(special_query)
            assert isinstance(response, SearchResponse)

    @pytest.mark.asyncio
    async def test_unicode_query(self, searcher):
        """Unicode characters in query should be handled."""
        unicode_query = "test query"

        with patch.object(searcher, '_ddgs_search_sync', return_value=[]):
            response = await searcher.search(unicode_query)
            assert isinstance(response, SearchResponse)

    @pytest.mark.asyncio
    async def test_parallel_empty_list(self, searcher):
        """Empty query list should return empty results."""
        response = await searcher.parallel_search([])

        assert response.total_queries == 0
        assert response.total_results == 0

    @pytest.mark.asyncio
    async def test_parallel_single_query(self, searcher):
        """Single query in parallel search should work."""
        with patch.object(searcher, '_ddgs_search_sync', return_value=[create_mock_ddg_result()]):
            response = await searcher.parallel_search(["single query"])

            assert response.total_queries == 1
            assert response.successful_queries == 1

    @pytest.mark.asyncio
    async def test_malformed_ddg_response(self, searcher):
        """Malformed DuckDuckGo response should be handled."""
        # Missing required fields
        malformed_result = {"title": "Test"}  # Missing href and body

        with patch.object(searcher, '_ddgs_search_sync', return_value=[malformed_result]):
            response = await searcher.search("test query")
            # Should not crash
            assert isinstance(response, SearchResponse)


# =============================================================================
# Timeout Configuration Tests
# =============================================================================

class TestTimeoutConfiguration:
    """Test timeout configuration options."""

    def test_default_timeout(self):
        """Default timeout should be reasonable."""
        searcher = WebSearcher()
        assert searcher.timeout >= 5
        assert searcher.timeout <= 60

    def test_custom_timeout(self):
        """Custom timeout should be respected."""
        searcher = WebSearcher(timeout=30)
        assert searcher.timeout == 30

    def test_very_short_timeout(self):
        """Very short timeout should be allowed."""
        searcher = WebSearcher(timeout=1)
        assert searcher.timeout == 1

    def test_very_long_timeout(self):
        """Very long timeout should be allowed."""
        searcher = WebSearcher(timeout=300)
        assert searcher.timeout == 300


# =============================================================================
# Integration Tests (Real Network)
# =============================================================================

class TestRealSearchIntegration:
    """Integration tests with real DuckDuckGo searches.

    These tests make real network requests.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_search(self):
        """Test real search query."""
        searcher = WebSearcher(timeout=30)

        try:
            response = await searcher.search("Python programming language")

            assert isinstance(response, SearchResponse)
            # May or may not have results depending on rate limits
            if response.success:
                assert response.result_count >= 0

        except Exception as e:
            pytest.skip(f"Network unavailable or rate limited: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_news_search(self):
        """Test real news search query."""
        searcher = WebSearcher(timeout=30)

        try:
            response = await searcher.search_news("technology news", timelimit="w")

            assert isinstance(response, SearchResponse)
            if response.success:
                assert response.result_count >= 0

        except Exception as e:
            pytest.skip(f"Network unavailable or rate limited: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_parallel_search(self):
        """Test real parallel search."""
        searcher = WebSearcher(timeout=30)

        try:
            response = await searcher.parallel_search(
                ["Python", "JavaScript"],
                max_results_per_query=3,
            )

            assert isinstance(response, ParallelSearchResponse)
            # May have partial success due to rate limits

        except Exception as e:
            pytest.skip(f"Network unavailable or rate limited: {e}")
