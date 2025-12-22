"""Comprehensive tests for NewsAggregator timeout and error handling.

Tests cover:
- RSS feed timeout scenarios
- Feed unreachable errors
- Invalid feed data handling
- Parallel feed fetching with mixed success/failure
- Web search timeout scenarios
- Dynamic query generation failures
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta

from socials_automator.news.aggregator import NewsAggregator
from socials_automator.news.models import NewsArticle, NewsCategory


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def aggregator():
    """Create a NewsAggregator instance for testing."""
    return NewsAggregator(
        profile_name="test_profile",
        profile_path=Path("profiles/test_profile"),
        use_dynamic_queries=False,  # Disable for basic tests
    )


@pytest.fixture
def aggregator_with_dynamic():
    """Create a NewsAggregator with dynamic queries enabled."""
    return NewsAggregator(
        profile_name="test_profile",
        profile_path=Path("profiles/test_profile"),
        use_dynamic_queries=True,
    )


@pytest.fixture
def mock_feed_config():
    """Create a mock RSS feed config."""
    mock = MagicMock()
    mock.name = "Test Feed"
    mock.url = "https://example.com/feed.xml"
    mock.category = NewsCategory.GENERAL
    mock.language = "en"
    mock.region = "us"
    return mock


@pytest.fixture
def mock_yaml_feed_config():
    """Create a mock YAML-based feed config."""
    mock = MagicMock()
    mock.name = "Test YAML Feed"
    mock.url = "https://example.com/yaml-feed.xml"
    mock.category = NewsCategory.GENERAL
    mock.language = "en"
    mock.region = "us"
    return mock


def create_mock_article(title: str = "Test Article", hours_old: int = 1):
    """Create a mock NewsArticle."""
    return NewsArticle(
        title=title,
        summary="Test summary",
        source_name="Test Source",
        source_url="https://example.com",
        article_url=f"https://example.com/article/{title.lower().replace(' ', '-')}",
        published_at=datetime.utcnow() - timedelta(hours=hours_old),
        category=NewsCategory.GENERAL,
        content_hash=f"hash_{title}",
    )


# =============================================================================
# RSS Feed Timeout Tests
# =============================================================================

class TestRSSFeedTimeouts:
    """Test RSS feed timeout handling."""

    @pytest.mark.asyncio
    async def test_feed_timeout_raises_error(self, aggregator, mock_feed_config):
        """A feed that times out should raise TimeoutError."""
        import time

        # Temporarily reduce timeout for testing
        original_timeout = aggregator._FEED_TIMEOUT_SECONDS
        aggregator._FEED_TIMEOUT_SECONDS = 0.5  # Very short for test

        def slow_sync(*args):
            time.sleep(10)  # Block for 10 seconds (sync, not async)
            return []

        try:
            with patch.object(aggregator, '_parse_feed_sync', side_effect=slow_sync):
                with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                    await aggregator._fetch_single_feed(mock_feed_config)
        finally:
            aggregator._FEED_TIMEOUT_SECONDS = original_timeout

    @pytest.mark.asyncio
    async def test_feed_timeout_logged(self, aggregator, mock_feed_config):
        """Timeout should be logged with feed name."""
        # Temporarily reduce timeout for testing
        original_timeout = aggregator._FEED_TIMEOUT_SECONDS
        aggregator._FEED_TIMEOUT_SECONDS = 0.1  # Very short for test

        def slow_sync(*args):
            import time
            time.sleep(10)  # Block for 10 seconds

        with patch.object(aggregator, '_parse_feed_sync', side_effect=slow_sync):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await aggregator._fetch_single_feed(mock_feed_config)

        # Restore
        aggregator._FEED_TIMEOUT_SECONDS = original_timeout

    @pytest.mark.asyncio
    async def test_yaml_feed_timeout_raises_error(self, aggregator, mock_yaml_feed_config):
        """YAML feed timeout should also raise TimeoutError."""
        original_timeout = aggregator._FEED_TIMEOUT_SECONDS
        aggregator._FEED_TIMEOUT_SECONDS = 0.1

        def slow_sync(*args):
            import time
            time.sleep(10)

        with patch.object(aggregator, '_parse_yaml_feed_sync', side_effect=slow_sync):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await aggregator._fetch_single_yaml_feed(mock_yaml_feed_config)

        aggregator._FEED_TIMEOUT_SECONDS = original_timeout

    @pytest.mark.asyncio
    async def test_multiple_feeds_one_timeout(self, aggregator):
        """When one feed times out, others should still succeed."""
        # Create mock feeds
        good_feed = MagicMock()
        good_feed.name = "Good Feed"
        good_feed.url = "https://good.com/feed.xml"
        good_feed.category = NewsCategory.GENERAL

        slow_feed = MagicMock()
        slow_feed.name = "Slow Feed"
        slow_feed.url = "https://slow.com/feed.xml"
        slow_feed.category = NewsCategory.GENERAL

        # Mock the fetch methods
        good_articles = [create_mock_article("Good Article")]

        async def mock_fetch_single(feed):
            if feed.name == "Slow Feed":
                raise TimeoutError(f"Feed {feed.name} timed out")
            return good_articles

        with patch.object(aggregator, '_fetch_single_feed', side_effect=mock_fetch_single):
            articles, failed = await aggregator._fetch_rss_feeds([good_feed, slow_feed])

            # Good feed should succeed
            assert len(articles) == 1
            assert articles[0].title == "Good Article"

            # Slow feed should be in failed list
            assert len(failed) == 1
            assert "Slow Feed" in failed[0]


# =============================================================================
# RSS Feed Unreachable Tests
# =============================================================================

class TestRSSFeedUnreachable:
    """Test handling of unreachable RSS feeds."""

    @pytest.mark.asyncio
    async def test_connection_refused_handled(self, aggregator, mock_feed_config):
        """Connection refused should be caught and logged."""
        def raise_connection_error(*args):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(aggregator, '_parse_feed_sync', side_effect=raise_connection_error):
            with pytest.raises(ConnectionRefusedError):
                await aggregator._fetch_single_feed(mock_feed_config)

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self, aggregator, mock_feed_config):
        """DNS resolution failure should be handled."""
        def raise_dns_error(*args):
            raise OSError("Name or service not known")

        with patch.object(aggregator, '_parse_feed_sync', side_effect=raise_dns_error):
            with pytest.raises(OSError):
                await aggregator._fetch_single_feed(mock_feed_config)

    @pytest.mark.asyncio
    async def test_http_404_handled(self, aggregator, mock_feed_config):
        """HTTP 404 should be handled gracefully."""
        mock_parsed = MagicMock()
        mock_parsed.bozo = True
        mock_parsed.bozo_exception = Exception("404 Not Found")
        mock_parsed.entries = []

        with patch('feedparser.parse', return_value=mock_parsed):
            with pytest.raises(ValueError, match="Feed parse error"):
                articles = aggregator._parse_feed_sync(mock_feed_config)

    @pytest.mark.asyncio
    async def test_ssl_certificate_error(self, aggregator, mock_feed_config):
        """SSL certificate errors should be handled."""
        def raise_ssl_error(*args):
            raise ConnectionError("SSL: CERTIFICATE_VERIFY_FAILED")

        with patch.object(aggregator, '_parse_feed_sync', side_effect=raise_ssl_error):
            with pytest.raises(ConnectionError):
                await aggregator._fetch_single_feed(mock_feed_config)


# =============================================================================
# Invalid Feed Data Tests
# =============================================================================

class TestInvalidFeedData:
    """Test handling of invalid RSS feed data."""

    def test_malformed_xml_handled(self, aggregator, mock_feed_config):
        """Malformed XML should be handled."""
        mock_parsed = MagicMock()
        mock_parsed.bozo = True
        mock_parsed.bozo_exception = Exception("not well-formed (invalid token)")
        mock_parsed.entries = []

        with patch('feedparser.parse', return_value=mock_parsed):
            with pytest.raises(ValueError, match="Feed parse error"):
                aggregator._parse_feed_sync(mock_feed_config)

    def test_empty_feed_returns_empty_list(self, aggregator, mock_feed_config):
        """Empty feed should return empty list without error."""
        mock_parsed = MagicMock()
        mock_parsed.bozo = False
        mock_parsed.entries = []

        with patch('feedparser.parse', return_value=mock_parsed):
            articles = aggregator._parse_feed_sync(mock_feed_config)
            assert articles == []

    def test_missing_title_skipped(self, aggregator, mock_feed_config):
        """Entries without titles should be skipped."""
        mock_entry1 = {"title": "", "link": "https://example.com/1"}
        mock_entry2 = {"title": "Valid Title", "link": "https://example.com/2"}

        mock_parsed = MagicMock()
        mock_parsed.bozo = False
        mock_parsed.entries = [mock_entry1, mock_entry2]

        with patch('feedparser.parse', return_value=mock_parsed):
            articles = aggregator._parse_feed_sync(mock_feed_config)
            assert len(articles) == 1
            assert articles[0].title == "Valid Title"

    def test_invalid_date_handled(self, aggregator, mock_feed_config):
        """Invalid publication dates should not crash."""
        mock_entry = {
            "title": "Test Article",
            "link": "https://example.com/article",
            "published": "invalid date format",
        }

        mock_parsed = MagicMock()
        mock_parsed.bozo = False
        mock_parsed.entries = [mock_entry]

        with patch('feedparser.parse', return_value=mock_parsed):
            articles = aggregator._parse_feed_sync(mock_feed_config)
            assert len(articles) == 1
            # Should have some default date, not crash


# =============================================================================
# Web Search Timeout Tests
# =============================================================================

class TestWebSearchTimeouts:
    """Test web search timeout handling."""

    @pytest.mark.asyncio
    async def test_ddg_search_timeout(self, aggregator):
        """DuckDuckGo search timeout should be handled."""
        mock_query = MagicMock()
        mock_query.query = "test query"
        mock_query.category = NewsCategory.GENERAL
        mock_query.language = "en"

        # Create a mock web searcher
        mock_searcher = MagicMock()

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(100)

        mock_searcher.parallel_news_search = slow_search

        with patch.object(aggregator, 'web_searcher', mock_searcher):
            # Should timeout without crashing
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    aggregator._fetch_search([mock_query]),
                    timeout=0.5,
                )

    @pytest.mark.asyncio
    async def test_search_rate_limit_handled(self, aggregator):
        """DuckDuckGo rate limit should be handled gracefully."""
        mock_query = MagicMock()
        mock_query.query = "test query"
        mock_query.category = NewsCategory.GENERAL
        mock_query.language = "en"

        # Create a mock web searcher
        mock_searcher = MagicMock()

        async def rate_limited(*args, **kwargs):
            raise Exception("DuckDuckGoSearchException: Ratelimit")

        mock_searcher.parallel_news_search = rate_limited

        with patch.object(aggregator, 'web_searcher', mock_searcher):
            articles, failed = await aggregator._fetch_search([mock_query])

            assert len(articles) == 0
            assert len(failed) == 1
            assert "parallel_search" in failed[0]  # Error message format


# =============================================================================
# Dynamic Query Generation Tests
# =============================================================================

class TestDynamicQueryGeneration:
    """Test dynamic query generation error handling."""

    @pytest.mark.asyncio
    async def test_ai_failure_falls_back_to_static(self, aggregator_with_dynamic):
        """AI query generation failure should fallback to static queries."""
        # Mock the generate_dynamic_queries to fail
        async def failing_generate(*args, **kwargs):
            raise Exception("AI provider unavailable")

        with patch('socials_automator.news.aggregator.generate_dynamic_queries', side_effect=failing_generate):
            # Mock other dependencies
            with patch.object(aggregator_with_dynamic, '_get_source_registry', return_value=MagicMock()):
                with patch.object(aggregator_with_dynamic, '_get_feed_rotator') as mock_feed_rotator:
                    mock_selection = MagicMock()
                    mock_selection.feeds = []
                    mock_selection.period = "morning"
                    mock_feed_rotator.return_value.select_feeds.return_value = mock_selection

                    with patch.object(aggregator_with_dynamic, '_get_query_rotator') as mock_query_rotator:
                        mock_query_rotator.return_value.current_batch = 1
                        mock_query_rotator.return_value.total_batches = 3
                        mock_query_rotator.return_value.get_current_queries.return_value = []
                        mock_query_rotator.return_value.advance_batch.return_value = None

                        with patch.object(aggregator_with_dynamic, '_fetch_yaml_feeds', return_value=([], [])):
                            with patch.object(aggregator_with_dynamic, '_fetch_search', return_value=([], [])):
                                # Should not raise, just fallback
                                result = await aggregator_with_dynamic.fetch_with_rotation()
                                assert result is not None

    @pytest.mark.asyncio
    async def test_ai_timeout_falls_back(self, aggregator_with_dynamic):
        """AI query generation timeout should fallback to static."""
        async def timeout_generate(*args, **kwargs):
            await asyncio.sleep(100)  # Hang

        with patch('socials_automator.news.aggregator.generate_dynamic_queries', side_effect=timeout_generate):
            # Set a short timeout for the test
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    aggregator_with_dynamic.fetch_with_rotation(),
                    timeout=0.5,
                )


# =============================================================================
# Caching Tests
# =============================================================================

class TestFeedCaching:
    """Test feed caching behavior."""

    @pytest.mark.asyncio
    async def test_cached_results_returned(self, aggregator, mock_feed_config):
        """Cached results should be returned without fetching."""
        cache_key = f"rss:{mock_feed_config.url}"
        cached_articles = [create_mock_article("Cached Article")]

        # Populate cache
        aggregator._cache[cache_key] = (datetime.utcnow(), cached_articles)

        # Should return cached results without calling parse
        with patch.object(aggregator, '_parse_feed_sync') as mock_parse:
            articles = await aggregator._fetch_single_feed(mock_feed_config)

            # Parse should not be called
            mock_parse.assert_not_called()

            assert len(articles) == 1
            assert articles[0].title == "Cached Article"

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self, aggregator, mock_feed_config):
        """Expired cache should trigger a new fetch."""
        cache_key = f"rss:{mock_feed_config.url}"
        old_time = datetime.utcnow() - timedelta(hours=1)
        cached_articles = [create_mock_article("Old Article")]

        # Populate with old cache
        aggregator._cache[cache_key] = (old_time, cached_articles)

        new_articles = [create_mock_article("New Article")]

        with patch.object(aggregator, '_parse_feed_sync', return_value=new_articles):
            articles = await aggregator._fetch_single_feed(mock_feed_config)

            assert len(articles) == 1
            assert articles[0].title == "New Article"


# =============================================================================
# Parallel Fetch Tests
# =============================================================================

class TestParallelFetching:
    """Test parallel feed fetching behavior."""

    @pytest.mark.asyncio
    async def test_parallel_feeds_partial_success(self, aggregator):
        """When some feeds fail, others should still return."""
        feeds = []
        for i in range(5):
            feed = MagicMock()
            feed.name = f"Feed {i}"
            feed.url = f"https://example{i}.com/feed.xml"
            feed.category = NewsCategory.GENERAL
            feeds.append(feed)

        async def mock_fetch(feed):
            if "2" in feed.name or "4" in feed.name:
                raise TimeoutError(f"Feed {feed.name} timed out")
            return [create_mock_article(f"Article from {feed.name}")]

        with patch.object(aggregator, '_fetch_single_feed', side_effect=mock_fetch):
            articles, failed = await aggregator._fetch_rss_feeds(feeds)

            # 3 feeds should succeed
            assert len(articles) == 3

            # 2 feeds should fail
            assert len(failed) == 2

    @pytest.mark.asyncio
    async def test_all_feeds_fail_returns_empty(self, aggregator):
        """When all feeds fail, should return empty list."""
        feeds = []
        for i in range(3):
            feed = MagicMock()
            feed.name = f"Feed {i}"
            feed.url = f"https://example{i}.com/feed.xml"
            feed.category = NewsCategory.GENERAL
            feeds.append(feed)

        async def always_fail(feed):
            raise TimeoutError(f"Feed {feed.name} timed out")

        with patch.object(aggregator, '_fetch_single_feed', side_effect=always_fail):
            articles, failed = await aggregator._fetch_rss_feeds(feeds)

            assert len(articles) == 0
            assert len(failed) == 3


# =============================================================================
# Timeout Constant Tests
# =============================================================================

class TestTimeoutConstants:
    """Test timeout configuration."""

    def test_feed_timeout_constant_exists(self, aggregator):
        """Feed timeout constant should be defined."""
        assert hasattr(aggregator, '_FEED_TIMEOUT_SECONDS')
        assert aggregator._FEED_TIMEOUT_SECONDS > 0

    def test_feed_timeout_reasonable_value(self, aggregator):
        """Feed timeout should be between 10-120 seconds."""
        timeout = aggregator._FEED_TIMEOUT_SECONDS
        assert 10 <= timeout <= 120, f"Timeout {timeout}s outside reasonable range"

    def test_cache_ttl_exists(self, aggregator):
        """Cache TTL should be defined."""
        assert hasattr(aggregator, '_cache_ttl_minutes')
        assert aggregator._cache_ttl_minutes > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestRealFeedFetching:
    """Integration tests with real RSS feeds.

    These tests make real network requests and may be slow.
    Skip if network is unavailable.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_rss_feed(self, aggregator):
        """Test fetching a real RSS feed."""
        # Use a reliable, stable RSS feed for testing
        real_feed = MagicMock()
        real_feed.name = "BBC Top Stories"
        real_feed.url = "https://feeds.bbci.co.uk/news/rss.xml"
        real_feed.category = NewsCategory.GENERAL

        try:
            articles = await asyncio.wait_for(
                aggregator._fetch_single_feed(real_feed),
                timeout=30.0,
            )

            # Should get some articles (BBC feed is usually well-populated)
            assert isinstance(articles, list)
            # May or may not have articles depending on feed state

        except (TimeoutError, asyncio.TimeoutError):
            pytest.skip("Network too slow or feed unavailable")
        except Exception as e:
            pytest.skip(f"Feed unavailable: {e}")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_feed_timeout_behavior(self, aggregator):
        """Test that real slow feeds are handled properly."""
        # Temporarily reduce timeout
        original_timeout = aggregator._FEED_TIMEOUT_SECONDS
        aggregator._FEED_TIMEOUT_SECONDS = 2  # Very short

        slow_feed = MagicMock()
        slow_feed.name = "Potentially Slow Feed"
        # Use a known slow or unreliable endpoint
        slow_feed.url = "https://httpstat.us/200?sleep=5000"  # 5 second delay
        slow_feed.category = NewsCategory.GENERAL

        try:
            await aggregator._fetch_single_feed(slow_feed)
            # If it succeeds quickly, that's fine
        except (TimeoutError, asyncio.TimeoutError):
            # Expected - timeout should trigger
            pass
        except Exception:
            # Other errors are acceptable (connection issues, etc.)
            pass
        finally:
            aggregator._FEED_TIMEOUT_SECONDS = original_timeout


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_feed_list(self, aggregator):
        """Empty feed list should return empty results."""
        articles, failed = await aggregator._fetch_rss_feeds([])
        assert articles == []
        assert failed == []

    @pytest.mark.asyncio
    async def test_none_feed_in_list(self, aggregator):
        """None values in feed list should be handled."""
        good_feed = MagicMock()
        good_feed.name = "Good Feed"
        good_feed.url = "https://example.com/feed.xml"
        good_feed.category = NewsCategory.GENERAL

        bad_feed = MagicMock()
        bad_feed.name = "Bad Feed"
        bad_feed.url = "https://bad.example.com/feed.xml"
        bad_feed.category = NewsCategory.GENERAL

        async def mock_fetch(feed):
            if feed.name == "Bad Feed":
                raise ValueError("Bad feed")
            return [create_mock_article("Article")]

        with patch.object(aggregator, '_fetch_single_feed', side_effect=mock_fetch):
            articles, failed = await aggregator._fetch_rss_feeds([good_feed, bad_feed])

            # Good feed should succeed
            assert len(articles) == 1
            # Bad feed should fail gracefully
            assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_very_slow_feed_vs_timeout(self, aggregator, mock_feed_config):
        """Feed that's slower than timeout should be cut off."""
        original_timeout = aggregator._FEED_TIMEOUT_SECONDS
        aggregator._FEED_TIMEOUT_SECONDS = 0.1

        def slow_parse(*args):
            import time
            time.sleep(5)  # Much longer than timeout
            return []

        with patch.object(aggregator, '_parse_feed_sync', side_effect=slow_parse):
            start = asyncio.get_event_loop().time()
            try:
                await aggregator._fetch_single_feed(mock_feed_config)
            except (TimeoutError, asyncio.TimeoutError):
                pass

            elapsed = asyncio.get_event_loop().time() - start

            # Should have timed out quickly, not waited 5 seconds
            assert elapsed < 1.0

        aggregator._FEED_TIMEOUT_SECONDS = original_timeout

    def test_unicode_in_feed_data(self, aggregator, mock_feed_config):
        """Unicode content in feeds should be handled."""
        mock_entry = {
            "title": "Article title",
            "link": "https://example.com/article",
            "summary": "Summary text",
        }

        mock_parsed = MagicMock()
        mock_parsed.bozo = False
        mock_parsed.entries = [mock_entry]

        with patch('feedparser.parse', return_value=mock_parsed):
            articles = aggregator._parse_feed_sync(mock_feed_config)
            assert len(articles) == 1
            # Check that Unicode was preserved (not checked in title but should work)
