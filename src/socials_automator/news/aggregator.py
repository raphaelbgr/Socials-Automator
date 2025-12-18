"""News aggregator that fetches articles from RSS feeds and web search."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

try:
    import feedparser
except ImportError:
    feedparser = None

from socials_automator.news.models import (
    NewsArticle,
    NewsCategory,
    AggregationResult,
)
from socials_automator.news.sources import (
    RSSFeedConfig,
    SearchQueryConfig,
    get_all_enabled_feeds,
    get_all_enabled_queries,
    get_feeds_by_priority,
    get_queries_by_priority,
)
from socials_automator.research.web_search import WebSearcher, SearchResult

logger = logging.getLogger("ai_calls")


def _generate_content_hash(title: str, source: str) -> str:
    """Generate a hash for deduplication based on title and source."""
    # Normalize: lowercase, remove punctuation, strip whitespace
    normalized = title.lower().strip()
    # Remove common noise words for better dedup
    noise = ["breaking", "just in", "update", "exclusive", "report"]
    for word in noise:
        normalized = normalized.replace(word, "")
    content = f"{normalized}|{source.lower()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _parse_rss_date(date_str: str | None) -> datetime:
    """Parse RSS date string to datetime, with fallback to now."""
    if not date_str:
        return datetime.utcnow()

    try:
        # feedparser's parsed date is a time tuple
        import email.utils
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.replace(tzinfo=None)  # Remove timezone for consistency
    except Exception:
        pass

    # Try common formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue

    return datetime.utcnow()


def _extract_image_from_rss(entry: dict) -> Optional[str]:
    """Extract image URL from RSS entry if available."""
    # Try media:content
    if hasattr(entry, "media_content") and entry.media_content:
        for media in entry.media_content:
            if media.get("medium") == "image" or media.get("type", "").startswith("image"):
                return media.get("url")

    # Try media:thumbnail
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        return entry.media_thumbnail[0].get("url")

    # Try enclosure
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("type", "").startswith("image"):
                return enc.get("href") or enc.get("url")

    # Try content with img tag
    content = entry.get("content", [{}])[0].get("value", "") if entry.get("content") else ""
    if "<img" in content:
        import re
        match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    return None


class NewsAggregator:
    """Aggregates news from RSS feeds and DuckDuckGo search.

    Usage:
        aggregator = NewsAggregator()

        # Fetch all news
        result = await aggregator.fetch()

        # Fetch with filters
        result = await aggregator.fetch(
            max_age_hours=12,
            categories=[NewsCategory.CELEBRITY, NewsCategory.MUSIC],
            priority_level=2,
        )

        # Fetch from specific sources only
        result = await aggregator.fetch(
            use_rss=True,
            use_search=False,  # Skip DuckDuckGo
        )
    """

    def __init__(
        self,
        feeds: list[RSSFeedConfig] | None = None,
        queries: list[SearchQueryConfig] | None = None,
        web_searcher: WebSearcher | None = None,
    ):
        """Initialize the aggregator.

        Args:
            feeds: Custom RSS feeds (uses defaults if None).
            queries: Custom search queries (uses defaults if None).
            web_searcher: Custom WebSearcher instance.
        """
        if feedparser is None:
            raise ImportError(
                "feedparser not installed. Install with: pip install feedparser"
            )

        self.feeds = feeds or get_all_enabled_feeds()
        self.queries = queries or get_all_enabled_queries()
        self.web_searcher = web_searcher or WebSearcher(
            timeout=15,
            max_results_per_query=10,
        )

        # Cache for recent fetches (to avoid hammering sources)
        self._cache: dict[str, tuple[datetime, list[NewsArticle]]] = {}
        self._cache_ttl_minutes = 15

    async def fetch(
        self,
        max_age_hours: int = 24,
        categories: list[NewsCategory] | None = None,
        priority_level: int = 3,
        use_rss: bool = True,
        use_search: bool = True,
        max_articles: int = 100,
    ) -> AggregationResult:
        """Fetch news articles from all configured sources.

        Args:
            max_age_hours: Maximum article age in hours.
            categories: Filter to specific categories (None = all).
            priority_level: Max priority level for sources (1-3).
            use_rss: Whether to fetch from RSS feeds.
            use_search: Whether to use DuckDuckGo news search.
            max_articles: Maximum articles to return after dedup.

        Returns:
            AggregationResult with all fetched articles.
        """
        start_time = time.time()

        all_articles: list[NewsArticle] = []
        failed_feeds: list[str] = []
        failed_queries: list[str] = []
        rss_count = 0
        search_count = 0

        # Filter feeds and queries by priority
        feeds = [f for f in self.feeds if f.priority <= priority_level]
        queries = [q for q in self.queries if q.priority <= priority_level]

        # Filter by category if specified
        if categories:
            feeds = [f for f in feeds if f.category in categories]
            queries = [q for q in queries if q.category in categories]

        # Fetch from RSS feeds
        if use_rss and feeds:
            rss_articles, rss_failed = await self._fetch_rss_feeds(feeds)
            all_articles.extend(rss_articles)
            failed_feeds.extend(rss_failed)
            rss_count = len(rss_articles)

        # Fetch from DuckDuckGo news search
        if use_search and queries:
            search_articles, search_failed = await self._fetch_search(queries)
            all_articles.extend(search_articles)
            failed_queries.extend(search_failed)
            search_count = len(search_articles)

        total_before_dedup = len(all_articles)

        # Filter by age
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        all_articles = [a for a in all_articles if a.published_at >= cutoff_time]

        # Deduplicate
        deduped_articles = self._deduplicate(all_articles)
        duplicates_removed = len(all_articles) - len(deduped_articles)

        # Sort by recency (newest first)
        deduped_articles.sort(key=lambda a: a.published_at, reverse=True)

        # Limit to max_articles
        if len(deduped_articles) > max_articles:
            deduped_articles = deduped_articles[:max_articles]

        duration_ms = int((time.time() - start_time) * 1000)

        result = AggregationResult(
            articles=deduped_articles,
            rss_articles_count=rss_count,
            search_articles_count=search_count,
            total_before_dedup=total_before_dedup,
            duplicates_removed=duplicates_removed,
            failed_feeds=failed_feeds,
            failed_queries=failed_queries,
        )

        logger.info(
            f"NEWS_AGGREGATOR | rss:{rss_count} | search:{search_count} | "
            f"deduped:{len(deduped_articles)} | removed:{duplicates_removed} | "
            f"failed_feeds:{len(failed_feeds)} | failed_queries:{len(failed_queries)} | "
            f"{duration_ms}ms"
        )

        return result

    async def _fetch_rss_feeds(
        self,
        feeds: list[RSSFeedConfig],
    ) -> tuple[list[NewsArticle], list[str]]:
        """Fetch articles from RSS feeds in parallel.

        Returns:
            Tuple of (articles, failed_feed_names)
        """
        tasks = [self._fetch_single_feed(feed) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[NewsArticle] = []
        failed: list[str] = []

        for feed, result in zip(feeds, results):
            if isinstance(result, Exception):
                failed.append(f"{feed.name}: {result}")
                logger.warning(f"RSS_FEED_ERROR | {feed.name} | {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
                logger.debug(f"RSS_FEED_OK | {feed.name} | {len(result)} articles")

        return all_articles, failed

    async def _fetch_single_feed(self, feed: RSSFeedConfig) -> list[NewsArticle]:
        """Fetch articles from a single RSS feed."""
        # Check cache
        cache_key = f"rss:{feed.url}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(minutes=self._cache_ttl_minutes):
                return cached_articles

        # Fetch in thread pool (feedparser is sync)
        articles = await asyncio.get_event_loop().run_in_executor(
            None,
            self._parse_feed_sync,
            feed,
        )

        # Update cache
        self._cache[cache_key] = (datetime.utcnow(), articles)

        return articles

    def _parse_feed_sync(self, feed: RSSFeedConfig) -> list[NewsArticle]:
        """Synchronous RSS feed parsing."""
        try:
            parsed = feedparser.parse(feed.url)

            if parsed.bozo and not parsed.entries:
                raise ValueError(f"Feed parse error: {parsed.bozo_exception}")

            articles = []
            for entry in parsed.entries[:20]:  # Limit per feed
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Get summary/description
                summary = entry.get("summary", "") or entry.get("description", "")
                if summary:
                    # Strip HTML tags for cleaner text
                    import re
                    summary = re.sub(r"<[^>]+>", "", summary)
                    summary = summary.strip()[:500]

                # Parse published date
                pub_date_str = entry.get("published") or entry.get("updated")
                published_at = _parse_rss_date(pub_date_str)

                # Get article URL
                article_url = entry.get("link", "")

                # Get image if available
                image_url = _extract_image_from_rss(entry)

                # Get author
                author = entry.get("author")

                # Generate content hash for deduplication
                content_hash = _generate_content_hash(title, feed.name)

                article = NewsArticle(
                    title=title,
                    summary=summary,
                    source_name=feed.name,
                    source_url=feed.url,
                    article_url=article_url,
                    published_at=published_at,
                    category=feed.category,
                    image_url=image_url,
                    author=author,
                    content_hash=content_hash,
                )

                articles.append(article)

            return articles

        except Exception as e:
            logger.warning(f"RSS parse error for {feed.name}: {e}")
            raise

    async def _fetch_search(
        self,
        queries: list[SearchQueryConfig],
    ) -> tuple[list[NewsArticle], list[str]]:
        """Fetch articles from DuckDuckGo news search.

        Returns:
            Tuple of (articles, failed_query_strings)
        """
        all_articles: list[NewsArticle] = []
        failed: list[str] = []

        # Group queries by their string to avoid duplicates
        query_map: dict[str, SearchQueryConfig] = {q.query: q for q in queries}

        # Execute news searches
        query_strings = list(query_map.keys())

        try:
            response = await self.web_searcher.parallel_news_search(
                queries=query_strings,
                max_results=10,
            )

            # Convert search results to NewsArticle
            for query_response in response.queries:
                if not query_response.success:
                    failed.append(f"{query_response.query}: {query_response.error}")
                    continue

                config = query_map.get(query_response.query)
                category = config.category if config else NewsCategory.GENERAL

                for result in query_response.results:
                    article = self._search_result_to_article(result, category)
                    all_articles.append(article)

        except Exception as e:
            logger.error(f"Search fetch error: {e}")
            failed.append(f"parallel_search: {e}")

        return all_articles, failed

    def _search_result_to_article(
        self,
        result: SearchResult,
        category: NewsCategory,
    ) -> NewsArticle:
        """Convert a SearchResult to NewsArticle."""
        # Extract source name from domain
        domain = result.domain
        source_name = domain.replace(".com", "").replace(".org", "").title()

        # Generate content hash
        content_hash = _generate_content_hash(result.title, source_name)

        return NewsArticle(
            title=result.title,
            summary=result.snippet,
            source_name=source_name,
            source_url=f"https://{domain}",
            article_url=result.url,
            published_at=datetime.utcnow(),  # Search doesn't give exact time
            category=category,
            image_url=None,
            content_hash=content_hash,
        )

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove duplicate articles based on content hash and title similarity."""
        seen_hashes: set[str] = set()
        seen_titles: set[str] = set()
        unique_articles: list[NewsArticle] = []

        for article in articles:
            # Check content hash
            if article.content_hash in seen_hashes:
                continue

            # Check normalized title (catch near-duplicates)
            normalized_title = article.title.lower().strip()[:50]
            if normalized_title in seen_titles:
                continue

            seen_hashes.add(article.content_hash)
            seen_titles.add(normalized_title)
            unique_articles.append(article)

        return unique_articles

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()


# Module-level convenience functions
_default_aggregator: NewsAggregator | None = None


def get_news_aggregator() -> NewsAggregator:
    """Get or create the default news aggregator instance."""
    global _default_aggregator
    if _default_aggregator is None:
        _default_aggregator = NewsAggregator()
    return _default_aggregator


async def fetch_news(
    max_age_hours: int = 24,
    categories: list[NewsCategory] | None = None,
    max_articles: int = 100,
) -> AggregationResult:
    """Convenience function to fetch news articles."""
    return await get_news_aggregator().fetch(
        max_age_hours=max_age_hours,
        categories=categories,
        max_articles=max_articles,
    )
