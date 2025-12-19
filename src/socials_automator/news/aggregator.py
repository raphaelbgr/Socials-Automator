"""News aggregator that fetches articles from RSS feeds and web search.

Enhanced with global source support, rotation, and enhanced logging.
"""

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

# Legacy imports for backwards compatibility
from socials_automator.news.sources_legacy import (
    RSSFeedConfig,
    SearchQueryConfig,
    get_all_enabled_feeds,
    get_all_enabled_queries,
    get_feeds_by_priority,
    get_queries_by_priority,
)

# New YAML-based source registry
from socials_automator.news.sources import (
    SourceRegistry,
    get_source_registry,
    FeedConfig,
    QueryConfig,
    FeedRotator,
    get_feed_rotator,
    QueryRotator,
    get_query_rotator,
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

    Supports both legacy hardcoded sources and new YAML-based global sources.

    Usage:
        aggregator = NewsAggregator()

        # Fetch all news with automatic source rotation
        result = await aggregator.fetch()

        # Fetch with filters
        result = await aggregator.fetch(
            max_age_hours=12,
            categories=[NewsCategory.CELEBRITY, NewsCategory.MUSIC],
            priority_level=2,
        )

        # Use YAML-based global sources with rotation
        result = await aggregator.fetch_with_rotation()

        # Profile-scoped usage (state saved to profile/data/)
        aggregator = NewsAggregator(profile_path=Path("profiles/news.but.quick"))
    """

    def __init__(
        self,
        feeds: list[RSSFeedConfig] | None = None,
        queries: list[SearchQueryConfig] | None = None,
        web_searcher: WebSearcher | None = None,
        use_yaml_sources: bool = True,
        profile_path: Path | None = None,
    ):
        """Initialize the aggregator.

        Args:
            feeds: Custom RSS feeds (uses defaults if None).
            queries: Custom search queries (uses defaults if None).
            web_searcher: Custom WebSearcher instance.
            use_yaml_sources: If True, use new YAML-based sources with rotation.
            profile_path: Profile directory for profile-scoped state storage.
        """
        if feedparser is None:
            raise ImportError(
                "feedparser not installed. Install with: pip install feedparser"
            )

        self.use_yaml_sources = use_yaml_sources
        self.profile_path = profile_path

        # Legacy sources (for backwards compatibility)
        self.feeds = feeds or get_all_enabled_feeds()
        self.queries = queries or get_all_enabled_queries()

        # New YAML-based sources
        self._source_registry: SourceRegistry | None = None
        self._feed_rotator: FeedRotator | None = None
        self._query_rotator: QueryRotator | None = None

        self.web_searcher = web_searcher or WebSearcher(
            timeout=15,
            max_results_per_query=10,
        )

        # Cache for recent fetches (to avoid hammering sources)
        self._cache: dict[str, tuple[datetime, list[NewsArticle]]] = {}
        self._cache_ttl_minutes = 15

    def _get_source_registry(self) -> SourceRegistry:
        """Get or create source registry (lazy loading)."""
        if self._source_registry is None:
            try:
                self._source_registry = get_source_registry()
            except FileNotFoundError:
                logger.warning("YAML sources not found, using legacy sources")
                self.use_yaml_sources = False
                return None
        return self._source_registry

    def _get_feed_rotator(self) -> FeedRotator:
        """Get or create feed rotator (lazy loading)."""
        if self._feed_rotator is None:
            self._feed_rotator = get_feed_rotator()
        return self._feed_rotator

    def _get_query_rotator(self) -> QueryRotator:
        """Get or create query rotator (lazy loading)."""
        if self._query_rotator is None:
            self._query_rotator = get_query_rotator(profile_path=self.profile_path)
        return self._query_rotator

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

    # =========================================================================
    # YAML-based sources with rotation
    # =========================================================================

    async def fetch_with_rotation(
        self,
        max_age_hours: int = 24,
        max_feeds: int = 40,
        max_articles: int = 100,
        advance_query_batch: bool = True,
    ) -> AggregationResult:
        """Fetch news using YAML-based sources with time-weighted rotation.

        This method uses the new global source configuration with:
        - Time-weighted feed selection based on current UTC hour
        - Round-robin query batch rotation
        - Region and language metadata on articles

        Args:
            max_age_hours: Maximum article age in hours.
            max_feeds: Maximum number of feeds to fetch.
            max_articles: Maximum articles to return after dedup.
            advance_query_batch: If True, advance to next query batch after fetch.

        Returns:
            AggregationResult with enhanced metadata.
        """
        start_time = time.time()

        # Get rotators
        feed_rotator = self._get_feed_rotator()
        query_rotator = self._get_query_rotator()
        registry = self._get_source_registry()

        if registry is None:
            # Fall back to legacy fetch
            logger.warning("YAML sources unavailable, using legacy fetch")
            return await self.fetch(max_age_hours=max_age_hours, max_articles=max_articles)

        # Select feeds based on current time
        feed_selection = feed_rotator.select_feeds(max_feeds=max_feeds)
        selected_feeds = feed_selection.feeds

        # Get current batch queries
        current_batch = query_rotator.current_batch
        selected_queries = query_rotator.get_current_queries()

        logger.info(
            f"AGGREGATOR_ROTATION | period={feed_selection.period} | "
            f"feeds={len(selected_feeds)} | queries={len(selected_queries)} | "
            f"batch={current_batch}/{query_rotator.total_batches}"
        )

        all_articles: list[NewsArticle] = []
        failed_feeds: list[str] = []
        failed_queries: list[str] = []
        rss_count = 0
        search_count = 0

        # Fetch from RSS feeds using new FeedConfig
        if selected_feeds:
            rss_articles, rss_failed = await self._fetch_yaml_feeds(selected_feeds)
            all_articles.extend(rss_articles)
            failed_feeds.extend(rss_failed)
            rss_count = len(rss_articles)

        # Fetch from search queries using new QueryConfig
        if selected_queries:
            search_articles, search_failed = await self._fetch_yaml_queries(selected_queries)
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

        # Collect region/language statistics
        regions_used = list(set(a.region for a in deduped_articles))
        languages_used = list(set(a.source_language for a in deduped_articles))
        feeds_fetched = [f.name for f in selected_feeds]

        # Advance query batch if requested
        if advance_query_batch and selected_queries:
            query_rotator.mark_batch_used()
            if query_rotator.is_cooldown_expired():
                query_rotator.advance_batch()

        duration_ms = int((time.time() - start_time) * 1000)

        result = AggregationResult(
            articles=deduped_articles,
            rss_articles_count=rss_count,
            search_articles_count=search_count,
            total_before_dedup=total_before_dedup,
            duplicates_removed=duplicates_removed,
            failed_feeds=failed_feeds,
            failed_queries=failed_queries,
            regions_used=regions_used,
            languages_used=languages_used,
            feeds_fetched=feeds_fetched,
            query_batch=current_batch,
        )

        logger.info(
            f"NEWS_AGGREGATOR_V2 | rss:{rss_count} | search:{search_count} | "
            f"deduped:{len(deduped_articles)} | removed:{duplicates_removed} | "
            f"regions:{len(regions_used)} | languages:{len(languages_used)} | "
            f"batch:{current_batch} | {duration_ms}ms"
        )

        return result

    async def _fetch_yaml_feeds(
        self,
        feeds: list[FeedConfig],
    ) -> tuple[list[NewsArticle], list[str]]:
        """Fetch articles from YAML-based RSS feeds.

        Args:
            feeds: List of FeedConfig from YAML sources.

        Returns:
            Tuple of (articles, failed_feed_names)
        """
        tasks = [self._fetch_single_yaml_feed(feed) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[NewsArticle] = []
        failed: list[str] = []

        for feed, result in zip(feeds, results):
            if isinstance(result, Exception):
                failed.append(f"{feed.name}: {result}")
                logger.warning(f"RSS_FEED_ERROR | {feed.name} | {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
                logger.debug(
                    f"RSS_FEED_OK | {feed.name} | {len(result)} articles | "
                    f"region={feed.region} | lang={feed.language}"
                )

        return all_articles, failed

    async def _fetch_single_yaml_feed(self, feed: FeedConfig) -> list[NewsArticle]:
        """Fetch articles from a single YAML-based feed."""
        # Check cache
        cache_key = f"rss:{feed.url}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(minutes=self._cache_ttl_minutes):
                return cached_articles

        # Fetch in thread pool (feedparser is sync)
        articles = await asyncio.get_event_loop().run_in_executor(
            None,
            self._parse_yaml_feed_sync,
            feed,
        )

        # Update cache
        self._cache[cache_key] = (datetime.utcnow(), articles)

        return articles

    def _parse_yaml_feed_sync(self, feed: FeedConfig) -> list[NewsArticle]:
        """Synchronous RSS feed parsing for YAML-based feed."""
        try:
            parsed = feedparser.parse(feed.url)

            if parsed.bozo and not parsed.entries:
                raise ValueError(f"Feed parse error: {parsed.bozo_exception}")

            # Get region config for timezone
            registry = self._get_source_registry()
            region_tz = "UTC"
            country_code = ""
            if registry:
                region_config = registry.config.get_region(feed.region)
                if region_config:
                    region_tz = region_config.timezone
                    country_code = region_config.country_code

            articles = []
            for entry in parsed.entries[:20]:  # Limit per feed
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Get summary/description
                summary = entry.get("summary", "") or entry.get("description", "")
                if summary:
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

                # Map YAML category to NewsCategory enum
                category = self._map_yaml_category(feed.category)

                article = NewsArticle(
                    title=title,
                    summary=summary,
                    source_name=feed.name,
                    source_url=feed.url,
                    article_url=article_url,
                    published_at=published_at,
                    category=category,
                    image_url=image_url,
                    author=author,
                    content_hash=content_hash,
                    # Enhanced metadata
                    region=feed.region,
                    source_language=feed.language,
                    source_timezone=region_tz,
                    was_translated=feed.translate,
                    country_code=country_code,
                )

                articles.append(article)

            return articles

        except Exception as e:
            logger.warning(f"RSS parse error for {feed.name}: {e}")
            raise

    def _map_yaml_category(self, yaml_category) -> NewsCategory:
        """Map YAML category to NewsCategory enum."""
        # The YAML category is already a NewsCategory enum from pydantic parsing
        # But we need to map it to the models.NewsCategory
        try:
            return NewsCategory(yaml_category.value)
        except (ValueError, AttributeError):
            return NewsCategory.GENERAL

    async def _fetch_yaml_queries(
        self,
        queries: list[QueryConfig],
    ) -> tuple[list[NewsArticle], list[str]]:
        """Fetch articles from YAML-based search queries.

        Args:
            queries: List of QueryConfig from YAML sources.

        Returns:
            Tuple of (articles, failed_query_strings)
        """
        all_articles: list[NewsArticle] = []
        failed: list[str] = []

        # Group queries by their string to avoid duplicates
        query_map: dict[str, QueryConfig] = {q.query: q for q in queries}

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
                if config is None:
                    continue

                category = self._map_yaml_category(config.category)

                for result in query_response.results:
                    article = NewsArticle(
                        title=result.title,
                        summary=result.snippet,
                        source_name=result.domain.replace(".com", "").replace(".org", "").title(),
                        source_url=f"https://{result.domain}",
                        article_url=result.url,
                        published_at=datetime.utcnow(),
                        category=category,
                        image_url=None,
                        content_hash=_generate_content_hash(result.title, result.domain),
                        # Enhanced metadata from query config
                        source_language=config.language,
                        was_translated=config.translate,
                    )
                    all_articles.append(article)

                logger.debug(
                    f"SEARCH_OK | query='{config.query[:30]}...' | "
                    f"results={len(query_response.results)} | lang={config.language}"
                )

        except Exception as e:
            logger.error(f"Search fetch error: {e}")
            failed.append(f"parallel_search: {e}")

        return all_articles, failed


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
