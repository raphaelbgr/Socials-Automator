"""Web search using DuckDuckGo - adapted from InfiniteResearch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None  # Will raise on first use

logger = logging.getLogger("ai_calls")


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str

    @property
    def domain(self) -> str:
        """Get the domain from URL."""
        return urlparse(self.url).netloc.replace("www.", "")


@dataclass
class SearchResponse:
    """Response from a web search."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    duration_ms: int = 0

    @property
    def result_count(self) -> int:
        return len(self.results)


@dataclass
class ParallelSearchResponse:
    """Response from parallel web searches."""

    total_queries: int = 0
    successful_queries: int = 0
    total_results: int = 0
    all_sources: list[SearchResult] = field(default_factory=list)
    queries: list[SearchResponse] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def unique_domains(self) -> list[str]:
        """Get unique domains from all sources."""
        domains: dict[str, int] = {}
        for source in self.all_sources:
            domain = source.domain
            domains[domain] = domains.get(domain, 0) + 1
        return sorted(domains.keys(), key=lambda d: domains[d], reverse=True)

    def to_context_string(self, max_sources: int = 10) -> str:
        """Convert search results to a context string for AI prompts.

        Args:
            max_sources: Maximum number of sources to include.

        Returns:
            Formatted string with search results for AI context.
        """
        if not self.all_sources:
            return "No search results found."

        lines = [f"Web Search Results ({len(self.all_sources)} sources found):"]
        lines.append("")

        for i, source in enumerate(self.all_sources[:max_sources], 1):
            lines.append(f"{i}. {source.title}")
            lines.append(f"   Source: {source.domain}")
            if source.snippet:
                # Truncate long snippets
                snippet = source.snippet[:200] + "..." if len(source.snippet) > 200 else source.snippet
                lines.append(f"   {snippet}")
            lines.append("")

        if len(self.all_sources) > max_sources:
            lines.append(f"... and {len(self.all_sources) - max_sources} more sources")

        return "\n".join(lines)


class WebSearcher:
    """DuckDuckGo web search with parallel query support.

    Adapted from InfiniteResearch's ParallelDuckDuckGoSearch.

    Usage:
        searcher = WebSearcher()

        # Single search
        result = await searcher.search("AI productivity tools 2024")

        # Parallel search (multiple queries at once)
        results = await searcher.parallel_search([
            "AI tools for content creation",
            "ChatGPT alternatives 2024",
            "best AI writing assistants",
        ])

        # Get context string for AI
        context = results.to_context_string()
    """

    def __init__(
        self,
        timeout: int = 10,
        max_results_per_query: int = 5,
        proxy: str | None = None,
        verify_ssl: bool = True,
    ):
        """Initialize the web searcher.

        Args:
            timeout: Timeout per search in seconds.
            max_results_per_query: Default max results per query.
            proxy: Optional proxy for requests.
            verify_ssl: Whether to verify SSL certificates.
        """
        if DDGS is None:
            raise ImportError(
                "ddgs not installed. Install with: pip install ddgs"
            )

        self.timeout = timeout
        self.max_results_per_query = max_results_per_query
        self.proxy = proxy
        self.verify_ssl = verify_ssl

        # Track last search for debugging
        self._last_search: ParallelSearchResponse | None = None

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> SearchResponse:
        """Execute a single web search.

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            SearchResponse with results.
        """
        max_results = max_results or self.max_results_per_query
        start_time = time.time()

        try:
            # Run sync DDGS in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._ddgs_search_sync,
                query,
                max_results,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            search_results = [
                SearchResult(
                    title=r.get("title", "Unknown"),
                    url=r.get("href", ""),
                    snippet=r.get("body", "")[:300],
                )
                for r in results
                if r.get("href")
            ]

            logger.info(f"WEB_SEARCH | query:{query[:50]}... | results:{len(search_results)} | {duration_ms}ms")

            return SearchResponse(
                query=query,
                results=search_results,
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"WEB_SEARCH_ERROR | query:{query[:50]}... | error:{e}")

            return SearchResponse(
                query=query,
                results=[],
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def parallel_search(
        self,
        queries: list[str],
        max_results: int | None = None,
    ) -> ParallelSearchResponse:
        """Execute multiple searches in parallel.

        Args:
            queries: List of search queries (max 25).
            max_results: Max results per query.

        Returns:
            ParallelSearchResponse with all results.
        """
        if not queries:
            return ParallelSearchResponse()

        if len(queries) > 25:
            queries = queries[:25]
            logger.warning(f"Truncated to 25 queries (was {len(queries)})")

        max_results = max_results or self.max_results_per_query
        start_time = time.time()

        # Execute all searches in parallel
        tasks = [self.search(query, max_results) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        duration_ms = int((time.time() - start_time) * 1000)

        # Build response
        response = ParallelSearchResponse(
            total_queries=len(queries),
            duration_ms=duration_ms,
        )

        seen_urls: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                response.queries.append(SearchResponse(
                    query="unknown",
                    success=False,
                    error=str(result),
                ))
            elif isinstance(result, SearchResponse):
                response.queries.append(result)

                if result.success:
                    response.successful_queries += 1
                    response.total_results += result.result_count

                    # Add unique sources
                    for source in result.results:
                        if source.url not in seen_urls:
                            seen_urls.add(source.url)
                            response.all_sources.append(source)

        self._last_search = response

        # Log summary
        domains = response.unique_domains[:5]
        domain_str = ", ".join(domains) if domains else "none"
        logger.info(
            f"PARALLEL_SEARCH | queries:{response.total_queries} | "
            f"success:{response.successful_queries} | "
            f"results:{response.total_results} | "
            f"unique:{len(response.all_sources)} | "
            f"domains:{domain_str} | {duration_ms}ms"
        )

        return response

    async def search_news(
        self,
        query: str,
        max_results: int | None = None,
    ) -> SearchResponse:
        """Search news articles.

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            SearchResponse with news results.
        """
        max_results = max_results or self.max_results_per_query
        start_time = time.time()

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._ddgs_news_sync,
                query,
                max_results,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            search_results = [
                SearchResult(
                    title=r.get("title", "Unknown"),
                    url=r.get("url", ""),
                    snippet=r.get("body", "")[:300],
                )
                for r in results
                if r.get("url")
            ]

            logger.info(f"NEWS_SEARCH | query:{query[:50]}... | results:{len(search_results)} | {duration_ms}ms")

            return SearchResponse(
                query=query,
                results=search_results,
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"NEWS_SEARCH_ERROR | query:{query[:50]}... | error:{e}")

            return SearchResponse(
                query=query,
                results=[],
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def parallel_news_search(
        self,
        queries: list[str],
        max_results: int | None = None,
    ) -> ParallelSearchResponse:
        """Execute multiple news searches in parallel.

        Args:
            queries: List of search queries (max 25).
            max_results: Max results per query.

        Returns:
            ParallelSearchResponse with all news results.
        """
        if not queries:
            return ParallelSearchResponse()

        if len(queries) > 25:
            queries = queries[:25]

        max_results = max_results or self.max_results_per_query
        start_time = time.time()

        tasks = [self.search_news(query, max_results) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        duration_ms = int((time.time() - start_time) * 1000)

        response = ParallelSearchResponse(
            total_queries=len(queries),
            duration_ms=duration_ms,
        )

        seen_urls: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                response.queries.append(SearchResponse(
                    query="unknown",
                    success=False,
                    error=str(result),
                ))
            elif isinstance(result, SearchResponse):
                response.queries.append(result)

                if result.success:
                    response.successful_queries += 1
                    response.total_results += result.result_count

                    for source in result.results:
                        if source.url not in seen_urls:
                            seen_urls.add(source.url)
                            response.all_sources.append(source)

        logger.info(
            f"PARALLEL_NEWS | queries:{response.total_queries} | "
            f"results:{response.total_results} | {duration_ms}ms"
        )

        return response

    def _ddgs_search_sync(self, query: str, max_results: int) -> list[dict]:
        """Synchronous DuckDuckGo web search."""
        try:
            with DDGS(
                proxy=self.proxy,
                timeout=self.timeout,
                verify=self.verify_ssl,
            ) as ddgs:
                return list(ddgs.text(
                    query=query,
                    max_results=max_results,
                ))
        except Exception as e:
            logger.debug(f"DDG search failed for '{query}': {e}")
            return []

    def _ddgs_news_sync(self, query: str, max_results: int) -> list[dict]:
        """Synchronous DuckDuckGo news search."""
        try:
            with DDGS(
                proxy=self.proxy,
                timeout=self.timeout,
                verify=self.verify_ssl,
            ) as ddgs:
                return list(ddgs.news(
                    query=query,
                    max_results=max_results,
                ))
        except Exception as e:
            logger.debug(f"DDG news search failed for '{query}': {e}")
            return []

    @property
    def last_search(self) -> ParallelSearchResponse | None:
        """Get the last parallel search response."""
        return self._last_search


# Module-level convenience functions
_default_searcher: WebSearcher | None = None


def get_web_searcher() -> WebSearcher:
    """Get or create the default web searcher instance."""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = WebSearcher()
    return _default_searcher


async def web_search(query: str, max_results: int = 5) -> SearchResponse:
    """Convenience function for single web search."""
    return await get_web_searcher().search(query, max_results)


async def parallel_web_search(queries: list[str], max_results: int = 5) -> ParallelSearchResponse:
    """Convenience function for parallel web search."""
    return await get_web_searcher().parallel_search(queries, max_results)
