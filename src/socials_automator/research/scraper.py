"""Web scraper using Trafilatura for content extraction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import trafilatura
from trafilatura.settings import use_config


@dataclass
class ScrapedContent:
    """Content extracted from a web page."""

    url: str
    title: str | None = None
    text: str | None = None
    author: str | None = None
    date: str | None = None
    description: str | None = None
    keywords: list[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: str | None = None

    @property
    def summary(self) -> str:
        """Get a short summary of the content."""
        if self.text:
            # First 500 chars
            return self.text[:500] + "..." if len(self.text) > 500 else self.text
        return self.description or ""


class WebScraper:
    """Web scraper for extracting article content.

    Uses Trafilatura for high-quality content extraction.
    Trafilatura has the best F1 score (0.937) among Python extraction libraries.

    Usage:
        scraper = WebScraper()

        # Scrape a single URL
        content = await scraper.scrape("https://example.com/article")

        # Scrape multiple URLs
        results = await scraper.scrape_many([
            "https://example.com/article1",
            "https://example.com/article2",
        ])

        # Search and scrape
        results = await scraper.search_and_scrape("AI productivity tools 2024")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_concurrent: int = 5,
        user_agent: str = "SocialsAutomator/0.1.0 (+https://github.com/socials-automator)",
    ):
        """Initialize the scraper.

        Args:
            timeout: HTTP request timeout in seconds.
            max_concurrent: Maximum concurrent requests.
            user_agent: User agent string for requests.
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent

        # Configure Trafilatura
        self._trafilatura_config = use_config()
        self._trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")

        self._http_client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def scrape(self, url: str) -> ScrapedContent:
        """Scrape content from a URL.

        Args:
            url: URL to scrape.

        Returns:
            ScrapedContent with extracted data.
        """
        async with self._semaphore:
            try:
                client = await self._get_client()
                response = await client.get(url)
                response.raise_for_status()
                html = response.text

                # Extract content with Trafilatura
                # Run in thread pool since trafilatura is synchronous
                extracted = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    include_comments=False,
                    include_tables=False,
                    include_images=False,
                    include_links=False,
                    output_format="txt",
                    config=self._trafilatura_config,
                )

                # Also get metadata
                metadata = await asyncio.to_thread(
                    trafilatura.extract_metadata,
                    html,
                )

                return ScrapedContent(
                    url=url,
                    title=metadata.title if metadata else None,
                    text=extracted,
                    author=metadata.author if metadata else None,
                    date=metadata.date if metadata else None,
                    description=metadata.description if metadata else None,
                    keywords=metadata.tags if metadata and metadata.tags else [],
                    success=bool(extracted),
                )

            except httpx.HTTPStatusError as e:
                return ScrapedContent(
                    url=url,
                    success=False,
                    error=f"HTTP {e.response.status_code}",
                )
            except Exception as e:
                return ScrapedContent(
                    url=url,
                    success=False,
                    error=str(e),
                )

    async def scrape_many(self, urls: list[str]) -> list[ScrapedContent]:
        """Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape.

        Returns:
            List of ScrapedContent objects.
        """
        tasks = [self.scrape(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def fetch_rss(self, feed_url: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch items from an RSS feed.

        Args:
            feed_url: URL of the RSS feed.
            limit: Maximum items to return.

        Returns:
            List of feed items with title, link, summary, published.
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser is required. Install with: pip install feedparser")

        try:
            client = await self._get_client()
            response = await client.get(feed_url)
            response.raise_for_status()

            # Parse feed (synchronous)
            feed = await asyncio.to_thread(feedparser.parse, response.text)

            items = []
            for entry in feed.entries[:limit]:
                items.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                })

            return items

        except Exception as e:
            return []

    async def scrape_with_context(
        self,
        url: str,
        context_prompt: str,
    ) -> dict[str, Any]:
        """Scrape a URL and extract specific information using AI.

        Args:
            url: URL to scrape.
            context_prompt: What to extract from the content.

        Returns:
            Dictionary with scraped content and AI-extracted info.
        """
        content = await self.scrape(url)

        if not content.success or not content.text:
            return {
                "url": url,
                "success": False,
                "error": content.error or "No content extracted",
            }

        return {
            "url": url,
            "success": True,
            "title": content.title,
            "text": content.text,
            "summary": content.summary,
            "keywords": content.keywords,
            # AI extraction would be done by the caller
            "raw_content": content.text,
        }
