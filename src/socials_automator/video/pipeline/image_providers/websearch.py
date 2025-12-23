"""Web Search Image Provider (DuckDuckGo).

Searches for images via DuckDuckGo with optional Tor proxy support.
No API key required. Rate limits are reasonable (~50 req/min).

Features:
- DuckDuckGo image search (no API key)
- Optional Tor proxy for anonymity
- Rich metadata (title, source, dimensions)
- Automatic retry with backoff

Usage:
    provider = WebSearchImageProvider(use_tor=True)
    results = await provider.search("taylor swift eras tour")
"""

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from .base import IImageSearchProvider, ImageSearchResult
from .tor_helper import get_tor_helper, is_tor_available, rotate_tor_ip, close_tor

logger = logging.getLogger("video.pipeline")


class WebSearchImageProvider(IImageSearchProvider):
    """DuckDuckGo image search provider with Tor support."""

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    RETRYABLE_EXCEPTIONS = (
        httpx.HTTPError,
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        use_tor: bool = False,
        safesearch: str = "moderate",  # off, moderate, strict
    ):
        """Initialize web search provider.

        Args:
            use_tor: Route requests through Tor proxy.
            safesearch: SafeSearch level (off, moderate, strict).
        """
        self.use_tor = use_tor
        self.safesearch = safesearch
        self._ddgs = None
        self._search_count = 0
        self._last_search_time = 0.0

        # Check Tor availability if requested
        if use_tor:
            if is_tor_available():
                logger.info("WebSearch: Tor proxy enabled")
            else:
                logger.warning(
                    "WebSearch: Tor requested but not available. "
                    "Falling back to direct connection."
                )
                self.use_tor = False

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "websearch"

    @property
    def cache_folder_name(self) -> str:
        """Return cache folder name."""
        return "image-cache-websearch"

    def _get_proxy(self) -> Optional[str]:
        """Get proxy URL if Tor is enabled."""
        if self.use_tor and is_tor_available():
            return get_tor_helper().proxy_url
        return None

    def _generate_image_id(self, query: str, index: int) -> str:
        """Generate unique image ID.

        Format: ws_{date}_{query_hash}_{index}
        Example: ws_20251223_a3f2b1c_001

        Args:
            query: Search query.
            index: Result index.

        Returns:
            Unique image ID.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:7]
        return f"ws_{date_str}_{query_hash}_{index:03d}"

    async def search(
        self,
        query: str,
        per_page: int = 5,
        orientation: Optional[str] = None,
    ) -> list[ImageSearchResult]:
        """Search for images on DuckDuckGo.

        Args:
            query: Search query string.
            per_page: Number of results (max ~100).
            orientation: Not supported by DDG, ignored.

        Returns:
            List of ImageSearchResult objects.
        """
        # Rate limiting: minimum 1 second between searches to avoid DDG rate limits
        now = time.time()
        time_since_last = now - self._last_search_time
        min_delay = 1.0  # 1 second minimum between searches
        if time_since_last < min_delay:
            await asyncio.sleep(min_delay - time_since_last)

        self._last_search_time = time.time()
        self._search_count += 1

        # Retry with Tor IP rotation on rate limit
        max_attempts = 3 if self.use_tor else 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                from ddgs import DDGS

                proxy = self._get_proxy()

                # Run sync DDG search in thread pool
                def do_search():
                    with DDGS(proxy=proxy, timeout=30) as ddgs:
                        results = list(ddgs.images(
                            query,
                            max_results=min(per_page, 50),
                            safesearch=self.safesearch,
                        ))
                    return results

                # Run in executor to not block event loop
                loop = asyncio.get_event_loop()
                ddg_results = await loop.run_in_executor(None, do_search)

                if not ddg_results:
                    logger.debug(f"WebSearch: No results for '{query}'")
                    return []

                results = []
                for i, item in enumerate(ddg_results):
                    image_id = self._generate_image_id(query, i)

                    results.append(ImageSearchResult(
                        id=image_id,
                        url=item.get("image", ""),
                        thumbnail_url=item.get("thumbnail", ""),
                        width=item.get("width", 0),
                        height=item.get("height", 0),
                        description=item.get("title", ""),
                        photographer=item.get("source", "Unknown"),
                        photographer_url=item.get("url", ""),  # Source page URL
                        source="websearch",
                    ))

                logger.debug(
                    f"WebSearch: Found {len(results)} images for '{query}' "
                    f"(via {'Tor' if proxy else 'direct'})"
                )
                return results

            except ImportError:
                logger.error(
                    "ddgs not installed. Run: pip install ddgs"
                )
                return []
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit error
                if "ratelimit" in error_str or "403" in error_str:
                    if self.use_tor and attempt < max_attempts - 1:
                        logger.warning(
                            f"WebSearch rate limited, reconnecting Tor (attempt {attempt + 1}/{max_attempts})..."
                        )
                        # Close and get fresh Tor connection
                        close_tor()
                        await asyncio.sleep(3.0)  # Wait for new circuit
                        continue

                logger.warning(f"WebSearch failed for '{query}': {e}")
                if attempt < max_attempts - 1:
                    continue
                return []

        # All attempts failed
        if last_error:
            logger.warning(f"WebSearch failed after {max_attempts} attempts: {last_error}")
        return []

    async def download(
        self,
        image_id: str,
        url: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Download an image from the web.

        Args:
            image_id: Image ID.
            url: URL to download from.
            output_path: Path to save the image.

        Returns:
            Path to downloaded image, or None if failed.
        """
        proxy = self._get_proxy()

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(
                    proxy=proxy,
                    timeout=30.0,
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://duckduckgo.com/",
                        "Sec-Fetch-Dest": "image",
                        "Sec-Fetch-Mode": "no-cors",
                        "Sec-Fetch-Site": "cross-site",
                    },
                ) as client:
                    response = await client.get(url)

                    if response.status_code != 200:
                        logger.warning(
                            f"WebSearch download failed: HTTP {response.status_code} for {url[:80]}..."
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_BASE_DELAY * (2 ** attempt))
                            continue
                        return None

                    # Determine file extension from content type
                    content_type = response.headers.get("content-type", "")
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = ".jpg"
                    elif "png" in content_type:
                        ext = ".png"
                    elif "webp" in content_type:
                        ext = ".webp"
                    elif "gif" in content_type:
                        ext = ".gif"
                    else:
                        # Try to infer from URL
                        url_lower = url.lower()
                        if ".png" in url_lower:
                            ext = ".png"
                        elif ".webp" in url_lower:
                            ext = ".webp"
                        elif ".gif" in url_lower:
                            ext = ".gif"
                        else:
                            ext = ".jpg"

                    # Update output path with correct extension
                    if not output_path.suffix:
                        output_path = output_path.with_suffix(ext)

                    # Validate it's actually an image (check magic bytes)
                    content = response.content
                    if not self._is_valid_image(content):
                        # Log what we got instead
                        content_preview = content[:100].decode('utf-8', errors='replace') if content else ""
                        logger.warning(
                            f"WebSearch: Invalid image data (not JPEG/PNG/GIF/WebP) from {url[:60]}... "
                            f"Content-Type: {content_type}, First bytes: {content_preview[:50]}..."
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            continue
                        return None

                    # Save to output path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(content)

                    return output_path

            except self.RETRYABLE_EXCEPTIONS as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"WebSearch download retry {attempt + 1}/{self.MAX_RETRIES}: {e}")
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"WebSearch download failed after {self.MAX_RETRIES} attempts: {e}")
                return None
            except Exception as e:
                logger.warning(f"WebSearch download error: {e}")
                return None

        return None

    def _is_valid_image(self, data: bytes) -> bool:
        """Check if data is a valid image by magic bytes.

        Args:
            data: Raw image data.

        Returns:
            True if data appears to be a valid image.
        """
        if len(data) < 8:
            return False

        # Check magic bytes
        # JPEG: FF D8 FF
        if data[:3] == b'\xff\xd8\xff':
            return True
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        # GIF: GIF87a or GIF89a
        if data[:6] in (b'GIF87a', b'GIF89a'):
            return True
        # WebP: RIFF....WEBP
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return True

        return False

    async def close(self) -> None:
        """Close any open connections."""
        pass  # No persistent connections

    def get_search_stats(self) -> dict:
        """Get search statistics.

        Returns:
            Dict with search count and Tor status.
        """
        return {
            "provider": "websearch",
            "engine": "duckduckgo",
            "search_count": self._search_count,
            "tor_enabled": self.use_tor,
            "tor_available": is_tor_available() if self.use_tor else None,
        }
