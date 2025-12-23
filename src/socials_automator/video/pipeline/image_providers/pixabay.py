"""Pixabay Image Provider.

Implements IImageSearchProvider for Pixabay API.
API Reference: https://pixabay.com/api/docs/

Rate Limits: 100 requests per 60 seconds (free tier)
Note: Hotlinking not allowed - images must be downloaded to local storage.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import httpx

from .base import IImageSearchProvider, ImageSearchResult

logger = logging.getLogger("video.pipeline")


class PixabayImageProvider(IImageSearchProvider):
    """Pixabay image search provider."""

    PIXABAY_API_URL = "https://pixabay.com/api/"

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """Initialize Pixabay provider.

        Args:
            api_key: Pixabay API key. If None, reads from PIXABAY_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("PIXABAY_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "pixabay"

    @property
    def cache_folder_name(self) -> str:
        """Return cache folder name."""
        return "image-cache-pixabay"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Pixabay API key not found. Set PIXABAY_API_KEY environment variable."
                )
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        url: str,
        params: dict,
    ) -> Optional[dict]:
        """Make API request with retry and exponential backoff."""
        client = await self._get_client()

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.get(url, params=params)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Pixabay rate limited, waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable error
                logger.warning(f"Pixabay API error: {response.status_code}")
                return None

            except httpx.HTTPError as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Pixabay HTTP error: {e}")
                return None

        return None

    async def search(
        self,
        query: str,
        per_page: int = 5,
        orientation: Optional[str] = None,
    ) -> list[ImageSearchResult]:
        """Search for images on Pixabay.

        Args:
            query: Search query string.
            per_page: Number of results (3-200).
            orientation: Filter by orientation ('horizontal', 'vertical').
                         Note: Pixabay uses 'horizontal'/'vertical' not 'landscape'/'portrait'.

        Returns:
            List of ImageSearchResult objects.
        """
        # Map standard orientation names to Pixabay's names
        orientation_map = {
            "landscape": "horizontal",
            "portrait": "vertical",
            "horizontal": "horizontal",
            "vertical": "vertical",
        }

        params = {
            "key": self.api_key,
            "q": query[:100],  # Max 100 chars
            "per_page": max(3, min(per_page, 200)),
            "page": 1,
            "image_type": "photo",  # Only photos, not illustrations/vectors
            "safesearch": "true",
        }

        if orientation and orientation.lower() in orientation_map:
            params["orientation"] = orientation_map[orientation.lower()]

        result = await self._request_with_retry(self.PIXABAY_API_URL, params)

        if not result or "hits" not in result:
            return []

        results = []
        for hit in result["hits"]:
            # Use largeImageURL for best quality (1280px width)
            # webformatURL is 640px, previewURL is 150px
            results.append(ImageSearchResult(
                id=str(hit["id"]),
                url=hit.get("largeImageURL", hit.get("webformatURL", "")),
                thumbnail_url=hit.get("previewURL", ""),
                width=hit.get("imageWidth", 0),
                height=hit.get("imageHeight", 0),
                description=hit.get("tags", ""),
                photographer=hit.get("user", "Unknown"),
                photographer_url=f"https://pixabay.com/users/{hit.get('user', '')}-{hit.get('user_id', '')}",
                source="pixabay",
            ))

        return results

    async def download(
        self,
        image_id: str,
        url: str,
        output_path: Path,
        source_page_url: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Path]:
        """Download an image from Pixabay.

        Note: Pixabay requires downloading images - hotlinking is not allowed.

        Args:
            image_id: Pixabay image ID.
            url: URL to download from.
            output_path: Path to save the image.
            source_page_url: Unused (Pixabay API provides direct access).
            log_callback: Unused (Pixabay downloads are straightforward).

        Returns:
            Path to downloaded image, or None if failed.
        """
        # Pixabay provides reliable direct access - no fallbacks needed
        _ = source_page_url, log_callback  # Mark as intentionally unused
        try:
            client = await self._get_client()
            response = await client.get(url, follow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"Pixabay download failed: {response.status_code}")
                return None

            # Determine file extension from content type
            content_type = response.headers.get("content-type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            else:
                # Try to get from URL
                url_lower = url.lower()
                if ".png" in url_lower:
                    ext = ".png"
                elif ".webp" in url_lower:
                    ext = ".webp"
                else:
                    ext = ".jpg"

            # Update output path with correct extension
            if not output_path.suffix:
                output_path = output_path.with_suffix(ext)

            # Save to output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path

        except (httpx.HTTPError, IOError) as e:
            logger.warning(f"Pixabay download error: {e}")
            return None
