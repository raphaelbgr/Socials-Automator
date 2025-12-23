"""Pexels Image Provider.

Implements IImageSearchProvider for Pexels API.
API Reference: https://www.pexels.com/api/documentation/#photos-search

Rate Limits: 200 requests/hour, 20,000 requests/month (free tier)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from .base import IImageSearchProvider, ImageSearchResult

logger = logging.getLogger("video.pipeline")


class PexelsImageProvider(IImageSearchProvider):
    """Pexels image search provider."""

    PEXELS_API_URL = "https://api.pexels.com/v1"

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """Initialize Pexels provider.

        Args:
            api_key: Pexels API key. If None, reads from PEXELS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "pexels"

    @property
    def cache_folder_name(self) -> str:
        """Return cache folder name."""
        return "image-cache"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Pexels API key not found. Set PEXELS_API_KEY environment variable."
                )
            self._client = httpx.AsyncClient(
                headers={"Authorization": self.api_key},
                timeout=30.0,
            )
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
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable error
                logger.warning(f"Pexels API error: {response.status_code}")
                return None

            except httpx.HTTPError as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Pexels HTTP error: {e}")
                return None

        return None

    async def search(
        self,
        query: str,
        per_page: int = 5,
        orientation: Optional[str] = None,
    ) -> list[ImageSearchResult]:
        """Search for images on Pexels.

        Args:
            query: Search query string.
            per_page: Number of results (max 80).
            orientation: Filter by orientation ('landscape', 'portrait', 'square').

        Returns:
            List of ImageSearchResult objects.
        """
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": 1,
        }

        if orientation:
            params["orientation"] = orientation

        url = f"{self.PEXELS_API_URL}/search"
        result = await self._request_with_retry(url, params)

        if not result or "photos" not in result:
            return []

        results = []
        for photo in result["photos"]:
            src = photo.get("src", {})
            results.append(ImageSearchResult(
                id=str(photo["id"]),
                url=src.get("large2x") or src.get("large") or src.get("original", ""),
                thumbnail_url=src.get("medium", ""),
                width=photo.get("width", 0),
                height=photo.get("height", 0),
                description=photo.get("alt", "") or photo.get("url", ""),
                photographer=photo.get("photographer", "Unknown"),
                photographer_url=photo.get("photographer_url", ""),
                source="pexels",
            ))

        return results

    async def download(
        self,
        image_id: str,
        url: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Download an image from Pexels.

        Args:
            image_id: Pexels photo ID.
            url: URL to download from.
            output_path: Path to save the image.

        Returns:
            Path to downloaded image, or None if failed.
        """
        try:
            client = await self._get_client()
            response = await client.get(url, follow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"Pexels download failed: {response.status_code}")
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
            logger.warning(f"Pexels download error: {e}")
            return None
