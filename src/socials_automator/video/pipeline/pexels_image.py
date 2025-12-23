"""Pexels Image API client.

Provides async methods to search and download images from Pexels.
Uses the same API key as the video client.

API Reference: https://www.pexels.com/api/documentation/#photos-search
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import httpx

from .image_cache import PexelsImageCache


class PexelsImageClient:
    """Async client for Pexels Image API."""

    PEXELS_API_URL = "https://api.pexels.com/v1"

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[PexelsImageCache] = None,
    ):
        """Initialize Pexels image client.

        Args:
            api_key: Pexels API key. If None, reads from PEXELS_API_KEY env var.
            cache: Optional image cache instance.
        """
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None
        self._cache = cache or PexelsImageCache()

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
        """Make API request with retry and exponential backoff.

        Args:
            url: Request URL.
            params: Query parameters.

        Returns:
            Response JSON or None if all retries failed.
        """
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
                return None

            except httpx.HTTPError:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                return None

        return None

    async def search(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: Optional[str] = None,
        size: Optional[str] = None,
    ) -> list[dict]:
        """Search for images.

        Args:
            query: Search query.
            per_page: Results per page (max 80).
            page: Page number.
            orientation: Filter by orientation (landscape, portrait, square).
            size: Filter by size (large, medium, small).

        Returns:
            List of image data dicts.
        """
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
        }

        if orientation:
            params["orientation"] = orientation
        if size:
            params["size"] = size

        url = f"{self.PEXELS_API_URL}/search"
        result = await self._request_with_retry(url, params)

        if result and "photos" in result:
            return result["photos"]

        return []

    async def get_photo(self, photo_id: int) -> Optional[dict]:
        """Get a single photo by ID.

        Args:
            photo_id: Pexels photo ID.

        Returns:
            Photo data dict or None if not found.
        """
        url = f"{self.PEXELS_API_URL}/photos/{photo_id}"
        return await self._request_with_retry(url, {})

    async def download_image(
        self,
        image_data: dict,
        output_path: Path,
        quality: str = "large",
    ) -> Optional[Path]:
        """Download an image to local path.

        Args:
            image_data: Pexels photo data dict.
            output_path: Path to save image.
            quality: Image quality (original, large, medium, small).

        Returns:
            Path to downloaded image or None if failed.
        """
        pexels_id = image_data.get("id")
        if not pexels_id:
            return None

        # Check cache first
        if self._cache.has_image(pexels_id):
            cached_path = self._cache.get_image_path(pexels_id)
            if cached_path:
                # Copy from cache to output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(cached_path, output_path)
                return output_path

        # Get download URL
        src = image_data.get("src", {})
        url = src.get(quality) or src.get("large") or src.get("original")

        if not url:
            return None

        try:
            client = await self._get_client()
            response = await client.get(url, follow_redirects=True)

            if response.status_code != 200:
                return None

            # Determine file extension
            content_type = response.headers.get("content-type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            else:
                ext = ".jpg"

            # Update output path with correct extension if needed
            if not output_path.suffix:
                output_path = output_path.with_suffix(ext)

            # Save to output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Add to cache
            self._cache.add_image(
                pexels_id=pexels_id,
                source_path=output_path,
                image_data=image_data,
                query_used=image_data.get("_query_used", ""),
            )

            return output_path

        except (httpx.HTTPError, IOError):
            return None

    async def search_and_select_best(
        self,
        query: str,
        prefer_orientation: Optional[str] = None,
        exclude_ids: Optional[set[int]] = None,
    ) -> Optional[dict]:
        """Search and return the best matching image.

        Args:
            query: Search query.
            prefer_orientation: Preferred orientation.
            exclude_ids: Set of Pexels IDs to exclude.

        Returns:
            Best matching image data or None.
        """
        exclude_ids = exclude_ids or set()

        images = await self.search(
            query=query,
            per_page=15,
            orientation=prefer_orientation,
        )

        # Filter out excluded IDs
        images = [img for img in images if img.get("id") not in exclude_ids]

        if not images:
            return None

        # Return first (most relevant) result
        # Add query used for cache tracking
        best = images[0]
        best["_query_used"] = query
        return best

    @property
    def cache(self) -> PexelsImageCache:
        """Get the image cache instance."""
        return self._cache
