"""Image downloading from Pexels with caching.

Downloads Pexels images to the cache and updates overlay paths.
Uses cache-first approach to avoid redundant downloads.

Cache Structure:
    pexels/image-cache/
        index.json
        12345678.jpg
        ...
"""

import logging
from pathlib import Path
from typing import Optional

from .base import (
    IImageDownloader,
    ImageOverlay,
    ImageOverlayScript,
    ImageOverlayError,
    PipelineContext,
)
from .image_cache import PexelsImageCache
from .pexels_image import PexelsImageClient

logger = logging.getLogger("ai_calls")


def format_file_size(size_bytes: int) -> str:
    """Format file size for display.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "245 KB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class ImageDownloader(IImageDownloader):
    """Downloads Pexels images with caching."""

    def __init__(
        self,
        pexels_client: Optional[PexelsImageClient] = None,
        cache: Optional[PexelsImageCache] = None,
    ):
        """Initialize image downloader.

        Args:
            pexels_client: Optional Pexels client.
            cache: Optional image cache.
        """
        super().__init__()
        self._pexels_client = pexels_client
        self._cache = cache or PexelsImageCache()
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_pexels_client(self) -> PexelsImageClient:
        """Get or create Pexels client."""
        if self._pexels_client is None:
            self._pexels_client = PexelsImageClient(cache=self._cache)
        return self._pexels_client

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute image download step.

        Args:
            context: Pipeline context with overlay script.

        Returns:
            Updated context with downloaded images.
        """
        if not context.image_overlays:
            return context

        # Count how many need downloading
        pexels_overlays = [
            o for o in context.image_overlays.overlays
            if o.source == "pexels" and o.pexels_id
        ]

        if not pexels_overlays:
            self.log_progress("No Pexels images to download")
            return context

        self.log_start(f"Downloading {len(pexels_overlays)} Pexels images")

        try:
            # Reset stats
            self._cache_hits = 0
            self._cache_misses = 0

            overlay_script = await self.download_images(
                overlay_script=context.image_overlays,
                output_dir=context.temp_dir / "overlay_images",
            )

            context.image_overlays = overlay_script

            # Log results
            self._log_results(overlay_script)

            # Log cache stats
            total = self._cache_hits + self._cache_misses
            if total > 0:
                hit_rate = (self._cache_hits / total) * 100
                self.log_progress(
                    f"  Cache: {self._cache_hits} hits, {self._cache_misses} misses "
                    f"({hit_rate:.0f}% hit rate)"
                )

            # Calculate total size
            total_size = sum(
                o.file_size_bytes for o in overlay_script.overlays
                if o.source == "pexels" and o.file_size_bytes
            )
            self.log_success(f"Downloaded {len(pexels_overlays)} images ({format_file_size(total_size)})")

            return context

        except Exception as e:
            self.log_error(f"Image download failed: {e}")
            raise ImageOverlayError(f"Failed to download images: {e}") from e

    async def download_images(
        self,
        overlay_script: ImageOverlayScript,
        output_dir: Path,
    ) -> ImageOverlayScript:
        """Download Pexels images to cache.

        Args:
            overlay_script: Script with Pexels image IDs.
            output_dir: Directory for downloaded images.

        Returns:
            Updated ImageOverlayScript with local paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for overlay in overlay_script.overlays:
            if overlay.source == "pexels" and overlay.pexels_id:
                await self._download_single(overlay, output_dir)

        return overlay_script

    async def _download_single(
        self,
        overlay: ImageOverlay,
        output_dir: Path,
    ) -> None:
        """Download a single image.

        Args:
            overlay: Overlay with pexels_id.
            output_dir: Output directory.
        """
        pexels_id = overlay.pexels_id
        if not pexels_id:
            return

        # Check cache first
        if self._cache.has_image(pexels_id):
            cached_path = self._cache.get_image_path(pexels_id)
            if cached_path:
                overlay.image_path = cached_path
                overlay.file_size_bytes = cached_path.stat().st_size
                self._cache_hits += 1

                self.log_detail(
                    f"[HIT] pexels:{pexels_id} - {format_file_size(overlay.file_size_bytes)}"
                )
                return

        # Need to download
        self._cache_misses += 1

        try:
            client = self._get_pexels_client()

            # Get photo details
            photo_data = await client.get_photo(pexels_id)
            if not photo_data:
                self.log_warning(f"Could not get photo data for pexels:{pexels_id}")
                overlay.source = None  # Mark as failed
                return

            # Download to temp path first
            temp_path = output_dir / f"{pexels_id}.jpg"

            downloaded_path = await client.download_image(
                image_data=photo_data,
                output_path=temp_path,
                quality="large",
            )

            if downloaded_path and downloaded_path.exists():
                overlay.image_path = self._cache.get_image_path(pexels_id) or downloaded_path
                overlay.file_size_bytes = overlay.image_path.stat().st_size
                overlay.width = photo_data.get("width", 0)
                overlay.height = photo_data.get("height", 0)
                overlay.alt_text = photo_data.get("alt", overlay.alt_text)

                self.log_detail(
                    f"[MISS] pexels:{pexels_id} - {format_file_size(overlay.file_size_bytes)} "
                    f"- \"{overlay.alt_text[:40]}...\""
                )
            else:
                self.log_warning(f"Download failed for pexels:{pexels_id}")
                overlay.source = None

        except Exception as e:
            logger.warning(f"Failed to download pexels:{pexels_id}: {e}")
            overlay.source = None

    def _log_results(self, overlay_script: ImageOverlayScript) -> None:
        """Log download results in table format.

        Args:
            overlay_script: Overlay script with downloaded images.
        """
        pexels_overlays = [
            o for o in overlay_script.overlays
            if o.source == "pexels" and o.pexels_id
        ]

        if not pexels_overlays:
            return

        self.log_progress("")
        self.log_progress("  --- Downloads ---")
        self.log_progress("  | Image           | Status | Size   | Description                      |")
        self.log_progress("  |-----------------|--------|--------|----------------------------------|")

        for overlay in pexels_overlays:
            image_id = f"pexels:{overlay.pexels_id}"[:15].ljust(15)

            if overlay.image_path and overlay.image_path.exists():
                # Determine if it was a cache hit
                # We can check by seeing if the path is in cache dir
                cache_dir = self._cache.cache_dir
                if overlay.image_path.parent == cache_dir:
                    status = "[HIT] "
                else:
                    status = "[MISS]"
                size = format_file_size(overlay.file_size_bytes).ljust(6)
            else:
                status = "[FAIL]"
                size = "-".ljust(6)

            desc = (overlay.alt_text or "-")[:32].ljust(32)

            self.log_progress(f"  | {image_id} | {status} | {size} | {desc} |")

        self.log_progress("")

    async def close(self) -> None:
        """Close any open connections."""
        if self._pexels_client:
            await self._pexels_client.close()
