"""Image downloading with caching for multiple providers.

Downloads images from Pexels, Pixabay, or other providers to cache.
Uses cache-first approach to avoid redundant downloads.

Cache Structure:
    pexels/
        image-cache/            # Pexels images
            index.json
            12345678.jpg
        image-cache-pixabay/    # Pixabay images
            index.json
            87654321.jpg
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
from .image_cache import ImageCache, get_image_cache
from .image_providers import (
    IImageSearchProvider,
    get_image_provider,
)

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
    """Downloads images from configured provider with caching."""

    def __init__(
        self,
        image_provider: Optional[str] = None,
        use_tor: bool = False,
    ):
        """Initialize image downloader.

        Args:
            image_provider: Image provider name (websearch, pexels, pixabay).
                            Defaults to "websearch" if not specified.
            use_tor: Route websearch provider through Tor for anonymity.
        """
        super().__init__()
        self._provider_name = image_provider or "websearch"
        self._use_tor = use_tor
        self._provider: Optional[IImageSearchProvider] = None
        self._cache: Optional[ImageCache] = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_provider(self) -> IImageSearchProvider:
        """Get or create image provider."""
        if self._provider is None:
            self._provider = get_image_provider(
                self._provider_name,
                use_tor=self._use_tor,
            )
        return self._provider

    def _get_cache(self) -> ImageCache:
        """Get or create image cache."""
        if self._cache is None:
            self._cache = get_image_cache(self._provider_name)
        return self._cache

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute image download step.

        Args:
            context: Pipeline context with overlay script.

        Returns:
            Updated context with downloaded images.
        """
        if not context.image_overlays:
            return context

        # Count how many need downloading (any non-local source with an ID)
        provider_overlays = [
            o for o in context.image_overlays.overlays
            if o.source and o.source != "local" and o.pexels_id
        ]

        if not provider_overlays:
            self.log_progress(f"No {self._provider_name} images to download")
            return context

        self.log_start(f"Downloading {len(provider_overlays)} {self._provider_name.title()} images")

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
                if o.source and o.source != "local" and o.file_size_bytes
            )
            self.log_success(f"Downloaded {len(provider_overlays)} images ({format_file_size(total_size)})")

            return context

        except Exception as e:
            self.log_error(f"Image download failed: {e}")
            raise ImageOverlayError(f"Failed to download images: {e}") from e

    async def download_images(
        self,
        overlay_script: ImageOverlayScript,
        output_dir: Path,
    ) -> ImageOverlayScript:
        """Download images from provider to cache.

        Args:
            overlay_script: Script with image IDs.
            output_dir: Directory for downloaded images.

        Returns:
            Updated ImageOverlayScript with local paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for overlay in overlay_script.overlays:
            # Download any non-local source with an ID
            if overlay.source and overlay.source != "local" and overlay.pexels_id:
                await self._download_single(overlay, output_dir)

        return overlay_script

    async def _download_single(
        self,
        overlay: ImageOverlay,
        output_dir: Path,
    ) -> None:
        """Download a single image.

        Args:
            overlay: Overlay with image ID and download URL.
            output_dir: Output directory.
        """
        image_id = overlay.pexels_id
        if not image_id:
            return

        cache = self._get_cache()
        provider = self._get_provider()

        # Check cache first
        if cache.has_image(image_id):
            cached_path = cache.get_image_path(image_id)
            if cached_path:
                overlay.image_path = cached_path
                overlay.file_size_bytes = cached_path.stat().st_size
                self._cache_hits += 1

                self.log_detail(
                    f"[HIT] {self._provider_name}:{image_id} - {format_file_size(overlay.file_size_bytes)}"
                )
                return

        # Need to download
        self._cache_misses += 1

        if not overlay.download_url:
            self.log_warning(f"No download URL for {self._provider_name}:{image_id}")
            overlay.source = None
            return

        try:
            # Download to temp path first
            temp_path = output_dir / f"{image_id}.jpg"

            downloaded_path = await provider.download(
                image_id=image_id,
                url=overlay.download_url,
                output_path=temp_path,
            )

            if downloaded_path and downloaded_path.exists():
                # Add to cache
                cached_path = cache.add_image(
                    image_id=image_id,
                    source_path=downloaded_path,
                    metadata={
                        "width": overlay.width,
                        "height": overlay.height,
                        "description": overlay.alt_text or "",
                        "download_url": overlay.download_url,
                    },
                )

                overlay.image_path = cached_path
                overlay.file_size_bytes = cached_path.stat().st_size

                alt_preview = (overlay.alt_text or "")[:40]
                self.log_detail(
                    f"[MISS] {self._provider_name}:{image_id} - {format_file_size(overlay.file_size_bytes)} "
                    f'- "{alt_preview}..."'
                )
            else:
                self.log_warning(f"Download failed for {self._provider_name}:{image_id}")
                overlay.source = None

        except Exception as e:
            logger.warning(f"Failed to download {self._provider_name}:{image_id}: {e}")
            overlay.source = None

    def _log_results(self, overlay_script: ImageOverlayScript) -> None:
        """Log download results in table format.

        Args:
            overlay_script: Overlay script with downloaded images.
        """
        # Filter to provider overlays (any non-local source)
        provider_overlays = [
            o for o in overlay_script.overlays
            if o.source and o.source != "local" and o.pexels_id
        ]

        if not provider_overlays:
            return

        cache = self._get_cache()

        self.log_progress("")
        self.log_progress("  --- Downloads ---")
        self.log_progress("  | Image           | Status | Size   | Description                      |")
        self.log_progress("  |-----------------|--------|--------|----------------------------------|")

        for overlay in provider_overlays:
            image_id = f"{overlay.source}:{overlay.pexels_id}"[:15].ljust(15)

            if overlay.image_path and overlay.image_path.exists():
                # Determine if it was a cache hit
                # We can check by seeing if the path is in cache dir
                if overlay.image_path.parent == cache.cache_dir:
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
        if self._provider:
            await self._provider.close()
