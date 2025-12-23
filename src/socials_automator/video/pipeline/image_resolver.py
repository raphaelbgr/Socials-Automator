"""Image resolution for overlays.

Resolves each planned image overlay to an actual image file:
1. First checks the profile's local image library
2. Falls back to configured image provider (Pexels, Pixabay, etc.)
3. Skips if no good match found (especially for exact matches)

Local Image Library Structure:
    profiles/{name}/assets/images/
        stranger-things/
            image.jpg           # The actual image
            metadata.json       # {"aliases": [...], "tags": [...]}
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .base import (
    IImageResolver,
    ImageOverlay,
    ImageOverlayScript,
    ImageOverlayError,
    LocalImageMetadata,
    PipelineContext,
)
from .image_providers import (
    IImageSearchProvider,
    ImageSearchResult,
    get_image_provider,
    AVAILABLE_PROVIDERS,
)

logger = logging.getLogger("ai_calls")


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class ImageResolver(IImageResolver):
    """Resolves image overlays to actual files.

    Supports multiple image providers (Pexels, Pixabay, etc.) via the
    --image-provider CLI flag.
    """

    def __init__(
        self,
        image_provider: Optional[str] = None,
        use_tor: bool = False,
    ):
        """Initialize image resolver.

        Args:
            image_provider: Image provider name (websearch, pexels, pixabay).
                            Defaults to "websearch" if not specified.
            use_tor: Route websearch provider through Tor for anonymity.
        """
        super().__init__()
        self._provider_name = image_provider or "websearch"
        self._use_tor = use_tor
        self._provider: Optional[IImageSearchProvider] = None
        self._used_image_ids: set[str] = set()

    def _get_provider(self) -> IImageSearchProvider:
        """Get or create image provider."""
        if self._provider is None:
            self._provider = get_image_provider(
                self._provider_name,
                use_tor=self._use_tor,
            )
        return self._provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute image resolution step.

        Args:
            context: Pipeline context with overlay script.

        Returns:
            Updated context with resolved images.
        """
        if not context.image_overlays:
            self.log_progress("No overlays to resolve")
            return context

        self.log_start(f"Resolving image sources (local library + {self._provider_name})")

        try:
            overlay_script = await self.resolve_images(
                overlay_script=context.image_overlays,
                profile_path=context.profile_path,
            )

            context.image_overlays = overlay_script

            # Log results
            self._log_results(overlay_script)

            # Count by source
            local_count = sum(1 for o in overlay_script.overlays if o.source == "local")
            provider_count = sum(
                1 for o in overlay_script.overlays
                if o.source and o.source not in ("local", None)
            )
            skipped_count = sum(1 for o in overlay_script.overlays if o.source is None)

            self.log_success(
                f"{local_count} local, {provider_count} {self._provider_name.title()}, {skipped_count} skipped"
            )

            return context

        except Exception as e:
            self.log_error(f"Image resolution failed: {e}")
            raise ImageOverlayError(f"Failed to resolve images: {e}") from e

    async def resolve_images(
        self,
        overlay_script: ImageOverlayScript,
        profile_path: Optional[Path] = None,
    ) -> ImageOverlayScript:
        """Resolve image sources for each overlay.

        Args:
            overlay_script: Script with planned overlays.
            profile_path: Path to profile for local image library.

        Returns:
            Updated ImageOverlayScript with resolved sources.
        """
        self.log_progress(f"  [>] Resolving {len(overlay_script.overlays)} images...")

        for overlay in overlay_script.overlays:
            await self._resolve_single(overlay, profile_path)

        return overlay_script

    async def _resolve_single(
        self,
        overlay: ImageOverlay,
        profile_path: Optional[Path],
    ) -> None:
        """Resolve a single overlay to an image.

        Args:
            overlay: Overlay to resolve.
            profile_path: Path to profile directory.
        """
        # Try local library first
        if profile_path and overlay.local_hint:
            local_result = self._find_local_image(
                profile_path=profile_path,
                hint=overlay.local_hint,
                topic=overlay.topic,
            )

            if local_result:
                path, match_reason = local_result
                overlay.source = "local"
                overlay.image_path = path
                overlay.file_size_bytes = path.stat().st_size if path.exists() else 0

                # Get dimensions if possible
                self._update_image_dimensions(overlay)

                self.log_detail(
                    f"LOCAL match: {overlay.topic} -> {path.name} ({match_reason})"
                )
                return

        # Try configured image provider for non-exact or if local not found
        if overlay.pexels_query:
            search_result = await self._search_provider(overlay)

            if search_result:
                overlay.source = self._provider_name
                overlay.pexels_id = search_result.id  # Use generic ID field
                overlay.download_url = search_result.url  # URL for downloading
                overlay.width = search_result.width
                overlay.height = search_result.height
                overlay.alt_text = search_result.description or overlay.alt_text

                # Mark as used to avoid duplicates
                if search_result.id:
                    self._used_image_ids.add(search_result.id)

                self.log_detail(
                    f"{self._provider_name.upper()} match: {overlay.topic} -> {self._provider_name}:{search_result.id}"
                )
                return

        # No match found
        if overlay.match_type == "exact":
            # Exact matches require real images - skip if not found
            overlay.source = None
            self.log_detail(
                f"SKIP (exact, no match): {overlay.topic}"
            )
        else:
            # Illustrative can be skipped gracefully
            overlay.source = None
            self.log_detail(
                f"SKIP (no good match): {overlay.topic}"
            )

    def _find_local_image(
        self,
        profile_path: Path,
        hint: str,
        topic: str,
    ) -> Optional[tuple[Path, str]]:
        """Find image in profile's local library.

        Search order:
        1. Exact folder name match (hint)
        2. Alias match in metadata.json
        3. Tag match in metadata.json

        Args:
            profile_path: Path to profile directory.
            hint: Folder name hint (e.g., "stranger-things").
            topic: Topic text for alias/tag matching.

        Returns:
            Tuple of (image_path, match_reason) or None.
        """
        assets_dir = profile_path / "assets" / "images"
        if not assets_dir.exists():
            return None

        # Normalize hint for comparison
        hint_lower = hint.lower().replace(" ", "-")
        topic_lower = topic.lower()

        # Try exact folder match first
        for folder in assets_dir.iterdir():
            if not folder.is_dir():
                continue

            if folder.name.lower() == hint_lower:
                image_path = self._get_image_from_folder(folder)
                if image_path:
                    return (image_path, "folder exact match")

        # Search all folders for alias/tag match
        for folder in assets_dir.iterdir():
            if not folder.is_dir():
                continue

            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                metadata = LocalImageMetadata(**data)

                # Check aliases
                for alias in metadata.aliases:
                    alias_lower = alias.lower()
                    if alias_lower == hint_lower or alias_lower == topic_lower:
                        image_path = self._get_image_from_folder(folder)
                        if image_path:
                            return (image_path, f'alias "{alias}"')

                    # Partial match in topic
                    if alias_lower in topic_lower:
                        image_path = self._get_image_from_folder(folder)
                        if image_path:
                            return (image_path, f'alias partial "{alias}"')

                # Check tags (less strict)
                topic_words = set(topic_lower.split())
                for tag in metadata.tags:
                    if tag.lower() in topic_words:
                        image_path = self._get_image_from_folder(folder)
                        if image_path:
                            return (image_path, f'tag "{tag}"')

            except (json.JSONDecodeError, IOError, TypeError):
                continue

        return None

    def _get_image_from_folder(self, folder: Path) -> Optional[Path]:
        """Get the image file from a folder.

        Looks for files named 'image.*' or any image file.

        Args:
            folder: Folder to search.

        Returns:
            Path to image file or None.
        """
        # First look for 'image.*'
        for ext in IMAGE_EXTENSIONS:
            image_path = folder / f"image{ext}"
            if image_path.exists():
                return image_path

        # Then look for any image file
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                return file

        return None

    async def _search_provider(self, overlay: ImageOverlay) -> Optional[ImageSearchResult]:
        """Search configured image provider for an image.

        Args:
            overlay: Overlay with pexels_query (used as search query).

        Returns:
            ImageSearchResult or None.
        """
        if not overlay.pexels_query:
            return None

        try:
            provider = self._get_provider()
            results = await provider.search(
                query=overlay.pexels_query,
                per_page=10,
            )

            # Filter out already used images
            for result in results:
                if result.id not in self._used_image_ids:
                    return result

            # If all results used, return first anyway
            return results[0] if results else None

        except Exception as e:
            logger.warning(f"{self._provider_name} search failed: {e}")
            return None

    def _update_image_dimensions(self, overlay: ImageOverlay) -> None:
        """Update overlay with image dimensions.

        Args:
            overlay: Overlay with image_path set.
        """
        if not overlay.image_path or not overlay.image_path.exists():
            return

        try:
            from PIL import Image
            with Image.open(overlay.image_path) as img:
                overlay.width, overlay.height = img.size
        except Exception:
            pass

    def _log_results(self, overlay_script: ImageOverlayScript) -> None:
        """Log resolution results in table format.

        Args:
            overlay_script: Resolved overlay script.
        """
        resolved = [o for o in overlay_script.overlays if o.source]
        if not resolved:
            self.log_progress("  No images resolved")
            return

        self.log_progress("")
        self.log_progress("  --- Resolution ---")
        self.log_progress("  | Seg | Source | Match Reason              | File                      |")
        self.log_progress("  |-----|--------|---------------------------|---------------------------|")

        for overlay in overlay_script.overlays:
            if overlay.source == "local":
                source = "LOCAL "
                match_reason = "local library"[:25].ljust(25)
                file_info = str(overlay.image_path.name)[:25] if overlay.image_path else "?"
            elif overlay.source and overlay.source != "local":
                source = overlay.source.upper()[:6].ljust(6)
                match_reason = f"query match"[:25].ljust(25)
                file_info = f"{overlay.source}:{overlay.pexels_id}"[:25]
            else:
                source = "SKIP  "
                match_reason = "no match"[:25].ljust(25)
                file_info = "-"[:25]

            self.log_progress(
                f"  | {overlay.segment_index:3d} | {source} | {match_reason} | {file_info.ljust(25)} |"
            )

        self.log_progress("")

    async def close(self) -> None:
        """Close any open connections."""
        if self._provider:
            await self._provider.close()
