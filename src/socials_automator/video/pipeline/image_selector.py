"""AI-powered image selection using vision models.

Fetches multiple candidate images per overlay topic, sends them to a vision
model for comparison, and selects the best match based on narration context.

This step runs when --smart-pick is enabled, replacing the default first-result
selection with AI-powered visual analysis.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Optional

from .base import (
    PipelineStep,
    ImageOverlay,
    ImageOverlayScript,
    ImageOverlayError,
    PipelineContext,
)
from .image_cache import ImageCache, get_image_cache
from .image_providers import (
    IImageSearchProvider,
    ImageSearchResult,
    get_image_provider,
)
from ...providers.text import TextProvider

logger = logging.getLogger("ai_calls")


# System prompt for image selection
IMAGE_SELECTION_SYSTEM_PROMPT = """You are an AI that analyzes images to find the best match for video narration segments.

You will receive:
1. A narration segment (what the video is discussing at this moment)
2. Multiple candidate images to choose from

Your task:
1. Analyze each image visually
2. Rate how well each image matches the narration context (0-100%)
3. Provide a brief description of what you see in each image
4. Select the best match
5. Write a detailed, searchable description of the winning image for future cache matching

## Response Format

Return valid JSON:
{
  "candidates": [
    {
      "index": 0,
      "score": 95,
      "assessment": "Main cast of Stranger Things with Eleven prominent, Netflix logo visible"
    },
    {
      "index": 1,
      "score": 40,
      "assessment": "Just Netflix logo on red background, not specific to the show"
    }
  ],
  "winner": {
    "index": 0,
    "score": 95,
    "reason": "Shows the actual show content with recognizable characters",
    "ai_description": "Stranger Things Season 5 promotional poster featuring the main cast including Eleven (Millie Bobby Brown) in the center, Dustin, Mike, Lucas, and Will standing in front of the Upside Down portal with red lightning. Netflix original series logo at bottom."
  }
}

## Important Guidelines

1. Score based on RELEVANCE to the narration, not image quality
2. For exact matches (specific shows, products, people), prioritize images that clearly show the subject
3. The ai_description should be detailed and include:
   - Main subjects/characters visible
   - Text/logos in the image
   - Setting/background
   - Any notable visual elements
   - Keywords that would help find this image again
4. Be specific - "woman using laptop" is better than "person with computer"
"""


class ImageSelector(PipelineStep):
    """Selects best images using AI vision analysis."""

    def __init__(
        self,
        image_provider: Optional[str] = None,
        use_tor: bool = False,
        candidate_count: int = 10,
        text_provider: Optional[TextProvider] = None,
        preferred_provider: Optional[str] = None,
    ):
        """Initialize image selector.

        Args:
            image_provider: Image search provider name (websearch, pexels, pixabay).
            use_tor: Route websearch through Tor.
            candidate_count: Number of candidates to fetch per overlay.
            text_provider: TextProvider for AI calls (should support vision).
            preferred_provider: Preferred AI provider name.
        """
        super().__init__()
        self._provider_name = image_provider or "websearch"
        self._use_tor = use_tor
        self._candidate_count = candidate_count
        self._text_provider = text_provider
        self._preferred_provider = preferred_provider
        self._provider: Optional[IImageSearchProvider] = None
        self._cache: Optional[ImageCache] = None
        self._used_image_ids: set[str] = set()

    def _get_provider(self) -> IImageSearchProvider:
        """Get or create image search provider."""
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

    def _get_text_provider(self) -> TextProvider:
        """Get or create text provider."""
        if self._text_provider is None:
            self._text_provider = TextProvider(
                provider_override=self._preferred_provider
            )
        return self._text_provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute AI-powered image selection.

        Args:
            context: Pipeline context with overlay script.

        Returns:
            Updated context with selected images.
        """
        if not context.image_overlays:
            self.log_progress("No overlays for image selection")
            return context

        overlays_to_process = [
            o for o in context.image_overlays.overlays
            if o.pexels_query and not o.source  # Not yet resolved
        ]

        if not overlays_to_process:
            self.log_progress("No overlays need AI selection")
            return context

        self.log_start(f"AI image selection for {len(overlays_to_process)} overlays")

        try:
            for overlay in overlays_to_process:
                await self._select_for_overlay(overlay, context)

            # Count results
            selected_count = sum(
                1 for o in overlays_to_process if o.source
            )

            self.log_success(f"Selected {selected_count}/{len(overlays_to_process)} images with AI")

            return context

        except Exception as e:
            self.log_error(f"Image selection failed: {e}")
            raise ImageOverlayError(f"Failed to select images: {e}") from e

    async def _select_for_overlay(
        self,
        overlay: ImageOverlay,
        context: PipelineContext,
    ) -> None:
        """Select best image for a single overlay.

        Args:
            overlay: Overlay to select image for.
            context: Pipeline context.
        """
        self.log_progress(f"\n  Topic: \"{overlay.topic}\" ({overlay.start_time:.1f}s - {overlay.end_time:.1f}s)")

        # Get narration text for this segment
        narration = self._get_narration_for_overlay(overlay, context)
        self.log_progress(f"  Narration: \"{narration[:60]}...\"")

        # Fetch candidates
        self.log_progress(f"  [>] Fetching {self._candidate_count} candidates...")
        candidates = await self._fetch_candidates(overlay)

        if not candidates:
            self.log_progress(f"  [X] No candidates found")
            overlay.source = None
            return

        self.log_progress(f"  [OK] Found {len(candidates)} candidates")

        # Download candidates to temp
        self.log_progress(f"  [>] Downloading candidates for analysis...")
        downloaded = await self._download_candidates(candidates, context.temp_dir)

        if not downloaded:
            self.log_progress(f"  [X] Failed to download candidates")
            overlay.source = None
            return

        # Send to AI for selection
        self.log_progress(f"  [>] Sending to AI vision model...")
        selection = await self._ai_select_best(
            candidates=downloaded,
            narration=narration,
            topic=overlay.topic,
        )

        if not selection:
            self.log_progress(f"  [X] AI selection failed, using first result")
            # Fallback to first candidate
            first = downloaded[0]
            self._apply_selection(overlay, first["result"], first["path"], None)
            return

        # Log results table
        self._log_selection_results(downloaded, selection)

        # Apply winner
        winner_idx = selection.get("winner", {}).get("index", 0)
        if 0 <= winner_idx < len(downloaded):
            winner = downloaded[winner_idx]
            ai_desc = selection.get("winner", {}).get("ai_description", "")
            self._apply_selection(overlay, winner["result"], winner["path"], ai_desc)

            self.log_progress(
                f"  [OK] Selected #{winner_idx + 1} ({selection['winner'].get('score', 0)}% match)"
            )
        else:
            # Invalid index, use first
            first = downloaded[0]
            self._apply_selection(overlay, first["result"], first["path"], None)

    def _get_narration_for_overlay(
        self,
        overlay: ImageOverlay,
        context: PipelineContext,
    ) -> str:
        """Get narration text for the overlay's time segment.

        Args:
            overlay: Overlay with timing.
            context: Pipeline context with script.

        Returns:
            Narration text for this segment.
        """
        if not context.script:
            return overlay.topic

        # Find segments that overlap with this overlay
        texts = []
        for seg in context.script.segments:
            # Check overlap
            if seg.end_time > overlay.start_time and seg.start_time < overlay.end_time:
                texts.append(seg.text)

        if texts:
            return " ".join(texts)

        # Fallback to topic
        return overlay.topic

    async def _fetch_candidates(
        self,
        overlay: ImageOverlay,
    ) -> list[ImageSearchResult]:
        """Fetch candidate images from provider.

        Args:
            overlay: Overlay with search query.

        Returns:
            List of search results.
        """
        if not overlay.pexels_query:
            return []

        try:
            provider = self._get_provider()
            results = await provider.search(
                query=overlay.pexels_query,
                per_page=self._candidate_count,
            )

            # Filter out already used
            filtered = [r for r in results if r.id not in self._used_image_ids]

            return filtered[:self._candidate_count]

        except Exception as e:
            logger.warning(f"Failed to fetch candidates: {e}")
            return []

    async def _download_candidates(
        self,
        candidates: list[ImageSearchResult],
        temp_dir: Path,
    ) -> list[dict]:
        """Download candidate images for analysis.

        Args:
            candidates: Search results to download.
            temp_dir: Temp directory for downloads.

        Returns:
            List of dicts with result and local path.
        """
        output_dir = temp_dir / "candidate_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        provider = self._get_provider()
        downloaded = []

        for i, result in enumerate(candidates):
            try:
                output_path = output_dir / f"candidate_{i}_{result.id}.jpg"

                # Check cache first
                cache = self._get_cache()
                if cache.has_image(result.id):
                    cached_path = cache.get_image_path(result.id)
                    if cached_path:
                        downloaded.append({
                            "index": i,
                            "result": result,
                            "path": cached_path,
                            "cached": True,
                        })
                        continue

                # Download
                path = await provider.download(
                    image_id=result.id,
                    url=result.url,
                    output_path=output_path,
                )

                if path and path.exists():
                    downloaded.append({
                        "index": i,
                        "result": result,
                        "path": path,
                        "cached": False,
                    })

            except Exception as e:
                logger.warning(f"Failed to download candidate {i}: {e}")
                continue

        return downloaded

    async def _ai_select_best(
        self,
        candidates: list[dict],
        narration: str,
        topic: str,
    ) -> Optional[dict]:
        """Use AI vision to select best matching image.

        Args:
            candidates: Downloaded candidates with paths.
            narration: Narration text for this segment.
            topic: Topic being illustrated.

        Returns:
            Selection result dict or None.
        """
        try:
            # Build prompt with images
            images_content = []

            for i, candidate in enumerate(candidates):
                # Read and encode image
                image_path = candidate["path"]
                if not image_path.exists():
                    continue

                # Read image bytes
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # Encode to base64
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # Determine media type
                suffix = image_path.suffix.lower()
                media_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }.get(suffix, "image/jpeg")

                images_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    }
                })

                # Add text label
                images_content.append({
                    "type": "text",
                    "text": f"Image #{i + 1} (ID: {candidate['result'].id})"
                })

            if not images_content:
                return None

            # Build the prompt
            prompt_text = f"""Analyze these {len(candidates)} images and select the best match for this video segment:

TOPIC: {topic}
NARRATION: "{narration}"

Compare all images and return JSON with scores and the winner.
The ai_description for the winner should be detailed enough to find this image again by text search."""

            # Add prompt at the end
            images_content.append({
                "type": "text",
                "text": prompt_text,
            })

            # Call AI with vision
            provider = self._get_text_provider()

            response = await provider.generate(
                prompt=images_content,
                system=IMAGE_SELECTION_SYSTEM_PROMPT,
                task_name="image_selection",
            )

            # Parse response
            return self._parse_selection_response(response)

        except Exception as e:
            logger.warning(f"AI selection failed: {e}")
            return None

    def _parse_selection_response(self, response: str) -> Optional[dict]:
        """Parse AI selection response.

        Args:
            response: AI response text.

        Returns:
            Parsed selection dict or None.
        """
        import re

        # Try to find JSON in response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return None

        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    def _apply_selection(
        self,
        overlay: ImageOverlay,
        result: ImageSearchResult,
        local_path: Path,
        ai_description: Optional[str],
    ) -> None:
        """Apply selection to overlay and cache.

        Args:
            overlay: Overlay to update.
            result: Selected search result.
            local_path: Local path to image.
            ai_description: AI-generated description for cache.
        """
        # Update overlay
        overlay.source = self._provider_name
        overlay.pexels_id = result.id
        overlay.download_url = result.url
        overlay.width = result.width
        overlay.height = result.height
        overlay.alt_text = ai_description or result.description or overlay.alt_text

        # Mark as used
        self._used_image_ids.add(result.id)

        # Add to cache with AI description
        cache = self._get_cache()

        if not cache.has_image(result.id):
            cache.add_image(
                image_id=result.id,
                source_path=local_path,
                metadata={
                    "width": result.width,
                    "height": result.height,
                    "description": result.description or "",
                    "ai_description": ai_description or "",
                    "download_url": result.url,
                    "query_used": overlay.pexels_query or "",
                },
            )
        elif ai_description:
            # Update existing cache entry with AI description
            str_id = str(result.id)
            if str_id in cache._index:
                cache._index[str_id]["ai_description"] = ai_description
                cache._save_index()

        # Set image path to cache location
        overlay.image_path = cache.get_image_path(result.id) or local_path

    def _log_selection_results(
        self,
        candidates: list[dict],
        selection: dict,
    ) -> None:
        """Log selection results in table format.

        Args:
            candidates: Downloaded candidates.
            selection: AI selection result.
        """
        self.log_progress("")
        self.log_progress("  | # | Score | Assessment                                          |")
        self.log_progress("  |---|-------|-----------------------------------------------------|")

        ai_candidates = selection.get("candidates", [])
        winner_idx = selection.get("winner", {}).get("index", -1)

        for i, candidate in enumerate(candidates):
            # Find AI assessment
            ai_data = next(
                (c for c in ai_candidates if c.get("index") == i),
                {}
            )

            score = ai_data.get("score", 0)
            assessment = ai_data.get("assessment", "-")[:51].ljust(51)

            marker = "[*]" if i == winner_idx else f" {i + 1} "

            self.log_progress(f"  |{marker}| {score:>4}% | {assessment} |")

        self.log_progress("")

    async def close(self) -> None:
        """Close any open connections."""
        if self._provider:
            await self._provider.close()
