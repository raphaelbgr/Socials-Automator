"""Video search using Pexels API.

Searches for vertical (9:16) stock videos matching script segments.
"""

import os
from typing import Optional

import httpx

from .base import (
    IVideoSearcher,
    PipelineContext,
    VideoScript,
    VideoSearchError,
)


class VideoSearcher(IVideoSearcher):
    """Searches Pexels for stock videos matching script segments."""

    PEXELS_API_URL = "https://api.pexels.com/videos"

    def __init__(
        self,
        api_key: Optional[str] = None,
        prefer_portrait: bool = True,
    ):
        """Initialize video searcher.

        Args:
            api_key: Pexels API key. If None, reads from PEXELS_API_KEY env var.
            prefer_portrait: Whether to prefer portrait (9:16) videos.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        self.prefer_portrait = prefer_portrait
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.api_key:
                raise VideoSearchError(
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

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute video search step.

        Args:
            context: Pipeline context with script.

        Returns:
            Updated context with search results stored in context.
        """
        if not context.script:
            raise VideoSearchError("No script available for video search")

        self.log_start(f"Searching videos for {len(context.script.segments)} segments")

        try:
            search_results = await self.search_videos(context.script)

            # Store results in context for download step
            # We'll store them temporarily - the downloader will use them
            context._search_results = search_results

            self.log_success(f"Found videos for {len(search_results)} segments")
            return context

        except Exception as e:
            self.log_error(f"Video search failed: {e}")
            raise VideoSearchError(f"Failed to search videos: {e}") from e
        finally:
            await self.close()

    async def search_videos(self, script: VideoScript) -> list[dict]:
        """Search for videos matching script segments.

        Args:
            script: Video script with segments.

        Returns:
            List of search results, one per segment.
        """
        results = []

        for segment in script.segments:
            self.log_progress(
                f"Searching segment {segment.index}: {segment.keywords[:2]}"
            )

            # Search for this segment
            video = await self._search_for_segment(segment)

            if video:
                results.append({
                    "segment_index": segment.index,
                    "video": video,
                    "keywords_used": segment.keywords,
                    "duration_needed": segment.duration_seconds,
                })
            else:
                self.log_progress(f"No video found for segment {segment.index}, using fallback")
                fallback = await self._search_fallback()
                results.append({
                    "segment_index": segment.index,
                    "video": fallback,
                    "keywords_used": ["abstract", "technology"],
                    "duration_needed": segment.duration_seconds,
                })

        return results

    async def _search_for_segment(self, segment) -> Optional[dict]:
        """Search for a video matching a segment.

        Args:
            segment: Video segment with keywords.

        Returns:
            Best matching video or None.
        """
        client = await self._get_client()

        # Try keywords in order of specificity
        for i in range(len(segment.keywords)):
            query = " ".join(segment.keywords[: len(segment.keywords) - i])
            if not query:
                continue

            video = await self._search_query(
                client,
                query,
                segment.duration_seconds,
            )

            if video:
                return video

        return None

    async def _search_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        target_duration: float,
    ) -> Optional[dict]:
        """Search Pexels with a specific query.

        Args:
            client: HTTP client.
            query: Search query.
            target_duration: Target video duration.

        Returns:
            Best matching video or None.
        """
        try:
            # Search portrait first
            if self.prefer_portrait:
                video = await self._search_with_orientation(
                    client, query, "portrait", target_duration
                )
                if video:
                    return video

            # Fallback to landscape (will be cropped)
            video = await self._search_with_orientation(
                client, query, "landscape", target_duration
            )
            return video

        except httpx.HTTPError as e:
            self.log_progress(f"Search error for '{query}': {e}")
            return None

    async def _search_with_orientation(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: str,
        target_duration: float,
    ) -> Optional[dict]:
        """Search with specific orientation.

        Args:
            client: HTTP client.
            query: Search query.
            orientation: Video orientation (portrait/landscape).
            target_duration: Target duration in seconds.

        Returns:
            Best matching video or None.
        """
        params = {
            "query": query,
            "orientation": orientation,
            "size": "medium",  # 1080p
            "per_page": 15,
        }

        response = await client.get(f"{self.PEXELS_API_URL}/search", params=params)
        response.raise_for_status()

        data = response.json()
        videos = data.get("videos", [])

        if not videos:
            return None

        # For portrait, prefer exact 9:16 ratio
        if orientation == "portrait":
            perfect_matches = [v for v in videos if self._is_9_16(v)]
            if perfect_matches:
                return self._select_best_duration(perfect_matches, target_duration)

        # Return best duration match
        return self._select_best_duration(videos, target_duration)

    async def _search_fallback(self) -> dict:
        """Search for generic fallback video.

        Returns:
            Fallback video.
        """
        client = await self._get_client()

        fallback_queries = [
            "abstract technology",
            "digital background",
            "blue abstract",
            "gradient motion",
        ]

        for query in fallback_queries:
            video = await self._search_query(client, query, 10.0)
            if video:
                return video

        raise VideoSearchError("Could not find any fallback videos")

    def _is_9_16(self, video: dict) -> bool:
        """Check if video is 9:16 aspect ratio.

        Args:
            video: Pexels video data.

        Returns:
            True if video is approximately 9:16.
        """
        video_files = video.get("video_files", [])
        if not video_files:
            return False

        vf = video_files[0]
        width = vf.get("width", 0)
        height = vf.get("height", 0)

        if width == 0 or height == 0:
            return False

        ratio = width / height
        target = 9 / 16  # 0.5625
        return abs(ratio - target) < 0.1

    def _select_best_duration(
        self,
        videos: list[dict],
        target_duration: float,
    ) -> Optional[dict]:
        """Select video with duration closest to target.

        Args:
            videos: List of videos.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video.
        """
        if not videos:
            return None

        return min(
            videos,
            key=lambda v: abs(v.get("duration", 0) - target_duration),
        )
