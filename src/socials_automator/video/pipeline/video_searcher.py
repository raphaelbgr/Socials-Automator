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
        ai_client: Optional[object] = None,
    ):
        """Initialize video searcher.

        Args:
            api_key: Pexels API key. If None, reads from PEXELS_API_KEY env var.
            prefer_portrait: Whether to prefer portrait (9:16) videos.
            ai_client: Optional AI client for enhanced keyword generation.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        self.prefer_portrait = prefer_portrait
        self.ai_client = ai_client
        self._client: Optional[httpx.AsyncClient] = None
        # Track used video IDs to prevent duplicates
        self._used_video_ids: set[int] = set()

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

        # Reset used video tracking for this run
        self._used_video_ids.clear()

        try:
            search_results = await self.search_videos(context.script)

            # Store results in context for download step
            # We'll store them temporarily - the downloader will use them
            context._search_results = search_results

            # Log deduplication stats
            self.log_progress(f"Used {len(self._used_video_ids)} unique videos (no duplicates)")
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

        # Generate enhanced keywords using AI if available
        enhanced_keywords = await self._get_enhanced_keywords(script)

        for segment in script.segments:
            # Use enhanced keywords if available, otherwise use segment keywords
            keywords = enhanced_keywords.get(segment.index, segment.keywords)

            self.log_progress(
                f"Searching segment {segment.index}: {keywords[:3]}"
            )

            # Search for this segment (will skip already-used videos)
            video = await self._search_for_segment(segment, keywords)

            if video:
                # Mark video as used
                video_id = video.get("id")
                if video_id:
                    self._used_video_ids.add(video_id)

                results.append({
                    "segment_index": segment.index,
                    "video": video,
                    "keywords_used": keywords,
                    "duration_needed": segment.duration_seconds,
                })
            else:
                self.log_progress(f"No video found for segment {segment.index}, using fallback")
                fallback = await self._search_fallback()
                if fallback:
                    video_id = fallback.get("id")
                    if video_id:
                        self._used_video_ids.add(video_id)
                results.append({
                    "segment_index": segment.index,
                    "video": fallback,
                    "keywords_used": ["abstract", "technology"],
                    "duration_needed": segment.duration_seconds,
                })

        return results

    async def _get_enhanced_keywords(self, script: VideoScript) -> dict[int, list[str]]:
        """Use AI to generate better video search keywords.

        Args:
            script: Video script with segments.

        Returns:
            Dict mapping segment index to enhanced keywords.
        """
        if not self.ai_client:
            return {}

        try:
            import json

            # Build segment info
            segments_info = "\n".join(
                f"Segment {s.index}: \"{s.text}\" (current keywords: {s.keywords[:3]})"
                for s in script.segments
            )

            prompt = f"""Generate UNIQUE video search keywords for each segment of this video script.

Segments:
{segments_info}

CRITICAL RULES:
1. Each segment MUST have DIFFERENT keywords - no repeats across segments!
2. Keywords must be VISUAL concepts that can be filmed (not abstract)
3. Use 2-word search phrases that work on stock video sites
4. Think: "What specific scene would I see in a video?"

GOOD examples: "laptop typing", "woman smiling", "city night", "hand writing", "coffee shop"
BAD examples: "productivity", "success", "AI", "concept" (too abstract)

Format as JSON:
{{
    "1": ["keyword1", "keyword2", "keyword3"],
    "2": ["keyword1", "keyword2", "keyword3"],
    ...
}}

Each segment MUST have unique keywords. Check your output for duplicates!
Respond with ONLY valid JSON."""

            response = await self.ai_client.generate(prompt)

            # Parse JSON
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)

            # Convert to dict with int keys
            result = {}
            for key, value in data.items():
                result[int(key)] = value

            self.log_progress(f"AI enhanced keywords for {len(result)} segments")
            return result

        except Exception as e:
            self.log_progress(f"AI keyword enhancement failed: {e}, using defaults")
            return {}

    async def _search_for_segment(
        self,
        segment,
        keywords: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Search for a video matching a segment.

        Args:
            segment: Video segment with keywords.
            keywords: Optional enhanced keywords to use instead of segment keywords.

        Returns:
            Best matching video or None (excludes already-used videos).
        """
        client = await self._get_client()

        # Use provided keywords or fall back to segment keywords
        search_keywords = keywords if keywords else segment.keywords

        # Try each keyword individually first (more likely to find unique videos)
        for keyword in search_keywords:
            video = await self._search_query(
                client,
                keyword,
                segment.duration_seconds,
            )
            if video:
                return video

        # Then try combined keywords
        for i in range(len(search_keywords)):
            query = " ".join(search_keywords[: len(search_keywords) - i])
            if not query or len(query.split()) < 2:
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
        """Select best video preferring LONGER clips (excluding already-used).

        Prefers videos that are longer than needed (can be trimmed) over
        shorter ones (would need slow motion).

        Args:
            videos: List of videos.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video that hasn't been used yet.
        """
        if not videos:
            return None

        # Filter out already-used videos
        available_videos = [
            v for v in videos
            if v.get("id") not in self._used_video_ids
        ]

        if not available_videos:
            self.log_progress("All videos already used, allowing reuse as last resort")
            available_videos = videos

        # Prefer videos LONGER than target (can be trimmed cleanly)
        longer_videos = [v for v in available_videos if v.get("duration", 0) >= target_duration]

        if longer_videos:
            # Among longer videos, pick the one closest to target (minimize waste)
            return min(
                longer_videos,
                key=lambda v: v.get("duration", 0) - target_duration,
            )
        else:
            # No videos long enough - pick the longest available (minimize slow motion)
            return max(
                available_videos,
                key=lambda v: v.get("duration", 0),
            )
