"""Stock footage retrieval from Pexels API."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from .config import PexelsConfig, get_keyword_fallbacks
from .models import StockFootageError, VideoClip

logger = logging.getLogger(__name__)


class PexelsClient:
    """Client for Pexels video API."""

    BASE_URL = "https://api.pexels.com/videos"

    def __init__(self, config: Optional[PexelsConfig] = None):
        """Initialize Pexels client.

        Args:
            config: Pexels configuration. Uses defaults if not provided.
        """
        self.config = config or PexelsConfig()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def api_key(self) -> str:
        """Get API key, raising error if not set."""
        key = self.config.api_key
        if not key:
            raise StockFootageError(
                f"Pexels API key not found. Set {self.config.api_key_env} "
                "environment variable. Get free key at: https://www.pexels.com/api/"
            )
        return key

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": self.api_key},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_videos(
        self,
        query: str,
        orientation: Optional[str] = None,
        size: Optional[str] = None,
        per_page: Optional[int] = None,
    ) -> dict:
        """Search for videos on Pexels.

        Args:
            query: Search query.
            orientation: 'portrait', 'landscape', or 'square'.
            size: 'large', 'medium', or 'small'.
            per_page: Number of results per page.

        Returns:
            Search results dictionary.
        """
        client = await self._get_client()

        params = {"query": query, "per_page": per_page or self.config.per_page}

        if orientation:
            params["orientation"] = orientation
        if size:
            params["size"] = size

        try:
            response = await client.get(f"{self.BASE_URL}/search", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise StockFootageError(f"Pexels API error: {e}") from e

    async def download_video(
        self,
        video: dict,
        output_path: Path,
        quality: Optional[str] = None,
    ) -> Path:
        """Download a video file.

        Args:
            video: Video data from Pexels API.
            output_path: Path to save the video.
            quality: Preferred quality ('hd', 'sd', etc.).

        Returns:
            Path to downloaded video.
        """
        quality = quality or self.config.quality
        video_files = video.get("video_files", [])

        if not video_files:
            raise StockFootageError("No video files available")

        # Find preferred quality
        video_file = next(
            (f for f in video_files if f.get("quality") == quality),
            video_files[0],
        )

        url = video_file.get("link")
        if not url:
            raise StockFootageError("No download URL available")

        client = await self._get_client()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)

            logger.info(f"Downloaded: {output_path.name}")
            return output_path

        except httpx.HTTPError as e:
            raise StockFootageError(f"Download failed: {e}") from e


def is_9_16(video: dict) -> bool:
    """Check if video is close to 9:16 aspect ratio.

    Args:
        video: Video data from Pexels API.

    Returns:
        True if video is approximately 9:16.
    """
    video_files = video.get("video_files", [])
    if not video_files:
        return False

    # Get dimensions from first file
    vf = video_files[0]
    width = vf.get("width", 0)
    height = vf.get("height", 0)

    if width == 0 or height == 0:
        return False

    ratio = width / height
    target = 9 / 16  # 0.5625
    return abs(ratio - target) < 0.1


def get_video_dimensions(video: dict) -> tuple[int, int]:
    """Get video dimensions.

    Args:
        video: Video data from Pexels API.

    Returns:
        Tuple of (width, height).
    """
    video_files = video.get("video_files", [])
    if not video_files:
        return (0, 0)

    vf = video_files[0]
    return (vf.get("width", 0), vf.get("height", 0))


class StockFootageService:
    """Service for finding and downloading stock footage."""

    def __init__(self, config: Optional[PexelsConfig] = None):
        """Initialize stock footage service.

        Args:
            config: Pexels configuration.
        """
        self.config = config or PexelsConfig()
        self.client = PexelsClient(config)

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()

    async def find_video(
        self,
        keywords: list[str],
        target_duration: float,
        scene_index: int,
        output_dir: Path,
    ) -> VideoClip:
        """Find and download best matching video for keywords.

        Uses priority order:
        1. Native 9:16 portrait videos
        2. Other portrait videos
        3. Landscape videos (will be cropped)

        Args:
            keywords: Keywords to search for.
            target_duration: Desired video duration in seconds.
            scene_index: Scene number (for filename).
            output_dir: Directory to save downloaded video.

        Returns:
            VideoClip with downloaded video information.

        Raises:
            StockFootageError: If no suitable video found.
        """
        output_dir = Path(output_dir)
        output_path = output_dir / f"scene_{scene_index:02d}.mp4"

        # Try portrait orientation first
        video = await self._search_with_priority(keywords, target_duration)

        if not video:
            # Try fallback keywords
            video = await self._search_with_fallback(keywords, target_duration)

        if not video:
            raise StockFootageError(
                f"No videos found for keywords: {keywords}"
            )

        # Download the video
        await self.client.download_video(video, output_path)

        # Get video info
        width, height = get_video_dimensions(video)
        duration = video.get("duration", 0)

        return VideoClip(
            path=output_path,
            source_url=video.get("url", ""),
            duration_seconds=duration,
            width=width,
            height=height,
            scene_index=scene_index,
            keywords_used=keywords,
        )

    async def _search_with_priority(
        self,
        keywords: list[str],
        target_duration: float,
    ) -> Optional[dict]:
        """Search with orientation priority.

        Args:
            keywords: Keywords to search for.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video or None.
        """
        query = " ".join(keywords)

        # Priority 1: Portrait orientation, prefer 9:16
        results = await self.client.search_videos(
            query=query,
            orientation="portrait",
            size="medium",
        )

        videos = results.get("videos", [])

        # Filter for exact 9:16
        perfect_matches = [v for v in videos if is_9_16(v)]
        if perfect_matches:
            return self._select_best_duration(perfect_matches, target_duration)

        # Any portrait video
        if videos:
            return self._select_best_duration(videos, target_duration)

        # Priority 2: Landscape (will crop to 9:16)
        results = await self.client.search_videos(
            query=query,
            orientation="landscape",
            size="medium",
        )

        videos = results.get("videos", [])
        if videos:
            return self._select_best_duration(videos, target_duration)

        return None

    async def _search_with_fallback(
        self,
        keywords: list[str],
        target_duration: float,
    ) -> Optional[dict]:
        """Search with keyword fallbacks.

        Args:
            keywords: Original keywords.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video or None.
        """
        # Try first keyword only
        if keywords:
            video = await self._search_with_priority(
                [keywords[0]], target_duration
            )
            if video:
                return video

        # Try fallback keywords
        for keyword in keywords:
            fallbacks = get_keyword_fallbacks(keyword)
            for fallback in fallbacks:
                video = await self._search_with_priority(
                    [fallback], target_duration
                )
                if video:
                    return video

        # Last resort: generic tech footage
        return await self._search_with_priority(
            ["technology abstract"], target_duration
        )

    def _select_best_duration(
        self,
        videos: list[dict],
        target_duration: float,
    ) -> Optional[dict]:
        """Select video with duration closest to target.

        Args:
            videos: List of video data.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video or None.
        """
        if not videos:
            return None

        return min(
            videos,
            key=lambda v: abs(v.get("duration", 0) - target_duration),
        )

    async def find_videos_for_scenes(
        self,
        scenes: list[tuple[list[str], float]],
        output_dir: Path,
    ) -> list[VideoClip]:
        """Find videos for multiple scenes.

        Args:
            scenes: List of (keywords, duration) tuples.
            output_dir: Directory to save downloaded videos.

        Returns:
            List of VideoClip objects.
        """
        clips = []

        for i, (keywords, duration) in enumerate(scenes):
            try:
                clip = await self.find_video(
                    keywords=keywords,
                    target_duration=duration,
                    scene_index=i + 1,
                    output_dir=output_dir,
                )
                clips.append(clip)
            except StockFootageError as e:
                logger.warning(f"Scene {i + 1}: {e}")
                # Try generic fallback
                try:
                    clip = await self.find_video(
                        keywords=["abstract background"],
                        target_duration=duration,
                        scene_index=i + 1,
                        output_dir=output_dir,
                    )
                    clips.append(clip)
                except StockFootageError:
                    raise StockFootageError(
                        f"Could not find video for scene {i + 1}"
                    )

        return clips
