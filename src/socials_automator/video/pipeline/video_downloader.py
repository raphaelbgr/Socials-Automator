"""Video downloading from Pexels.

Downloads video clips to a temporary folder for assembly.
"""

from pathlib import Path
from typing import Optional

import httpx

from .base import (
    IVideoDownloader,
    PipelineContext,
    VideoClipInfo,
    VideoDownloadError,
)


class VideoDownloader(IVideoDownloader):
    """Downloads videos from Pexels search results."""

    def __init__(self, quality: str = "hd"):
        """Initialize video downloader.

        Args:
            quality: Preferred video quality (hd, sd).
        """
        super().__init__()
        self.quality = quality
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute video download step.

        Args:
            context: Pipeline context with search results.

        Returns:
            Updated context with downloaded clips.
        """
        # Get search results from previous step
        search_results = getattr(context, "_search_results", None)
        if not search_results:
            raise VideoDownloadError("No search results available for download")

        self.log_start(f"Downloading {len(search_results)} video clips")

        try:
            clips_dir = context.temp_dir / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            clips = await self.download_videos(search_results, clips_dir)
            context.clips = clips

            self.log_success(f"Downloaded {len(clips)} clips to {clips_dir}")
            return context

        except Exception as e:
            self.log_error(f"Video download failed: {e}")
            raise VideoDownloadError(f"Failed to download videos: {e}") from e
        finally:
            await self.close()

    async def download_videos(
        self,
        search_results: list[dict],
        output_dir: Path,
    ) -> list[VideoClipInfo]:
        """Download videos from search results.

        Args:
            search_results: List of search results from VideoSearcher.
            output_dir: Directory to save videos.

        Returns:
            List of VideoClipInfo for downloaded clips.
        """
        clips = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in search_results:
            segment_index = result["segment_index"]
            video_data = result["video"]
            keywords = result["keywords_used"]

            self.log_progress(f"Downloading clip for segment {segment_index}...")

            try:
                clip = await self._download_video(
                    video_data,
                    segment_index,
                    output_dir,
                    keywords,
                )
                clips.append(clip)

            except Exception as e:
                self.log_error(f"Failed to download segment {segment_index}: {e}")
                raise VideoDownloadError(
                    f"Failed to download video for segment {segment_index}: {e}"
                ) from e

        return clips

    async def _download_video(
        self,
        video_data: dict,
        segment_index: int,
        output_dir: Path,
        keywords: list[str],
    ) -> VideoClipInfo:
        """Download a single video.

        Args:
            video_data: Pexels video data.
            segment_index: Index of the segment.
            output_dir: Output directory.
            keywords: Keywords used for search.

        Returns:
            VideoClipInfo for the downloaded clip.
        """
        # Get best quality video file
        video_file = self._select_video_file(video_data)

        if not video_file:
            raise VideoDownloadError("No suitable video file found")

        url = video_file.get("link")
        if not url:
            raise VideoDownloadError("No download URL available")

        # Download the file
        output_path = output_dir / f"segment_{segment_index:02d}.mp4"

        client = await self._get_client()

        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(8192):
                    f.write(chunk)

        self.log_progress(f"Downloaded: {output_path.name}")

        return VideoClipInfo(
            segment_index=segment_index,
            path=output_path,
            source_url=video_data.get("url", ""),
            pexels_id=video_data.get("id", 0),
            title=video_data.get("user", {}).get("name", "Unknown"),
            duration_seconds=video_data.get("duration", 0),
            width=video_file.get("width", 0),
            height=video_file.get("height", 0),
            keywords_used=keywords,
        )

    def _select_video_file(self, video_data: dict) -> Optional[dict]:
        """Select the best video file from available options.

        Args:
            video_data: Pexels video data.

        Returns:
            Best video file or None.
        """
        video_files = video_data.get("video_files", [])

        if not video_files:
            return None

        # Prefer HD quality
        for vf in video_files:
            if vf.get("quality") == self.quality:
                return vf

        # Fallback to first available
        return video_files[0]
