"""Image uploader for Instagram posting using Cloudinary."""

import asyncio
from pathlib import Path
from typing import Callable, Awaitable
import cloudinary
import cloudinary.uploader

from .models import InstagramConfig


class CloudinaryUploader:
    """Upload images to Cloudinary to get public URLs for Instagram API.

    Instagram Graph API requires images to be at publicly accessible URLs.
    Cloudinary provides a generous free tier (25GB storage, 25k transformations/month).
    """

    def __init__(
        self,
        config: InstagramConfig,
        progress_callback: Callable[[str, int, int], Awaitable[None]] | None = None,
    ):
        """Initialize Cloudinary uploader.

        Args:
            config: Instagram configuration with Cloudinary credentials
            progress_callback: Async callback(step, current, total) for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback
        self._uploaded_public_ids: list[str] = []

        # Configure Cloudinary
        cloudinary.config(
            cloud_name=config.cloudinary_cloud_name,
            api_key=config.cloudinary_api_key,
            api_secret=config.cloudinary_api_secret,
            secure=True,
        )

    async def _report_progress(self, step: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            await self.progress_callback(step, current, total)

    def upload_image(self, image_path: Path, folder: str = "socials-automator") -> str:
        """Upload a single image to Cloudinary.

        Args:
            image_path: Path to the image file
            folder: Cloudinary folder to organize uploads

        Returns:
            Public URL of the uploaded image
        """
        result = cloudinary.uploader.upload(
            str(image_path),
            folder=folder,
            resource_type="image",
            overwrite=True,
            # Use original filename as public_id for easier cleanup
            public_id=f"{folder}/{image_path.stem}",
        )

        # Track for cleanup
        self._uploaded_public_ids.append(result["public_id"])

        return result["secure_url"]

    def upload_video(self, video_path: Path, folder: str = "socials-automator") -> str:
        """Upload a video to Cloudinary.

        Args:
            video_path: Path to the video file (MP4 recommended)
            folder: Cloudinary folder to organize uploads

        Returns:
            Public URL of the uploaded video
        """
        result = cloudinary.uploader.upload(
            str(video_path),
            folder=folder,
            resource_type="video",
            overwrite=True,
            # Use original filename as public_id for easier cleanup
            public_id=f"{folder}/{video_path.stem}",
        )

        # Track for cleanup (need to specify resource_type for videos)
        self._uploaded_public_ids.append((result["public_id"], "video"))

        return result["secure_url"]

    async def upload_video_async(
        self,
        video_path: Path,
        folder: str = "socials-automator",
    ) -> str:
        """Upload a video asynchronously.

        Args:
            video_path: Path to the video file
            folder: Cloudinary folder to organize uploads

        Returns:
            Public URL of the uploaded video
        """
        await self._report_progress("Uploading video to Cloudinary", 0, 1)

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: self.upload_video(video_path, folder),
        )

        await self._report_progress("Uploading video to Cloudinary", 1, 1)
        return url

    async def upload_batch(
        self,
        image_paths: list[Path],
        folder: str = "socials-automator",
    ) -> list[str]:
        """Upload multiple images in parallel.

        Args:
            image_paths: List of paths to image files
            folder: Cloudinary folder to organize uploads

        Returns:
            List of public URLs in the same order as input paths
        """
        total = len(image_paths)
        urls: list[str] = []

        await self._report_progress("Uploading images to Cloudinary", 0, total)

        # Upload images sequentially to avoid rate limits
        # Could be parallelized with asyncio.gather if needed
        for i, path in enumerate(image_paths):
            # Run sync upload in thread pool to not block
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                lambda p=path: self.upload_image(p, folder),
            )
            urls.append(url)
            await self._report_progress("Uploading images to Cloudinary", i + 1, total)

        return urls

    def cleanup(self) -> int:
        """Delete all uploaded files from Cloudinary.

        Call this after successful Instagram posting to free up storage.

        Returns:
            Number of files deleted
        """
        if not self._uploaded_public_ids:
            return 0

        deleted = 0
        for item in self._uploaded_public_ids:
            try:
                # Handle both old format (just public_id) and new format (tuple with resource_type)
                if isinstance(item, tuple):
                    public_id, resource_type = item
                    cloudinary.uploader.destroy(public_id, resource_type=resource_type)
                else:
                    cloudinary.uploader.destroy(item)
                deleted += 1
            except Exception:
                # Ignore cleanup errors
                pass

        self._uploaded_public_ids.clear()
        return deleted

    async def cleanup_async(self) -> int:
        """Async version of cleanup."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cleanup)
