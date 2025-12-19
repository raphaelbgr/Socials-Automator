"""Image uploader for Instagram posting using Cloudinary."""

import asyncio
import io
from pathlib import Path
from typing import Callable, Awaitable
import cloudinary
import cloudinary.uploader

from .models import InstagramConfig


class ProgressFileWrapper(io.IOBase):
    """File wrapper that reports read progress."""

    def __init__(
        self,
        file_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        self._file = open(file_path, "rb")
        self._size = file_path.stat().st_size
        self._read = 0
        self._callback = progress_callback

    def read(self, size: int = -1) -> bytes:
        data = self._file.read(size)
        self._read += len(data)
        if self._callback:
            self._callback(self._read, self._size)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()

    def close(self) -> None:
        self._file.close()

    @property
    def size(self) -> int:
        return self._size


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

    def upload_image(
        self,
        image_path: Path,
        folder: str = "socials-automator",
        profile: str | None = None,
        post_id: str | None = None,
    ) -> str:
        """Upload a single image to Cloudinary.

        Args:
            image_path: Path to the image file
            folder: Base Cloudinary folder to organize uploads
            profile: Profile name for scoped folder (e.g., "ai.for.mortals")
            post_id: Post/reel ID for scoped folder (e.g., "18-003-news")

        Returns:
            Public URL of the uploaded image
        """
        # Build scoped folder path: folder/profile/post_id
        scoped_folder = folder
        if profile:
            scoped_folder = f"{folder}/{profile}"
            if post_id:
                scoped_folder = f"{scoped_folder}/{post_id}"

        result = cloudinary.uploader.upload(
            str(image_path),
            folder=scoped_folder,
            resource_type="image",
            overwrite=True,
            # Use scoped path + filename as public_id for easier cleanup
            public_id=f"{scoped_folder}/{image_path.stem}",
        )

        # Track for cleanup
        self._uploaded_public_ids.append(result["public_id"])

        return result["secure_url"]

    def upload_video(
        self,
        video_path: Path,
        folder: str = "socials-automator",
        progress_callback: Callable[[int, int], None] | None = None,
        profile: str | None = None,
        post_id: str | None = None,
    ) -> str:
        """Upload a video to Cloudinary with progress tracking.

        Args:
            video_path: Path to the video file (MP4 recommended)
            folder: Base Cloudinary folder to organize uploads
            progress_callback: Optional callback(bytes_uploaded, total_bytes) for progress
            profile: Profile name for scoped folder (e.g., "ai.for.mortals")
            post_id: Post/reel ID for scoped folder (e.g., "18-003-news")

        Returns:
            Public URL of the uploaded video
        """
        # Build scoped folder path: folder/profile/post_id
        scoped_folder = folder
        if profile:
            scoped_folder = f"{folder}/{profile}"
            if post_id:
                scoped_folder = f"{scoped_folder}/{post_id}"

        file_size = video_path.stat().st_size

        # For large files (>20MB), use chunked upload
        if file_size > 20_000_000:
            result = cloudinary.uploader.upload_large(
                str(video_path),
                folder=scoped_folder,
                resource_type="video",
                overwrite=True,
                public_id=f"{scoped_folder}/{video_path.stem}",
                chunk_size=6_000_000,  # 6MB chunks
            )
        else:
            # For smaller files, use regular upload with progress wrapper
            wrapper = ProgressFileWrapper(video_path, progress_callback)
            try:
                result = cloudinary.uploader.upload(
                    wrapper,
                    folder=scoped_folder,
                    resource_type="video",
                    overwrite=True,
                    public_id=f"{scoped_folder}/{video_path.stem}",
                )
            finally:
                wrapper.close()

        # Track for cleanup (need to specify resource_type for videos)
        self._uploaded_public_ids.append((result["public_id"], "video"))

        return result["secure_url"]

    async def upload_video_async(
        self,
        video_path: Path,
        folder: str = "socials-automator",
        profile: str | None = None,
        post_id: str | None = None,
    ) -> str:
        """Upload a video asynchronously.

        Args:
            video_path: Path to the video file
            folder: Base Cloudinary folder to organize uploads
            profile: Profile name for scoped folder (e.g., "ai.for.mortals")
            post_id: Post/reel ID for scoped folder (e.g., "18-003-news")

        Returns:
            Public URL of the uploaded video
        """
        await self._report_progress("Uploading video to Cloudinary", 0, 1)

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: self.upload_video(video_path, folder, profile=profile, post_id=post_id),
        )

        await self._report_progress("Uploading video to Cloudinary", 1, 1)
        return url

    async def upload_batch(
        self,
        image_paths: list[Path],
        folder: str = "socials-automator",
        profile: str | None = None,
        post_id: str | None = None,
    ) -> list[str]:
        """Upload multiple images in parallel.

        Args:
            image_paths: List of paths to image files
            folder: Base Cloudinary folder to organize uploads
            profile: Profile name for scoped folder (e.g., "ai.for.mortals")
            post_id: Post/reel ID for scoped folder (e.g., "18-003-news")

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
                lambda p=path: self.upload_image(p, folder, profile=profile, post_id=post_id),
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
