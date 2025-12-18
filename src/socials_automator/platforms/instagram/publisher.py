"""Instagram platform publisher.

Wraps the existing InstagramClient to implement the PlatformPublisher interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..base import PlatformPublisher, PublishResult, ProgressCallback
from .config import InstagramConfig


class InstagramPublisher(PlatformPublisher):
    """Instagram publisher implementing the platform interface.

    Uses the existing InstagramClient and CloudinaryUploader under the hood.
    """

    def __init__(
        self,
        config: InstagramConfig,
        progress_callback: ProgressCallback = None,
    ):
        """Initialize Instagram publisher.

        Args:
            config: Instagram platform configuration.
            progress_callback: Optional callback for progress updates.
        """
        super().__init__(config, progress_callback)
        self._config: InstagramConfig = config
        self._client = None
        self._uploader = None

    @property
    def platform_name(self) -> str:
        return "instagram"

    def _get_client(self):
        """Lazy-initialize the Instagram client."""
        if self._client is None:
            from socials_automator.instagram.client import InstagramClient

            legacy_config = self._config.to_legacy_config()
            self._client = InstagramClient(legacy_config)

        return self._client

    def _get_uploader(self):
        """Lazy-initialize the Cloudinary uploader."""
        if self._uploader is None:
            from socials_automator.instagram.uploader import CloudinaryUploader

            legacy_config = self._config.to_legacy_config()
            self._uploader = CloudinaryUploader(legacy_config)

        return self._uploader

    async def publish_reel(
        self,
        video_path: Path,
        caption: str,
        thumbnail_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> PublishResult:
        """Publish a video reel to Instagram.

        Args:
            video_path: Path to the video file.
            caption: Caption for the reel.
            thumbnail_path: Optional custom thumbnail.
            **kwargs: Additional options:
                - share_to_feed: bool (default True)
                - max_retries: int (default 3)

        Returns:
            PublishResult with success status and details.
        """
        # Validate config
        is_valid, error_msg = self._config.validate()
        if not is_valid:
            return self._make_result(success=False, error=error_msg)

        # Check video exists
        if not video_path.exists():
            return self._make_result(
                success=False,
                error=f"Video file not found: {video_path}",
            )

        try:
            # Step 1: Upload video to Cloudinary
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            await self._emit_progress(
                "upload", 0.0,
                f"Step 1/4: Uploading video to Cloudinary ({file_size_mb:.1f} MB)..."
            )

            uploader = self._get_uploader()
            video_url = uploader.upload_video(video_path)

            await self._emit_progress(
                "upload", 40.0,
                f"Step 1/4: Video uploaded to Cloudinary"
            )

            # Step 2: Upload thumbnail if provided
            cover_url = None
            if thumbnail_path and thumbnail_path.exists():
                await self._emit_progress(
                    "upload", 45.0,
                    "Step 2/4: Uploading thumbnail..."
                )
                cover_url = uploader.upload_image(thumbnail_path)
                await self._emit_progress(
                    "upload", 50.0,
                    "Step 2/4: Thumbnail uploaded"
                )
            else:
                await self._emit_progress(
                    "upload", 50.0,
                    "Step 2/4: Using auto-generated thumbnail"
                )

            # Step 3: Create container and wait for processing
            await self._emit_progress(
                "publish", 55.0,
                "Step 3/4: Creating Instagram container..."
            )

            # Publish to Instagram
            client = self._get_client()
            result = await client.publish_reel(
                video_url=video_url,
                caption=caption,
                share_to_feed=kwargs.get("share_to_feed", True),
                cover_url=cover_url,
                max_retries=kwargs.get("max_retries", 3),
            )

            # Convert to PublishResult
            if result.success:
                await self._emit_progress(
                    "complete", 100.0,
                    "Step 4/4: Published successfully!"
                )
                return self._make_result(
                    success=True,
                    media_id=result.media_id,
                    permalink=result.permalink,
                    video_url=video_url,
                    cover_url=cover_url,
                )
            else:
                return self._make_result(
                    success=False,
                    error=result.error_message,
                    video_url=video_url,
                )

        except Exception as e:
            return self._make_result(
                success=False,
                error=f"Instagram publish failed: {str(e)}",
            )

    async def check_credentials(self) -> tuple[bool, str]:
        """Verify Instagram credentials are valid.

        Returns:
            Tuple of (is_valid, message).
        """
        # First validate config has all required fields
        is_valid, error_msg = self._config.validate()
        if not is_valid:
            return False, error_msg

        try:
            client = self._get_client()
            result = await client.validate_token()
            username = result.get("username", "unknown")
            return True, f"Connected to @{username}"

        except Exception as e:
            return False, f"Credential check failed: {str(e)}"
