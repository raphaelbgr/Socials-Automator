"""TikTok platform publisher.

Implements TikTok Content Posting API for video uploads.

API Reference:
- https://developers.tiktok.com/doc/content-posting-api-get-started
- https://developers.tiktok.com/doc/content-posting-api-reference-direct-post
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any, Optional

import httpx

from ..base import PlatformPublisher, PublishResult, ProgressCallback
from .config import TikTokConfig


_logger = logging.getLogger("tiktok_api")


# TikTok video requirements
TIKTOK_MAX_VIDEO_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
TIKTOK_MIN_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB minimum chunk
TIKTOK_MAX_CHUNK_SIZE = 64 * 1024 * 1024  # 64MB max chunk
TIKTOK_CHUNK_THRESHOLD = 64 * 1024 * 1024  # Files > 64MB need chunked upload


class TikTokAPIError(Exception):
    """TikTok API error."""

    def __init__(self, message: str, error_code: str = None, log_id: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.log_id = log_id


class TikTokPublisher(PlatformPublisher):
    """TikTok publisher implementing the platform interface.

    Uses TikTok's Content Posting API for direct video uploads.

    Note: Posts are private until your app passes TikTok's audit.
    """

    def __init__(
        self,
        config: TikTokConfig,
        progress_callback: ProgressCallback = None,
    ):
        """Initialize TikTok publisher.

        Args:
            config: TikTok platform configuration.
            progress_callback: Optional callback for progress updates.
        """
        super().__init__(config, progress_callback)
        self._config: TikTokConfig = config

    @property
    def platform_name(self) -> str:
        return "tiktok"

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: dict = None,
        json_data: dict = None,
        data: bytes = None,
        timeout: float = 60.0,
    ) -> dict:
        """Make a request to the TikTok API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (full URL or path).
            headers: Request headers.
            json_data: JSON body.
            data: Raw bytes body (for uploads).
            timeout: Request timeout.

        Returns:
            JSON response as dict.

        Raises:
            TikTokAPIError: If the API returns an error.
        """
        if not endpoint.startswith("http"):
            url = f"{self._config.api_base_url}/{endpoint}"
        else:
            url = endpoint

        # Default headers
        if headers is None:
            headers = {}

        # Add authorization
        headers["Authorization"] = f"Bearer {self._config.access_token}"

        _logger.info(f"TikTok API: {method} {endpoint}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                if json_data:
                    headers["Content-Type"] = "application/json; charset=UTF-8"
                    response = await client.post(url, headers=headers, json=json_data)
                elif data:
                    response = await client.post(url, headers=headers, content=data)
                else:
                    response = await client.post(url, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, headers=headers, content=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Handle response
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    return {"status": "ok"}  # Some endpoints don't return JSON

                # Check for API error in response
                if result.get("error", {}).get("code"):
                    error = result["error"]
                    raise TikTokAPIError(
                        message=error.get("message", "Unknown error"),
                        error_code=error.get("code"),
                        log_id=error.get("log_id"),
                    )

                return result

            # Handle HTTP errors
            try:
                error_data = response.json()
                error = error_data.get("error", {})
                raise TikTokAPIError(
                    message=error.get("message", f"HTTP {response.status_code}"),
                    error_code=error.get("code"),
                    log_id=error.get("log_id"),
                )
            except Exception:
                raise TikTokAPIError(f"HTTP {response.status_code}: {response.text[:200]}")

    async def _init_video_upload(
        self,
        video_size: int,
        chunk_size: int = TIKTOK_MAX_CHUNK_SIZE,
        total_chunk_count: int = 1,
    ) -> dict:
        """Initialize video upload and get upload URL.

        Args:
            video_size: Total video file size in bytes.
            chunk_size: Size of each chunk.
            total_chunk_count: Number of chunks.

        Returns:
            Dict with publish_id and upload_url.
        """
        # Determine source type
        if video_size <= TIKTOK_CHUNK_THRESHOLD:
            source = "FILE_UPLOAD"
        else:
            source = "FILE_UPLOAD"  # Chunked upload uses same source

        post_info = {
            "title": "",  # Set later with caption
            "privacy_level": "MUTUAL_FOLLOW_FRIENDS",  # Safe default until audit
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        }

        source_info = {
            "source": source,
            "video_size": video_size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunk_count,
        }

        result = await self._make_request(
            "POST",
            "post/publish/video/init/",
            json_data={
                "post_info": post_info,
                "source_info": source_info,
            },
        )

        data = result.get("data", {})
        return {
            "publish_id": data.get("publish_id"),
            "upload_url": data.get("upload_url"),
        }

    async def _upload_video_chunk(
        self,
        upload_url: str,
        chunk_data: bytes,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> bool:
        """Upload a video chunk.

        Args:
            upload_url: URL from init response.
            chunk_data: Chunk bytes.
            chunk_index: Index of this chunk (0-based).
            total_chunks: Total number of chunks.

        Returns:
            True if upload successful.
        """
        headers = {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(chunk_data)),
        }

        # For chunked uploads, add range header
        if total_chunks > 1:
            # Calculate byte range
            start_byte = chunk_index * len(chunk_data)
            end_byte = start_byte + len(chunk_data) - 1
            # Note: TikTok expects Content-Range format
            headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/*"

        await self._make_request(
            "PUT",
            upload_url,
            headers=headers,
            data=chunk_data,
            timeout=300.0,  # 5 min timeout for large chunks
        )
        return True

    async def _check_publish_status(self, publish_id: str) -> dict:
        """Check the status of a video publish.

        Args:
            publish_id: ID from init response.

        Returns:
            Status dict with status and optional error.
        """
        result = await self._make_request(
            "POST",
            "post/publish/status/fetch/",
            json_data={"publish_id": publish_id},
        )

        data = result.get("data", {})
        return {
            "status": data.get("status"),
            "fail_reason": data.get("fail_reason"),
            "publicaly_available_post_id": data.get("publicaly_available_post_id", []),
        }

    async def _wait_for_publish(
        self,
        publish_id: str,
        max_wait_seconds: int = 600,
        poll_interval: float = 10.0,
    ) -> dict:
        """Wait for video to finish publishing.

        Args:
            publish_id: ID from init response.
            max_wait_seconds: Maximum wait time.
            poll_interval: Time between status checks.

        Returns:
            Final status dict.

        Raises:
            TikTokAPIError: If publishing fails.
        """
        elapsed = 0.0

        while elapsed < max_wait_seconds:
            status = await self._check_publish_status(publish_id)
            status_code = status.get("status", "").upper()

            if status_code == "PUBLISH_COMPLETE":
                return status
            elif status_code == "FAILED":
                raise TikTokAPIError(
                    f"Video publish failed: {status.get('fail_reason', 'Unknown reason')}"
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            await self._emit_progress(
                "processing",
                min(90.0, 50.0 + (40.0 * elapsed / max_wait_seconds)),
                f"Processing video... ({int(elapsed)}s)",
            )

        raise TikTokAPIError("Video publish timed out")

    async def publish_reel(
        self,
        video_path: Path,
        caption: str,
        thumbnail_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> PublishResult:
        """Publish a video to TikTok.

        Args:
            video_path: Path to the video file.
            caption: Caption/title for the video.
            thumbnail_path: Optional thumbnail (TikTok auto-generates if not provided).
            **kwargs: Additional options:
                - privacy_level: str (default "MUTUAL_FOLLOW_FRIENDS")
                - disable_comment: bool (default False)
                - disable_duet: bool (default False)
                - disable_stitch: bool (default False)

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

        # Get video size
        video_size = video_path.stat().st_size

        if video_size > TIKTOK_MAX_VIDEO_SIZE:
            return self._make_result(
                success=False,
                error=f"Video too large: {video_size / 1024 / 1024:.1f}MB (max 4GB)",
            )

        try:
            await self._emit_progress("init", 5.0, "Initializing TikTok upload...")

            # Calculate chunks
            if video_size <= TIKTOK_CHUNK_THRESHOLD:
                chunk_size = video_size
                total_chunks = 1
            else:
                chunk_size = TIKTOK_MAX_CHUNK_SIZE
                total_chunks = math.ceil(video_size / chunk_size)

            # Initialize upload
            init_result = await self._init_video_upload(
                video_size=video_size,
                chunk_size=chunk_size,
                total_chunk_count=total_chunks,
            )

            publish_id = init_result["publish_id"]
            upload_url = init_result["upload_url"]

            if not upload_url:
                return self._make_result(
                    success=False,
                    error="Failed to get upload URL from TikTok",
                )

            await self._emit_progress("upload", 10.0, "Uploading video...")

            # Upload video chunks
            with open(video_path, "rb") as f:
                for chunk_index in range(total_chunks):
                    chunk_data = f.read(chunk_size)
                    await self._upload_video_chunk(
                        upload_url,
                        chunk_data,
                        chunk_index,
                        total_chunks,
                    )

                    progress = 10.0 + (40.0 * (chunk_index + 1) / total_chunks)
                    await self._emit_progress(
                        "upload",
                        progress,
                        f"Uploaded chunk {chunk_index + 1}/{total_chunks}",
                    )

            await self._emit_progress("processing", 50.0, "Processing video...")

            # Wait for publish to complete
            final_status = await self._wait_for_publish(publish_id)

            # Get post ID if available
            post_ids = final_status.get("publicaly_available_post_id", [])
            post_id = post_ids[0] if post_ids else None

            await self._emit_progress("complete", 100.0, "Published to TikTok!")

            return self._make_result(
                success=True,
                media_id=post_id or publish_id,
                permalink=f"https://www.tiktok.com/@user/video/{post_id}" if post_id else None,
                publish_id=publish_id,
            )

        except TikTokAPIError as e:
            _logger.error(f"TikTok API error: {e}")
            return self._make_result(
                success=False,
                error=str(e),
                error_code=e.error_code,
                log_id=e.log_id,
            )

        except Exception as e:
            _logger.error(f"TikTok publish error: {e}")
            return self._make_result(
                success=False,
                error=f"TikTok publish failed: {str(e)}",
            )

    async def check_credentials(self) -> tuple[bool, str]:
        """Verify TikTok credentials are valid.

        Returns:
            Tuple of (is_valid, message).
        """
        # First validate config has all required fields
        is_valid, error_msg = self._config.validate()
        if not is_valid:
            return False, error_msg

        try:
            # Try to get user info
            result = await self._make_request(
                "GET",
                "user/info/",
                headers={"Content-Type": "application/json"},
            )

            data = result.get("data", {}).get("user", {})
            display_name = data.get("display_name", "unknown")

            return True, f"Connected to TikTok as {display_name}"

        except TikTokAPIError as e:
            return False, f"TikTok credential check failed: {e}"
        except Exception as e:
            return False, f"TikTok connection error: {str(e)}"
