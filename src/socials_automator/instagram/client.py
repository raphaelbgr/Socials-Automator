"""Instagram Graph API client for publishing carousel posts."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

import httpx

from .models import (
    InstagramConfig,
    InstagramProgress,
    InstagramPublishResult,
    InstagramPostStatus,
)

# Set up file logging for API calls
_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_api_logger = logging.getLogger("instagram_api")
_api_logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(_log_dir / "instagram_api.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
_api_logger.addHandler(_file_handler)


class InstagramAPIError(Exception):
    """Base exception for Instagram API errors."""

    def __init__(self, message: str, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


class InstagramClient:
    """Instagram Graph API client for publishing carousel posts.

    Implements the container-based publishing workflow:
    1. Create media container for each image (requires public URL)
    2. Create carousel container referencing all child containers
    3. Publish the carousel

    API Reference:
    https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/content-publishing
    """

    def __init__(
        self,
        config: InstagramConfig,
        progress_callback: Callable[[InstagramProgress], Awaitable[None]] | None = None,
    ):
        """Initialize Instagram client.

        Args:
            config: Instagram API configuration
            progress_callback: Async callback for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback
        self.base_url = f"https://graph.facebook.com/{config.api_version}"

        # Progress tracking
        self._progress = InstagramProgress()

        # API call counter for logging
        self._api_call_count = 0
        _api_logger.info(f"=== NEW SESSION === Instagram User ID: {config.instagram_user_id}")

    async def _report_progress(
        self,
        status: InstagramPostStatus,
        step: str,
        percent: float = 0.0,
    ) -> None:
        """Update and report progress."""
        self._progress.status = status
        self._progress.current_step = step
        self._progress.progress_percent = percent
        if self.progress_callback:
            await self.progress_callback(self._progress)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
    ) -> dict:
        """Make a request to the Instagram Graph API.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body for POST

        Returns:
            JSON response as dict

        Raises:
            InstagramAPIError: If the API returns an error
        """
        self._api_call_count += 1
        url = f"{self.base_url}/{endpoint}"

        # Add access token to params
        if params is None:
            params = {}
        params["access_token"] = self.config.access_token

        # Log the API call (without token)
        log_params = {k: v for k, v in params.items() if k != "access_token"}
        _api_logger.info(f"API CALL #{self._api_call_count} | {method} {endpoint} | params: {log_params}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, params=params, data=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            result = response.json()

            # Log the response
            if "error" in result:
                _api_logger.error(f"API CALL #{self._api_call_count} | ERROR: {result['error']}")
            else:
                _api_logger.info(f"API CALL #{self._api_call_count} | SUCCESS: {list(result.keys())}")

            # Check for API errors
            if "error" in result:
                error = result["error"]
                raise InstagramAPIError(
                    message=error.get("message", "Unknown API error"),
                    error_code=error.get("code"),
                )

            return result

    async def create_image_container(
        self,
        image_url: str,
        is_carousel_item: bool = True,
    ) -> str:
        """Create a media container for a single image.

        Args:
            image_url: Public URL of the image
            is_carousel_item: Whether this is part of a carousel

        Returns:
            Container ID (creation_id)
        """
        endpoint = f"{self.config.instagram_user_id}/media"
        params = {
            "image_url": image_url,
            "is_carousel_item": "true" if is_carousel_item else "false",
        }

        result = await self._make_request("POST", endpoint, params=params)
        return result["id"]

    async def create_carousel_container(
        self,
        children_ids: list[str],
        caption: str,
    ) -> str:
        """Create a carousel container from child containers.

        Args:
            children_ids: List of child container IDs
            caption: Post caption (max 2200 chars)

        Returns:
            Carousel container ID
        """
        endpoint = f"{self.config.instagram_user_id}/media"
        params = {
            "media_type": "CAROUSEL",
            "children": ",".join(children_ids),
            "caption": caption[:2200],  # Instagram caption limit
        }

        result = await self._make_request("POST", endpoint, params=params)
        return result["id"]

    async def check_container_status(self, container_id: str) -> dict:
        """Check if a container is ready for publishing.

        Args:
            container_id: The container ID to check

        Returns:
            Dict with status_code and status fields
        """
        endpoint = container_id
        params = {"fields": "status_code,status"}

        return await self._make_request("GET", endpoint, params=params)

    async def wait_for_container(
        self,
        container_id: str,
        max_wait_seconds: int = 300,
        poll_interval: float = 10.0,  # Increased from 5s to reduce API calls
    ) -> bool:
        """Wait for a container to be ready for publishing.

        Args:
            container_id: The container ID to wait for
            max_wait_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            True if container is ready, False if timeout

        Raises:
            InstagramAPIError: If container processing failed
        """
        elapsed = 0.0
        while elapsed < max_wait_seconds:
            status = await self.check_container_status(container_id)
            status_code = status.get("status_code", "").upper()

            if status_code == "FINISHED":
                return True
            elif status_code == "ERROR":
                raise InstagramAPIError(
                    f"Container processing failed: {status.get('status', 'Unknown error')}"
                )
            elif status_code == "EXPIRED":
                raise InstagramAPIError("Container expired before publishing")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return False

    async def publish_container(self, creation_id: str) -> str:
        """Publish a container to Instagram.

        Args:
            creation_id: The container ID to publish

        Returns:
            Media ID of the published post
        """
        endpoint = f"{self.config.instagram_user_id}/media_publish"
        params = {"creation_id": creation_id}

        result = await self._make_request("POST", endpoint, params=params)
        return result["id"]

    async def get_media_permalink(self, media_id: str) -> str | None:
        """Get the permalink for a published media.

        Args:
            media_id: The media ID

        Returns:
            Permalink URL or None if not available
        """
        try:
            params = {"fields": "permalink"}
            result = await self._make_request("GET", media_id, params=params)
            return result.get("permalink")
        except InstagramAPIError:
            return None

    async def publish_carousel(
        self,
        image_urls: list[str],
        caption: str,
    ) -> InstagramPublishResult:
        """High-level method to publish a complete carousel post.

        This orchestrates the full publishing workflow:
        1. Create container for each image
        2. Wait for all containers to be ready
        3. Create carousel container
        4. Wait for carousel container to be ready
        5. Publish

        Args:
            image_urls: List of public image URLs
            caption: Post caption with hashtags

        Returns:
            InstagramPublishResult with success status and details
        """
        self._progress = InstagramProgress(
            total_images=len(image_urls),
            image_urls=image_urls,
        )

        try:
            # Step 1: Create image containers
            await self._report_progress(
                InstagramPostStatus.CREATING_CONTAINERS,
                "Creating image containers...",
                10.0,
            )

            container_ids = []
            for i, url in enumerate(image_urls):
                container_id = await self.create_image_container(url)
                container_ids.append(container_id)
                self._progress.containers_created = i + 1
                self._progress.container_ids = container_ids

                await self._report_progress(
                    InstagramPostStatus.CREATING_CONTAINERS,
                    f"Created container {i + 1}/{len(image_urls)}",
                    10.0 + (30.0 * (i + 1) / len(image_urls)),
                )

            # Step 2: Wait for containers to be ready
            await self._report_progress(
                InstagramPostStatus.CREATING_CONTAINERS,
                "Waiting for containers to process...",
                45.0,
            )

            for i, container_id in enumerate(container_ids):
                ready = await self.wait_for_container(container_id)
                if not ready:
                    raise InstagramAPIError(
                        f"Container {i + 1} timed out waiting to be ready"
                    )

            # Step 3: Create carousel container
            await self._report_progress(
                InstagramPostStatus.CREATING_CONTAINERS,
                "Creating carousel container...",
                60.0,
            )

            carousel_id = await self.create_carousel_container(container_ids, caption)
            self._progress.carousel_container_id = carousel_id

            # Step 4: Wait for carousel to be ready
            await self._report_progress(
                InstagramPostStatus.CREATING_CONTAINERS,
                "Waiting for carousel to process...",
                70.0,
            )

            ready = await self.wait_for_container(carousel_id)
            if not ready:
                raise InstagramAPIError("Carousel container timed out waiting to be ready")

            # Step 5: Publish
            await self._report_progress(
                InstagramPostStatus.PUBLISHING,
                "Publishing to Instagram...",
                85.0,
            )

            media_id = await self.publish_container(carousel_id)

            # Get permalink
            permalink = await self.get_media_permalink(media_id)

            # Success
            await self._report_progress(
                InstagramPostStatus.PUBLISHED,
                "Published successfully!",
                100.0,
            )

            _api_logger.info(f"=== SESSION COMPLETE === Total API calls: {self._api_call_count}")

            return InstagramPublishResult(
                success=True,
                media_id=media_id,
                permalink=permalink,
                container_ids=container_ids,
                image_urls=image_urls,
                published_at=datetime.now(),
            )

        except InstagramAPIError as e:
            # Check for "ghost publish" - post went through despite error
            ghost_media_id = None
            ghost_permalink = None

            if "rate limit" in str(e).lower() or e.error_code == 4:
                _api_logger.warning("Rate limit error - checking if post was actually published...")
                await asyncio.sleep(2)  # Brief wait before checking
                recent = await self.get_recent_media(limit=1)
                if recent:
                    recent_post = recent[0]
                    # Check if this was posted in the last 60 seconds
                    from datetime import timezone
                    recent_time = datetime.fromisoformat(recent_post.get("timestamp", "").replace("Z", "+00:00"))
                    if (datetime.now(timezone.utc) - recent_time).total_seconds() < 60:
                        ghost_media_id = recent_post.get("id")
                        ghost_permalink = recent_post.get("permalink")
                        _api_logger.warning(f"GHOST PUBLISH DETECTED! Post went through: {ghost_permalink}")

            if ghost_media_id:
                # Post actually succeeded!
                await self._report_progress(
                    InstagramPostStatus.PUBLISHED,
                    "Published (detected after rate limit error)",
                    100.0,
                )
                _api_logger.info(f"=== SESSION COMPLETE (GHOST) === Total API calls: {self._api_call_count}")
                return InstagramPublishResult(
                    success=True,
                    media_id=ghost_media_id,
                    permalink=ghost_permalink,
                    container_ids=container_ids,
                    image_urls=image_urls,
                    published_at=datetime.now(),
                )

            self._progress.error = str(e)
            await self._report_progress(
                InstagramPostStatus.FAILED,
                f"Failed: {e}",
                self._progress.progress_percent,
            )

            _api_logger.error(f"=== SESSION FAILED === Total API calls: {self._api_call_count} | Error: {e}")

            return InstagramPublishResult(
                success=False,
                error_message=str(e),
                container_ids=self._progress.container_ids,
                image_urls=image_urls,
            )

        except Exception as e:
            self._progress.error = str(e)
            await self._report_progress(
                InstagramPostStatus.FAILED,
                f"Unexpected error: {e}",
                self._progress.progress_percent,
            )

            return InstagramPublishResult(
                success=False,
                error_message=f"Unexpected error: {e}",
                container_ids=self._progress.container_ids,
                image_urls=image_urls,
            )

    async def get_recent_media(self, limit: int = 1) -> list[dict]:
        """Get recent media posts from the account.

        Args:
            limit: Number of recent posts to fetch

        Returns:
            List of media objects with id, timestamp, permalink
        """
        try:
            endpoint = f"{self.config.instagram_user_id}/media"
            params = {"fields": "id,timestamp,permalink,media_type", "limit": limit}
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("data", [])
        except InstagramAPIError:
            return []

    async def validate_token(self) -> dict:
        """Validate the access token and get account info.

        Returns:
            Dict with user_id and username

        Raises:
            InstagramAPIError: If token is invalid
        """
        endpoint = self.config.instagram_user_id
        params = {"fields": "id,username"}

        return await self._make_request("GET", endpoint, params=params)
