"""Instagram Graph API client for publishing carousel posts."""

import asyncio
import logging
import unicodedata
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

# Set up file logging for API calls (console output suppressed)
_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_api_logger = logging.getLogger("instagram_api")
_api_logger.setLevel(logging.DEBUG)
_api_logger.propagate = False  # Don't propagate to root logger (no console output)
_file_handler = logging.FileHandler(_log_dir / "instagram_api.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
_api_logger.addHandler(_file_handler)


def _sanitize_caption(caption: str) -> str:
    """Sanitize caption to remove problematic characters for Instagram API.

    Handles:
    - Unicode normalization (NFC)
    - Emojis (removed - can cause encoding issues on Windows)
    - Zero-width characters
    - Control characters (except newlines)
    - Smart quotes -> regular quotes
    - Non-breaking spaces -> regular spaces

    Args:
        caption: Raw caption text

    Returns:
        Sanitized caption safe for Instagram API
    """
    import re

    if not caption:
        return ""

    # First, remove emojis using regex (most reliable method)
    # This pattern covers most emoji ranges including:
    # - Emoticons, Dingbats, Symbols
    # - Supplementary planes (U+1F000 and above)
    # - Regional indicators, skin tones, etc.
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"  # Misc Symbols, Emoticons, etc.
        "\U0001FA00-\U0001FAFF"  # Extended-A symbols
        "\U00002702-\U000027B0"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation selectors
        "\U0001F1E0-\U0001F1FF"  # Flags (regional indicators)
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002300-\U000023FF"  # Misc technical
        "\U0000200D"             # Zero-width joiner (used in emoji sequences)
        "]+",
        flags=re.UNICODE
    )
    caption = emoji_pattern.sub('', caption)

    # Normalize Unicode to NFC (composed form)
    caption = unicodedata.normalize('NFC', caption)

    # Replace problematic characters with safe alternatives
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2014': '-',   # Em dash
        '\u2013': '-',   # En dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\ufeff': '',    # BOM
        '\u00ad': '',    # Soft hyphen
        '\u2028': '\n',  # Line separator
        '\u2029': '\n',  # Paragraph separator
    }

    for old, new in replacements.items():
        caption = caption.replace(old, new)

    # Remove remaining control/format characters and any chars above BMP
    cleaned = []
    for char in caption:
        if char in '\n\r\t':
            cleaned.append(char)
        elif unicodedata.category(char) == 'Cc':  # Control characters
            continue
        elif unicodedata.category(char) == 'Cf':  # Format characters
            continue
        elif unicodedata.category(char) == 'So':  # Symbol, Other
            continue
        elif ord(char) > 0xFFFF:  # Supplementary plane chars
            continue
        else:
            cleaned.append(char)

    caption = ''.join(cleaned)

    # Clean up double spaces left by emoji removal
    while '  ' in caption:
        caption = caption.replace('  ', ' ')

    # Normalize multiple newlines to max 2
    while '\n\n\n' in caption:
        caption = caption.replace('\n\n\n', '\n\n')

    return caption.strip()


class InstagramAPIError(Exception):
    """Base exception for Instagram API errors."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        error_subcode: int | None = None,
        is_retryable: bool = False,
        user_title: str | None = None,
        user_message: str | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.error_subcode = error_subcode
        self.is_retryable = is_retryable
        self.user_title = user_title
        self.user_message = user_message


# Known Instagram API error codes with explanations and retry guidance
INSTAGRAM_ERROR_CODES = {
    # Media processing errors (often retryable)
    2207032: {
        "name": "MEDIA_UPLOAD_FAILED",
        "description": "Instagram failed to process the media upload",
        "user_message": "Instagram's servers failed to process your images. This is usually temporary.",
        "is_retryable": True,
        "retry_delay": 30,  # seconds
    },
    2207026: {
        "name": "MEDIA_NOT_READY",
        "description": "Media container is not ready yet",
        "user_message": "Instagram is still processing your images. Waiting...",
        "is_retryable": True,
        "retry_delay": 10,
    },
    2207001: {
        "name": "MEDIA_TYPE_NOT_SUPPORTED",
        "description": "Unsupported media type",
        "user_message": "One of your images has an unsupported format. Instagram requires JPEG images.",
        "is_retryable": False,
    },
    2207003: {
        "name": "MEDIA_SIZE_ERROR",
        "description": "Media exceeds size limits",
        "user_message": "One of your images is too large. Instagram has a 8MB limit per image.",
        "is_retryable": False,
    },
    2207050: {
        "name": "CAROUSEL_MIN_CHILDREN",
        "description": "Carousel needs at least 2 items",
        "user_message": "Carousels require at least 2 images.",
        "is_retryable": False,
    },
    2207051: {
        "name": "CAROUSEL_MAX_CHILDREN",
        "description": "Carousel exceeds 10 items",
        "user_message": "Carousels cannot have more than 10 images.",
        "is_retryable": False,
    },
    # Rate limiting
    4: {
        "name": "RATE_LIMIT",
        "description": "Rate limit reached",
        "user_message": "Instagram rate limit reached. The post may have still gone through - checking...",
        "is_retryable": True,
        "retry_delay": 60,
    },
    9: {
        "name": "APP_RATE_LIMIT",
        "description": "Application request limit reached",
        "user_message": "App rate limit reached. Will retry with longer delay.",
        "is_retryable": True,
        "retry_delay": 300,  # 5 minutes - this is a more severe rate limit
    },
    17: {
        "name": "USER_RATE_LIMIT",
        "description": "User request limit reached",
        "user_message": "You've made too many requests. Please wait a few minutes.",
        "is_retryable": True,
        "retry_delay": 120,
    },
    # Auth errors (not retryable)
    190: {
        "name": "ACCESS_TOKEN_EXPIRED",
        "description": "Access token expired",
        "user_message": "Your Instagram access token has expired. Run: socials token <profile> --refresh",
        "is_retryable": False,
    },
    10: {
        "name": "PERMISSION_DENIED",
        "description": "Permission denied",
        "user_message": "Your app doesn't have permission to publish. Check your Instagram API setup.",
        "is_retryable": False,
    },
}

# Error SUBCODES - these provide more specific info when combined with error codes
# These take precedence over the main error code when present
INSTAGRAM_ERROR_SUBCODES = {
    2207069: {
        "name": "DAILY_POSTING_LIMIT",
        "description": "Content Publishing API daily limit exceeded",
        "user_message": (
            "You've reached Instagram's DAILY POSTING LIMIT.\n\n"
            "The Content Publishing API allows ~25 posts per day per account.\n"
            "This limit resets at midnight UTC.\n\n"
            "What to do:\n"
            "  - Wait until midnight UTC (or try again tomorrow)\n"
            "  - Each carousel counts as multiple actions (1 per image + carousel + publish)\n"
            "  - Failed retries also count towards the limit"
        ),
        "is_retryable": False,  # NOT retryable - must wait for daily reset
    },
}


def get_error_info(error_code: int | None, error_subcode: int | None = None) -> dict:
    """Get detailed error information for an Instagram error code.

    Args:
        error_code: Main error code from API response.
        error_subcode: Sub-error code (takes precedence if known).

    Returns:
        Dict with name, description, user_message, is_retryable, and optional retry_delay.
    """
    # Check subcode first - it provides more specific info
    if error_subcode is not None and error_subcode in INSTAGRAM_ERROR_SUBCODES:
        return INSTAGRAM_ERROR_SUBCODES[error_subcode]

    if error_code is None:
        return {
            "name": "UNKNOWN",
            "description": "Unknown error",
            "user_message": "An unknown error occurred with Instagram.",
            "is_retryable": False,
        }
    return INSTAGRAM_ERROR_CODES.get(error_code, {
        "name": f"ERROR_{error_code}",
        "description": f"Unknown error code: {error_code}",
        "user_message": f"Instagram returned error code {error_code}. This may be temporary - try again.",
        "is_retryable": True,  # Unknown errors might be retryable
        "retry_delay": 30,
    })


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
                error_code = error.get("code")
                error_subcode = error.get("error_subcode")
                error_info = get_error_info(error_code, error_subcode)
                raise InstagramAPIError(
                    message=error.get("message", "Unknown API error"),
                    error_code=error_code,
                    error_subcode=error_subcode,
                    is_retryable=error_info.get("is_retryable", False),
                    user_title=error.get("error_user_title"),
                    user_message=error_info.get("user_message"),
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
            "caption": _sanitize_caption(caption)[:2200],  # Sanitize + limit
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
        import re

        elapsed = 0.0
        while elapsed < max_wait_seconds:
            status = await self.check_container_status(container_id)
            status_code = status.get("status_code", "").upper()

            if status_code == "FINISHED":
                return True
            elif status_code == "ERROR":
                # Try to extract error code from status message
                status_msg = status.get("status", "Unknown error")
                error_code = None

                # Look for error code pattern like "error code 2207032"
                match = re.search(r"error code (\d+)", status_msg, re.IGNORECASE)
                if match:
                    error_code = int(match.group(1))

                error_info = get_error_info(error_code)
                raise InstagramAPIError(
                    f"Container processing failed: {status_msg}",
                    error_code=error_code,
                    is_retryable=error_info.get("is_retryable", False),
                )
            elif status_code == "EXPIRED":
                raise InstagramAPIError(
                    "Container expired before publishing",
                    is_retryable=True,  # Can retry with new containers
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return False

    async def wait_for_container_with_progress(
        self,
        container_id: str,
        max_wait_seconds: int = 600,
        poll_interval: float = 15.0,
    ) -> bool:
        """Wait for a container with progress reporting.

        Same as wait_for_container but emits progress updates.

        Args:
            container_id: The container ID to wait for
            max_wait_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            True if container is ready, False if timeout

        Raises:
            InstagramAPIError: If container processing failed
        """
        import re

        elapsed = 0.0
        check_count = 0
        while elapsed < max_wait_seconds:
            status = await self.check_container_status(container_id)
            status_code = status.get("status_code", "").upper()
            check_count += 1

            # Calculate progress (30% to 75% during processing)
            progress = 30.0 + (45.0 * min(elapsed / max_wait_seconds, 1.0))

            if status_code == "FINISHED":
                return True
            elif status_code == "ERROR":
                status_msg = status.get("status", "Unknown error")
                error_code = None

                match = re.search(r"error code (\d+)", status_msg, re.IGNORECASE)
                if match:
                    error_code = int(match.group(1))

                error_info = get_error_info(error_code)
                raise InstagramAPIError(
                    f"Container processing failed: {status_msg}",
                    error_code=error_code,
                    is_retryable=error_info.get("is_retryable", False),
                )
            elif status_code == "EXPIRED":
                raise InstagramAPIError(
                    "Container expired before publishing",
                    is_retryable=True,
                )
            elif status_code == "IN_PROGRESS":
                # Show progress with elapsed time
                await self._report_progress(
                    InstagramPostStatus.CREATING_CONTAINERS,
                    f"Processing... ({int(elapsed)}s elapsed, check #{check_count})",
                    progress,
                )
            else:
                # Unknown status - show it
                await self._report_progress(
                    InstagramPostStatus.CREATING_CONTAINERS,
                    f"Status: {status_code} ({int(elapsed)}s elapsed)",
                    progress,
                )

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
        max_retries: int = 3,
    ) -> InstagramPublishResult:
        """High-level method to publish a complete carousel post.

        This orchestrates the full publishing workflow:
        1. Create container for each image
        2. Wait for all containers to be ready
        3. Create carousel container
        4. Wait for carousel container to be ready
        5. Publish

        Includes automatic retry for transient errors (like 2207032).

        Args:
            image_urls: List of public image URLs
            caption: Post caption with hashtags
            max_retries: Maximum retry attempts for transient errors

        Returns:
            InstagramPublishResult with success status and details
        """
        last_error = None
        container_ids = []
        publish_attempted = False  # Track if we tried to publish - NEVER retry after this

        for attempt in range(max_retries + 1):
            self._progress = InstagramProgress(
                total_images=len(image_urls),
                image_urls=image_urls,
            )

            try:
                # Step 1: Create image containers
                if attempt > 0:
                    await self._report_progress(
                        InstagramPostStatus.CREATING_CONTAINERS,
                        f"Retry {attempt}/{max_retries}: Creating containers...",
                        5.0,
                    )
                else:
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
                            f"Container {i + 1} timed out waiting to be ready",
                            is_retryable=True,
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
                    raise InstagramAPIError(
                        "Carousel container timed out waiting to be ready",
                        is_retryable=True,
                    )

                # Step 5: Publish
                await self._report_progress(
                    InstagramPostStatus.PUBLISHING,
                    "Publishing to Instagram...",
                    85.0,
                )

                # CRITICAL: Mark that we're attempting publish - NEVER retry after this point
                # because the post might have gone through even if we get an error
                publish_attempted = True
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
                last_error = e
                error_info = get_error_info(e.error_code, e.error_subcode)

                # Log the error with details
                _api_logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: "
                    f"[{error_info['name']}] {e}"
                )

                # Check for DAILY POSTING LIMIT - don't retry, show clear message
                if e.error_subcode == 2207069:
                    _api_logger.error(f"DAILY POSTING LIMIT HIT - not retrying")
                    await self._report_progress(
                        InstagramPostStatus.FAILED,
                        "Daily posting limit reached!",
                        self._progress.progress_percent,
                    )
                    # Break out immediately - no point retrying
                    break

                # CHECK FOR GHOST PUBLISH only if we've attempted to publish
                # Ghost publish can ONLY happen after calling publish_container()
                # Rate limit during container creation does NOT mean post went through
                if publish_attempted:
                    _api_logger.warning(f"Checking for ghost publish (publish_attempted={publish_attempted})...")
                    await self._report_progress(
                        InstagramPostStatus.PUBLISHING,
                        "Checking if post went through...",
                        self._progress.progress_percent,
                    )
                    await asyncio.sleep(3)  # Brief wait

                    try:
                        recent = await self.get_recent_media(limit=1)
                        if recent:
                            recent_post = recent[0]
                            from datetime import timezone
                            recent_time = datetime.fromisoformat(
                                recent_post.get("timestamp", "").replace("Z", "+00:00")
                            )
                            seconds_ago = (datetime.now(timezone.utc) - recent_time).total_seconds()

                            if seconds_ago < 120:  # Posted in last 2 minutes = ghost publish!
                                _api_logger.warning(f"GHOST PUBLISH DETECTED! Post is live ({seconds_ago:.0f}s ago)")
                                await self._report_progress(
                                    InstagramPostStatus.PUBLISHED,
                                    "Published! (detected after error)",
                                    100.0,
                                )
                                return InstagramPublishResult(
                                    success=True,
                                    media_id=recent_post.get("id"),
                                    permalink=recent_post.get("permalink"),
                                    container_ids=container_ids,
                                    image_urls=image_urls,
                                    published_at=datetime.now(),
                                )
                            else:
                                _api_logger.info(f"Most recent post is {seconds_ago:.0f}s old - not ours")
                    except Exception as check_err:
                        _api_logger.warning(f"Ghost check failed: {check_err}")

                    # If we already attempted publish, don't retry (could cause duplicates)
                    if publish_attempted:
                        _api_logger.warning("Post not found after publish attempt - not retrying to avoid duplicates")
                        break

                # Before publish: retry is safe (we haven't published anything yet)
                if e.is_retryable and attempt < max_retries:
                    retry_delay = error_info.get("retry_delay", 30)
                    # Show clear message about rate limiting during container creation
                    retry_msg = f"Rate limited during container creation. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                    await self._report_progress(
                        InstagramPostStatus.CREATING_CONTAINERS,
                        retry_msg,
                        self._progress.progress_percent,
                    )
                    _api_logger.info(f"Waiting {retry_delay}s before retry (container creation phase)...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Not retryable or out of retries - fall through to error handling
                break

            except Exception as e:
                # Non-Instagram errors (network, etc) - don't retry
                last_error = InstagramAPIError(f"Unexpected error: {e}")
                break

        # All retries exhausted or error after publish
        # Final check: if we attempted publish, wait a bit more and verify
        if publish_attempted and last_error:
            _api_logger.info("Final Instagram check after publish error (waiting 10s)...")
            await self._report_progress(
                InstagramPostStatus.PUBLISHING,
                "Final check - waiting for Instagram to sync...",
                self._progress.progress_percent,
            )
            await asyncio.sleep(10)  # Extra wait for Instagram to sync

            try:
                recent = await self.get_recent_media(limit=1)
                if recent:
                    recent_post = recent[0]
                    from datetime import timezone
                    recent_time = datetime.fromisoformat(
                        recent_post.get("timestamp", "").replace("Z", "+00:00")
                    )
                    seconds_ago = (datetime.now(timezone.utc) - recent_time).total_seconds()
                    # Extended window - post could have been made up to 5 minutes ago
                    if seconds_ago < 300:
                        _api_logger.info(f"Post confirmed on Instagram ({seconds_ago:.0f}s ago)!")
                        await self._report_progress(
                            InstagramPostStatus.PUBLISHED,
                            "Published! (confirmed on final check)",
                            100.0,
                        )
                        return InstagramPublishResult(
                            success=True,
                            media_id=recent_post.get("id"),
                            permalink=recent_post.get("permalink"),
                            container_ids=container_ids,
                            image_urls=image_urls,
                            published_at=datetime.now(),
                        )
                    else:
                        _api_logger.warning(f"Most recent post is {seconds_ago:.0f}s old - not our post")
            except Exception as e:
                _api_logger.warning(f"Final check failed: {e}")

        # Build user-friendly error message
        if last_error:
            error_code = last_error.error_code if hasattr(last_error, 'error_code') else None
            error_subcode = last_error.error_subcode if hasattr(last_error, 'error_subcode') else None
            error_info = get_error_info(error_code, error_subcode)
            user_message = error_info.get("user_message", str(last_error))
            error_name = error_info.get("name", "UNKNOWN")

            self._progress.error = str(last_error)
            await self._report_progress(
                InstagramPostStatus.FAILED,
                f"[{error_name}] {user_message[:50]}..." if len(user_message) > 50 else f"[{error_name}] {user_message}",
                self._progress.progress_percent,
            )

            _api_logger.error(
                f"=== SESSION FAILED === Total API calls: {self._api_call_count} | "
                f"Error: [{error_name}] {last_error}"
            )

            return InstagramPublishResult(
                success=False,
                error_message=f"[{error_name}] {user_message}",
                container_ids=container_ids,
                image_urls=image_urls,
            )

        # Should never reach here, but just in case
        return InstagramPublishResult(
            success=False,
            error_message="Unknown error occurred",
            container_ids=container_ids,
            image_urls=image_urls,
        )

    async def create_reel_container(
        self,
        video_url: str,
        caption: str,
        share_to_feed: bool = True,
        thumb_offset_ms: int | None = None,
        cover_url: str | None = None,
    ) -> str:
        """Create a Reel container for publishing.

        Args:
            video_url: Public URL of the video (must be MP4, max 90s for API)
            caption: Post caption (max 2200 chars)
            share_to_feed: Whether to also share to main feed (default True)
            thumb_offset_ms: Thumbnail offset in milliseconds (optional, ignored if cover_url is set)
            cover_url: Custom thumbnail image URL (1080x1920, JPEG/PNG)

        Returns:
            Container ID (creation_id)
        """
        endpoint = f"{self.config.instagram_user_id}/media"
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": _sanitize_caption(caption)[:2200],  # Sanitize + limit
            "share_to_feed": "true" if share_to_feed else "false",
        }

        # cover_url takes precedence over thumb_offset
        if cover_url:
            params["cover_url"] = cover_url
        elif thumb_offset_ms is not None:
            params["thumb_offset"] = str(thumb_offset_ms)

        result = await self._make_request("POST", endpoint, params=params)
        return result["id"]

    async def publish_reel(
        self,
        video_url: str,
        caption: str,
        share_to_feed: bool = True,
        cover_url: str | None = None,
        max_retries: int = 3,
    ) -> InstagramPublishResult:
        """Publish a Reel to Instagram.

        Complete workflow:
        1. Create Reel container with video URL
        2. Wait for video processing (can take 1-5 minutes)
        3. Publish the container

        Args:
            video_url: Public URL of the video (Cloudinary, etc.)
            caption: Full caption including hashtags
            share_to_feed: Whether to share to main feed
            cover_url: Custom thumbnail image URL (optional)
            max_retries: Number of retries on failure

        Returns:
            InstagramPublishResult with success status and details
        """
        _api_logger.info(f"=== NEW SESSION === Instagram User ID: {self.config.instagram_user_id}")

        self._progress = InstagramProgress(
            total_images=1,  # Reels are single video
            image_urls=[video_url],
        )

        container_id = None
        last_error = None
        publish_attempted = False

        for attempt in range(max_retries + 1):
            try:
                # Step 1: Create Reel container
                if container_id is None:
                    await self._report_progress(
                        InstagramPostStatus.CREATING_CONTAINERS,
                        "Creating Reel container on Instagram...",
                        10.0,
                    )
                    container_id = await self.create_reel_container(
                        video_url=video_url,
                        caption=caption,
                        share_to_feed=share_to_feed,
                        cover_url=cover_url,
                    )
                    _api_logger.info(f"Reel container created: {container_id}")
                    await self._report_progress(
                        InstagramPostStatus.CREATING_CONTAINERS,
                        f"Container created: {container_id[:20]}...",
                        20.0,
                    )

                # Step 2: Wait for video processing (longer than images)
                await self._report_progress(
                    InstagramPostStatus.CREATING_CONTAINERS,
                    "Waiting for Instagram to process video (up to 10 min)...",
                    30.0,
                )
                ready = await self.wait_for_container_with_progress(
                    container_id,
                    max_wait_seconds=600,  # 10 minutes max for video
                    poll_interval=15.0,  # Check every 15s
                )

                if not ready:
                    raise InstagramAPIError(
                        "Video processing timed out after 10 minutes",
                        is_retryable=True,
                    )

                await self._report_progress(
                    InstagramPostStatus.CREATING_CONTAINERS,
                    "Video processing complete!",
                    75.0,
                )

                # Step 3: Publish
                await self._report_progress(
                    InstagramPostStatus.PUBLISHING,
                    "Publishing Reel to feed...",
                    80.0,
                )

                publish_attempted = True
                media_id = await self.publish_container(container_id)
                _api_logger.info(f"Reel published! Media ID: {media_id}")

                # Get permalink
                permalink = await self.get_media_permalink(media_id)

                await self._report_progress(
                    InstagramPostStatus.PUBLISHED,
                    "Reel published successfully!",
                    100.0,
                )

                _api_logger.info(f"=== SESSION COMPLETE === Total API calls: {self._api_call_count}")

                return InstagramPublishResult(
                    success=True,
                    media_id=media_id,
                    permalink=permalink,
                    container_ids=[container_id],
                    image_urls=[video_url],
                    published_at=datetime.now(),
                )

            except InstagramAPIError as e:
                last_error = e
                error_info = get_error_info(e.error_code, e.error_subcode)

                _api_logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: "
                    f"[{error_info['name']}] {e}"
                )

                # Check for DAILY POSTING LIMIT
                if e.error_subcode == 2207069:
                    _api_logger.error(f"DAILY POSTING LIMIT HIT - not retrying")
                    await self._report_progress(
                        InstagramPostStatus.FAILED,
                        "Daily posting limit reached!",
                        self._progress.progress_percent,
                    )
                    break

                # If publish was attempted, check for ghost publish
                if publish_attempted:
                    _api_logger.warning("Checking for ghost publish...")
                    await asyncio.sleep(5)

                    try:
                        recent = await self.get_recent_media(limit=1)
                        if recent:
                            recent_post = recent[0]
                            from datetime import timezone
                            recent_time = datetime.fromisoformat(
                                recent_post.get("timestamp", "").replace("Z", "+00:00")
                            )
                            seconds_ago = (datetime.now(timezone.utc) - recent_time).total_seconds()

                            if seconds_ago < 300:  # 5 minutes for video
                                _api_logger.warning(f"GHOST PUBLISH DETECTED! Reel is live ({seconds_ago:.0f}s ago)")
                                return InstagramPublishResult(
                                    success=True,
                                    media_id=recent_post.get("id"),
                                    permalink=recent_post.get("permalink"),
                                    container_ids=[container_id] if container_id else [],
                                    image_urls=[video_url],
                                    published_at=datetime.now(),
                                )
                    except Exception as check_err:
                        _api_logger.warning(f"Ghost check failed: {check_err}")

                    # Don't retry after publish attempt
                    break

                # Retry if possible
                if e.is_retryable and attempt < max_retries:
                    retry_delay = error_info.get("retry_delay", 60)
                    await self._report_progress(
                        InstagramPostStatus.CREATING_CONTAINERS,
                        f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})",
                        self._progress.progress_percent,
                    )
                    _api_logger.info(f"Waiting {retry_delay}s before retry...")
                    await asyncio.sleep(retry_delay)
                    container_id = None  # Reset container for fresh attempt
                    continue

                break

            except Exception as e:
                last_error = InstagramAPIError(f"Unexpected error: {e}")
                break

        # Build error result
        if last_error:
            error_code = last_error.error_code if hasattr(last_error, 'error_code') else None
            error_subcode = last_error.error_subcode if hasattr(last_error, 'error_subcode') else None
            error_info = get_error_info(error_code, error_subcode)
            user_message = error_info.get("user_message", str(last_error))
            error_name = error_info.get("name", "UNKNOWN")

            await self._report_progress(
                InstagramPostStatus.FAILED,
                f"[{error_name}] {user_message[:50]}...",
                self._progress.progress_percent,
            )

            _api_logger.error(
                f"=== SESSION FAILED === Total API calls: {self._api_call_count} | "
                f"Error: [{error_name}] {last_error}"
            )

            return InstagramPublishResult(
                success=False,
                error_message=f"[{error_name}] {user_message}",
                container_ids=[container_id] if container_id else [],
                image_urls=[video_url],
            )

        return InstagramPublishResult(
            success=False,
            error_message="Unknown error occurred",
            container_ids=[container_id] if container_id else [],
            image_urls=[video_url],
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
