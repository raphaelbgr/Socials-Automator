"""Abstract base classes for platform publishers.

This module defines the interfaces that all platform implementations must follow.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Awaitable


@dataclass
class PublishResult:
    """Unified result from publishing to any platform."""

    success: bool
    platform: str
    media_id: Optional[str] = None
    permalink: Optional[str] = None
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return f"[{self.platform}] Success: {self.permalink or self.media_id}"
        return f"[{self.platform}] Failed: {self.error}"


@dataclass
class PlatformConfig(ABC):
    """Base configuration for any platform.

    Each platform extends this with its specific credentials and settings.
    Supports loading values from environment variables using ENV:VAR_NAME syntax.
    """

    platform: str
    enabled: bool = True

    @classmethod
    @abstractmethod
    def from_profile(
        cls,
        profile_path: Path,
        platform_data: dict[str, Any],
    ) -> "PlatformConfig":
        """Load configuration from profile metadata.

        Args:
            profile_path: Path to the profile directory.
            platform_data: Platform-specific config from metadata.json.

        Returns:
            Configured instance of the platform config.
        """
        ...

    @staticmethod
    def resolve_env(value: str) -> str:
        """Resolve ENV:VAR_NAME to actual environment variable value.

        Args:
            value: String that may contain ENV:VAR_NAME reference.

        Returns:
            Resolved value from environment, or original value if not ENV: prefixed.

        Example:
            resolve_env("ENV:INSTAGRAM_USER_ID") -> "17841405793187218"
            resolve_env("literal_value") -> "literal_value"
        """
        if isinstance(value, str) and value.startswith("ENV:"):
            env_var = value[4:]
            return os.getenv(env_var, "")
        return value

    @classmethod
    def resolve_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve all ENV: references in a dictionary.

        Args:
            data: Dictionary with potential ENV: values.

        Returns:
            Dictionary with all ENV: values resolved.
        """
        resolved = {}
        for key, value in data.items():
            if isinstance(value, str):
                resolved[key] = cls.resolve_env(value)
            elif isinstance(value, dict):
                resolved[key] = cls.resolve_dict(value)
            else:
                resolved[key] = value
        return resolved


# Type for progress callbacks
ProgressCallback = Callable[[str, float, str], Awaitable[None]] | None


class PlatformPublisher(ABC):
    """Abstract base class for platform publishers.

    Each platform (Instagram, TikTok, etc.) implements this interface
    to provide a unified API for publishing content.
    """

    def __init__(
        self,
        config: PlatformConfig,
        progress_callback: ProgressCallback = None,
    ):
        """Initialize the publisher.

        Args:
            config: Platform-specific configuration.
            progress_callback: Optional callback for progress updates.
        """
        self.config = config
        self._progress_callback = progress_callback

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier (e.g., 'instagram', 'tiktok')."""
        ...

    @abstractmethod
    async def publish_reel(
        self,
        video_path: Path,
        caption: str,
        thumbnail_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> PublishResult:
        """Publish a video reel to the platform.

        Args:
            video_path: Path to the video file.
            caption: Caption/description for the video.
            thumbnail_path: Optional custom thumbnail image.
            **kwargs: Platform-specific additional options.

        Returns:
            PublishResult with success status and details.
        """
        ...

    @abstractmethod
    async def check_credentials(self) -> tuple[bool, str]:
        """Verify that credentials are valid.

        Returns:
            Tuple of (is_valid, message).
        """
        ...

    async def _emit_progress(self, stage: str, progress: float, message: str) -> None:
        """Emit a progress update if callback is set."""
        if self._progress_callback:
            await self._progress_callback(stage, progress, message)

    def _make_result(
        self,
        success: bool,
        media_id: Optional[str] = None,
        permalink: Optional[str] = None,
        error: Optional[str] = None,
        **details: Any,
    ) -> PublishResult:
        """Create a PublishResult for this platform."""
        return PublishResult(
            success=success,
            platform=self.platform_name,
            media_id=media_id,
            permalink=permalink,
            error=error,
            details=details,
        )
