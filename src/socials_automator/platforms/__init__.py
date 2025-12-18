"""Multi-platform publishing abstraction layer.

This module provides a unified interface for publishing content to multiple
social media platforms (Instagram, TikTok, etc.).

Usage:
    from socials_automator.platforms import PlatformRegistry, PublishResult

    # Get a publisher for a platform
    config = PlatformRegistry.load_config("instagram", profile_path)
    publisher = PlatformRegistry.get_publisher("instagram", config)

    # Publish content
    result = await publisher.publish_reel(video_path, caption)
"""

from .base import (
    PlatformConfig,
    PlatformPublisher,
    PublishResult,
)
from .registry import PlatformRegistry

__all__ = [
    "PlatformConfig",
    "PlatformPublisher",
    "PublishResult",
    "PlatformRegistry",
]
