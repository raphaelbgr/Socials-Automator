"""Platform registry for discovering and instantiating platform publishers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .base import PlatformConfig, PlatformPublisher, ProgressCallback


class PlatformRegistry:
    """Registry and factory for platform publishers.

    Handles discovery, registration, and instantiation of platform adapters.
    Platforms are registered at import time and can be retrieved by name.

    Usage:
        # Get available platforms
        platforms = PlatformRegistry.available_platforms()

        # Load config and create publisher
        config = PlatformRegistry.load_config("instagram", profile_path)
        publisher = PlatformRegistry.get_publisher("instagram", config)

        # Get enabled platforms from a profile
        enabled = PlatformRegistry.get_enabled_platforms(profile_path)
    """

    _platforms: dict[str, tuple[Type["PlatformPublisher"], Type["PlatformConfig"]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        publisher_cls: Type["PlatformPublisher"],
        config_cls: Type["PlatformConfig"],
    ) -> None:
        """Register a platform adapter.

        Args:
            name: Platform identifier (e.g., 'instagram', 'tiktok').
            publisher_cls: Publisher class implementing PlatformPublisher.
            config_cls: Config class implementing PlatformConfig.
        """
        cls._platforms[name.lower()] = (publisher_cls, config_cls)

    @classmethod
    def get_publisher(
        cls,
        name: str,
        config: "PlatformConfig",
        progress_callback: "ProgressCallback" = None,
    ) -> "PlatformPublisher":
        """Get a publisher instance for a platform.

        Args:
            name: Platform identifier.
            config: Platform configuration.
            progress_callback: Optional progress callback.

        Returns:
            Configured publisher instance.

        Raises:
            ValueError: If platform is not registered.
        """
        name = name.lower()
        if name not in cls._platforms:
            available = ", ".join(cls._platforms.keys())
            raise ValueError(f"Unknown platform: {name}. Available: {available}")

        publisher_cls, _ = cls._platforms[name]
        return publisher_cls(config, progress_callback=progress_callback)

    @classmethod
    def load_config(
        cls,
        name: str,
        profile_path: Path,
    ) -> "PlatformConfig":
        """Load platform configuration from a profile.

        Reads the profile's metadata.json and extracts platform-specific config.

        Args:
            name: Platform identifier.
            profile_path: Path to the profile directory.

        Returns:
            Configured platform config instance.

        Raises:
            ValueError: If platform is not registered or not configured in profile.
        """
        name = name.lower()
        if name not in cls._platforms:
            available = ", ".join(cls._platforms.keys())
            raise ValueError(f"Unknown platform: {name}. Available: {available}")

        # Load profile metadata
        metadata_path = profile_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Profile metadata not found: {metadata_path}")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Get platform config from metadata
        platforms_config = metadata.get("platforms", {})
        platform_data = platforms_config.get(name, {})

        # If no platform config in metadata, use empty dict (will fall back to env vars)
        _, config_cls = cls._platforms[name]
        return config_cls.from_profile(profile_path, platform_data)

    @classmethod
    def get_enabled_platforms(cls, profile_path: Path) -> list[str]:
        """Get list of enabled platforms for a profile.

        Args:
            profile_path: Path to the profile directory.

        Returns:
            List of enabled platform names.
        """
        metadata_path = profile_path / "metadata.json"
        if not metadata_path.exists():
            return ["instagram"]  # Default fallback

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        platforms_config = metadata.get("platforms", {})
        enabled = []

        for name, config in platforms_config.items():
            if config.get("enabled", True):
                enabled.append(name)

        # If no platforms configured, default to instagram
        return enabled if enabled else ["instagram"]

    @classmethod
    def get_default_platforms(cls, profile_path: Path) -> list[str]:
        """Get default platforms for a profile (used when no flags specified).

        Args:
            profile_path: Path to the profile directory.

        Returns:
            List of default platform names.
        """
        metadata_path = profile_path / "metadata.json"
        if not metadata_path.exists():
            return ["instagram"]

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Check for explicit default_platforms setting
        default = metadata.get("default_platforms")
        if default:
            return default

        # Fall back to just instagram for backwards compatibility
        return ["instagram"]

    @classmethod
    def available_platforms(cls) -> list[str]:
        """Get list of all registered platform names."""
        return list(cls._platforms.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a platform is registered."""
        return name.lower() in cls._platforms


def _register_platforms() -> None:
    """Register all available platform adapters.

    Called automatically on module import.
    """
    # Import and register Instagram
    try:
        from .instagram import InstagramPublisher, InstagramConfig
        PlatformRegistry.register("instagram", InstagramPublisher, InstagramConfig)
    except ImportError:
        pass  # Instagram module not available

    # Import and register TikTok
    try:
        from .tiktok import TikTokPublisher, TikTokConfig
        PlatformRegistry.register("tiktok", TikTokPublisher, TikTokConfig)
    except ImportError:
        pass  # TikTok module not available


# Auto-register platforms on import
_register_platforms()
