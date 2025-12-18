"""Instagram platform configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import PlatformConfig


@dataclass
class InstagramConfig(PlatformConfig):
    """Instagram-specific configuration.

    Credentials can be specified in profile metadata.json or fall back to
    environment variables.

    Profile metadata format:
        {
          "platforms": {
            "instagram": {
              "enabled": true,
              "user_id": "ENV:INSTAGRAM_USER_ID",
              "access_token": "ENV:INSTAGRAM_ACCESS_TOKEN",
              "cloudinary_cloud_name": "ENV:CLOUDINARY_CLOUD_NAME",
              "cloudinary_api_key": "ENV:CLOUDINARY_API_KEY",
              "cloudinary_api_secret": "ENV:CLOUDINARY_API_SECRET"
            }
          }
        }
    """

    platform: str = "instagram"

    # Instagram credentials
    user_id: str = ""
    access_token: str = ""

    # Cloudinary credentials (for media hosting)
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # API settings
    api_version: str = "v21.0"

    @classmethod
    def from_profile(
        cls,
        profile_path: Path,
        platform_data: dict[str, Any],
    ) -> "InstagramConfig":
        """Load configuration from profile metadata.

        IMPORTANT: Only falls back to generic environment variables if the
        profile does NOT define the credential at all. If a profile defines
        a credential (even as ENV:VAR_NAME that resolves to empty), we do NOT
        fall back - this prevents cross-account contamination.

        Falls back to environment variables only if not specified in profile.
        """
        # Resolve any ENV: references in the platform data
        resolved = cls.resolve_dict(platform_data)

        # Helper to get value with safe fallback
        # Only falls back to generic env var if key is NOT defined in profile
        def get_credential(key: str, fallback_env: str) -> str:
            if key in platform_data:
                # Profile explicitly defines this key - use resolved value, NO fallback
                # This prevents cross-account contamination when profile-specific
                # env vars are not set (they would fall back to wrong account)
                value = resolved.get(key, "")
                if not value:
                    # Profile defines the key but it's empty - this is an error
                    # Don't silently use another account's credentials!
                    import logging
                    logging.warning(
                        f"Instagram credential '{key}' is defined in profile but empty. "
                        f"Check that the environment variable is set correctly."
                    )
                return value
            else:
                # Profile doesn't define this key - fall back to generic env var
                return os.getenv(fallback_env, "")

        user_id = get_credential("user_id", "INSTAGRAM_USER_ID")
        access_token = get_credential("access_token", "INSTAGRAM_ACCESS_TOKEN")

        # Cloudinary credentials can be shared across profiles (same account)
        # so we use simpler fallback logic for these
        cloudinary_cloud_name = (
            resolved.get("cloudinary_cloud_name")
            or os.getenv("CLOUDINARY_CLOUD_NAME", "")
        )
        cloudinary_api_key = (
            resolved.get("cloudinary_api_key")
            or os.getenv("CLOUDINARY_API_KEY", "")
        )
        cloudinary_api_secret = (
            resolved.get("cloudinary_api_secret")
            or os.getenv("CLOUDINARY_API_SECRET", "")
        )

        return cls(
            enabled=resolved.get("enabled", True),
            user_id=user_id,
            access_token=access_token,
            cloudinary_cloud_name=cloudinary_cloud_name,
            cloudinary_api_key=cloudinary_api_key,
            cloudinary_api_secret=cloudinary_api_secret,
            api_version=resolved.get("api_version", "v21.0"),
        )

    def validate(self) -> tuple[bool, str]:
        """Validate that all required credentials are present.

        Returns:
            Tuple of (is_valid, error_message).
        """
        missing = []

        if not self.user_id:
            missing.append("user_id (or INSTAGRAM_USER_ID)")
        if not self.access_token:
            missing.append("access_token (or INSTAGRAM_ACCESS_TOKEN)")
        if not self.cloudinary_cloud_name:
            missing.append("cloudinary_cloud_name (or CLOUDINARY_CLOUD_NAME)")
        if not self.cloudinary_api_key:
            missing.append("cloudinary_api_key (or CLOUDINARY_API_KEY)")
        if not self.cloudinary_api_secret:
            missing.append("cloudinary_api_secret (or CLOUDINARY_API_SECRET)")

        if missing:
            return False, f"Missing Instagram credentials: {', '.join(missing)}"

        return True, "OK"

    def to_legacy_config(self) -> "LegacyInstagramConfig":
        """Convert to the legacy InstagramConfig format for existing client.

        This allows us to use the existing InstagramClient without modification.
        """
        from socials_automator.instagram.models import InstagramConfig as LegacyInstagramConfig

        return LegacyInstagramConfig(
            instagram_user_id=self.user_id,
            access_token=self.access_token,
            cloudinary_cloud_name=self.cloudinary_cloud_name,
            cloudinary_api_key=self.cloudinary_api_key,
            cloudinary_api_secret=self.cloudinary_api_secret,
            api_version=self.api_version,
        )
