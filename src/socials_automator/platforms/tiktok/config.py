"""TikTok platform configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import PlatformConfig


@dataclass
class TikTokConfig(PlatformConfig):
    """TikTok-specific configuration.

    Credentials can be specified in profile metadata.json or fall back to
    environment variables.

    Profile metadata format:
        {
          "platforms": {
            "tiktok": {
              "enabled": true,
              "client_key": "ENV:TIKTOK_CLIENT_KEY",
              "client_secret": "ENV:TIKTOK_CLIENT_SECRET",
              "access_token": "ENV:TIKTOK_ACCESS_TOKEN"
            }
          }
        }

    TikTok API Setup:
    1. Create app at https://developers.tiktok.com/
    2. Request 'video.upload' scope approval
    3. Complete OAuth flow to get access token
    4. Pass audit for public visibility (posts are private until audit)
    """

    platform: str = "tiktok"

    # OAuth credentials
    client_key: str = ""
    client_secret: str = ""
    access_token: str = ""

    # Optional: for token refresh
    refresh_token: str = ""

    # API settings
    api_base_url: str = "https://open.tiktokapis.com/v2"

    @classmethod
    def from_profile(
        cls,
        profile_path: Path,
        platform_data: dict[str, Any],
    ) -> "TikTokConfig":
        """Load configuration from profile metadata.

        Falls back to environment variables if not specified in profile.
        """
        # Resolve any ENV: references in the platform data
        resolved = cls.resolve_dict(platform_data)

        # Get values from profile or fall back to env vars
        client_key = resolved.get("client_key") or os.getenv("TIKTOK_CLIENT_KEY", "")
        client_secret = resolved.get("client_secret") or os.getenv("TIKTOK_CLIENT_SECRET", "")
        access_token = resolved.get("access_token") or os.getenv("TIKTOK_ACCESS_TOKEN", "")
        refresh_token = resolved.get("refresh_token") or os.getenv("TIKTOK_REFRESH_TOKEN", "")

        return cls(
            enabled=resolved.get("enabled", True),
            client_key=client_key,
            client_secret=client_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            api_base_url=resolved.get("api_base_url", "https://open.tiktokapis.com/v2"),
        )

    def validate(self) -> tuple[bool, str]:
        """Validate that all required credentials are present.

        Returns:
            Tuple of (is_valid, error_message).
        """
        missing = []

        if not self.client_key:
            missing.append("client_key (or TIKTOK_CLIENT_KEY)")
        if not self.client_secret:
            missing.append("client_secret (or TIKTOK_CLIENT_SECRET)")
        if not self.access_token:
            missing.append("access_token (or TIKTOK_ACCESS_TOKEN)")

        if missing:
            return False, f"Missing TikTok credentials: {', '.join(missing)}"

        return True, "OK"
