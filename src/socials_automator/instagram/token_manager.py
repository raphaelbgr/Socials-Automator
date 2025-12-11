"""Instagram/Facebook token management with auto-refresh."""

import os
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path

import httpx

_logger = logging.getLogger("instagram_api")

# Token info cache
_token_expiry_cache: dict[str, datetime] = {}


class TokenManager:
    """Manages Instagram/Facebook access tokens with automatic refresh.

    Features:
    - Check token validity and expiration
    - Exchange short-lived token for long-lived token (60 days)
    - Refresh long-lived tokens before expiration
    - Auto-update .env file with new tokens
    """

    GRAPH_API_VERSION = "v18.0"
    GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

    def __init__(
        self,
        access_token: str,
        app_id: str | None = None,
        app_secret: str | None = None,
        env_file_path: Path | None = None,
    ):
        """Initialize token manager.

        Args:
            access_token: Current Instagram/Facebook access token
            app_id: Facebook App ID (required for refresh)
            app_secret: Facebook App Secret (required for refresh)
            env_file_path: Path to .env file for auto-updating tokens
        """
        self.access_token = access_token
        self.app_id = app_id
        self.app_secret = app_secret
        self.env_file_path = env_file_path or self._find_env_file()

    @staticmethod
    def _find_env_file() -> Path | None:
        """Find .env file in project root."""
        current = Path.cwd()
        while current != current.parent:
            env_path = current / ".env"
            if env_path.exists():
                return env_path
            current = current.parent
        return None

    @classmethod
    def from_env(cls) -> "TokenManager":
        """Create TokenManager from environment variables."""
        from dotenv import load_dotenv
        load_dotenv()

        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        app_id = os.getenv("FACEBOOK_APP_ID", "")
        app_secret = os.getenv("FACEBOOK_APP_SECRET", "")

        if not access_token:
            raise ValueError("INSTAGRAM_ACCESS_TOKEN not set in environment")

        return cls(
            access_token=access_token,
            app_id=app_id if app_id else None,
            app_secret=app_secret if app_secret else None,
        )

    @property
    def can_refresh(self) -> bool:
        """Check if we have credentials to refresh the token."""
        return bool(self.app_id and self.app_secret)

    async def get_token_info(self) -> dict:
        """Get information about the current token.

        Returns:
            Dict with token info including expiration, scopes, etc.
        """
        url = f"{self.GRAPH_API_BASE}/debug_token"
        params = {
            "input_token": self.access_token,
            "access_token": f"{self.app_id}|{self.app_secret}" if self.can_refresh else self.access_token,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            data = response.json()

            if "error" in data:
                raise ValueError(f"Token debug failed: {data['error'].get('message', 'Unknown error')}")

            return data.get("data", {})

    async def check_token_validity(self) -> tuple[bool, str, int | None]:
        """Check if the current token is valid.

        Returns:
            Tuple of (is_valid, message, days_until_expiry)
        """
        try:
            info = await self.get_token_info()

            is_valid = info.get("is_valid", False)
            expires_at = info.get("expires_at", 0)

            if not is_valid:
                error_msg = info.get("error", {}).get("message", "Token is invalid")
                return False, error_msg, None

            if expires_at == 0:
                # Token doesn't expire (page token)
                return True, "Token is valid (no expiration)", None

            expiry_date = datetime.fromtimestamp(expires_at)
            now = datetime.now()
            days_left = (expiry_date - now).days

            if days_left < 0:
                return False, "Token has expired", days_left
            elif days_left < 7:
                return True, f"Token expires in {days_left} days - refresh recommended", days_left
            else:
                return True, f"Token is valid for {days_left} more days", days_left

        except Exception as e:
            return False, f"Could not validate token: {e}", None

    async def exchange_for_long_lived_token(self) -> str:
        """Exchange a short-lived token for a long-lived token (60 days).

        Returns:
            New long-lived access token
        """
        if not self.can_refresh:
            raise ValueError(
                "Cannot exchange token: FACEBOOK_APP_ID and FACEBOOK_APP_SECRET required. "
                "Add them to your .env file."
            )

        url = f"{self.GRAPH_API_BASE}/oauth/access_token"
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "fb_exchange_token": self.access_token,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            data = response.json()

            if "error" in data:
                raise ValueError(f"Token exchange failed: {data['error'].get('message', 'Unknown error')}")

            new_token = data.get("access_token")
            if not new_token:
                raise ValueError("No access_token in response")

            _logger.info("Successfully exchanged for long-lived token")
            return new_token

    async def refresh_token(self) -> str:
        """Refresh the current long-lived token.

        For long-lived tokens, this exchanges the token for a new one
        with a fresh 60-day expiration.

        Returns:
            New refreshed access token
        """
        # Long-lived token refresh uses the same endpoint as exchange
        return await self.exchange_for_long_lived_token()

    def update_env_file(self, new_token: str) -> bool:
        """Update the .env file with a new token.

        Args:
            new_token: The new access token to save

        Returns:
            True if successfully updated
        """
        if not self.env_file_path or not self.env_file_path.exists():
            _logger.warning("No .env file found - token not saved")
            return False

        try:
            content = self.env_file_path.read_text(encoding="utf-8")

            # Replace the token line
            pattern = r'(INSTAGRAM_ACCESS_TOKEN=)[^\n]*'
            new_content = re.sub(pattern, f'\\1{new_token}', content)

            if new_content == content:
                # Token line not found, append it
                new_content = content.rstrip() + f"\nINSTAGRAM_ACCESS_TOKEN={new_token}\n"

            self.env_file_path.write_text(new_content, encoding="utf-8")

            # Update instance token
            self.access_token = new_token

            _logger.info(f"Updated token in {self.env_file_path}")
            return True

        except Exception as e:
            _logger.error(f"Failed to update .env file: {e}")
            return False

    async def ensure_valid_token(self, refresh_if_expiring_days: int = 7) -> str:
        """Ensure we have a valid token, refreshing if needed.

        Args:
            refresh_if_expiring_days: Refresh if token expires within this many days

        Returns:
            Valid access token (may be refreshed)
        """
        is_valid, message, days_left = await self.check_token_validity()

        if not is_valid:
            if not self.can_refresh:
                raise ValueError(
                    f"Token is invalid ({message}) and cannot auto-refresh. "
                    "Set FACEBOOK_APP_ID and FACEBOOK_APP_SECRET in .env, "
                    "then get a new token from Graph API Explorer."
                )

            _logger.info(f"Token invalid ({message}), attempting refresh...")
            try:
                new_token = await self.refresh_token()
                self.update_env_file(new_token)
                return new_token
            except Exception as e:
                raise ValueError(
                    f"Token refresh failed: {e}. "
                    "Get a new token from Graph API Explorer and update .env"
                )

        # Token is valid, but check if it's expiring soon
        if days_left is not None and days_left < refresh_if_expiring_days and self.can_refresh:
            _logger.info(f"Token expires in {days_left} days, refreshing proactively...")
            try:
                new_token = await self.refresh_token()
                self.update_env_file(new_token)
                return new_token
            except Exception as e:
                _logger.warning(f"Proactive refresh failed ({e}), using current token")

        return self.access_token


async def get_valid_token() -> str:
    """Convenience function to get a valid Instagram token.

    Automatically refreshes if needed and credentials are available.

    Returns:
        Valid access token
    """
    manager = TokenManager.from_env()
    return await manager.ensure_valid_token()


def get_token_exchange_url(
    app_id: str,
    app_secret: str,
    short_lived_token: str,
) -> str:
    """Generate the URL to exchange a short-lived token for long-lived.

    Args:
        app_id: Facebook App ID
        app_secret: Facebook App Secret
        short_lived_token: The short-lived token to exchange

    Returns:
        URL to visit (or fetch) for token exchange
    """
    return (
        f"https://graph.facebook.com/v18.0/oauth/access_token?"
        f"grant_type=fb_exchange_token&"
        f"client_id={app_id}&"
        f"client_secret={app_secret}&"
        f"fb_exchange_token={short_lived_token}"
    )
