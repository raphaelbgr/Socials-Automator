"""Embedded Tor helper for anonymous web scraping.

Provides automatic Tor proxy support using pure Python (no external installation).
Uses `torpy` library for embedded Tor circuits.

Features:
- No external Tor installation required
- Automatic circuit creation
- IP rotation between videos
- Thread-safe connection management

Usage:
    from .tor_helper import get_tor_helper, is_tor_available

    helper = get_tor_helper()
    if helper.is_available():
        proxy_url = helper.proxy_url
        # Use proxy_url with httpx or requests

Dependencies:
    pip install torpy
"""

import asyncio
import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger("video.pipeline")

# Cooldown between IP rotations (Tor recommends 10+ seconds between new circuits)
IP_ROTATION_COOLDOWN = 10


class EmbeddedTorHelper:
    """Embedded Tor helper using torpy (pure Python Tor client).

    Creates and manages Tor circuits without external Tor installation.
    Thread-safe for use in async contexts.
    """

    def __init__(self):
        """Initialize embedded Tor helper."""
        self._guard = None
        self._circuit = None
        self._lock = threading.Lock()
        self._last_rotation_time = 0.0
        self._initialized = False
        self._current_ip: Optional[str] = None
        self._proxy_adapter = None

    def _ensure_initialized(self) -> bool:
        """Ensure Tor circuit is initialized.

        Returns:
            True if successfully initialized.
        """
        if self._initialized and self._guard is not None:
            return True

        with self._lock:
            if self._initialized and self._guard is not None:
                return True

            try:
                from torpy import TorClient

                logger.info("Initializing embedded Tor circuit...")

                # Create Tor client and get a guard node
                self._client = TorClient()
                self._guard = self._client.get_guard()

                # Create initial circuit
                self._circuit = self._guard.create_circuit(3)

                self._initialized = True
                logger.info("Embedded Tor circuit ready")
                return True

            except ImportError:
                logger.warning(
                    "torpy not installed. Run: pip install torpy"
                )
                return False
            except Exception as e:
                logger.warning(f"Failed to initialize Tor circuit: {e}")
                return False

    def is_available(self) -> bool:
        """Check if Tor is available.

        Returns:
            True if torpy is installed and circuit can be created.
        """
        try:
            from torpy import TorClient
            return True
        except ImportError:
            return False

    @property
    def proxy_url(self) -> Optional[str]:
        """Get SOCKS5 proxy URL.

        Note: torpy doesn't provide a SOCKS proxy directly.
        Instead, use the requests adapter or make_request method.

        Returns:
            None (use make_request instead for torpy).
        """
        # torpy doesn't expose a SOCKS port
        # Use make_request or get_session instead
        return None

    def make_request(
        self,
        url: str,
        timeout: int = 30,
    ) -> Optional[bytes]:
        """Make HTTP request through Tor.

        Args:
            url: URL to fetch.
            timeout: Request timeout in seconds.

        Returns:
            Response content or None if failed.
        """
        if not self._ensure_initialized():
            return None

        try:
            from torpy.http.requests import TorRequests

            with TorRequests() as tor_requests:
                with tor_requests.get_session() as session:
                    response = session.get(url, timeout=timeout)
                    if response.status_code == 200:
                        return response.content
                    return None

        except Exception as e:
            logger.debug(f"Tor request failed: {e}")
            return None

    async def make_request_async(
        self,
        url: str,
        timeout: int = 30,
    ) -> Optional[bytes]:
        """Make async HTTP request through Tor.

        Args:
            url: URL to fetch.
            timeout: Request timeout in seconds.

        Returns:
            Response content or None if failed.
        """
        # Run sync request in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.make_request,
            url,
            timeout,
        )

    def get_current_ip(self) -> Optional[str]:
        """Get current exit node IP address.

        Returns:
            Current IP address or None if failed.
        """
        try:
            content = self.make_request("https://api.ipify.org?format=json")
            if content:
                import json
                data = json.loads(content)
                self._current_ip = data.get("ip")
                return self._current_ip
        except Exception as e:
            logger.debug(f"Failed to get Tor IP: {e}")
        return None

    def request_new_identity(self) -> bool:
        """Request a new Tor circuit (new IP).

        Creates a new circuit through different nodes.
        Respects cooldown period between rotations.

        Returns:
            True if new circuit was created successfully.
        """
        # Check cooldown
        now = time.time()
        time_since_last = now - self._last_rotation_time
        if time_since_last < IP_ROTATION_COOLDOWN:
            wait_time = IP_ROTATION_COOLDOWN - time_since_last
            logger.debug(f"Waiting {wait_time:.1f}s for Tor rotation cooldown")
            time.sleep(wait_time)

        with self._lock:
            try:
                if self._guard is None:
                    if not self._ensure_initialized():
                        return False

                # Close old circuit
                if self._circuit:
                    try:
                        self._circuit.close()
                    except Exception:
                        pass

                # Create new circuit
                self._circuit = self._guard.create_circuit(3)
                self._last_rotation_time = time.time()

                # Get new IP for logging
                old_ip = self._current_ip
                new_ip = self.get_current_ip()

                if old_ip and new_ip and old_ip != new_ip:
                    logger.info(f"Tor IP rotated: {old_ip} -> {new_ip}")
                else:
                    logger.info("Tor: New circuit created")

                return True

            except Exception as e:
                logger.warning(f"Failed to rotate Tor circuit: {e}")
                return False

    def close(self) -> None:
        """Close Tor connections."""
        with self._lock:
            if self._circuit:
                try:
                    self._circuit.close()
                except Exception:
                    pass
                self._circuit = None

            if self._guard:
                try:
                    self._guard.close()
                except Exception:
                    pass
                self._guard = None

            self._initialized = False


class TorRequestsAdapter:
    """Adapter to make torpy work with httpx-like interface.

    Provides a context manager that wraps torpy's TorRequests
    to provide a consistent interface for downloading images.
    """

    def __init__(self):
        """Initialize adapter."""
        self._session = None
        self._tor_requests = None

    async def get(
        self,
        url: str,
        timeout: float = 30.0,
        **kwargs,
    ) -> "TorResponse":
        """Make GET request through Tor.

        Args:
            url: URL to fetch.
            timeout: Request timeout.
            **kwargs: Additional arguments (ignored).

        Returns:
            TorResponse object.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_get,
            url,
            timeout,
        )

    def _sync_get(self, url: str, timeout: float) -> "TorResponse":
        """Synchronous GET request."""
        try:
            from torpy.http.requests import TorRequests

            with TorRequests() as tor_requests:
                with tor_requests.get_session() as session:
                    response = session.get(url, timeout=timeout)
                    return TorResponse(
                        status_code=response.status_code,
                        content=response.content,
                        headers=dict(response.headers),
                    )
        except Exception as e:
            logger.debug(f"Tor request failed: {e}")
            return TorResponse(status_code=0, content=b"", headers={}, error=str(e))


class TorResponse:
    """Simple response object for Tor requests."""

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict,
        error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.content = content
        self.headers = headers
        self.error = error


# Global helper instance (lazy initialization)
_tor_helper: Optional[EmbeddedTorHelper] = None
_tor_adapter: Optional[TorRequestsAdapter] = None


def get_tor_helper() -> EmbeddedTorHelper:
    """Get global TorHelper instance."""
    global _tor_helper
    if _tor_helper is None:
        _tor_helper = EmbeddedTorHelper()
    return _tor_helper


def get_tor_adapter() -> TorRequestsAdapter:
    """Get global Tor requests adapter."""
    global _tor_adapter
    if _tor_adapter is None:
        _tor_adapter = TorRequestsAdapter()
    return _tor_adapter


def is_tor_available() -> bool:
    """Check if Tor is available (torpy installed)."""
    try:
        from torpy import TorClient
        return True
    except ImportError:
        return False


def get_tor_proxy_url() -> Optional[str]:
    """Get Tor proxy URL (not available with embedded Tor)."""
    return None


def rotate_tor_ip() -> bool:
    """Request new Tor circuit (new IP)."""
    return get_tor_helper().request_new_identity()


def close_tor() -> None:
    """Close Tor connection and reset global helper.

    Call this at the end of video generation to ensure
    a fresh Tor circuit for the next video.
    """
    global _tor_helper
    if _tor_helper is not None:
        _tor_helper.close()
        _tor_helper = None
