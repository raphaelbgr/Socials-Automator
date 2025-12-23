"""Image search providers package.

Provides a unified interface for searching and downloading images
from multiple providers (DuckDuckGo web search, Pexels, Pixabay).

Usage:
    from socials_automator.video.pipeline.image_providers import (
        get_image_provider,
        IImageSearchProvider,
        ImageSearchResult,
    )

    # Default: Web search (no API key needed)
    provider = get_image_provider("websearch")
    results = await provider.search("stranger things poster")

    # Web search with Tor for anonymity
    provider = get_image_provider("websearch", use_tor=True)
    results = await provider.search("taylor swift eras tour")

    # Or use Pexels/Pixabay for stock photos
    provider = get_image_provider("pexels")
    results = await provider.search("sunset beach")
"""

from .base import (
    IImageSearchProvider,
    ImageSearchResult,
    get_image_provider,
)
from .pexels import PexelsImageProvider
from .pixabay import PixabayImageProvider
from .websearch import WebSearchImageProvider
from .tor_helper import (
    EmbeddedTorHelper,
    get_tor_helper,
    is_tor_available,
    rotate_tor_ip,
    close_tor,
)
from .headless_screenshot import (
    capture_image_screenshot,
    is_playwright_available,
    close_browser,
    get_browsers_info,
)

__all__ = [
    # Base
    "IImageSearchProvider",
    "ImageSearchResult",
    "get_image_provider",
    # Providers
    "PexelsImageProvider",
    "PixabayImageProvider",
    "WebSearchImageProvider",
    # Tor helper
    "EmbeddedTorHelper",
    "get_tor_helper",
    "is_tor_available",
    "rotate_tor_ip",
    "close_tor",
    # Headless screenshot fallback
    "capture_image_screenshot",
    "is_playwright_available",
    "close_browser",
    "get_browsers_info",
]

# Available providers for CLI help
AVAILABLE_PROVIDERS = ["pexels", "pixabay", "websearch"]
