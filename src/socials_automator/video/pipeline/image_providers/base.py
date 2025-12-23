"""Abstract base class for image search providers.

Defines the interface that all image providers must implement.
This allows swapping between Pexels, Pixabay, Unsplash, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImageSearchResult:
    """Result from an image search.

    Attributes:
        id: Provider-specific image ID.
        url: URL to download the image.
        thumbnail_url: URL for thumbnail preview.
        width: Image width in pixels.
        height: Image height in pixels.
        description: Image description/alt text.
        photographer: Photographer name (for attribution).
        photographer_url: Link to photographer's profile.
        source: Provider name (pexels, pixabay, unsplash).
    """
    id: str
    url: str
    thumbnail_url: str
    width: int
    height: int
    description: str
    photographer: str
    photographer_url: str
    source: str


class IImageSearchProvider(ABC):
    """Abstract interface for image search providers.

    All image providers (Pexels, Pixabay, Unsplash, etc.) must implement
    this interface to be used by the ImageResolver.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'pexels', 'pixabay')."""
        pass

    @property
    @abstractmethod
    def cache_folder_name(self) -> str:
        """Return the cache folder name for this provider.

        E.g., 'image-cache' for Pexels, 'image-cache-pixabay' for Pixabay.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        per_page: int = 5,
        orientation: Optional[str] = None,
    ) -> list[ImageSearchResult]:
        """Search for images matching the query.

        Args:
            query: Search query string.
            per_page: Number of results to return (max varies by provider).
            orientation: Image orientation filter ('landscape', 'portrait', 'square').

        Returns:
            List of ImageSearchResult objects.
        """
        pass

    @abstractmethod
    async def download(
        self,
        image_id: str,
        url: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Download an image to the specified path.

        Args:
            image_id: Provider-specific image ID.
            url: URL to download from.
            output_path: Path to save the image.

        Returns:
            Path to downloaded image, or None if download failed.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections/clients."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def get_image_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    use_tor: bool = False,
) -> IImageSearchProvider:
    """Factory function to get an image provider by name.

    Args:
        provider_name: Provider name ('pexels', 'pixabay', 'websearch').
        api_key: Optional API key (if not provided, reads from env var).
            Not needed for 'websearch' provider.
        use_tor: Route requests through Tor proxy (websearch only).

    Returns:
        An instance of the requested image provider.

    Raises:
        ValueError: If provider name is not recognized.
    """
    # Import here to avoid circular imports
    from .pexels import PexelsImageProvider
    from .pixabay import PixabayImageProvider
    from .websearch import WebSearchImageProvider

    provider_name = provider_name.lower()

    # Handle websearch separately (different constructor)
    if provider_name == "websearch":
        return WebSearchImageProvider(use_tor=use_tor)

    # API-based providers
    providers = {
        "pexels": PexelsImageProvider,
        "pixabay": PixabayImageProvider,
    }

    if provider_name not in providers:
        available = ", ".join(list(providers.keys()) + ["websearch"])
        raise ValueError(
            f"Unknown image provider: {provider_name}. "
            f"Available providers: {available}"
        )

    return providers[provider_name](api_key=api_key)
