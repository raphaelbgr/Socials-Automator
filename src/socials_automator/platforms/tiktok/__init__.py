"""TikTok platform adapter.

Implements TikTok Content Posting API for video uploads.
"""

from .publisher import TikTokPublisher
from .config import TikTokConfig

__all__ = [
    "TikTokPublisher",
    "TikTokConfig",
]
