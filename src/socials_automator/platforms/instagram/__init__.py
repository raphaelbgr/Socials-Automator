"""Instagram platform adapter.

Wraps the existing Instagram client to implement the PlatformPublisher interface.
"""

from .publisher import InstagramPublisher
from .config import InstagramConfig

__all__ = [
    "InstagramPublisher",
    "InstagramConfig",
]
