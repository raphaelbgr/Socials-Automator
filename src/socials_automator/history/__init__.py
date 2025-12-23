"""Unified content history module for uniqueness checking.

Provides profile-scoped history tracking for:
- Regular video reels (topics)
- News video reels (stories, themes)
- Future: Carousel posts

All history classes share common similarity checking and posted content scanning.
"""

from .base import BaseContentHistory
from .reel_history import ReelContentHistory
from .news_history import NewsContentHistory
from .similarity import normalize_text, jaccard_similarity, is_similar
from .posted_scanner import PostedContentScanner

__all__ = [
    "BaseContentHistory",
    "ReelContentHistory",
    "NewsContentHistory",
    "PostedContentScanner",
    "normalize_text",
    "jaccard_similarity",
    "is_similar",
]
