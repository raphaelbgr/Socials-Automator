"""News source configuration and registry.

This package provides centralized configuration for news sources
loaded from config/news_sources.yaml, with support for:
- Time-weighted feed selection
- Round-robin query batch rotation
- Region and category balancing

Usage:
    from socials_automator.news.sources import (
        SourceRegistry,
        get_source_registry,
        QueryRotator,
        get_query_rotator,
        FeedRotator,
        get_feed_rotator,
    )

    # Get cached registry instance
    registry = get_source_registry()

    # Get feeds for current time with rotation
    rotator = get_feed_rotator()
    feeds = rotator.select_feeds(max_feeds=40)

    # Get queries with batch rotation
    query_rotator = get_query_rotator()
    queries = query_rotator.get_current_queries()
"""

from .models import (
    NewsSourcesConfig,
    FeedConfig,
    QueryConfig,
    RegionConfig,
    NewsCategory,
    TimeWeights,
    RotationConfig,
    SettingsConfig,
    FreshnessConfig,
    CategoryTargets,
    QueryRotationConfig,
)

from .registry import (
    SourceRegistry,
    get_source_registry,
)

from .query_rotator import (
    QueryRotator,
    RotationState,
    get_query_rotator,
)

from .feed_rotator import (
    FeedRotator,
    FeedSelection,
    get_feed_rotator,
)

__all__ = [
    # Models
    "NewsSourcesConfig",
    "FeedConfig",
    "QueryConfig",
    "RegionConfig",
    "NewsCategory",
    "TimeWeights",
    "RotationConfig",
    "SettingsConfig",
    "FreshnessConfig",
    "CategoryTargets",
    "QueryRotationConfig",
    # Registry
    "SourceRegistry",
    "get_source_registry",
    # Query Rotation
    "QueryRotator",
    "RotationState",
    "get_query_rotator",
    # Feed Rotation
    "FeedRotator",
    "FeedSelection",
    "get_feed_rotator",
]
