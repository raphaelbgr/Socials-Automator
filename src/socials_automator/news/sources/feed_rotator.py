"""Feed rotation with time-based and region-based weighting.

This module manages feed selection based on:
- Current UTC time (regions have different weights for different periods)
- Feed priority
- Category balance targets

Usage:
    from socials_automator.news.sources import FeedRotator

    rotator = FeedRotator()

    # Get feeds for current time
    feeds = rotator.get_feeds()

    # Get feeds for a specific edition
    feeds = rotator.get_feeds_for_edition("morning")

    # Get balanced feeds across categories
    feeds = rotator.get_balanced_feeds(max_feeds=30)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from socials_automator.news.sources.models import (
    FeedConfig,
    NewsCategory,
    TimeWeights,
)
from socials_automator.news.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger("ai_calls")


def _get_current_period() -> str:
    """Get current time period based on UTC hour.

    Returns:
        Period name: 'morning', 'afternoon', 'evening', or 'night'.
    """
    hour = datetime.utcnow().hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


@dataclass
class FeedSelection:
    """Result of feed selection with metadata."""

    feeds: list[FeedConfig]
    period: str
    regions_used: list[str]
    categories_used: list[str]
    selection_timestamp: datetime

    @property
    def feed_count(self) -> int:
        """Number of feeds selected."""
        return len(self.feeds)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "feed_count": self.feed_count,
            "period": self.period,
            "regions": self.regions_used,
            "categories": self.categories_used,
            "timestamp": self.selection_timestamp.isoformat(),
        }


class FeedRotator:
    """Manages time-weighted feed selection.

    Feeds are selected based on:
    1. Current time period (morning/afternoon/evening/night)
    2. Region weights for that period
    3. Feed priority (1=highest, 3=lowest)
    4. Category balance targets

    Example:
        rotator = FeedRotator()

        # Get optimal feeds for current time
        selection = rotator.select_feeds(max_feeds=40)
        feeds = selection.feeds

        # Get feeds emphasizing a specific region
        feeds = rotator.get_feeds_for_region("korea", max_feeds=10)
    """

    def __init__(self, registry: Optional[SourceRegistry] = None):
        """Initialize the rotator.

        Args:
            registry: SourceRegistry instance. Uses default if None.
        """
        self._registry = registry or get_source_registry()

    def select_feeds(
        self,
        max_feeds: int = 50,
        period: Optional[str] = None,
        include_priority_3: bool = True,
    ) -> FeedSelection:
        """Select feeds based on current time and weights.

        Args:
            max_feeds: Maximum number of feeds to select.
            period: Time period override. Uses current if None.
            include_priority_3: Whether to include lowest priority feeds.

        Returns:
            FeedSelection with selected feeds and metadata.
        """
        if period is None:
            period = _get_current_period()

        # Get time weights for this period
        weights = self._registry.config.get_time_weights_for_period(period)

        # Get all enabled feeds
        max_priority = 3 if include_priority_3 else 2
        all_feeds = self._registry.get_feeds_by_priority(max_priority)

        # Score each feed
        scored_feeds = self._score_feeds(all_feeds, weights)

        # Sort by score (higher is better)
        scored_feeds.sort(key=lambda x: x[0], reverse=True)

        # Select top feeds
        selected = [feed for _, feed in scored_feeds[:max_feeds]]

        # Collect metadata
        regions_used = list(set(f.region for f in selected))
        categories_used = list(set(f.category.value for f in selected))

        selection = FeedSelection(
            feeds=selected,
            period=period,
            regions_used=regions_used,
            categories_used=categories_used,
            selection_timestamp=datetime.utcnow(),
        )

        logger.info(
            f"FEED_ROTATOR | period={period} | selected={len(selected)} | "
            f"regions={len(regions_used)} | categories={len(categories_used)}"
        )

        return selection

    def _score_feeds(
        self,
        feeds: list[FeedConfig],
        weights: TimeWeights,
    ) -> list[tuple[float, FeedConfig]]:
        """Score feeds based on priority and region weights.

        Args:
            feeds: List of feeds to score.
            weights: Time-based weights for regions.

        Returns:
            List of (score, feed) tuples.
        """
        scored: list[tuple[float, FeedConfig]] = []

        for feed in feeds:
            # Base score from priority (invert so priority 1 = score 3)
            priority_score = 4 - feed.priority  # 1->3, 2->2, 3->1

            # Region weight (0.0 to 1.0)
            region_weight = weights.get_weight(feed.region)

            # Combined score
            score = priority_score * region_weight

            # Small random factor for variety (0-0.1)
            score += random.random() * 0.1

            scored.append((score, feed))

        return scored

    def get_feeds_for_edition(
        self,
        edition: str,
        max_feeds: int = 50,
    ) -> list[FeedConfig]:
        """Get feeds for a specific edition.

        Args:
            edition: Edition name (morning, midday, evening, night).

        Returns:
            List of selected feeds.
        """
        # Map edition to period
        period_map = {
            "morning": "morning",
            "midday": "afternoon",
            "evening": "evening",
            "night": "night",
        }
        period = period_map.get(edition, "morning")

        selection = self.select_feeds(max_feeds=max_feeds, period=period)
        return selection.feeds

    def get_feeds_for_region(
        self,
        region: str,
        max_feeds: int = 10,
    ) -> list[FeedConfig]:
        """Get feeds for a specific region.

        Args:
            region: Region ID (e.g., 'us', 'korea').
            max_feeds: Maximum feeds to return.

        Returns:
            List of feeds from that region.
        """
        all_region_feeds = self._registry.get_feeds_by_region(region)

        # Sort by priority
        all_region_feeds.sort(key=lambda f: f.priority)

        return all_region_feeds[:max_feeds]

    def get_balanced_feeds(
        self,
        max_feeds: int = 30,
        period: Optional[str] = None,
    ) -> list[FeedConfig]:
        """Get feeds balanced across categories.

        Uses category targets from config to balance selection.

        Args:
            max_feeds: Total feeds to select.
            period: Time period. Uses current if None.

        Returns:
            Balanced list of feeds.
        """
        if period is None:
            period = _get_current_period()

        targets = self._registry.config.rotation.category_targets
        weights = self._registry.config.get_time_weights_for_period(period)

        # Calculate target count per category
        category_counts = {
            NewsCategory.CELEBRITY: int(max_feeds * targets.celebrity),
            NewsCategory.MUSIC: int(max_feeds * targets.music),
            NewsCategory.MOVIES: int(max_feeds * targets.movies),
            NewsCategory.STREAMING: int(max_feeds * targets.streaming),
            NewsCategory.VIRAL: int(max_feeds * targets.viral),
        }

        # Fill remaining with general
        allocated = sum(category_counts.values())
        category_counts[NewsCategory.GENERAL] = max_feeds - allocated

        # Select feeds per category
        selected: list[FeedConfig] = []

        for category, count in category_counts.items():
            if count <= 0:
                continue

            category_feeds = self._registry.get_feeds_by_category(category)

            # Score by region weight and priority
            scored = self._score_feeds(category_feeds, weights)
            scored.sort(key=lambda x: x[0], reverse=True)

            for _, feed in scored[:count]:
                selected.append(feed)

        logger.debug(
            f"FEED_ROTATOR | balanced selection | total={len(selected)} | "
            f"targets={category_counts}"
        )

        return selected

    def get_high_priority_feeds(self, max_feeds: int = 20) -> list[FeedConfig]:
        """Get only priority 1 feeds.

        Args:
            max_feeds: Maximum feeds to return.

        Returns:
            List of high-priority feeds.
        """
        priority_1 = self._registry.get_feeds_by_priority(1)
        return priority_1[:max_feeds]

    def get_summary(self) -> dict:
        """Get summary of feed distribution.

        Returns:
            Dictionary with distribution info.
        """
        all_feeds = self._registry.get_all_feeds()

        by_region: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_priority: dict[int, int] = {}

        for feed in all_feeds:
            by_region[feed.region] = by_region.get(feed.region, 0) + 1
            by_category[feed.category.value] = by_category.get(feed.category.value, 0) + 1
            by_priority[feed.priority] = by_priority.get(feed.priority, 0) + 1

        return {
            "total_feeds": len(all_feeds),
            "by_region": by_region,
            "by_category": by_category,
            "by_priority": by_priority,
            "current_period": _get_current_period(),
        }


# Module-level cached instance
_rotator_instance: Optional[FeedRotator] = None


def get_feed_rotator(reload: bool = False) -> FeedRotator:
    """Get or create the default FeedRotator instance.

    Args:
        reload: If True, recreate the instance.

    Returns:
        FeedRotator instance.
    """
    global _rotator_instance

    if _rotator_instance is None or reload:
        _rotator_instance = FeedRotator()

    return _rotator_instance
