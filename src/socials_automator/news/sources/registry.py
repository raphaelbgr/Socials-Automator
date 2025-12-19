"""Source registry for loading and managing news source configuration.

This module provides the SourceRegistry class which loads the YAML configuration
and provides convenient methods for accessing feeds and queries based on various
criteria like time of day, region, category, etc.

Usage:
    from socials_automator.news.sources import SourceRegistry

    registry = SourceRegistry.load()

    # Get feeds for current time
    feeds = registry.get_feeds_for_edition()

    # Get queries for a specific batch
    queries = registry.get_queries_for_batch(1)

    # Get region-weighted feeds
    feeds = registry.get_weighted_feeds_for_period("morning")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from socials_automator.news.sources.models import (
    NewsSourcesConfig,
    FeedConfig,
    QueryConfig,
    NewsCategory,
    TimeWeights,
)

logger = logging.getLogger("ai_calls")


def _get_time_period(hour: int) -> str:
    """Convert UTC hour to time period name.

    Args:
        hour: Hour in UTC (0-23).

    Returns:
        Period name: 'morning', 'afternoon', 'evening', or 'night'.
    """
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


class SourceRegistry:
    """Registry for managing news source configuration.

    Loads configuration from YAML and provides methods for:
    - Getting feeds based on time, region, category, priority
    - Getting queries based on batch, language, category
    - Time-weighted feed selection for different regions

    Example:
        registry = SourceRegistry.load()

        # Get high-priority feeds
        priority_feeds = registry.get_feeds_by_priority(1)

        # Get K-pop feeds
        kpop_feeds = registry.get_feeds_by_region("korea")

        # Get current batch queries
        queries = registry.get_current_batch_queries()
    """

    # Default path for config file
    DEFAULT_CONFIG_PATH = Path("config/news_sources.yaml")

    def __init__(self, config: NewsSourcesConfig):
        """Initialize with configuration.

        Args:
            config: Parsed configuration object.
        """
        self._config = config
        self._loaded_at = datetime.utcnow()

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "SourceRegistry":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config. Uses default if None.

        Returns:
            SourceRegistry instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If YAML is invalid.
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH

        # Make path absolute if relative
        if not path.is_absolute():
            # Try relative to current directory
            if not path.exists():
                # Try relative to project root
                project_root = Path(__file__).parents[4]  # src/socials_automator/news/sources -> project root
                path = project_root / path

        if not path.exists():
            raise FileNotFoundError(f"News sources config not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        try:
            config = NewsSourcesConfig(**data)
        except Exception as e:
            raise ValueError(f"Invalid config structure in {path}: {e}")

        logger.info(
            f"SOURCE_REGISTRY | loaded={path.name} | feeds={config.enabled_feed_count} | "
            f"queries={config.enabled_query_count} | regions={config.region_count}"
        )

        return cls(config)

    @property
    def config(self) -> NewsSourcesConfig:
        """Get the underlying configuration."""
        return self._config

    # =========================================================================
    # Feed Access Methods
    # =========================================================================

    def get_all_feeds(self, enabled_only: bool = True) -> list[FeedConfig]:
        """Get all configured feeds.

        Args:
            enabled_only: If True, only return enabled feeds.

        Returns:
            List of feed configurations.
        """
        if enabled_only:
            return [f for f in self._config.feeds if f.enabled]
        return list(self._config.feeds)

    def get_feeds_by_region(self, region: str) -> list[FeedConfig]:
        """Get feeds for a specific region.

        Args:
            region: Region ID (e.g., 'us', 'korea', 'uk').

        Returns:
            List of feeds for that region.
        """
        return self._config.get_feeds_by_region(region)

    def get_feeds_by_category(self, category: NewsCategory) -> list[FeedConfig]:
        """Get feeds for a specific category.

        Args:
            category: News category enum.

        Returns:
            List of feeds for that category.
        """
        return self._config.get_feeds_by_category(category)

    def get_feeds_by_priority(self, max_priority: int = 2) -> list[FeedConfig]:
        """Get feeds up to a certain priority level.

        Args:
            max_priority: Maximum priority (1=highest, 3=lowest).

        Returns:
            List of feeds with priority <= max_priority.
        """
        return self._config.get_feeds_by_priority(max_priority)

    def get_feeds_for_edition(
        self,
        utc_hour: Optional[int] = None,
        max_feeds: int = 50,
    ) -> list[FeedConfig]:
        """Get feeds appropriate for the current time edition.

        Uses time-based weighting to prioritize regions that are active
        during the current time period.

        Args:
            utc_hour: Hour in UTC (0-23). Uses current hour if None.
            max_feeds: Maximum number of feeds to return.

        Returns:
            List of feeds weighted by current time period.
        """
        if utc_hour is None:
            utc_hour = datetime.utcnow().hour

        period = _get_time_period(utc_hour)
        return self.get_weighted_feeds_for_period(period, max_feeds)

    def get_weighted_feeds_for_period(
        self,
        period: str,
        max_feeds: int = 50,
    ) -> list[FeedConfig]:
        """Get feeds weighted by region for a time period.

        Higher-weighted regions get more feeds included.

        Args:
            period: Time period ('morning', 'afternoon', 'evening', 'night').
            max_feeds: Maximum feeds to return.

        Returns:
            List of weighted feeds.
        """
        weights = self._config.get_time_weights_for_period(period)
        all_feeds = self.get_all_feeds()

        # Calculate effective priority for each feed
        # priority * region_weight -> lower is better
        feed_scores: list[tuple[float, FeedConfig]] = []

        for feed in all_feeds:
            region_weight = weights.get_weight(feed.region)
            # Invert weight so higher weight = lower score = more priority
            effective_score = feed.priority / (region_weight + 0.01)
            feed_scores.append((effective_score, feed))

        # Sort by score (lower is better)
        feed_scores.sort(key=lambda x: x[0])

        # Return top feeds
        selected = [feed for _, feed in feed_scores[:max_feeds]]

        # Log distribution
        region_counts = {}
        for feed in selected:
            region_counts[feed.region] = region_counts.get(feed.region, 0) + 1

        logger.debug(
            f"FEED_SELECTION | period={period} | selected={len(selected)} | "
            f"regions={region_counts}"
        )

        return selected

    # =========================================================================
    # Query Access Methods
    # =========================================================================

    def get_all_queries(self, enabled_only: bool = True) -> list[QueryConfig]:
        """Get all configured queries.

        Args:
            enabled_only: If True, only return enabled queries.

        Returns:
            List of query configurations.
        """
        if enabled_only:
            return [q for q in self._config.queries if q.enabled]
        return list(self._config.queries)

    def get_queries_by_batch(self, batch: int) -> list[QueryConfig]:
        """Get queries for a specific batch.

        Args:
            batch: Batch number (1-3 typically).

        Returns:
            List of queries in that batch.
        """
        return self._config.get_queries_by_batch(batch)

    def get_queries_by_language(self, language: str) -> list[QueryConfig]:
        """Get queries for a specific language.

        Args:
            language: Language code (e.g., 'en', 'es', 'ko').

        Returns:
            List of queries in that language.
        """
        return self._config.get_queries_by_language(language)

    def get_queries_by_category(self, category: NewsCategory) -> list[QueryConfig]:
        """Get queries for a specific category.

        Args:
            category: News category enum.

        Returns:
            List of queries for that category.
        """
        return [
            q for q in self._config.queries
            if q.category == category and q.enabled
        ]

    def get_queries_needing_translation(self) -> list[QueryConfig]:
        """Get queries that require translation to English.

        Returns:
            List of queries with translate=True.
        """
        return self._config.get_queries_needing_translation()

    def get_batch_count(self) -> int:
        """Get the number of query batches configured.

        Returns:
            Number of batches (typically 3).
        """
        return self._config.rotation.query_rotation.batches

    def get_cooldown_hours(self) -> int:
        """Get the cooldown period between batch rotations.

        Returns:
            Cooldown in hours.
        """
        return self._config.rotation.query_rotation.cooldown_hours

    # =========================================================================
    # Region Access Methods
    # =========================================================================

    def get_all_regions(self) -> list[str]:
        """Get all region IDs.

        Returns:
            List of region IDs (e.g., ['us', 'uk', 'korea']).
        """
        return list(self._config.regions.keys())

    def get_region_timezone(self, region: str) -> str:
        """Get timezone for a region.

        Args:
            region: Region ID.

        Returns:
            Timezone abbreviation (e.g., 'EST', 'KST').
        """
        region_config = self._config.get_region(region)
        return region_config.timezone if region_config else "UTC"

    def get_region_languages(self, region: str) -> list[str]:
        """Get languages for a region.

        Args:
            region: Region ID.

        Returns:
            List of language codes.
        """
        region_config = self._config.get_region(region)
        return region_config.languages if region_config else ["en"]

    # =========================================================================
    # Settings Access
    # =========================================================================

    def get_max_age_hours(self) -> int:
        """Get default maximum article age in hours."""
        return self._config.settings.default_max_age_hours

    def get_dedup_threshold(self) -> float:
        """Get semantic deduplication similarity threshold."""
        return self._config.settings.semantic_dedup_threshold

    def is_translation_enabled(self) -> bool:
        """Check if translation is enabled globally."""
        return self._config.settings.translation_enabled

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_summary(self) -> dict:
        """Get a summary of the configuration.

        Returns:
            Dictionary with counts and statistics.
        """
        feeds = self.get_all_feeds()
        queries = self.get_all_queries()

        feeds_by_region = {}
        for feed in feeds:
            feeds_by_region[feed.region] = feeds_by_region.get(feed.region, 0) + 1

        feeds_by_category = {}
        for feed in feeds:
            cat = feed.category.value
            feeds_by_category[cat] = feeds_by_category.get(cat, 0) + 1

        queries_by_batch = {}
        for query in queries:
            queries_by_batch[query.batch] = queries_by_batch.get(query.batch, 0) + 1

        queries_by_language = {}
        for query in queries:
            queries_by_language[query.language] = queries_by_language.get(query.language, 0) + 1

        return {
            "version": self._config.version,
            "loaded_at": self._loaded_at.isoformat(),
            "feeds": {
                "total": len(feeds),
                "by_region": feeds_by_region,
                "by_category": feeds_by_category,
            },
            "queries": {
                "total": len(queries),
                "by_batch": queries_by_batch,
                "by_language": queries_by_language,
                "need_translation": len(self.get_queries_needing_translation()),
            },
            "regions": self.get_all_regions(),
            "settings": {
                "max_age_hours": self.get_max_age_hours(),
                "dedup_threshold": self.get_dedup_threshold(),
                "translation_enabled": self.is_translation_enabled(),
                "cooldown_hours": self.get_cooldown_hours(),
            },
        }


# Module-level cached instance
_registry_instance: Optional[SourceRegistry] = None


def get_source_registry(reload: bool = False) -> SourceRegistry:
    """Get or create the default SourceRegistry instance.

    Args:
        reload: If True, reload from YAML even if already loaded.

    Returns:
        SourceRegistry instance.
    """
    global _registry_instance

    if _registry_instance is None or reload:
        _registry_instance = SourceRegistry.load()

    return _registry_instance
