"""Pydantic models for news source configuration.

These models provide type-safe access to the YAML configuration
in config/news_sources.yaml.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class NewsCategory(str, Enum):
    """News story categories matching the existing NewsCategory enum."""

    CELEBRITY = "celebrity"
    MOVIES = "movies"
    TV = "tv"
    MUSIC = "music"
    STREAMING = "streaming"
    VIRAL = "viral"
    GENERAL = "general"


class RegionConfig(BaseModel):
    """Configuration for a geographic region."""

    id: str
    display_name: str
    languages: list[str]
    timezone: str
    priority_weight: float = 1.0
    country_code: str = ""


class FeedConfig(BaseModel):
    """Configuration for a single RSS feed."""

    name: str
    url: str
    region: str
    category: NewsCategory
    priority: int = Field(default=2, ge=1, le=3)
    language: str = "en"
    translate: bool = False
    enabled: bool = True

    @property
    def is_high_priority(self) -> bool:
        """Check if this is a high-priority feed."""
        return self.priority == 1


class QueryConfig(BaseModel):
    """Configuration for a search query."""

    query: str
    category: NewsCategory
    language: str = "en"
    priority: int = Field(default=2, ge=1, le=3)
    translate: bool = False
    batch: int = Field(default=1, ge=1, le=10)
    enabled: bool = True
    max_results: int = 10


class TimeWeights(BaseModel):
    """Time-based weights for region selection."""

    us: float = 1.0
    uk: float = 1.0
    korea: float = 1.0
    japan: float = 1.0
    latam: float = 1.0
    europe: float = 1.0
    india: float = 1.0
    australia: float = 1.0

    def get_weight(self, region: str) -> float:
        """Get weight for a specific region."""
        return getattr(self, region, 1.0)


class CategoryTargets(BaseModel):
    """Target percentages for category balance."""

    celebrity: float = 0.30
    music: float = 0.25
    movies: float = 0.20
    streaming: float = 0.15
    viral: float = 0.10


class FreshnessConfig(BaseModel):
    """Settings for article freshness scoring."""

    breaking_window_hours: int = 6
    breaking_boost: float = 1.5
    stale_penalty_per_hour: float = 0.02
    max_age_hours: int = 24


class QueryRotationConfig(BaseModel):
    """Settings for query rotation."""

    batches: int = 3
    queries_per_batch: int = 12
    cooldown_hours: int = 6


class RotationConfig(BaseModel):
    """Full rotation configuration."""

    time_weights: dict[str, TimeWeights] = Field(default_factory=dict)
    category_targets: CategoryTargets = Field(default_factory=CategoryTargets)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    query_rotation: QueryRotationConfig = Field(default_factory=QueryRotationConfig)

    @field_validator("time_weights", mode="before")
    @classmethod
    def parse_time_weights(cls, v: dict) -> dict[str, TimeWeights]:
        """Parse time weights from YAML format."""
        if isinstance(v, dict):
            return {k: TimeWeights(**vv) if isinstance(vv, dict) else vv for k, vv in v.items()}
        return v


class SettingsConfig(BaseModel):
    """Global settings."""

    default_max_age_hours: int = 24
    parallel_fetch_limit: int = 50
    translation_enabled: bool = True
    semantic_dedup_threshold: float = 0.85
    query_batch_size: int = 10
    query_cooldown_hours: int = 6


class NewsSourcesConfig(BaseModel):
    """Root configuration model for news_sources.yaml."""

    version: str = "1.0"
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    regions: dict[str, RegionConfig] = Field(default_factory=dict)
    feeds: list[FeedConfig] = Field(default_factory=list)
    queries: list[QueryConfig] = Field(default_factory=list)
    rotation: RotationConfig = Field(default_factory=RotationConfig)

    @field_validator("regions", mode="before")
    @classmethod
    def parse_regions(cls, v: dict) -> dict[str, RegionConfig]:
        """Parse regions from YAML format."""
        if isinstance(v, dict):
            result = {}
            for k, vv in v.items():
                if isinstance(vv, dict):
                    # Ensure 'id' field is set
                    if "id" not in vv:
                        vv["id"] = k
                    result[k] = RegionConfig(**vv)
                else:
                    result[k] = vv
            return result
        return v

    # Statistics properties

    @property
    def feed_count(self) -> int:
        """Total number of configured feeds."""
        return len(self.feeds)

    @property
    def enabled_feed_count(self) -> int:
        """Number of enabled feeds."""
        return len([f for f in self.feeds if f.enabled])

    @property
    def query_count(self) -> int:
        """Total number of configured queries."""
        return len(self.queries)

    @property
    def enabled_query_count(self) -> int:
        """Number of enabled queries."""
        return len([q for q in self.queries if q.enabled])

    @property
    def region_count(self) -> int:
        """Number of configured regions."""
        return len(self.regions)

    @property
    def languages(self) -> set[str]:
        """All languages covered by feeds and queries."""
        langs = set()
        for feed in self.feeds:
            langs.add(feed.language)
        for query in self.queries:
            langs.add(query.language)
        return langs

    # Query methods

    def get_feeds_by_region(self, region: str) -> list[FeedConfig]:
        """Get all feeds for a specific region."""
        return [f for f in self.feeds if f.region == region and f.enabled]

    def get_feeds_by_category(self, category: NewsCategory) -> list[FeedConfig]:
        """Get all feeds for a specific category."""
        return [f for f in self.feeds if f.category == category and f.enabled]

    def get_feeds_by_priority(self, max_priority: int = 2) -> list[FeedConfig]:
        """Get feeds up to a certain priority level."""
        return [f for f in self.feeds if f.priority <= max_priority and f.enabled]

    def get_queries_by_batch(self, batch: int) -> list[QueryConfig]:
        """Get queries for a specific batch."""
        return [q for q in self.queries if q.batch == batch and q.enabled]

    def get_queries_by_language(self, language: str) -> list[QueryConfig]:
        """Get queries for a specific language."""
        return [q for q in self.queries if q.language == language and q.enabled]

    def get_queries_needing_translation(self) -> list[QueryConfig]:
        """Get queries that require translation."""
        return [q for q in self.queries if q.translate and q.enabled]

    def get_time_weights_for_period(self, period: str) -> TimeWeights:
        """Get time weights for a specific period (morning, afternoon, evening, night)."""
        return self.rotation.time_weights.get(period, TimeWeights())

    def get_region(self, region_id: str) -> Optional[RegionConfig]:
        """Get region configuration by ID."""
        return self.regions.get(region_id)
