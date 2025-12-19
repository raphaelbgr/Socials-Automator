"""Data models for news aggregation and curation."""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


class NewsEdition(str, Enum):
    """Time-based news edition types."""

    MORNING = "morning"
    MIDDAY = "midday"
    EVENING = "evening"
    NIGHT = "night"

    @classmethod
    def from_hour(cls, hour: int) -> "NewsEdition":
        """Determine edition based on hour (UTC)."""
        if 6 <= hour < 12:
            return cls.MORNING
        elif 12 <= hour < 17:
            return cls.MIDDAY
        elif 17 <= hour < 22:
            return cls.EVENING
        else:
            return cls.NIGHT

    @property
    def display_name(self) -> str:
        """Human-readable edition name."""
        names = {
            self.MORNING: "Morning Briefing",
            self.MIDDAY: "Midday Pulse",
            self.EVENING: "Evening Wrap",
            self.NIGHT: "Night Edition",
        }
        return names[self]

    @property
    def theme(self) -> str:
        """Content theme for this edition."""
        themes = {
            self.MORNING: "Overnight news and what to expect today",
            self.MIDDAY: "Breaking stories and trending topics",
            self.EVENING: "Today's biggest stories summarized",
            self.NIGHT: "Entertainment highlights and tomorrow's preview",
        }
        return themes[self]


class NewsCategory(str, Enum):
    """News story categories."""

    CELEBRITY = "celebrity"
    MOVIES = "movies"
    TV = "tv"
    MUSIC = "music"
    STREAMING = "streaming"
    VIRAL = "viral"
    GENERAL = "general"


@dataclass
class NewsArticle:
    """Raw article fetched from RSS feed or search."""

    title: str
    summary: str
    source_name: str
    source_url: str
    article_url: str
    published_at: datetime
    category: NewsCategory = NewsCategory.GENERAL
    image_url: Optional[str] = None
    author: Optional[str] = None

    # Metadata for deduplication and ranking
    content_hash: str = ""
    fetch_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Location and language metadata (for global news support)
    region: str = "us"  # Region ID (us, uk, korea, japan, latam, europe, india, australia)
    source_language: str = "en"  # Original language code
    source_timezone: str = "UTC"  # Source timezone abbreviation (EST, KST, etc.)
    was_translated: bool = False  # Whether content was translated to English
    country_code: str = ""  # ISO country code if known (US, GB, KR, etc.)

    @property
    def age_hours(self) -> float:
        """Hours since publication."""
        delta = datetime.utcnow() - self.published_at
        return delta.total_seconds() / 3600

    @property
    def is_fresh(self) -> bool:
        """Check if article is within 24 hours."""
        return self.age_hours <= 24

    @property
    def is_breaking(self) -> bool:
        """Check if article is within 6 hours (breaking news window)."""
        return self.age_hours <= 6

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "summary": self.summary,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "article_url": self.article_url,
            "published_at": self.published_at.isoformat(),
            "category": self.category.value,
            "image_url": self.image_url,
            "author": self.author,
            "age_hours": round(self.age_hours, 1),
            "is_fresh": self.is_fresh,
            "is_breaking": self.is_breaking,
            # Location and language metadata
            "region": self.region,
            "source_language": self.source_language,
            "source_timezone": self.source_timezone,
            "was_translated": self.was_translated,
            "country_code": self.country_code,
        }


@dataclass
class NewsStory:
    """Curated, AI-processed story ready for script generation.

    This is the output of the NewsCurator - a cleaned up, summarized
    version of a NewsArticle with additional context for video creation.
    """

    headline: str  # Punchy, rewritten headline for video
    summary: str  # 2-3 sentences, casual tone
    why_it_matters: str  # "Why you should care" angle
    source_name: str  # Attribution (TMZ, Variety, etc.)
    category: NewsCategory
    visual_keywords: list[str]  # For stock video search (2-word phrases)

    # Original article reference
    original_url: str = ""
    original_title: str = ""
    published_at: Optional[datetime] = None

    # Ranking metadata
    relevance_score: float = 0.0  # 0-1, how relevant/important
    virality_score: float = 0.0  # 0-1, how shareable/trending
    usefulness_score: float = 0.0  # 0-1, how actionable for viewer

    # Location and language metadata
    region: str = "us"  # Region ID
    source_language: str = "en"  # Original language code
    was_translated: bool = False  # Whether content was translated

    @property
    def total_score(self) -> float:
        """Combined ranking score."""
        # Weight: relevance 40%, virality 30%, usefulness 30%
        return (
            self.relevance_score * 0.4 +
            self.virality_score * 0.3 +
            self.usefulness_score * 0.3
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "headline": self.headline,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "source_name": self.source_name,
            "category": self.category.value,
            "visual_keywords": self.visual_keywords,
            "original_url": self.original_url,
            "original_title": self.original_title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "scores": {
                "relevance": round(self.relevance_score, 2),
                "virality": round(self.virality_score, 2),
                "usefulness": round(self.usefulness_score, 2),
                "total": round(self.total_score, 2),
            },
            # Location and language metadata
            "region": self.region,
            "source_language": self.source_language,
            "was_translated": self.was_translated,
        }


@dataclass
class NewsBrief:
    """Collection of curated stories for one video.

    This is the final output of the news curation pipeline,
    ready to be passed to the script planner.
    """

    edition: NewsEdition
    date: date
    stories: list[NewsStory]
    theme: str  # e.g., "Today's entertainment buzz"

    # Generation metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    total_articles_fetched: int = 0
    total_articles_after_dedup: int = 0

    @property
    def story_count(self) -> int:
        """Number of stories in this brief."""
        return len(self.stories)

    @property
    def categories_covered(self) -> list[str]:
        """List of unique categories in this brief."""
        return list(set(s.category.value for s in self.stories))

    @property
    def sources_cited(self) -> list[str]:
        """List of unique sources cited."""
        return list(set(s.source_name for s in self.stories))

    def get_hook_text(self) -> str:
        """Generate hook text for the video (spoken narration)."""
        count = self.story_count
        if count == 1:
            return f"The one story you need to know today"
        else:
            return f"{count} things in entertainment you need to know today"

    def get_thumbnail_text(self, max_stories: int = 3) -> str:
        """Generate teaser list for thumbnail/visual hook.

        Creates a visual teaser showing key headlines like:
            JUST IN
            -> Netflix new movie
            -> Metallica tour
            -> Taylor Swift album

        Args:
            max_stories: Maximum headlines to show (default 3 for readability)

        Returns:
            Formatted teaser text for thumbnail display.
        """
        if not self.stories:
            return "BREAKING NEWS"

        lines = ["JUST IN"]

        for story in self.stories[:max_stories]:
            # Get short version of headline (first 4-5 words)
            headline = story.headline
            words = headline.split()

            # Truncate to ~25 chars for thumbnail readability
            short_headline = ""
            for word in words:
                if len(short_headline) + len(word) + 1 > 25:
                    break
                short_headline += (" " if short_headline else "") + word

            if not short_headline:
                short_headline = words[0] if words else "News"

            lines.append(f"-> {short_headline}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "edition": self.edition.value,
            "edition_display": self.edition.display_name,
            "date": self.date.isoformat(),
            "theme": self.theme,
            "story_count": self.story_count,
            "stories": [s.to_dict() for s in self.stories],
            "categories_covered": self.categories_covered,
            "sources_cited": self.sources_cited,
            "metadata": {
                "generated_at": self.generated_at.isoformat(),
                "total_articles_fetched": self.total_articles_fetched,
                "total_articles_after_dedup": self.total_articles_after_dedup,
            },
        }


@dataclass
class AggregationResult:
    """Result from NewsAggregator.fetch() operation."""

    articles: list[NewsArticle]
    fetch_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Source statistics
    rss_articles_count: int = 0
    search_articles_count: int = 0
    total_before_dedup: int = 0
    duplicates_removed: int = 0

    # Error tracking
    failed_feeds: list[str] = field(default_factory=list)
    failed_queries: list[str] = field(default_factory=list)

    # Region/language statistics (for enhanced logging)
    regions_used: list[str] = field(default_factory=list)
    languages_used: list[str] = field(default_factory=list)
    feeds_fetched: list[str] = field(default_factory=list)
    query_batch: int = 0  # Which query batch was used

    @property
    def total_articles(self) -> int:
        """Total articles after deduplication."""
        return len(self.articles)

    @property
    def success_rate(self) -> float:
        """Percentage of sources that succeeded."""
        total_sources = (
            self.rss_articles_count + self.search_articles_count +
            len(self.failed_feeds) + len(self.failed_queries)
        )
        if total_sources == 0:
            return 0.0
        successful = self.rss_articles_count + self.search_articles_count
        return successful / total_sources

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            "total_articles": self.total_articles,
            "fetch_timestamp": self.fetch_timestamp.isoformat(),
            "sources": {
                "rss_count": self.rss_articles_count,
                "search_count": self.search_articles_count,
                "total_before_dedup": self.total_before_dedup,
                "duplicates_removed": self.duplicates_removed,
                "feeds_fetched": self.feeds_fetched,
                "query_batch": self.query_batch,
            },
            "coverage": {
                "regions": self.regions_used,
                "languages": self.languages_used,
            },
            "errors": {
                "failed_feeds": self.failed_feeds,
                "failed_queries": self.failed_queries,
            },
            "success_rate": round(self.success_rate, 2),
        }
