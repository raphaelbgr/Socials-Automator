"""News source configurations for RSS feeds and search queries."""

from dataclasses import dataclass
from typing import Optional

from socials_automator.news.models import NewsCategory


@dataclass(frozen=True)
class RSSFeedConfig:
    """Configuration for a single RSS feed source."""

    name: str
    url: str
    category: NewsCategory
    priority: int = 2  # 1 = highest, 3 = lowest
    enabled: bool = True


@dataclass(frozen=True)
class SearchQueryConfig:
    """Configuration for a DuckDuckGo search query."""

    query: str
    category: NewsCategory
    priority: int = 2
    max_results: int = 10
    enabled: bool = True


# =============================================================================
# RSS Feed Sources
# =============================================================================

CELEBRITY_FEEDS = [
    RSSFeedConfig(
        name="TMZ",
        url="https://www.tmz.com/rss.xml",
        category=NewsCategory.CELEBRITY,
        priority=1,
    ),
    RSSFeedConfig(
        name="E! News",
        url="https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml",
        category=NewsCategory.CELEBRITY,
        priority=2,
    ),
    RSSFeedConfig(
        name="People",
        url="https://people.com/feed/",
        category=NewsCategory.CELEBRITY,
        priority=2,
    ),
    RSSFeedConfig(
        name="Page Six",
        url="https://pagesix.com/feed/",
        category=NewsCategory.CELEBRITY,
        priority=3,
    ),
    RSSFeedConfig(
        name="Us Weekly",
        url="https://www.usmagazine.com/feed/",
        category=NewsCategory.CELEBRITY,
        priority=3,
    ),
]

MOVIES_TV_FEEDS = [
    RSSFeedConfig(
        name="Variety",
        url="https://variety.com/feed/",
        category=NewsCategory.MOVIES,
        priority=1,
    ),
    RSSFeedConfig(
        name="Deadline",
        url="https://deadline.com/feed/",
        category=NewsCategory.MOVIES,
        priority=1,
    ),
    RSSFeedConfig(
        name="Hollywood Reporter",
        url="https://www.hollywoodreporter.com/feed/",
        category=NewsCategory.MOVIES,
        priority=2,
    ),
    RSSFeedConfig(
        name="Entertainment Weekly",
        url="https://ew.com/feed/",
        category=NewsCategory.TV,
        priority=2,
    ),
    RSSFeedConfig(
        name="Screen Rant",
        url="https://screenrant.com/feed/",
        category=NewsCategory.MOVIES,
        priority=3,
    ),
]

MUSIC_FEEDS = [
    RSSFeedConfig(
        name="Rolling Stone",
        url="https://www.rollingstone.com/feed/",
        category=NewsCategory.MUSIC,
        priority=1,
    ),
    RSSFeedConfig(
        name="Billboard",
        url="https://www.billboard.com/feed/",
        category=NewsCategory.MUSIC,
        priority=1,
    ),
    RSSFeedConfig(
        name="Pitchfork",
        url="https://pitchfork.com/feed/feed-news/rss",
        category=NewsCategory.MUSIC,
        priority=2,
    ),
    RSSFeedConfig(
        name="NME",
        url="https://www.nme.com/feed",
        category=NewsCategory.MUSIC,
        priority=3,
    ),
]

STREAMING_FEEDS = [
    RSSFeedConfig(
        name="What's on Netflix",
        url="https://www.whats-on-netflix.com/feed/",
        category=NewsCategory.STREAMING,
        priority=2,
    ),
    RSSFeedConfig(
        name="Decider",
        url="https://decider.com/feed/",
        category=NewsCategory.STREAMING,
        priority=2,
    ),
]

GENERAL_ENTERTAINMENT_FEEDS = [
    RSSFeedConfig(
        name="The Verge Entertainment",
        url="https://www.theverge.com/entertainment/rss/index.xml",
        category=NewsCategory.GENERAL,
        priority=2,
    ),
    RSSFeedConfig(
        name="AV Club",
        url="https://www.avclub.com/rss",
        category=NewsCategory.GENERAL,
        priority=3,
    ),
]

# Combined list of all feeds
ALL_RSS_FEEDS = (
    CELEBRITY_FEEDS +
    MOVIES_TV_FEEDS +
    MUSIC_FEEDS +
    STREAMING_FEEDS +
    GENERAL_ENTERTAINMENT_FEEDS
)


# =============================================================================
# DuckDuckGo Search Queries
# =============================================================================

ENTERTAINMENT_SEARCH_QUERIES = [
    # Breaking/trending
    SearchQueryConfig(
        query="celebrity news today",
        category=NewsCategory.CELEBRITY,
        priority=1,
        max_results=10,
    ),
    SearchQueryConfig(
        query="entertainment news breaking",
        category=NewsCategory.GENERAL,
        priority=1,
        max_results=10,
    ),
    SearchQueryConfig(
        query="viral celebrity moment today",
        category=NewsCategory.VIRAL,
        priority=1,
        max_results=5,
    ),
    # Movies & TV
    SearchQueryConfig(
        query="new movie releases this week",
        category=NewsCategory.MOVIES,
        priority=2,
        max_results=8,
    ),
    SearchQueryConfig(
        query="tv show premiere this week",
        category=NewsCategory.TV,
        priority=2,
        max_results=8,
    ),
    # Streaming
    SearchQueryConfig(
        query="streaming news netflix disney plus",
        category=NewsCategory.STREAMING,
        priority=2,
        max_results=8,
    ),
    SearchQueryConfig(
        query="new on netflix this week",
        category=NewsCategory.STREAMING,
        priority=2,
        max_results=5,
    ),
    # Music
    SearchQueryConfig(
        query="new music releases this week",
        category=NewsCategory.MUSIC,
        priority=2,
        max_results=8,
    ),
    SearchQueryConfig(
        query="concert tour announcement 2025",
        category=NewsCategory.MUSIC,
        priority=3,
        max_results=5,
    ),
    # Drama/viral (lower priority - only if needed)
    SearchQueryConfig(
        query="celebrity drama today",
        category=NewsCategory.VIRAL,
        priority=3,
        max_results=5,
    ),
]


# =============================================================================
# Source Management Functions
# =============================================================================

def get_feeds_by_category(category: NewsCategory) -> list[RSSFeedConfig]:
    """Get all RSS feeds for a specific category."""
    return [f for f in ALL_RSS_FEEDS if f.category == category and f.enabled]


def get_feeds_by_priority(max_priority: int = 2) -> list[RSSFeedConfig]:
    """Get RSS feeds up to a certain priority level.

    Args:
        max_priority: Maximum priority level (1=highest, 3=lowest)

    Returns:
        List of feeds with priority <= max_priority
    """
    return [f for f in ALL_RSS_FEEDS if f.priority <= max_priority and f.enabled]


def get_queries_by_category(category: NewsCategory) -> list[SearchQueryConfig]:
    """Get all search queries for a specific category."""
    return [q for q in ENTERTAINMENT_SEARCH_QUERIES if q.category == category and q.enabled]


def get_queries_by_priority(max_priority: int = 2) -> list[SearchQueryConfig]:
    """Get search queries up to a certain priority level."""
    return [q for q in ENTERTAINMENT_SEARCH_QUERIES if q.priority <= max_priority and q.enabled]


def get_all_enabled_feeds() -> list[RSSFeedConfig]:
    """Get all enabled RSS feeds."""
    return [f for f in ALL_RSS_FEEDS if f.enabled]


def get_all_enabled_queries() -> list[SearchQueryConfig]:
    """Get all enabled search queries."""
    return [q for q in ENTERTAINMENT_SEARCH_QUERIES if q.enabled]


# =============================================================================
# Source Categories Summary
# =============================================================================

def get_source_summary() -> dict:
    """Get a summary of all configured sources."""
    feeds = get_all_enabled_feeds()
    queries = get_all_enabled_queries()

    return {
        "rss_feeds": {
            "total": len(feeds),
            "by_category": {
                cat.value: len([f for f in feeds if f.category == cat])
                for cat in NewsCategory
            },
            "by_priority": {
                f"priority_{p}": len([f for f in feeds if f.priority == p])
                for p in [1, 2, 3]
            },
        },
        "search_queries": {
            "total": len(queries),
            "by_category": {
                cat.value: len([q for q in queries if q.category == cat])
                for cat in NewsCategory
            },
        },
        "feed_names": [f.name for f in feeds],
    }
