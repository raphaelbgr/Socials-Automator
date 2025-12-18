"""News aggregation and curation module for news-based content profiles."""

from socials_automator.news.models import (
    NewsArticle,
    NewsStory,
    NewsBrief,
    NewsEdition,
)
from socials_automator.news.aggregator import NewsAggregator
from socials_automator.news.curator import NewsCurator

__all__ = [
    "NewsArticle",
    "NewsStory",
    "NewsBrief",
    "NewsEdition",
    "NewsAggregator",
    "NewsCurator",
]
