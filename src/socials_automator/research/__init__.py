"""Research module for web scraping and trend detection."""

from .scraper import WebScraper
from .reddit import RedditResearcher
from .aggregator import ResearchAggregator

__all__ = [
    "WebScraper",
    "RedditResearcher",
    "ResearchAggregator",
]
