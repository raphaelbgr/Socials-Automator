"""Research module for web scraping, search, and trend detection."""

from .scraper import WebScraper
from .reddit import RedditResearcher
from .aggregator import ResearchAggregator
from .web_search import (
    WebSearcher,
    SearchResult,
    SearchResponse,
    ParallelSearchResponse,
    get_web_searcher,
    web_search,
    parallel_web_search,
)

__all__ = [
    "WebScraper",
    "RedditResearcher",
    "ResearchAggregator",
    # Web search
    "WebSearcher",
    "SearchResult",
    "SearchResponse",
    "ParallelSearchResponse",
    "get_web_searcher",
    "web_search",
    "parallel_web_search",
]
