"""Research aggregator combining multiple sources."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .scraper import WebScraper, ScrapedContent
from .reddit import RedditResearcher, RedditPost, RedditTrend


@dataclass
class ResearchResult:
    """Aggregated research result from multiple sources."""

    topic: str
    relevance_score: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    summary: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_reddit_source(self, post: RedditPost) -> None:
        """Add a Reddit post as a source."""
        self.sources.append({
            "type": "reddit",
            "title": post.title,
            "url": post.full_url,
            "subreddit": post.subreddit,
            "score": post.score,
            "comments": post.num_comments,
            "content": post.selftext[:500] if post.selftext else None,
        })
        self.relevance_score += post.engagement_score / 100

    def add_web_source(self, content: ScrapedContent) -> None:
        """Add a web article as a source."""
        self.sources.append({
            "type": "web",
            "title": content.title,
            "url": content.url,
            "summary": content.summary,
            "keywords": content.keywords,
        })
        self.relevance_score += 5  # Base score for web articles


@dataclass
class TrendingTopic:
    """A trending topic identified from research."""

    topic: str
    keywords: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    source_count: int = 0
    sample_titles: list[str] = field(default_factory=list)
    suggested_angles: list[str] = field(default_factory=list)


class ResearchAggregator:
    """Aggregate research from multiple sources.

    Combines:
    - Reddit (trending posts, discussions)
    - Web articles (via RSS feeds and scraping)
    - News sources

    Usage:
        aggregator = ResearchAggregator(profile_config)
        await aggregator.initialize()

        # Get trending topics for content ideas
        topics = await aggregator.get_trending_topics()

        # Research a specific topic
        research = await aggregator.research_topic("ChatGPT productivity tips")

        await aggregator.close()
    """

    def __init__(self, profile_config: dict[str, Any] | None = None):
        """Initialize the research aggregator.

        Args:
            profile_config: Profile configuration with research sources.
        """
        self.profile_config = profile_config or {}
        self.research_sources = self.profile_config.get("research_sources", {})

        self._scraper = WebScraper()
        self._reddit = RedditResearcher()

    async def initialize(self) -> None:
        """Initialize all research sources."""
        # Reddit initialization happens on first use
        pass

    async def close(self) -> None:
        """Close all research sources."""
        await self._scraper.close()
        await self._reddit.close()

    async def get_trending_topics(
        self,
        subreddits: list[str] | None = None,
        rss_feeds: list[str] | None = None,
        limit: int = 10,
    ) -> list[TrendingTopic]:
        """Get trending topics from all sources.

        Args:
            subreddits: Subreddits to check. Defaults to config.
            rss_feeds: RSS feeds to check. Defaults to config.
            limit: Maximum topics to return.

        Returns:
            List of trending topics.
        """
        subreddits = subreddits or self.research_sources.get("subreddits", [])
        rss_feeds = rss_feeds or self.research_sources.get("rss_feeds", [])

        # Gather from all sources concurrently
        tasks = []

        if subreddits:
            tasks.append(self._get_reddit_trends(subreddits))

        if rss_feeds:
            tasks.append(self._get_rss_trends(rss_feeds))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all topics
        all_topics: dict[str, TrendingTopic] = {}

        for result in results:
            if isinstance(result, Exception):
                continue
            for topic in result:
                key = topic.topic.lower()[:50]
                if key not in all_topics:
                    all_topics[key] = topic
                else:
                    # Merge scores
                    all_topics[key].relevance_score += topic.relevance_score
                    all_topics[key].source_count += topic.source_count
                    all_topics[key].sample_titles.extend(topic.sample_titles[:2])

        # Sort by relevance
        sorted_topics = sorted(
            all_topics.values(),
            key=lambda t: t.relevance_score,
            reverse=True,
        )

        return sorted_topics[:limit]

    async def _get_reddit_trends(self, subreddits: list[str]) -> list[TrendingTopic]:
        """Get trending topics from Reddit."""
        try:
            trends = await self._reddit.get_trending_topics(subreddits)

            topics = []
            for trend in trends[:20]:
                topic = TrendingTopic(
                    topic=trend.topic,
                    relevance_score=trend.relevance_score / 100,
                    source_count=len(trend.posts),
                    sample_titles=[p.title for p in trend.posts[:3]],
                )
                topics.append(topic)

            return topics

        except Exception:
            return []

    async def _get_rss_trends(self, feeds: list[str]) -> list[TrendingTopic]:
        """Get trending topics from RSS feeds."""
        topics = []

        for feed_url in feeds:
            try:
                items = await self._scraper.fetch_rss(feed_url, limit=10)

                for item in items:
                    topic = TrendingTopic(
                        topic=item.get("title", ""),
                        relevance_score=3.0,  # Base score for RSS items
                        source_count=1,
                        sample_titles=[item.get("title", "")],
                    )
                    topics.append(topic)

            except Exception:
                continue

        return topics

    async def research_topic(
        self,
        topic: str,
        subreddits: list[str] | None = None,
        include_web: bool = True,
        max_sources: int = 10,
    ) -> ResearchResult:
        """Deep research on a specific topic.

        Args:
            topic: Topic to research.
            subreddits: Subreddits to search.
            include_web: Whether to include web search.
            max_sources: Maximum sources to include.

        Returns:
            ResearchResult with all found sources.
        """
        subreddits = subreddits or self.research_sources.get("subreddits", [])
        result = ResearchResult(topic=topic)

        # Search Reddit
        if subreddits:
            try:
                posts = await self._reddit.search(
                    query=topic,
                    subreddits=subreddits,
                    limit=max_sources,
                    time_filter="month",
                )

                for post in posts:
                    result.add_reddit_source(post)

            except Exception:
                pass

        # Extract common keywords from sources
        all_text = " ".join([
            s.get("title", "") + " " + (s.get("content", "") or "")
            for s in result.sources
        ])

        # Simple keyword extraction (in production, use NLP)
        words = all_text.lower().split()
        word_freq: dict[str, int] = {}
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        result.keywords = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:20]

        return result

    async def get_content_ideas(
        self,
        content_pillars: list[str],
        avoid_topics: list[str] | None = None,
        count: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate content ideas based on trending topics.

        Args:
            content_pillars: Content pillars to focus on.
            avoid_topics: Topics to avoid (recently used).
            count: Number of ideas to generate.

        Returns:
            List of content ideas with topic, angle, and sources.
        """
        avoid_topics = avoid_topics or []
        avoid_lower = [t.lower() for t in avoid_topics]

        # Get trending topics
        trends = await self.get_trending_topics(limit=30)

        # Filter out avoided topics
        filtered = [
            t for t in trends
            if not any(avoid in t.topic.lower() for avoid in avoid_lower)
        ]

        # Select top ideas
        ideas = []
        for trend in filtered[:count * 2]:  # Get extra for filtering
            if len(ideas) >= count:
                break

            idea = {
                "topic": trend.topic,
                "keywords": trend.keywords,
                "relevance_score": trend.relevance_score,
                "source_count": trend.source_count,
                "sample_titles": trend.sample_titles,
                "suggested_pillar": self._match_pillar(trend.topic, content_pillars),
            }
            ideas.append(idea)

        return ideas

    def _match_pillar(self, topic: str, pillars: list[str]) -> str | None:
        """Match a topic to a content pillar."""
        topic_lower = topic.lower()

        # Simple keyword matching
        pillar_keywords = {
            "tool_tutorials": ["how to", "tutorial", "guide", "use", "learn"],
            "productivity_hacks": ["tips", "tricks", "hacks", "save time", "productivity"],
            "tool_comparisons": ["vs", "comparison", "best", "alternative", "compare"],
            "ai_news_simplified": ["new", "update", "release", "announced", "launch"],
            "prompt_templates": ["prompt", "template", "example", "copy"],
        }

        for pillar in pillars:
            keywords = pillar_keywords.get(pillar, [pillar.replace("_", " ")])
            if any(kw in topic_lower for kw in keywords):
                return pillar

        return pillars[0] if pillars else None
