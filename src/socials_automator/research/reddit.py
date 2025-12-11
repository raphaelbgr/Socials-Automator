"""Reddit research using AsyncPRAW."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import asyncpraw
from asyncpraw.models import Subreddit, Submission


@dataclass
class RedditPost:
    """A Reddit post with relevant data."""

    id: str
    title: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: float
    selftext: str | None = None
    author: str | None = None
    permalink: str | None = None
    is_self: bool = False

    @property
    def created_at(self) -> datetime:
        """Get creation datetime."""
        return datetime.fromtimestamp(self.created_utc)

    @property
    def full_url(self) -> str:
        """Get full Reddit URL."""
        return f"https://reddit.com{self.permalink}" if self.permalink else self.url

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score (score + comments weighted)."""
        return self.score + (self.num_comments * 2)


@dataclass
class RedditTrend:
    """A trending topic from Reddit."""

    topic: str
    posts: list[RedditPost] = field(default_factory=list)
    total_score: int = 0
    total_comments: int = 0
    subreddits: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def relevance_score(self) -> float:
        """Calculate overall relevance score."""
        return self.total_score + (self.total_comments * 2) + (len(self.posts) * 10)


class RedditResearcher:
    """Research trending topics on Reddit using AsyncPRAW.

    Usage:
        researcher = RedditResearcher()
        await researcher.initialize()

        # Get hot posts from subreddits
        posts = await researcher.get_hot_posts(
            subreddits=["ChatGPT", "productivity"],
            limit=20,
        )

        # Search for specific topics
        posts = await researcher.search(
            query="AI productivity tips",
            subreddits=["ChatGPT"],
        )

        # Get trending topics
        trends = await researcher.get_trending_topics(
            subreddits=["ChatGPT", "artificial"],
        )

        await researcher.close()
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
    ):
        """Initialize Reddit researcher.

        Args:
            client_id: Reddit API client ID. Defaults to env var.
            client_secret: Reddit API client secret. Defaults to env var.
            user_agent: User agent string. Defaults to env var.
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT",
            "SocialsAutomator/0.1.0 (by /u/socials-automator)"
        )

        self._reddit: asyncpraw.Reddit | None = None

    async def initialize(self) -> None:
        """Initialize the Reddit client."""
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Reddit API credentials required. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
            )

        self._reddit = asyncpraw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

    async def close(self) -> None:
        """Close the Reddit client."""
        if self._reddit:
            await self._reddit.close()
            self._reddit = None

    def _submission_to_post(self, submission: Submission) -> RedditPost:
        """Convert a PRAW submission to RedditPost."""
        return RedditPost(
            id=submission.id,
            title=submission.title,
            url=submission.url,
            subreddit=str(submission.subreddit),
            score=submission.score,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            selftext=submission.selftext if submission.is_self else None,
            author=str(submission.author) if submission.author else None,
            permalink=submission.permalink,
            is_self=submission.is_self,
        )

    async def get_hot_posts(
        self,
        subreddits: list[str],
        limit: int = 20,
        time_filter: str = "day",
    ) -> list[RedditPost]:
        """Get hot posts from subreddits.

        Args:
            subreddits: List of subreddit names.
            limit: Maximum posts per subreddit.
            time_filter: Time filter (hour, day, week, month, year, all).

        Returns:
            List of hot posts.
        """
        if not self._reddit:
            await self.initialize()

        posts = []

        for subreddit_name in subreddits:
            try:
                subreddit = await self._reddit.subreddit(subreddit_name)
                async for submission in subreddit.hot(limit=limit):
                    posts.append(self._submission_to_post(submission))
            except Exception as e:
                # Skip failed subreddits
                continue

        # Sort by engagement score
        posts.sort(key=lambda p: p.engagement_score, reverse=True)

        return posts

    async def get_top_posts(
        self,
        subreddits: list[str],
        limit: int = 20,
        time_filter: str = "week",
    ) -> list[RedditPost]:
        """Get top posts from subreddits.

        Args:
            subreddits: List of subreddit names.
            limit: Maximum posts per subreddit.
            time_filter: Time filter (hour, day, week, month, year, all).

        Returns:
            List of top posts.
        """
        if not self._reddit:
            await self.initialize()

        posts = []

        for subreddit_name in subreddits:
            try:
                subreddit = await self._reddit.subreddit(subreddit_name)
                async for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    posts.append(self._submission_to_post(submission))
            except Exception:
                continue

        posts.sort(key=lambda p: p.engagement_score, reverse=True)
        return posts

    async def search(
        self,
        query: str,
        subreddits: list[str] | None = None,
        limit: int = 20,
        time_filter: str = "week",
        sort: str = "relevance",
    ) -> list[RedditPost]:
        """Search Reddit for posts matching a query.

        Args:
            query: Search query.
            subreddits: Optional list of subreddits to search in.
            limit: Maximum results.
            time_filter: Time filter.
            sort: Sort order (relevance, hot, top, new, comments).

        Returns:
            List of matching posts.
        """
        if not self._reddit:
            await self.initialize()

        posts = []

        if subreddits:
            # Search within specific subreddits
            subreddit_str = "+".join(subreddits)
            subreddit = await self._reddit.subreddit(subreddit_str)
        else:
            # Search all of Reddit
            subreddit = await self._reddit.subreddit("all")

        try:
            async for submission in subreddit.search(
                query,
                sort=sort,
                time_filter=time_filter,
                limit=limit,
            ):
                posts.append(self._submission_to_post(submission))
        except Exception:
            pass

        return posts

    async def get_trending_topics(
        self,
        subreddits: list[str],
        limit_per_sub: int = 30,
        min_score: int = 50,
    ) -> list[RedditTrend]:
        """Analyze subreddits to find trending topics.

        Args:
            subreddits: Subreddits to analyze.
            limit_per_sub: Posts to fetch per subreddit.
            min_score: Minimum score to consider.

        Returns:
            List of trending topics with metadata.
        """
        # Get hot posts
        all_posts = await self.get_hot_posts(subreddits, limit=limit_per_sub)

        # Filter by score
        filtered = [p for p in all_posts if p.score >= min_score]

        # Group by common keywords/themes (simple approach)
        # In production, you'd use NLP or AI for better grouping
        trends: dict[str, RedditTrend] = {}

        for post in filtered:
            # Use the title as the topic (simplified)
            # A real implementation would extract keywords
            topic = post.title[:100]  # Truncate long titles

            if topic not in trends:
                trends[topic] = RedditTrend(topic=topic)

            trend = trends[topic]
            trend.posts.append(post)
            trend.total_score += post.score
            trend.total_comments += post.num_comments
            if post.subreddit not in trend.subreddits:
                trend.subreddits.append(post.subreddit)

        # Sort by relevance
        sorted_trends = sorted(
            trends.values(),
            key=lambda t: t.relevance_score,
            reverse=True,
        )

        return sorted_trends[:20]  # Top 20 trends

    async def get_post_content(self, post_id: str) -> RedditPost | None:
        """Get full content of a specific post.

        Args:
            post_id: Reddit post ID.

        Returns:
            RedditPost with full content, or None if not found.
        """
        if not self._reddit:
            await self.initialize()

        try:
            submission = await self._reddit.submission(id=post_id)
            return self._submission_to_post(submission)
        except Exception:
            return None
