"""Dynamic query generation using AI to avoid topic repetition.

This module generates fresh search queries based on what topics have
already been covered, ensuring diverse news content across reels.

Usage:
    from socials_automator.news.dynamic_queries import DynamicQueryGenerator

    generator = DynamicQueryGenerator(profile_name="news.but.quick")
    queries = await generator.generate_queries(count=10)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from socials_automator.providers.text import TextProvider

logger = logging.getLogger("ai_calls")


# =============================================================================
# Topic History
# =============================================================================

def get_topic_history_path(profile_name: str) -> Path:
    """Get path to topic history file for a profile."""
    from socials_automator.constants import get_profiles_dir
    return get_profiles_dir() / profile_name / "topic_history.json"


def get_query_history_path(profile_name: str) -> Path:
    """Get path to query history file for a profile."""
    from socials_automator.constants import get_profiles_dir
    return get_profiles_dir() / profile_name / "query_history.json"


def load_topic_history(profile_name: str, hours: int = 48) -> list[dict]:
    """Load recently covered topics.

    Args:
        profile_name: Profile to load history for.
        hours: How far back to look (default 48h).

    Returns:
        List of topic entries with entities and categories.
    """
    history_path = get_topic_history_path(profile_name)
    if not history_path.exists():
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_history = json.load(f)

        # Filter to recent topics
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []

        for entry in all_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time > cutoff:
                recent.append(entry)

        return recent
    except Exception as e:
        logger.warning(f"Could not load topic history: {e}")
        return []


def save_topics_to_history(profile_name: str, topics: list[dict]) -> None:
    """Save covered topics to history.

    Args:
        profile_name: Profile to save history for.
        topics: List of topic dicts with 'entities', 'categories', 'headlines'.
    """
    history_path = get_topic_history_path(profile_name)

    try:
        # Load existing history
        all_history = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                all_history = json.load(f)

        # Add new entry
        timestamp = datetime.now().isoformat()
        all_history.append({
            "timestamp": timestamp,
            "topics": topics,
        })

        # Keep only last 100 entries
        all_history = all_history[-100:]

        # Save
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(all_history, f, indent=2)

        logger.info(f"TOPIC_HISTORY | saved {len(topics)} topics")

    except Exception as e:
        logger.warning(f"Could not save topic history: {e}")


def load_query_history(profile_name: str, hours: int = 72) -> list[str]:
    """Load recently used search queries.

    Args:
        profile_name: Profile to load history for.
        hours: How far back to look (default 72h for queries).

    Returns:
        List of query strings that have been used recently.
    """
    history_path = get_query_history_path(profile_name)
    if not history_path.exists():
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_history = json.load(f)

        # Filter to recent queries
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_queries = []

        for entry in all_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time > cutoff:
                recent_queries.extend(entry.get("queries", []))

        return list(set(recent_queries))  # Dedupe
    except Exception as e:
        logger.warning(f"Could not load query history: {e}")
        return []


def save_queries_to_history(profile_name: str, queries: list[str]) -> None:
    """Save used search queries to history.

    This prevents the same queries from being regenerated, even if
    the articles they found weren't selected for the final video.

    Args:
        profile_name: Profile to save history for.
        queries: List of query strings that were used.
    """
    history_path = get_query_history_path(profile_name)

    try:
        # Load existing history
        all_history = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                all_history = json.load(f)

        # Add new entry
        timestamp = datetime.now().isoformat()
        all_history.append({
            "timestamp": timestamp,
            "queries": queries,
        })

        # Keep only last 50 entries
        all_history = all_history[-50:]

        # Save
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(all_history, f, indent=2)

        logger.info(f"QUERY_HISTORY | saved {len(queries)} queries")

    except Exception as e:
        logger.warning(f"Could not save query history: {e}")


def extract_topics_from_stories(stories: list) -> list[dict]:
    """Extract topic information from stories for history tracking.

    Args:
        stories: List of NewsStory objects.

    Returns:
        List of topic dicts.
    """
    topics = []
    for story in stories:
        headline = story.headline if hasattr(story, 'headline') else str(story)
        category = story.category.value if hasattr(story, 'category') else 'general'

        # Extract likely entities (capitalized words, quoted terms)
        entities = []
        # Find capitalized words (2+ chars, not at sentence start)
        words = headline.split()
        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) >= 2 and clean[0].isupper() and i > 0:
                entities.append(clean)

        # Find quoted terms
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", headline)
        entities.extend(quoted)

        topics.append({
            "headline": headline,
            "category": category,
            "entities": list(set(entities))[:5],  # Top 5 unique entities
        })

    return topics


# =============================================================================
# AI Query Generator
# =============================================================================

QUERY_GENERATION_SYSTEM = """You are a news search query optimizer for an entertainment news account.

Your job is to generate diverse, effective search queries that will find FRESH news stories.

CRITICAL RULES:
1. AVOID topics that have been recently covered (provided in context)
2. Generate queries that target DIFFERENT artists, shows, movies, events
3. Mix categories: celebrity, music, movies, TV, streaming, viral
4. Include trending/timely terms (2025, this week, new, announces, etc.)
5. Each query should be 3-6 words, specific enough to get relevant results
6. NO duplicate queries or minor variations of the same query

GOOD QUERY EXAMPLES:
- "Taylor Swift tour announcement 2025"
- "Marvel movie release this week"
- "Grammy nominations 2025 surprises"
- "Netflix new series premiere December"
- "viral TikTok celebrity moment"

BAD QUERY EXAMPLES (too generic or repetitive):
- "entertainment news" (too generic)
- "celebrity news today" (too generic)
- "BTS news" (if BTS was recently covered)
- "music news" (too generic)"""

QUERY_GENERATION_PROMPT = """Generate {count} fresh search queries for entertainment news.

RECENTLY COVERED TOPICS (AVOID THESE):
{covered_topics}

RECENTLY USED QUERIES (DO NOT REGENERATE OR USE SIMILAR):
{recent_queries}

UNDERREPRESENTED CATEGORIES (prioritize these):
{gap_categories}

CURRENT DATE: {current_date}

IMPORTANT:
- Do NOT generate queries similar to the "RECENTLY USED QUERIES" list above
- Vary the artists, shows, and topics - don't keep searching for the same celebrities
- Each query should target a DIFFERENT subject than previous queries

Return a JSON array of query objects:
[
  {{"query": "search query here", "category": "music|movies|tv|celebrity|streaming|viral", "reason": "why this query"}},
  ...
]

Generate {count} diverse queries. Return ONLY valid JSON, no markdown."""


@dataclass
class GeneratedQuery:
    """A dynamically generated search query."""
    query: str
    category: str
    reason: str


class DynamicQueryGenerator:
    """Generates search queries using AI based on coverage gaps.

    Usage:
        generator = DynamicQueryGenerator(profile_name="news.but.quick")
        queries = await generator.generate_queries(count=10)

        # Get query strings for search
        query_strings = [q.query for q in queries]
    """

    # Category definitions for gap analysis
    CATEGORIES = ["celebrity", "music", "movies", "tv", "streaming", "viral"]

    def __init__(
        self,
        profile_name: str,
        text_provider: Optional[TextProvider] = None,
        history_hours: int = 24,
        query_history_hours: int = 72,
    ):
        """Initialize the generator.

        Args:
            profile_name: Profile name for topic history.
            text_provider: AI provider for query generation.
            history_hours: How far back to check for covered topics.
            query_history_hours: How far back to check for used queries (longer).
        """
        self.profile_name = profile_name
        self.history_hours = history_hours
        self.query_history_hours = query_history_hours
        self._text_provider = text_provider

    @property
    def text_provider(self) -> TextProvider:
        """Lazy-load the text provider."""
        if self._text_provider is None:
            self._text_provider = TextProvider()
        return self._text_provider

    def _get_covered_topics_text(self) -> str:
        """Get formatted text of recently covered topics."""
        history = load_topic_history(self.profile_name, self.history_hours)

        if not history:
            return "No recent topics covered yet."

        # Extract all entities and headlines
        all_entities = set()
        all_headlines = []
        category_counts = {cat: 0 for cat in self.CATEGORIES}

        for entry in history:
            for topic in entry.get("topics", []):
                headline = topic.get("headline", "")
                if headline:
                    all_headlines.append(headline)
                entities = topic.get("entities", [])
                all_entities.update(entities)
                cat = topic.get("category", "general")
                if cat in category_counts:
                    category_counts[cat] += 1

        # Format for prompt
        lines = []
        if all_entities:
            lines.append(f"Entities covered: {', '.join(sorted(all_entities)[:30])}")
        if all_headlines:
            lines.append(f"\nRecent headlines ({len(all_headlines)} total):")
            for h in all_headlines[-15:]:  # Last 15 headlines
                lines.append(f"  - {h[:80]}")

        return "\n".join(lines) if lines else "No specific topics tracked yet."

    def _get_gap_categories(self) -> str:
        """Identify underrepresented categories."""
        history = load_topic_history(self.profile_name, self.history_hours)

        # Count categories
        category_counts = {cat: 0 for cat in self.CATEGORIES}
        for entry in history:
            for topic in entry.get("topics", []):
                cat = topic.get("category", "general")
                if cat in category_counts:
                    category_counts[cat] += 1

        # Find gaps (categories with fewer mentions)
        if not any(category_counts.values()):
            return "All categories need coverage (no history yet)"

        total = sum(category_counts.values())
        if total == 0:
            return "All categories need coverage"

        # Sort by count (ascending) to find gaps
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1])
        gaps = [f"{cat} ({count}/{total})" for cat, count in sorted_cats[:3]]

        return f"Underrepresented: {', '.join(gaps)}"

    def _get_recent_queries_text(self) -> str:
        """Get formatted text of recently used search queries."""
        recent_queries = load_query_history(self.profile_name, self.query_history_hours)

        if not recent_queries:
            return "No recent queries (this is the first run)."

        # Show the queries (limit to 30 for prompt size)
        lines = ["The following queries were recently used - DO NOT use these or similar:"]
        for q in recent_queries[:30]:
            lines.append(f"  - {q}")

        return "\n".join(lines)

    async def generate_queries(self, count: int = 10) -> list[GeneratedQuery]:
        """Generate fresh search queries using AI.

        Args:
            count: Number of queries to generate.

        Returns:
            List of GeneratedQuery objects.
        """
        covered = self._get_covered_topics_text()
        gaps = self._get_gap_categories()
        recent_queries = self._get_recent_queries_text()

        prompt = QUERY_GENERATION_PROMPT.format(
            count=count,
            covered_topics=covered,
            recent_queries=recent_queries,
            gap_categories=gaps,
            current_date=datetime.now().strftime("%B %d, %Y"),
        )

        try:
            response = await self.text_provider.generate(
                prompt=prompt,
                system=QUERY_GENERATION_SYSTEM,
                task="dynamic_query_generation",
                temperature=0.8,  # Higher creativity
                max_tokens=1500,
            )

            queries = self._parse_response(response)

            # Save the generated queries to history (prevents regeneration)
            if queries:
                query_strings = [q.query for q in queries]
                save_queries_to_history(self.profile_name, query_strings)

            logger.info(f"DYNAMIC_QUERIES | generated {len(queries)} queries")
            return queries

        except Exception as e:
            logger.error(f"DYNAMIC_QUERIES | generation failed: {e}")
            return self._get_fallback_queries(count)

    def _parse_response(self, response: str) -> list[GeneratedQuery]:
        """Parse AI response into GeneratedQuery objects."""
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"DYNAMIC_QUERIES | JSON parse error: {e}")
            return []

        queries = []
        for item in data:
            if isinstance(item, dict) and "query" in item:
                queries.append(GeneratedQuery(
                    query=item.get("query", ""),
                    category=item.get("category", "general"),
                    reason=item.get("reason", ""),
                ))

        return queries

    def _get_fallback_queries(self, count: int) -> list[GeneratedQuery]:
        """Get fallback queries if AI generation fails."""
        # Generic but useful fallback queries
        fallbacks = [
            GeneratedQuery("breaking entertainment news today", "general", "fallback"),
            GeneratedQuery("new movie trailer release", "movies", "fallback"),
            GeneratedQuery("music album announcement 2025", "music", "fallback"),
            GeneratedQuery("tv series premiere this week", "tv", "fallback"),
            GeneratedQuery("streaming platform new content", "streaming", "fallback"),
            GeneratedQuery("celebrity viral moment today", "viral", "fallback"),
            GeneratedQuery("award show nominations 2025", "general", "fallback"),
            GeneratedQuery("concert tour announcement", "music", "fallback"),
            GeneratedQuery("box office weekend results", "movies", "fallback"),
            GeneratedQuery("trending celebrity news", "celebrity", "fallback"),
        ]
        return fallbacks[:count]


# =============================================================================
# Module-level convenience
# =============================================================================

async def generate_dynamic_queries(
    profile_name: str,
    count: int = 10,
    text_provider: Optional[TextProvider] = None,
) -> list[str]:
    """Convenience function to generate query strings.

    Args:
        profile_name: Profile name for topic history.
        count: Number of queries to generate.
        text_provider: Optional AI provider.

    Returns:
        List of query strings ready for search.
    """
    generator = DynamicQueryGenerator(
        profile_name=profile_name,
        text_provider=text_provider,
    )
    queries = await generator.generate_queries(count=count)
    return [q.query for q in queries]
