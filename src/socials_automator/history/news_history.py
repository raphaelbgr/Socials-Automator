"""Content history for news video reels.

Tracks:
- News stories/headlines used
- Edition themes for variety
- Dynamic query topics

Replaces the old story_history.json, theme_history.json, topic_history.json
with unified session-based storage.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from .base import BaseContentHistory
from .posted_scanner import PostedContent
from .similarity import is_similar, normalize_text

logger = logging.getLogger("history")


class NewsContentHistory(BaseContentHistory):
    """Content history for news video reels.

    Tracks stories, headlines, and themes to ensure unique content.
    Provides theme rotation for edition variety.
    """

    SESSION_TYPE = "news_stories"
    HISTORY_FILE = "story_history.json"
    DEFAULT_LOOKBACK_DAYS = 7  # News is more time-sensitive
    SIMILARITY_THRESHOLD = 0.35  # Lower threshold to catch similar headlines
    STORY_COOLDOWN_HOURS = 48  # Stories stay in cooldown for 48h

    # Key entities that should only appear once per cooldown period
    # These are extracted from headlines for entity-level deduplication
    ENTITY_COOLDOWN_HOURS = 24

    def __init__(
        self,
        profile_path: Path,
        lookback_days: Optional[int] = None,
    ):
        """Initialize news content history.

        Args:
            profile_path: Path to profile directory.
            lookback_days: How far back to look (default 7).
        """
        super().__init__(profile_path, lookback_days)

        # Theme tracking
        self._theme_history: Optional[list[dict]] = None

    def get_content_text(self, item: dict | PostedContent) -> str:
        """Extract headline/story text from item.

        Args:
            item: Session dict or PostedContent.

        Returns:
            Headline or story text.
        """
        if isinstance(item, PostedContent):
            # For news, headlines are more important than topic
            if item.headlines:
                return " | ".join(item.headlines)
            return item.topic or ""

        # Session dict
        return (
            item.get("content")
            or item.get("headline")
            or item.get("headline_normalized")
            or ""
        )

    def get_constraints(self) -> list[str]:
        """Generate constraints for news curation AI.

        Returns:
            List of constraint strings.
        """
        constraints = []

        # Recent stories to avoid
        recent = self.get_recent_headlines()
        if recent:
            sample = recent[-5:]
            constraints.append(
                f"AVOID these recent stories: {', '.join(h[:40] + '...' if len(h) > 40 else h for h in sample)}"
            )

        # Theme variety
        recent_themes = self._get_recent_themes()
        if recent_themes:
            constraints.append(
                f"Recent themes used: {', '.join(recent_themes[-3:])} - try something different"
            )

        return constraints

    # =========================================================================
    # Story/headline tracking
    # =========================================================================

    def add_story(
        self,
        headline: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a news story to history.

        Args:
            headline: Story headline.
            metadata: Optional metadata (source, category, etc.)
        """
        entry_metadata = metadata or {}
        entry_metadata["headline_normalized"] = self._normalize_headline(headline)

        self.add(headline, entry_metadata)

    def add_stories(self, stories: list[Any]) -> None:
        """Add multiple stories to history.

        Args:
            stories: List of story objects (with headline attribute) or strings.
        """
        for story in stories:
            if hasattr(story, "headline"):
                headline = story.headline
                metadata = {
                    "source": getattr(story, "source", None),
                    "category": getattr(story, "category", None),
                }
            else:
                headline = str(story)
                metadata = {}

            self.add_story(headline, metadata)

    def is_story_recent(
        self,
        headline: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if a story headline is too similar to recent stories.

        Uses multiple detection methods:
        1. Jaccard similarity checking (catches similar wording)
        2. Exact match on normalized text
        3. Key entity overlap (catches same subject with different wording)

        Args:
            headline: Headline to check.
            threshold: Similarity threshold.

        Returns:
            True if story should be filtered out.
        """
        # Check similarity
        is_similar_result, _ = self.is_recent(headline, threshold)
        if is_similar_result:
            return True

        # Also check normalized exact match
        normalized = self._normalize_headline(headline)
        recent_normalized = self._get_recent_normalized_headlines()

        if normalized in recent_normalized:
            return True

        # Check key entity overlap (catches same subject with different headlines)
        if self._has_recent_entity_overlap(headline):
            return True

        return False

    def _has_recent_entity_overlap(self, headline: str) -> bool:
        """Check if headline shares key entities with recent stories.

        Extracts key entities (proper nouns, show names, product names) and
        checks if any appear in recent headlines.

        Args:
            headline: Headline to check.

        Returns:
            True if headline shares key entity with recent story.
        """
        # Extract key entities from new headline
        new_entities = self._extract_key_entities(headline)
        if not new_entities:
            return False

        # Get entities from recent headlines
        recent_entities = self._get_recent_entities()

        # Check for overlap
        overlap = new_entities & recent_entities
        if overlap:
            logger.debug(f"ENTITY_OVERLAP | {headline[:40]}... shares: {overlap}")
            return True

        return False

    def _extract_key_entities(self, headline: str) -> set[str]:
        """Extract key entities from a headline.

        Looks for:
        - Multi-word proper nouns (Stranger Things, Ranveer Singh)
        - Known entity patterns (Season X, Part X)
        - Capitalized phrases

        Args:
            headline: Headline text.

        Returns:
            Set of normalized entity strings.
        """
        entities = set()

        # Clean headline
        clean = headline.strip()

        # Extract capitalized multi-word phrases (2-3 words)
        # This catches "Stranger Things", "Ranveer Singh", "Black Mirror" etc.
        words = clean.split()
        for i in range(len(words) - 1):
            # 2-word entities
            if (words[i][0:1].isupper() and words[i+1][0:1].isupper()
                and len(words[i]) > 2 and len(words[i+1]) > 2):
                entity = f"{words[i].lower()} {words[i+1].lower()}"
                # Filter out common non-entity phrases
                if not self._is_common_phrase(entity):
                    entities.add(entity)

            # 3-word entities
            if i + 2 < len(words):
                if (words[i][0:1].isupper() and words[i+1][0:1].isupper()
                    and words[i+2][0:1].isupper()
                    and len(words[i]) > 2):
                    entity = f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
                    if not self._is_common_phrase(entity):
                        entities.add(entity)

        return entities

    def _is_common_phrase(self, phrase: str) -> bool:
        """Check if a phrase is a common non-entity expression."""
        common = {
            "box office", "breaking news", "season premiere",
            "series finale", "release date", "coming soon",
            "first look", "official trailer", "new episode",
            "latest update", "exclusive interview", "behind scenes",
        }
        return phrase in common

    def _get_recent_entities(self) -> set[str]:
        """Get entities from recent headlines.

        Returns:
            Set of entity strings from recent history.
        """
        self._ensure_cache()

        entities = set()

        # From session history
        for item in self._cached_session_items:
            content = item.get("content", "")
            if content:
                entities.update(self._extract_key_entities(content))

        # From posted content
        for item in self._cached_posted_items:
            for headline in item.headlines:
                entities.update(self._extract_key_entities(headline))

        return entities

    def get_recent_headlines(self) -> list[str]:
        """Get list of recent headline strings.

        Returns:
            List of headline strings.
        """
        return self.get_recent_content_texts()

    def get_stories_to_filter(self) -> set[str]:
        """Get set of normalized headlines to filter out.

        For efficient filtering during news curation.

        Returns:
            Set of normalized headline strings.
        """
        return self._get_recent_normalized_headlines()

    def _normalize_headline(self, headline: str) -> str:
        """Normalize headline for exact matching.

        Args:
            headline: Raw headline text.

        Returns:
            Normalized headline string.
        """
        # Lowercase, remove punctuation, collapse whitespace
        normalized = headline.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _get_recent_normalized_headlines(self) -> set[str]:
        """Get set of normalized headlines from recent history.

        Returns:
            Set of normalized headline strings.
        """
        self._ensure_cache()

        normalized = set()

        # From session history
        for item in self._cached_session_items:
            if "headline_normalized" in item.get("metadata", {}):
                normalized.add(item["metadata"]["headline_normalized"])
            elif item.get("content"):
                normalized.add(self._normalize_headline(item["content"]))

        # From posted content
        for item in self._cached_posted_items:
            for headline in item.headlines:
                normalized.add(self._normalize_headline(headline))

        return normalized

    # =========================================================================
    # Theme rotation
    # =========================================================================

    def add_theme(self, theme: str, edition: Optional[str] = None) -> None:
        """Record a theme that was used.

        Args:
            theme: Theme string (e.g., "tech", "economy", "entertainment").
            edition: Optional edition (morning, midday, evening, night).
        """
        themes = self._load_theme_history()

        themes.append({
            "theme": theme,
            "edition": edition,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep last 50 themes
        themes = themes[-50:]
        self._save_theme_history(themes)
        self._theme_history = themes

    def get_recommended_theme(
        self,
        available_themes: list[str],
        edition: Optional[str] = None,
    ) -> str:
        """Get recommended theme based on rotation.

        Recommends least-recently-used theme for variety.

        Args:
            available_themes: List of available theme options.
            edition: Optional edition for time-based filtering.

        Returns:
            Recommended theme string.
        """
        if not available_themes:
            return "general"

        recent = self._get_recent_themes(limit=10)

        # Find themes not used recently
        for theme in available_themes:
            if theme not in recent:
                return theme

        # All used recently, return least recent
        for theme in reversed(available_themes):
            if theme in recent:
                # This was used earliest among recent
                return available_themes[0]

        return available_themes[0]

    def is_theme_recent(self, theme: str, lookback: int = 3) -> bool:
        """Check if a theme was used recently.

        Args:
            theme: Theme to check.
            lookback: Number of recent themes to check.

        Returns:
            True if theme was used recently.
        """
        recent = self._get_recent_themes(limit=lookback)
        return theme.lower() in [t.lower() for t in recent]

    def _get_recent_themes(self, limit: int = 10) -> list[str]:
        """Get list of recently used themes.

        Args:
            limit: Maximum themes to return.

        Returns:
            List of theme strings.
        """
        themes = self._load_theme_history()
        return [t["theme"] for t in themes[-limit:]]

    def _load_theme_history(self) -> list[dict]:
        """Load theme history from file."""
        if self._theme_history is not None:
            return self._theme_history

        theme_path = self._history_dir / "theme_history.json"

        if not theme_path.exists():
            # Try to migrate from old location
            old_path = self.profile_path / "theme_history.json"
            if old_path.exists():
                try:
                    with open(old_path, "r", encoding="utf-8") as f:
                        self._theme_history = json.load(f)
                    # Save to new location
                    self._save_theme_history(self._theme_history)
                    logger.info(f"Migrated theme history from {old_path}")
                    return self._theme_history
                except Exception as e:
                    logger.warning(f"Could not migrate theme history: {e}")

            self._theme_history = []
            return self._theme_history

        try:
            with open(theme_path, "r", encoding="utf-8") as f:
                self._theme_history = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load theme history: {e}")
            self._theme_history = []

        return self._theme_history

    def _save_theme_history(self, themes: list[dict]) -> None:
        """Save theme history to file."""
        theme_path = self._history_dir / "theme_history.json"

        try:
            with open(theme_path, "w", encoding="utf-8") as f:
                json.dump(themes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save theme history: {e}")

    # =========================================================================
    # Dynamic query topic tracking
    # =========================================================================

    def add_query_topic(self, topic: str) -> None:
        """Record a topic used for dynamic search queries.

        Args:
            topic: Topic string used for query generation.
        """
        # Store in a separate file for query tracking
        query_path = self._history_dir / "query_topics.json"

        try:
            topics = []
            if query_path.exists():
                with open(query_path, "r", encoding="utf-8") as f:
                    topics = json.load(f)

            topics.append({
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
            })

            # Keep last 100
            topics = topics[-100:]

            with open(query_path, "w", encoding="utf-8") as f:
                json.dump(topics, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Could not save query topic: {e}")

    def get_recent_query_topics(self, days: int = 7) -> list[str]:
        """Get recently used query topics.

        Args:
            days: How far back to look.

        Returns:
            List of topic strings.
        """
        query_path = self._history_dir / "query_topics.json"

        if not query_path.exists():
            return []

        try:
            with open(query_path, "r", encoding="utf-8") as f:
                topics = json.load(f)

            cutoff = datetime.now() - timedelta(days=days)
            recent = []

            for item in topics:
                try:
                    timestamp = datetime.fromisoformat(item.get("timestamp", ""))
                    if timestamp >= cutoff:
                        recent.append(item["topic"])
                except (ValueError, TypeError):
                    recent.append(item["topic"])

            return recent

        except Exception as e:
            logger.warning(f"Could not load query topics: {e}")
            return []
