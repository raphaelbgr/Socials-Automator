"""Base class for content history tracking.

Provides common functionality for:
- Session-based history storage (ai_sessions pattern)
- Posted content scanning
- Similarity checking
- Constraint generation for AI prompts
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from .similarity import is_similar, normalize_text
from .posted_scanner import PostedContentScanner, PostedContent

logger = logging.getLogger("history")


class BaseContentHistory(ABC):
    """Abstract base class for content history tracking.

    Provides unified storage using the ai_sessions pattern with:
    - Profile-scoped JSON file storage
    - Posted content scanning (source of truth)
    - Similarity checking for deduplication
    - Constraint generation for AI prompts

    Subclasses implement content-type-specific logic for:
    - Reels (topics, hooks)
    - News (stories, themes, headlines)
    - Posts (carousel topics)
    """

    # Subclasses should override these
    SESSION_TYPE: str = "content_history"
    HISTORY_FILE: str = "content_history.json"
    DEFAULT_LOOKBACK_DAYS: int = 30
    SIMILARITY_THRESHOLD: float = 0.5

    def __init__(
        self,
        profile_path: Path,
        lookback_days: Optional[int] = None,
    ):
        """Initialize content history for a profile.

        Args:
            profile_path: Path to the profile directory.
            lookback_days: How far back to look for history.
        """
        self.profile_path = Path(profile_path)
        self.profile_name = self.profile_path.name
        self.lookback_days = lookback_days or self.DEFAULT_LOOKBACK_DAYS

        # Initialize components
        self._scanner = PostedContentScanner(self.profile_path)
        self._history_dir = self.profile_path / "ai_sessions" / self.SESSION_TYPE
        self._history_dir.mkdir(parents=True, exist_ok=True)

        # Cache for performance
        self._cache_valid = False
        self._cached_session_items: list[dict] = []
        self._cached_posted_items: list[PostedContent] = []

        logger.debug(
            f"ContentHistory initialized: {self.profile_name}/{self.SESSION_TYPE}"
        )

    # =========================================================================
    # Abstract methods - subclasses must implement
    # =========================================================================

    @abstractmethod
    def get_content_text(self, item: dict | PostedContent) -> str:
        """Extract the primary text content from an item for similarity checking.

        Args:
            item: Session dict or PostedContent object.

        Returns:
            Primary text to use for similarity (topic, headline, etc.)
        """
        pass

    @abstractmethod
    def get_constraints(self) -> list[str]:
        """Generate constraints for AI prompts based on history.

        Returns:
            List of constraint strings for the AI.
        """
        pass

    # =========================================================================
    # Core public methods
    # =========================================================================

    def is_recent(
        self,
        content: str,
        threshold: Optional[float] = None,
    ) -> tuple[bool, str | None]:
        """Check if content is too similar to recent history.

        Checks both session history and posted content.

        Args:
            content: Content text to check.
            threshold: Similarity threshold (default from class).

        Returns:
            Tuple of (is_recent, matching_content or None).
        """
        threshold = threshold or self.SIMILARITY_THRESHOLD

        # Get all recent content texts
        recent_texts = self.get_recent_content_texts()

        if not recent_texts:
            return False, None

        return is_similar(content, recent_texts, threshold)

    def add(self, content: str, metadata: Optional[dict] = None) -> None:
        """Add content to session history.

        Args:
            content: Primary content text (topic, headline, etc.)
            metadata: Optional additional metadata.
        """
        entry = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Load existing history
        history = self._load_session_history()
        history.append(entry)

        # Keep only recent entries (last 200)
        history = history[-200:]

        # Save
        self._save_session_history(history)

        # Invalidate cache
        self._cache_valid = False

        logger.debug(f"Added to history: {content[:50]}...")

    def get_recent_content_texts(self) -> list[str]:
        """Get all recent content texts for similarity checking.

        Combines session history with posted content.

        Returns:
            List of content text strings.
        """
        self._ensure_cache()

        texts = []

        # From session history
        for item in self._cached_session_items:
            text = self.get_content_text(item)
            if text:
                texts.append(text)

        # From posted content (source of truth)
        for item in self._cached_posted_items:
            text = self.get_content_text(item)
            if text:
                texts.append(text)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for text in texts:
            normalized = text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(text)

        return unique

    def get_recent_items(self) -> list[dict | PostedContent]:
        """Get all recent items (both session and posted).

        Returns:
            List of history items.
        """
        self._ensure_cache()
        return self._cached_session_items + self._cached_posted_items

    def get_context_for_prompt(self, max_items: int = 50) -> str:
        """Get formatted context for AI prompts.

        Args:
            max_items: Maximum items to include.

        Returns:
            Formatted string for AI prompt.
        """
        texts = self.get_recent_content_texts()

        if not texts:
            return ""

        # Take most recent items (but show variety)
        texts_to_show = texts[-max_items:]

        items_text = "\n".join(f"- {t}" for t in texts_to_show)

        return f"""
=== CONTENT ALREADY CREATED (DO NOT REPEAT) ===
The following {len(texts_to_show)} items have already been covered.
You MUST generate something COMPLETELY DIFFERENT.
DO NOT use the same topics, angles, or phrasing.

{items_text}

=== REQUIRED: GENERATE UNIQUE CONTENT ===
- Use a DIFFERENT subject than what's listed above
- Use a DIFFERENT angle/approach
- Use DIFFERENT phrasing
- Cover something NEW we haven't done"""

    def invalidate_cache(self) -> None:
        """Invalidate the cache to force reload on next access."""
        self._cache_valid = False

    # =========================================================================
    # Session storage methods (ai_sessions pattern)
    # =========================================================================

    def _get_history_file_path(self) -> Path:
        """Get path to the history JSON file."""
        return self._history_dir / self.HISTORY_FILE

    def _load_session_history(self) -> list[dict]:
        """Load session history from file."""
        history_path = self._get_history_file_path()

        if not history_path.exists():
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            return []

    def _save_session_history(self, history: list[dict]) -> None:
        """Save session history to file."""
        history_path = self._get_history_file_path()

        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

    def _get_recent_session_items(self) -> list[dict]:
        """Get session items within lookback window."""
        all_history = self._load_session_history()
        cutoff = datetime.now() - timedelta(days=self.lookback_days)

        recent = []
        for item in all_history:
            try:
                timestamp = datetime.fromisoformat(item.get("timestamp", ""))
                if timestamp >= cutoff:
                    recent.append(item)
            except (ValueError, TypeError):
                # Include items without valid timestamp
                recent.append(item)

        return recent

    # =========================================================================
    # Cache management
    # =========================================================================

    def _ensure_cache(self) -> None:
        """Ensure cache is populated."""
        if self._cache_valid:
            return

        # Load session history
        self._cached_session_items = self._get_recent_session_items()

        # Scan posted content
        self._cached_posted_items = self._scanner.scan_reels(self.lookback_days)

        self._cache_valid = True

        logger.debug(
            f"Cache loaded: {len(self._cached_session_items)} session + "
            f"{len(self._cached_posted_items)} posted"
        )
