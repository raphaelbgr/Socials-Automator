"""Content history for regular video reels.

Tracks:
- Topics generated and posted
- Hook patterns used (for variety)
- Duration feedback (for learning)

Replaces the old reel_topic_history.json with unified session-based storage.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from .base import BaseContentHistory
from .posted_scanner import PostedContent

logger = logging.getLogger("history")


# Hook type patterns for detection
HOOK_PATTERNS = {
    "question": [
        r"^\s*(do you|have you|ever wonder|did you know|what if|why do|how do|can you)",
        r"\?$",
    ],
    "number": [
        r"^\s*\d+\s+(ways|tips|tricks|secrets|reasons|things|steps|hacks|prompts|tools)",
    ],
    "statement": [
        r"^\s*(this is|here's|the secret|most people|stop doing|you need to|i just)",
    ],
    "story": [
        r"^\s*(i |my |when i|last week|yesterday|one day|imagine)",
    ],
}


class ReelContentHistory(BaseContentHistory):
    """Content history for regular video reels.

    Tracks topics and hooks to ensure unique, varied content.
    Uses both session history and posted content scanning.
    """

    SESSION_TYPE = "reel_topics"
    HISTORY_FILE = "topic_history.json"
    DEFAULT_LOOKBACK_DAYS = 30
    SIMILARITY_THRESHOLD = 0.35  # Lower = stricter (catches more similar topics)

    def __init__(
        self,
        profile_path: Path,
        lookback_days: Optional[int] = None,
    ):
        """Initialize reel content history.

        Args:
            profile_path: Path to profile directory.
            lookback_days: How far back to look (default 30).
        """
        super().__init__(profile_path, lookback_days)

        # Hook pattern tracking (loaded on demand)
        self._hook_counts: Optional[Counter] = None

    def get_content_text(self, item: dict | PostedContent) -> str:
        """Extract topic text from item.

        Args:
            item: Session dict or PostedContent.

        Returns:
            Topic string.
        """
        if isinstance(item, PostedContent):
            return item.topic or ""

        # Session dict
        return item.get("content") or item.get("topic") or ""

    def get_constraints(self) -> list[str]:
        """Generate constraints for AI prompts.

        Includes:
        - Topic variety constraints
        - Hook variety constraints

        Returns:
            List of constraint strings.
        """
        constraints = []

        # Topic constraints
        recent_topics = self.get_recent_content_texts()
        if recent_topics:
            # Show sample of recent topics to avoid
            sample = recent_topics[-5:]
            constraints.append(
                f"AVOID these recent topics: {', '.join(t[:30] + '...' if len(t) > 30 else t for t in sample)}"
            )

        # Hook variety constraints
        hook_counts = self._get_hook_counts()
        total_hooks = sum(hook_counts.values())

        if total_hooks > 0:
            # Find overused hook types (>40% of total)
            for hook_type, count in hook_counts.items():
                if count / total_hooks > 0.4:
                    constraints.append(
                        f"AVOID {hook_type} hooks (used {count}/{total_hooks} times recently)"
                    )

            # Suggest underused hook types
            for hook_type in HOOK_PATTERNS.keys():
                if hook_counts.get(hook_type, 0) == 0:
                    constraints.append(
                        f"Consider using {hook_type} hooks (not used recently)"
                    )

        return constraints

    # =========================================================================
    # Topic-specific methods
    # =========================================================================

    def add_topic(
        self,
        topic: str,
        hook_text: Optional[str] = None,
        hook_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a topic to history.

        Args:
            topic: Topic string.
            hook_text: Optional hook text for pattern tracking.
            hook_type: Optional hook type (question, number, statement, story).
            metadata: Optional additional metadata.
        """
        entry_metadata = metadata or {}

        if hook_text:
            entry_metadata["hook_text"] = hook_text

        if hook_type:
            entry_metadata["hook_type"] = hook_type
        elif hook_text:
            # Auto-detect hook type
            entry_metadata["hook_type"] = self._detect_hook_type(hook_text)

        self.add(topic, entry_metadata)

        # Invalidate hook counts cache
        self._hook_counts = None

    def is_topic_recent(
        self,
        topic: str,
        threshold: Optional[float] = None,
    ) -> tuple[bool, str | None]:
        """Check if a topic is too similar to recent topics.

        Args:
            topic: Topic to check.
            threshold: Similarity threshold.

        Returns:
            Tuple of (is_recent, matching_topic or None).
        """
        return self.is_recent(topic, threshold)

    def get_recent_topics(self) -> list[str]:
        """Get list of recent topic strings.

        Returns:
            List of topic strings.
        """
        return self.get_recent_content_texts()

    def get_recommended_hook_type(self) -> str:
        """Get recommended hook type based on recent usage.

        Returns least used hook type for variety.

        Returns:
            Hook type string.
        """
        hook_counts = self._get_hook_counts()
        all_types = set(HOOK_PATTERNS.keys())
        used_types = set(hook_counts.keys())
        unused = all_types - used_types

        if unused:
            return unused.pop()

        # Return least used
        if hook_counts:
            return min(hook_counts, key=hook_counts.get)

        return "question"  # Default

    # =========================================================================
    # Hook pattern tracking
    # =========================================================================

    def _get_hook_counts(self) -> Counter:
        """Get hook pattern usage counts.

        Returns:
            Counter of hook types.
        """
        if self._hook_counts is not None:
            return self._hook_counts

        counts = Counter()

        # From session history
        for item in self._get_recent_session_items():
            metadata = item.get("metadata", {})
            if "hook_type" in metadata:
                counts[metadata["hook_type"]] += 1
            elif "hook_text" in metadata:
                hook_type = self._detect_hook_type(metadata["hook_text"])
                counts[hook_type] += 1

        # From posted content
        self._ensure_cache()
        for item in self._cached_posted_items:
            if item.hook_type:
                counts[item.hook_type] += 1
            elif item.hook_text:
                hook_type = self._detect_hook_type(item.hook_text)
                counts[hook_type] += 1

        self._hook_counts = counts
        return counts

    def _detect_hook_type(self, hook_text: str) -> str:
        """Detect hook type from text using patterns.

        Args:
            hook_text: Hook text to analyze.

        Returns:
            Hook type string.
        """
        hook_lower = hook_text.lower().strip()

        for hook_type, patterns in HOOK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, hook_lower, re.IGNORECASE):
                    return hook_type

        return "statement"  # Default

    def record_feedback(
        self,
        topic: str,
        accepted: bool,
        actual_duration: Optional[float] = None,
        word_count: Optional[int] = None,
    ) -> None:
        """Record feedback for duration learning.

        Args:
            topic: Topic that was generated.
            accepted: Whether the duration was acceptable.
            actual_duration: Actual video duration in seconds.
            word_count: Actual word count.
        """
        # Find and update the matching entry
        history = self._load_session_history()

        for entry in reversed(history):
            if entry.get("content") == topic:
                entry.setdefault("feedback", {})
                entry["feedback"]["accepted"] = accepted
                if actual_duration:
                    entry["feedback"]["actual_duration"] = actual_duration
                if word_count:
                    entry["feedback"]["word_count"] = word_count
                break

        self._save_session_history(history)
