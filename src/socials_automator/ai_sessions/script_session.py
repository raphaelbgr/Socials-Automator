"""Script generation session with profile-scoped context.

Maintains conversation context for video script generation, enabling:
- Unique hooks (avoids recently used hook patterns)
- Varied segment structures
- Topic deduplication
- Duration learning from feedback
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from .profile_session import ProfileAISession, AIEventCallback

_logger = logging.getLogger("ai_sessions")


class VideoSegment(BaseModel):
    """A segment of the video script."""
    narration: str = Field(..., description="Narration text for this segment")
    keywords: list[str] = Field(default_factory=list, description="Visual keywords for stock footage")
    duration_hint: float = Field(default=10.0, description="Suggested duration in seconds")


class VideoScript(BaseModel):
    """Complete video script with hook, segments, and CTA."""
    hook: str = Field(..., description="Opening hook (3 seconds, attention-grabbing)")
    hook_type: str = Field(default="statement", description="Hook style: question, number, statement, story")
    segments: list[VideoSegment] = Field(..., description="Main content segments")
    cta: str = Field(..., description="Call to action (4 seconds)")
    total_words: int = Field(default=0, description="Total word count")
    estimated_duration: float = Field(default=60.0, description="Estimated duration in seconds")


class ScriptSession(ProfileAISession):
    """Session for video script generation with uniqueness constraints.

    Tracks:
    - Hook patterns used (question, number, statement, story)
    - Segment structures (tips-list, comparison, tutorial, story)
    - Topics covered
    - Duration accuracy from feedback

    Generates constraints to ensure unique, varied content.
    """

    SESSION_TYPE = "script_generation"

    # Hook type patterns for detection
    HOOK_PATTERNS = {
        "question": [r"^\s*(do you|have you|ever wonder|did you know|what if|why do|how do|can you)", r"\?$"],
        "number": [r"^\s*\d+\s+(ways|tips|tricks|secrets|reasons|things|steps|hacks)"],
        "statement": [r"^\s*(this is|here's|the secret|most people|stop doing|you need to)"],
        "story": [r"^\s*(i |my |when i|last week|yesterday|one day|imagine)"],
    }

    def __init__(
        self,
        profile_path: Path,
        target_duration: float = 60.0,
        provider_override: str | None = None,
        event_callback: AIEventCallback = None,
        history_days: int = 14,
    ):
        """Initialize script generation session.

        Args:
            profile_path: Path to the profile directory.
            target_duration: Target video duration in seconds.
            provider_override: Override LLM provider.
            event_callback: Optional callback for AI events.
            history_days: Days of history to analyze for constraints.
        """
        super().__init__(
            profile_path=profile_path,
            provider_override=provider_override,
            event_callback=event_callback,
            history_days=history_days,
        )
        self.target_duration = target_duration

        # Analyze history for patterns
        self._hook_counts = self._analyze_hook_patterns()
        self._recent_topics = self._get_recent_topics()
        self._segment_patterns = self._analyze_segment_patterns()

        _logger.info(
            f"ScriptSession initialized: "
            f"hooks={self._hook_counts}, "
            f"topics={len(self._recent_topics)}"
        )

    def _analyze_hook_patterns(self) -> dict[str, int]:
        """Analyze recent hooks to count pattern usage."""
        counts = Counter()

        sessions = self._storage.get_recent_sessions(
            self.SESSION_TYPE, days=self.history_days, limit=30
        )

        for session in sessions:
            metadata = session.get("metadata", {})
            if "hook_type" in metadata:
                counts[metadata["hook_type"]] += 1
            elif "hook_text" in metadata:
                # Detect hook type from text
                hook_type = self._detect_hook_type(metadata["hook_text"])
                counts[hook_type] += 1

        return dict(counts)

    def _detect_hook_type(self, hook_text: str) -> str:
        """Detect hook type from text using patterns."""
        hook_lower = hook_text.lower().strip()

        for hook_type, patterns in self.HOOK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, hook_lower, re.IGNORECASE):
                    return hook_type

        return "statement"  # Default

    def _get_recent_topics(self) -> list[str]:
        """Get list of recently covered topics."""
        topics = []

        sessions = self._storage.get_recent_sessions(
            self.SESSION_TYPE, days=self.history_days, limit=30
        )

        for session in sessions:
            metadata = session.get("metadata", {})
            if "topic" in metadata:
                topics.append(metadata["topic"].lower())

        return topics

    def _analyze_segment_patterns(self) -> dict[str, int]:
        """Analyze segment structure patterns."""
        counts = Counter()

        sessions = self._storage.get_recent_sessions(
            self.SESSION_TYPE, days=self.history_days, limit=30
        )

        for session in sessions:
            metadata = session.get("metadata", {})
            if "segment_structure" in metadata:
                counts[metadata["segment_structure"]] += 1

        return dict(counts)

    def get_constraints(self) -> list[str]:
        """Generate constraints from session history.

        Returns:
            List of constraint strings for the AI prompt.
        """
        constraints = []

        # Hook variety constraints
        total_hooks = sum(self._hook_counts.values())
        if total_hooks > 0:
            # Find overused hook types (>40% of total)
            for hook_type, count in self._hook_counts.items():
                if count / total_hooks > 0.4:
                    constraints.append(
                        f"AVOID {hook_type} hooks (used {count}/{total_hooks} times recently)"
                    )

            # Suggest underused hook types
            for hook_type in self.HOOK_PATTERNS.keys():
                if self._hook_counts.get(hook_type, 0) == 0:
                    constraints.append(
                        f"Consider using {hook_type} hooks (not used recently)"
                    )

        # Topic deduplication
        if self._recent_topics:
            recent_5 = self._recent_topics[:5]
            constraints.append(
                f"AVOID these recently covered topics: {', '.join(recent_5)}"
            )

        # Segment structure variety
        if self._segment_patterns:
            most_common = max(self._segment_patterns, key=self._segment_patterns.get)
            if self._segment_patterns[most_common] > 3:
                constraints.append(
                    f"VARY segment structure ('{most_common}' used too often)"
                )

        # Duration accuracy
        if self._history_summary.get("rejected_count", 0) > 2:
            constraints.append(
                "PAY ATTENTION to word count - recent scripts had duration issues"
            )

        return constraints

    def get_system_prompt(self) -> str:
        """Get system prompt for script generation."""
        return f"""You are a video script writer for short-form social media content.

Your task is to create engaging {self.target_duration}-second video scripts with:
1. An attention-grabbing HOOK (first 3 seconds)
2. Main content SEGMENTS with narration and visual keywords
3. A clear CALL TO ACTION (last 4 seconds)

PROFILE: {self.profile_name}

SCRIPT REQUIREMENTS:
- Target duration: {self.target_duration} seconds
- Speaking pace: 150 words per minute
- Target word count: {int(self.target_duration / 60 * 150)} words total
- Hook must grab attention immediately
- Each segment needs specific visual keywords for stock footage
- CTA should encourage engagement (follow, like, comment)

HOOK TYPES TO VARY:
- question: Start with a thought-provoking question
- number: "5 ways to...", "3 secrets..."
- statement: Bold claim or surprising fact
- story: Mini-story or personal anecdote opener"""

    async def generate_script(
        self,
        topic: str,
        research_context: str | None = None,
        content_pillar: str | None = None,
    ) -> VideoScript:
        """Generate a video script with session context.

        Args:
            topic: Topic to generate script about.
            research_context: Optional research/facts to include.
            content_pillar: Content pillar/category.

        Returns:
            VideoScript with hook, segments, and CTA.
        """
        # Build prompt
        prompt = f"Generate a {self.target_duration}-second video script about:\n\n"
        prompt += f"TOPIC: {topic}\n"

        if content_pillar:
            prompt += f"CONTENT PILLAR: {content_pillar}\n"

        if research_context:
            prompt += f"\nRESEARCH CONTEXT:\n{research_context}\n"

        prompt += f"""
Create a script with:
1. hook: Attention-grabbing opening (3 seconds, ~10-15 words)
2. hook_type: One of: question, number, statement, story
3. segments: {int((self.target_duration - 7) / 10)} content segments with narration and visual keywords
4. cta: Call to action (4 seconds, ~15-20 words)
5. total_words: Count all words in hook + segments + cta
6. estimated_duration: total_words / 2.5 (words per second at 150 wpm)

Each segment should have:
- narration: The spoken words (~25-30 words per 10-second segment)
- keywords: 3-5 keywords for stock footage search
- duration_hint: Suggested duration in seconds
"""

        # Generate structured output
        script = await self.generate_structured(
            prompt=prompt,
            response_model=VideoScript,
            task="script_generation",
            temperature=0.8,
            max_tokens=2048,
        )

        # Record metadata for future constraints
        self.set_metadata(
            topic=topic,
            hook_text=script.hook,
            hook_type=script.hook_type,
            segment_count=len(script.segments),
            word_count=script.total_words,
            target_duration=self.target_duration,
        )

        # Detect hook type if not provided correctly
        detected_type = self._detect_hook_type(script.hook)
        if detected_type != script.hook_type:
            _logger.debug(
                f"Hook type mismatch: declared={script.hook_type}, "
                f"detected={detected_type}"
            )

        return script

    def record_duration_result(
        self,
        actual_duration: float,
        word_count: int,
        accepted: bool,
    ) -> None:
        """Record duration accuracy for learning.

        Args:
            actual_duration: Actual audio duration in seconds.
            word_count: Actual word count.
            accepted: Whether the duration was acceptable.
        """
        metrics = {
            "actual_duration": actual_duration,
            "target_duration": self.target_duration,
            "duration_diff": actual_duration - self.target_duration,
            "word_count": word_count,
            "words_per_second": word_count / actual_duration if actual_duration > 0 else 0,
        }

        self.add_feedback(
            quality="accepted" if accepted else "rejected",
            notes=f"Duration: {actual_duration:.1f}s (target: {self.target_duration}s)",
            metrics=metrics,
        )

    def get_recommended_hook_type(self) -> str:
        """Get recommended hook type based on recent usage.

        Returns:
            Hook type that's least used recently.
        """
        all_types = set(self.HOOK_PATTERNS.keys())
        used_types = set(self._hook_counts.keys())
        unused = all_types - used_types

        if unused:
            return unused.pop()

        # Return least used
        if self._hook_counts:
            return min(self._hook_counts, key=self._hook_counts.get)

        return "question"  # Default

    def is_topic_recent(self, topic: str) -> bool:
        """Check if a topic was recently covered.

        Args:
            topic: Topic to check.

        Returns:
            True if topic or similar was covered recently.
        """
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        for recent_topic in self._recent_topics:
            recent_words = set(recent_topic.split())
            # Check for significant overlap
            overlap = len(topic_words & recent_words) / max(len(topic_words), 1)
            if overlap > 0.5:
                return True

        return topic_lower in self._recent_topics
