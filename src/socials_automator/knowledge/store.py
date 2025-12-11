"""Knowledge store for tracking post history and avoiding repetition."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import (
    PostRecord,
    TopicRecord,
    HookRecord,
    PromptLog,
    PromptLogEntry,
    PostsIndex,
    TopicsIndex,
    HooksIndex,
)


class KnowledgeStore:
    """Knowledge base for tracking generated content.

    Provides:
    - Post history tracking
    - Topic deduplication
    - Hook uniqueness checking
    - Prompt logging for learning
    - Semantic search via ChromaDB (optional)

    Usage:
        store = KnowledgeStore(profile_path=Path("profiles/ai.for.mortals"))

        # Check if topic was recently used
        if store.is_topic_recent("chatgpt email tips", days=7):
            # Skip this topic
            pass

        # Log a new post
        store.add_post(PostRecord(...))

        # Get context for AI
        context = store.get_recent_context(days=7)
    """

    def __init__(self, profile_path: Path):
        """Initialize knowledge store for a profile.

        Args:
            profile_path: Path to the profile directory.
        """
        self.profile_path = profile_path
        self.knowledge_path = profile_path / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)

        # File paths
        self.posts_index_path = self.knowledge_path / "posts_index.json"
        self.topics_index_path = self.knowledge_path / "topics_used.json"
        self.hooks_index_path = self.knowledge_path / "hooks_used.json"
        self.prompts_log_path = self.knowledge_path / "prompts_log.json"

        # ChromaDB for semantic search (lazy loaded)
        self._chroma_client = None
        self._hooks_collection = None
        self._topics_collection = None

        # Load indices
        self._posts_index: PostsIndex | None = None
        self._topics_index: TopicsIndex | None = None
        self._hooks_index: HooksIndex | None = None

    def _load_json(self, path: Path, model: type[BaseModel]) -> BaseModel:
        """Load a JSON file into a Pydantic model."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return model.model_validate(data)
        return model()

    def _save_json(self, path: Path, model: BaseModel) -> None:
        """Save a Pydantic model to JSON file."""
        with open(path, "w") as f:
            json.dump(model.model_dump(mode="json"), f, indent=2, default=str)

    @property
    def posts_index(self) -> PostsIndex:
        """Get posts index, loading from file if needed."""
        if self._posts_index is None:
            self._posts_index = self._load_json(self.posts_index_path, PostsIndex)
        return self._posts_index

    @property
    def topics_index(self) -> TopicsIndex:
        """Get topics index, loading from file if needed."""
        if self._topics_index is None:
            self._topics_index = self._load_json(self.topics_index_path, TopicsIndex)
        return self._topics_index

    @property
    def hooks_index(self) -> HooksIndex:
        """Get hooks index, loading from file if needed."""
        if self._hooks_index is None:
            self._hooks_index = self._load_json(self.hooks_index_path, HooksIndex)
        return self._hooks_index

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB for semantic search."""
        if self._chroma_client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.knowledge_path / "chromadb"),
                anonymized_telemetry=False,
            ))

            self._hooks_collection = self._chroma_client.get_or_create_collection(
                name="hooks",
                metadata={"hnsw:space": "cosine"},
            )

            self._topics_collection = self._chroma_client.get_or_create_collection(
                name="topics",
                metadata={"hnsw:space": "cosine"},
            )

        except ImportError:
            # ChromaDB not installed, skip semantic search
            pass

    # ==================== Post Management ====================

    def add_post(self, post: PostRecord) -> None:
        """Add a post to the index.

        Args:
            post: Post record to add.
        """
        index = self.posts_index
        index.posts.append(post)
        index.total_posts = len(index.posts)
        index.last_updated = datetime.now()
        self._save_json(self.posts_index_path, index)

        # Also add topic and hook
        self.add_topic(TopicRecord(
            topic=post.topic,
            date_used=post.date,
            post_id=post.id,
            content_pillar=post.content_pillar,
            keywords=post.keywords,
        ))

        self.add_hook(HookRecord(
            text=post.hook_text,
            hook_type=post.hook_type,
            date_used=post.date,
            post_id=post.id,
        ))

        # Update keyword frequency
        self._update_keyword_frequency(post.keywords)

    def get_recent_posts(self, days: int = 7) -> list[PostRecord]:
        """Get posts from the last N days.

        Args:
            days: Number of days to look back.

        Returns:
            List of recent posts.
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        return [
            post for post in self.posts_index.posts
            if post.date >= cutoff_str
        ]

    def get_posts_by_pillar(self, pillar: str, limit: int = 10) -> list[PostRecord]:
        """Get recent posts for a content pillar.

        Args:
            pillar: Content pillar name.
            limit: Maximum posts to return.

        Returns:
            List of posts for the pillar.
        """
        posts = [
            post for post in self.posts_index.posts
            if post.content_pillar == pillar
        ]
        return sorted(posts, key=lambda p: p.date, reverse=True)[:limit]

    # ==================== Topic Management ====================

    def add_topic(self, topic: TopicRecord) -> None:
        """Add a topic to the index.

        Args:
            topic: Topic record to add.
        """
        index = self.topics_index
        index.topics.append(topic)
        self._save_json(self.topics_index_path, index)

        # Add to ChromaDB for semantic search
        self._init_chromadb()
        if self._topics_collection is not None:
            self._topics_collection.add(
                documents=[topic.topic],
                ids=[f"{topic.post_id}_{topic.topic[:50]}"],
                metadatas=[{
                    "post_id": topic.post_id,
                    "date": topic.date_used,
                    "pillar": topic.content_pillar or "",
                }],
            )

    def is_topic_recent(self, topic: str, days: int = 7) -> bool:
        """Check if a topic was used recently.

        Args:
            topic: Topic to check.
            days: Number of days to consider "recent".

        Returns:
            True if topic was used within the specified days.
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        for record in self.topics_index.topics:
            if record.topic.lower() == topic.lower() and record.date_used >= cutoff_str:
                return True

        return False

    def find_similar_topics(self, topic: str, threshold: float = 0.8, limit: int = 5) -> list[TopicRecord]:
        """Find semantically similar topics using ChromaDB.

        Args:
            topic: Topic to search for.
            threshold: Similarity threshold (0-1).
            limit: Maximum results.

        Returns:
            List of similar topic records.
        """
        self._init_chromadb()
        if self._topics_collection is None:
            return []

        results = self._topics_collection.query(
            query_texts=[topic],
            n_results=limit,
        )

        similar = []
        if results["ids"] and results["distances"]:
            for i, (id_, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                # ChromaDB returns distance, not similarity
                similarity = 1 - distance
                if similarity >= threshold:
                    # Find the full record
                    post_id = results["metadatas"][0][i]["post_id"]
                    for record in self.topics_index.topics:
                        if record.post_id == post_id:
                            similar.append(record)
                            break

        return similar

    # ==================== Hook Management ====================

    def add_hook(self, hook: HookRecord) -> None:
        """Add a hook to the index.

        Args:
            hook: Hook record to add.
        """
        index = self.hooks_index
        index.hooks.append(hook)
        self._save_json(self.hooks_index_path, index)

        # Add to ChromaDB
        self._init_chromadb()
        if self._hooks_collection is not None:
            self._hooks_collection.add(
                documents=[hook.text],
                ids=[f"{hook.post_id}_{hook.hook_type}"],
                metadatas=[{
                    "post_id": hook.post_id,
                    "date": hook.date_used,
                    "type": hook.hook_type,
                }],
            )

    def is_hook_similar(self, hook_text: str, threshold: float = 0.85) -> bool:
        """Check if a hook is too similar to existing hooks.

        Args:
            hook_text: Hook text to check.
            threshold: Similarity threshold.

        Returns:
            True if hook is too similar to an existing one.
        """
        self._init_chromadb()
        if self._hooks_collection is None:
            # Fall back to exact match
            return any(h.text.lower() == hook_text.lower() for h in self.hooks_index.hooks)

        results = self._hooks_collection.query(
            query_texts=[hook_text],
            n_results=1,
        )

        if results["distances"] and results["distances"][0]:
            similarity = 1 - results["distances"][0][0]
            return similarity >= threshold

        return False

    # ==================== Keyword Management ====================

    def _update_keyword_frequency(self, keywords: list[str]) -> None:
        """Update keyword frequency counts."""
        index = self.topics_index
        for keyword in keywords:
            keyword_lower = keyword.lower()
            index.keywords_frequency[keyword_lower] = index.keywords_frequency.get(keyword_lower, 0) + 1
        self._save_json(self.topics_index_path, index)

    def get_keyword_frequency(self) -> dict[str, int]:
        """Get keyword usage frequency.

        Returns:
            Dictionary of keyword -> count.
        """
        return self.topics_index.keywords_frequency.copy()

    def get_underused_keywords(self, available_keywords: list[str], min_gap: int = 3) -> list[str]:
        """Get keywords that haven't been used recently.

        Args:
            available_keywords: List of keywords to check.
            min_gap: Minimum posts since last use.

        Returns:
            List of underused keywords.
        """
        freq = self.get_keyword_frequency()
        avg_freq = sum(freq.values()) / len(freq) if freq else 0

        underused = []
        for keyword in available_keywords:
            keyword_lower = keyword.lower()
            if freq.get(keyword_lower, 0) < avg_freq - min_gap:
                underused.append(keyword)

        return underused

    # ==================== Prompt Logging ====================

    def log_prompts(self, prompt_log: PromptLog) -> None:
        """Log prompts used for a post.

        Args:
            prompt_log: Prompt log to save.
        """
        # Load existing logs
        logs = []
        if self.prompts_log_path.exists():
            with open(self.prompts_log_path) as f:
                logs = json.load(f)

        logs.append(prompt_log.model_dump(mode="json"))

        with open(self.prompts_log_path, "w") as f:
            json.dump(logs, f, indent=2, default=str)

    def get_successful_prompts(self, prompt_type: str, limit: int = 5) -> list[PromptLogEntry]:
        """Get recent successful prompts of a type.

        Args:
            prompt_type: Type of prompt (e.g., "hook_generation").
            limit: Maximum prompts to return.

        Returns:
            List of prompt entries.
        """
        if not self.prompts_log_path.exists():
            return []

        with open(self.prompts_log_path) as f:
            logs = json.load(f)

        entries = []
        for log in reversed(logs):  # Most recent first
            for entry_data in log.get("entries", []):
                if entry_data.get("prompt_type") == prompt_type:
                    entries.append(PromptLogEntry.model_validate(entry_data))
                    if len(entries) >= limit:
                        return entries

        return entries

    # ==================== Context Generation ====================

    def get_recent_context(self, days: int = 7) -> str:
        """Generate context string for AI about recent posts.

        Args:
            days: Number of days to include.

        Returns:
            Context string for AI prompts.
        """
        recent = self.get_recent_posts(days)

        if not recent:
            return "No posts have been generated yet."

        lines = [f"Recent posts (last {days} days):"]
        for post in recent[-10:]:  # Last 10 posts
            lines.append(f"- {post.date}: {post.topic} ({post.content_pillar})")
            lines.append(f"  Hook: {post.hook_text}")

        # Add keyword stats
        freq = self.get_keyword_frequency()
        if freq:
            top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("\nMost used keywords:")
            for keyword, count in top_keywords:
                lines.append(f"- {keyword}: {count} times")

        return "\n".join(lines)

    def get_topics_to_avoid(self, days: int = 7) -> list[str]:
        """Get topics that should be avoided (recently used).

        Args:
            days: Number of days to consider.

        Returns:
            List of topics to avoid.
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        return [
            record.topic
            for record in self.topics_index.topics
            if record.date_used >= cutoff_str
        ]
