"""Data models for knowledge base."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PostRecord(BaseModel):
    """Record of a generated post."""

    id: str
    date: str  # YYYY-MM-DD
    topic: str
    content_pillar: str
    hook_type: str
    hook_text: str
    slides_count: int
    keywords: list[str] = Field(default_factory=list)
    path: str  # Relative path to post folder
    created_at: datetime = Field(default_factory=datetime.now)

    # Generation metadata
    text_provider: str | None = None
    image_provider: str | None = None
    generation_time_seconds: float | None = None
    total_cost_usd: float | None = None


class TopicRecord(BaseModel):
    """Record of a topic that has been used."""

    topic: str
    date_used: str  # YYYY-MM-DD
    post_id: str
    content_pillar: str | None = None
    can_revisit_after_days: int = 30  # Don't reuse topic for N days
    keywords: list[str] = Field(default_factory=list)


class HookRecord(BaseModel):
    """Record of a hook that has been used."""

    text: str
    hook_type: str
    date_used: str  # YYYY-MM-DD
    post_id: str
    embedding_id: str | None = None  # ChromaDB ID for similarity search


class PromptLogEntry(BaseModel):
    """Single prompt/response pair."""

    prompt_type: str  # research, hook_generation, content_planning, etc.
    system_prompt: str | None = None
    user_prompt: str
    model: str
    temperature: float | None = None
    response: Any  # Can be string or structured data
    tokens_used: int | None = None
    cost_usd: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PromptLog(BaseModel):
    """Log of all prompts used for a post."""

    post_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    entries: list[PromptLogEntry] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_entry(self, entry: PromptLogEntry) -> None:
        """Add a prompt entry and update totals."""
        self.entries.append(entry)
        if entry.tokens_used:
            self.total_tokens += entry.tokens_used
        if entry.cost_usd:
            self.total_cost_usd += entry.cost_usd


class PostsIndex(BaseModel):
    """Index of all posts for a profile."""

    total_posts: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)
    posts: list[PostRecord] = Field(default_factory=list)


class TopicsIndex(BaseModel):
    """Index of all topics used."""

    topics: list[TopicRecord] = Field(default_factory=list)
    keywords_frequency: dict[str, int] = Field(default_factory=dict)


class HooksIndex(BaseModel):
    """Index of all hooks used."""

    hooks: list[HookRecord] = Field(default_factory=list)


class KeywordStats(BaseModel):
    """Statistics about keyword usage."""

    keyword: str
    count: int
    last_used: str  # YYYY-MM-DD
    posts: list[str] = Field(default_factory=list)  # Post IDs
