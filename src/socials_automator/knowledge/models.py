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


# =============================================================================
# AI Tools Database Models
# =============================================================================

class AIToolCategory(BaseModel):
    """Category for AI tools."""

    id: str
    name: str
    description: str
    icon: str = ""


class AIToolRecord(BaseModel):
    """Record of an AI tool in the database.

    Used for:
    - Topic suggestions for AI-related content
    - Version verification during content generation
    - Hidden gem discovery for unique video topics
    """

    id: str
    name: str
    company: str
    category: str
    current_version: str
    version_date: str  # YYYY-MM or YYYY-MM-DD
    url: str
    pricing: str  # free | freemium | paid | enterprise
    hidden_gem: bool = False
    content_potential: str = "medium"  # low | medium | high
    features: list[str] = Field(default_factory=list)
    best_for: list[str] = Field(default_factory=list)
    video_ideas: list[str] = Field(default_factory=list)

    @property
    def is_free(self) -> bool:
        """Check if tool has a free tier."""
        return self.pricing in ("free", "freemium")

    @property
    def has_video_ideas(self) -> bool:
        """Check if tool has specific video ideas."""
        return len(self.video_ideas) > 0

    @property
    def content_score(self) -> int:
        """Numeric content potential score (1-3)."""
        scores = {"low": 1, "medium": 2, "high": 3}
        return scores.get(self.content_potential, 2)

    def to_prompt_context(self) -> str:
        """Format tool info for AI prompts."""
        features_str = ", ".join(self.features[:5])
        return (
            f"{self.name} ({self.company}) - {self.current_version} ({self.version_date})\n"
            f"  Category: {self.category} | Pricing: {self.pricing}\n"
            f"  Features: {features_str}\n"
            f"  Best for: {', '.join(self.best_for[:3])}"
        )


class AIToolsConfig(BaseModel):
    """Configuration for AI tools database."""

    version: str
    last_updated: str
    categories: list[AIToolCategory] = Field(default_factory=list)
    tools: list[AIToolRecord] = Field(default_factory=list)
    version_check_notes: str = ""
    content_hints: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tools(self) -> int:
        """Total number of tools in database."""
        return len(self.tools)

    @property
    def hidden_gems_count(self) -> int:
        """Number of hidden gem tools."""
        return sum(1 for t in self.tools if t.hidden_gem)

    @property
    def high_potential_count(self) -> int:
        """Number of high content potential tools."""
        return sum(1 for t in self.tools if t.content_potential == "high")

    def get_tools_by_category(self, category: str) -> list[AIToolRecord]:
        """Get all tools in a category."""
        return [t for t in self.tools if t.category == category]

    def get_hidden_gems(self) -> list[AIToolRecord]:
        """Get all hidden gem tools."""
        return [t for t in self.tools if t.hidden_gem]

    def get_high_potential_tools(self) -> list[AIToolRecord]:
        """Get tools with high content potential."""
        return [t for t in self.tools if t.content_potential == "high"]

    def get_tool_by_id(self, tool_id: str) -> AIToolRecord | None:
        """Get a tool by its ID."""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None

    def get_tools_with_video_ideas(self) -> list[AIToolRecord]:
        """Get tools that have specific video ideas."""
        return [t for t in self.tools if t.has_video_ideas]


class AIToolUsageRecord(BaseModel):
    """Track when AI tools are mentioned in content."""

    tool_id: str
    post_id: str
    date_used: str  # YYYY-MM-DD
    context: str = ""  # How it was mentioned (main topic, comparison, etc.)
