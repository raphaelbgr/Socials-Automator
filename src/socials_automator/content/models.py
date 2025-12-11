"""Data models for content generation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SlideType(str, Enum):
    """Type of slide in a carousel."""

    HOOK = "hook"
    CONTENT = "content"
    CTA = "cta"


class HookType(str, Enum):
    """Type of hook strategy."""

    CURIOSITY_GAP = "curiosity_gap"
    BOLD_STATEMENT = "bold_statement"
    NUMBER_BENEFIT = "number_benefit"
    QUESTION = "question"
    SOCIAL_PROOF = "social_proof"


class SlideContent(BaseModel):
    """Content for a single slide."""

    number: int
    slide_type: SlideType
    heading: str
    body: str | None = None
    subtext: str | None = None

    # Image settings
    has_background_image: bool = False
    image_prompt: str | None = None
    image_style: str | None = None

    # Generated outputs
    image_path: str | None = None
    image_bytes: bytes | None = None

    class Config:
        arbitrary_types_allowed = True


class PostPlan(BaseModel):
    """Plan for a carousel post before generation."""

    topic: str
    content_pillar: str
    hook_type: HookType
    hook_text: str
    hook_subtext: str | None = None
    target_slides: int = 6

    # Slide outlines
    slides: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    keywords: list[str] = Field(default_factory=list)
    research_sources: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class CarouselPost(BaseModel):
    """A complete carousel post ready for output."""

    id: str
    date: str  # YYYY-MM-DD
    slug: str
    topic: str
    content_pillar: str

    # Hook
    hook_type: HookType
    hook_text: str

    # Slides
    slides: list[SlideContent] = Field(default_factory=list)

    # Caption
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    alt_texts: list[str] = Field(default_factory=list)

    # Generation metadata
    generation_time_seconds: float | None = None
    total_cost_usd: float = 0.0
    text_provider: str | None = None
    image_provider: str | None = None

    # Status
    status: Literal["planned", "generating", "generated", "posted"] = "planned"
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def slides_count(self) -> int:
        """Get number of slides."""
        return len(self.slides)


class GenerationProgress(BaseModel):
    """Progress tracking for generation."""

    post_id: str
    status: str = "starting"
    current_step: str = ""
    current_action: str = ""  # Detailed action like "Validating AI output #1", "Generating content"
    total_steps: int = 0
    completed_steps: int = 0
    current_slide: int = 0
    total_slides: int = 0
    errors: list[str] = Field(default_factory=list)

    # Validation tracking
    validation_attempt: int = 0
    validation_max_attempts: int = 6
    validation_error: str | None = None

    # Detailed event info (legacy, for compatibility)
    event_type: str = ""  # text_call, text_response, text_error, image_call, image_response, image_error
    provider: str | None = None
    model: str | None = None
    prompt_preview: str | None = None  # First 200 chars of prompt
    response_preview: str | None = None  # First 200 chars of response
    duration_seconds: float | None = None
    cost_usd: float | None = None

    # Current text AI activity (persisted across events)
    text_provider: str | None = None
    text_model: str | None = None
    text_prompt_preview: str | None = None
    text_failed_providers: list[str] = Field(default_factory=list)

    # Current image AI activity
    image_provider: str | None = None
    image_model: str | None = None
    image_prompt_preview: str | None = None
    image_failed_providers: list[str] = Field(default_factory=list)

    # Accumulated stats
    total_text_calls: int = 0
    total_image_calls: int = 0
    total_cost_usd: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
