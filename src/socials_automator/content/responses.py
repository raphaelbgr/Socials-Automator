"""Pydantic response models for AI extraction.

These models define the exact structure expected from AI responses,
enabling Instructor to enforce schema compliance and retry on validation errors.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanningResponse(BaseModel):
    """Phase 1: Planning response - analyze topic and determine structure."""

    content_count: int = Field(
        ge=1, le=15, description="Number of content items to create (1-15)"
    )
    content_type: str = Field(
        description="Type of content (tip, prompt, tool, step, hack, way, etc.)"
    )
    refined_topic: str = Field(
        description="Polished, engaging version of the topic"
    )
    target_audience: str = Field(
        description="Who this content is for"
    )


class StructureResponse(BaseModel):
    """Phase 2: Structure response - create hook and slide titles."""

    hook_text: str = Field(
        max_length=80, description="Catchy hook headline (max 10 words)"
    )
    hook_subtext: str | None = Field(
        default=None, max_length=50, description="Optional 5-word subtext"
    )
    hook_image_description: str = Field(
        description="Description for the hook slide background image"
    )
    slide_titles: list[str] = Field(
        min_length=1, description="List of slide titles, one per content item"
    )


class ContentSlideResponse(BaseModel):
    """Phase 3: Content slide response - single slide content."""

    heading: str = Field(
        max_length=50, description="Short heading for slide (max 50 chars, ~6-8 words)"
    )
    body: str = Field(
        max_length=200, description="1-2 short sentences with actionable details (max 200 chars)"
    )


class CTAResponse(BaseModel):
    """Phase 4: CTA response - call-to-action for final slide."""

    cta_text: str = Field(
        max_length=30, description="Short punchy CTA (2-5 words)"
    )
    cta_subtext: str | None = Field(
        default=None, max_length=50, description="Optional extra line"
    )


class HookListResponse(BaseModel):
    """Response for hook generation - list of hook options."""

    hooks: list[str] = Field(
        min_length=1, max_length=10, description="List of hook text options"
    )


class CaptionResponse(BaseModel):
    """Response for caption generation."""

    caption: str = Field(
        max_length=280, description="Instagram caption (short for Threads compatibility)"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords extracted from content"
    )


class ContentValidationResponse(BaseModel):
    """Response for AI content validation - checks quality of generated text."""

    is_valid: bool = Field(
        description="True if content passes all quality checks"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found (empty if valid)"
    )
    severity: str = Field(
        default="none",
        description="Severity: 'none', 'minor', 'major', or 'critical'"
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested improvement if issues found"
    )
