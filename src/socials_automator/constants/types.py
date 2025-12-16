"""Type definitions for the Socials Automator.

This module contains all type aliases, TypedDicts, and protocols:
- Type aliases for common patterns
- TypedDict definitions for structured data
- Protocol classes for interface definitions
- Callback type signatures

AI CONTEXT:
-----------
These types provide static type checking and documentation.
Use these types instead of raw dicts for better IDE support and validation.

TypedDicts are preferred for:
- API responses
- Configuration data
- JSON-serializable structures

MODIFICATION GUIDE:
------------------
- Add new types at the end of relevant sections
- Use TypedDict for dict-like structures with known keys
- Use Protocol for interface definitions
- Include docstrings for all public types
"""

from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    TypeAlias,
    TypedDict,
    Union,
    runtime_checkable,
)


# =============================================================================
# BASIC TYPE ALIASES
# =============================================================================

JSON: TypeAlias = dict[str, Any]
"""Generic JSON object type."""

JSONValue: TypeAlias = Union[str, int, float, bool, None, list[Any], dict[str, Any]]
"""Valid JSON value types."""

PathLike: TypeAlias = Union[str, Path]
"""Path-like type (string or Path object)."""

Keywords: TypeAlias = list[str]
"""List of keyword strings."""

Hashtags: TypeAlias = list[str]
"""List of hashtag strings (with or without #)."""


# =============================================================================
# CALLBACK TYPE ALIASES
# =============================================================================

ProgressCallback: TypeAlias = Callable[[str, float, str], None]
"""Callback for progress updates: (stage, progress_0_to_1, message)."""

LogCallback: TypeAlias = Callable[[str, str], None]
"""Callback for logging: (level, message)."""

ValidationCallback: TypeAlias = Callable[[str], tuple[bool, str]]
"""Callback for validation: (content) -> (is_valid, feedback)."""


# =============================================================================
# PROFILE CONFIGURATION TYPES
# =============================================================================

class ProfileConfig(TypedDict, total=False):
    """Profile configuration from metadata.json.

    Attributes:
        id: Profile identifier (e.g., 'ai.for.mortals').
        name: Internal name.
        display_name: Human-readable display name.
        niche_id: Reference to niche definition.
        tagline: Short profile tagline.
        description: Full profile description.
        language: Content language code (default: 'en').
    """
    id: str
    name: str
    display_name: str
    niche_id: str
    tagline: str
    description: str
    language: str


class ContentPillar(TypedDict, total=False):
    """Content pillar definition.

    Attributes:
        id: Pillar identifier (e.g., 'tool_tutorials').
        name: Human-readable name.
        description: Description of this content type.
        examples: Example topics for this pillar.
        frequency_percent: Target posting frequency (0-100).
    """
    id: str
    name: str
    description: str
    examples: list[str]
    frequency_percent: int


# =============================================================================
# VIDEO PIPELINE TYPES
# =============================================================================

class VideoSegmentData(TypedDict):
    """Data for a single video segment.

    Attributes:
        index: Segment number (1-based).
        text: Narration text for this segment.
        duration_seconds: Target duration in seconds.
        keywords: Search keywords for video matching.
    """
    index: int
    text: str
    duration_seconds: float
    keywords: list[str]


class VideoClipData(TypedDict, total=False):
    """Data for a downloaded video clip.

    Attributes:
        segment_index: Which segment this clip belongs to.
        path: Local file path.
        source_url: Original source URL (e.g., Pexels page).
        pexels_id: Pexels video ID.
        title: Video title/description.
        duration_seconds: Actual clip duration.
        width: Video width in pixels.
        height: Video height in pixels.
        keywords_used: Keywords that matched this video.
    """
    segment_index: int
    path: str
    source_url: str
    pexels_id: int
    title: str
    duration_seconds: float
    width: int
    height: int
    keywords_used: list[str]


class VideoMetadataDict(TypedDict, total=False):
    """Video metadata for output files.

    Attributes:
        post_id: Unique post identifier.
        title: Video title.
        topic: Original topic.
        created_at: ISO timestamp of creation.
        duration_seconds: Final video duration.
        segments: List of segment timing data.
        clips_used: List of clips used.
        narration: Full narration text.
    """
    post_id: str
    title: str
    topic: str
    created_at: str
    duration_seconds: float
    segments: list[dict[str, Any]]
    clips_used: list[dict[str, Any]]
    narration: str


# =============================================================================
# PEXELS API TYPES
# =============================================================================

class PexelsVideoFile(TypedDict, total=False):
    """Pexels video file data from API.

    Attributes:
        id: File ID.
        quality: Quality label ('hd', 'sd', etc.).
        file_type: MIME type.
        width: Video width.
        height: Video height.
        link: Download URL.
    """
    id: int
    quality: str
    file_type: str
    width: int
    height: int
    link: str


class PexelsVideo(TypedDict, total=False):
    """Pexels video data from API.

    Attributes:
        id: Pexels video ID.
        width: Original width.
        height: Original height.
        duration: Duration in seconds.
        url: Pexels page URL.
        image: Thumbnail image URL.
        video_files: Available video files.
        user: Video author info.
    """
    id: int
    width: int
    height: int
    duration: int
    url: str
    image: str
    video_files: list[PexelsVideoFile]
    user: dict[str, Any]


class PexelsCacheEntry(TypedDict, total=False):
    """Pexels cache entry data.

    Attributes:
        pexels_id: Pexels video ID.
        filename: Local filename.
        pexels_url: Original Pexels page URL.
        duration_seconds: Video duration.
        width: Video width.
        height: Video height.
        orientation: 'portrait', 'landscape', or 'square'.
        keywords_matched: Keywords that led to this video.
        cached_at: ISO timestamp when cached.
        last_used: ISO timestamp of last use.
        hit_count: Number of times retrieved from cache.
        actual_width: Actual file width (may differ from API).
        actual_height: Actual file height.
        resolution: Resolution string (e.g., '1080x1920').
        aspect_ratio: Aspect ratio string (e.g., '9:16').
        quality_label: Quality label (e.g., 'HD 1080p').
    """
    pexels_id: int
    filename: str
    pexels_url: str
    duration_seconds: float
    width: int
    height: int
    orientation: str
    keywords_matched: list[str]
    cached_at: str
    last_used: str
    hit_count: int
    actual_width: int
    actual_height: int
    resolution: str
    aspect_ratio: str
    quality_label: str


# =============================================================================
# AI RESPONSE TYPES
# =============================================================================

class AIScriptResponse(TypedDict):
    """Expected response from AI script generation.

    Attributes:
        title: Video title.
        hook: Opening hook text.
        segments: List of segment data.
        cta: Call-to-action text.
        cta_ending: CTA ending phrase.
    """
    title: str
    hook: str
    segments: list[VideoSegmentData]
    cta: str
    cta_ending: str


class AICaptionResponse(TypedDict):
    """Expected response from AI caption generation.

    Attributes:
        caption: Generated caption text.
        hashtags: Space-separated hashtags.
    """
    caption: str
    hashtags: str


class AIValidationResponse(TypedDict):
    """Expected response from AI validation.

    Attributes:
        is_valid: Whether content passed validation.
        score: Quality score (1-10).
        feedback: Feedback message.
    """
    is_valid: bool
    score: int
    feedback: str


# =============================================================================
# PROTOCOL DEFINITIONS
# =============================================================================

@runtime_checkable
class AIClient(Protocol):
    """Protocol for AI text generation clients.

    Any object implementing this protocol can be used for AI generation.
    """

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated text response.
        """
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations.

    Generic caching interface for videos, images, etc.
    """

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        ...

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...


# =============================================================================
# RESULT TYPES
# =============================================================================

class Result(TypedDict, total=False):
    """Generic result type for operations.

    Attributes:
        success: Whether operation succeeded.
        data: Result data (if successful).
        error: Error message (if failed).
        error_code: Error code for programmatic handling.
    """
    success: bool
    data: Any
    error: str
    error_code: str


class ValidationResult(TypedDict):
    """Result of content validation.

    Attributes:
        is_valid: Whether validation passed.
        score: Quality score (0-10).
        feedback: Human-readable feedback.
        issues: List of specific issues found.
    """
    is_valid: bool
    score: int
    feedback: str
    issues: list[str]
