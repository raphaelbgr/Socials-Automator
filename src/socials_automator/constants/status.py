"""Status enums and state constants for the Socials Automator.

This module contains all status enums and state definitions:
- Pipeline execution states
- Post/content lifecycle states
- API response states
- Log levels and severity

AI CONTEXT:
-----------
Status enums represent the state machine for content lifecycle:
  PENDING -> GENERATING -> VALIDATING -> GENERATED -> PENDING_POST -> POSTED

Pipeline steps have their own status tracking for progress reporting.

MODIFICATION GUIDE:
------------------
- Add new enum values at the END to maintain backwards compatibility
- Each enum should have a human-readable description
- Use .value for string representation when needed
"""

from enum import Enum, auto
from typing import Final


# =============================================================================
# CONTENT LIFECYCLE STATUS
# =============================================================================

class ContentStatus(str, Enum):
    """Status of content through its lifecycle.

    Workflow:
        PENDING -> GENERATING -> GENERATED -> PENDING_POST -> POSTED
                      |                            |
                      v                            v
                   FAILED                       FAILED
    """

    PENDING = "pending"
    """Content is queued but not yet started."""

    GENERATING = "generating"
    """Content is currently being generated."""

    VALIDATING = "validating"
    """Content is being validated by AI."""

    GENERATED = "generated"
    """Content generation complete, ready for review."""

    PENDING_POST = "pending-post"
    """Content approved and queued for posting."""

    POSTING = "posting"
    """Content is being posted to platform."""

    POSTED = "posted"
    """Content successfully posted to platform."""

    FAILED = "failed"
    """Content generation or posting failed."""

    ARCHIVED = "archived"
    """Content archived (old or removed)."""


# =============================================================================
# PIPELINE STEP STATUS
# =============================================================================

class PipelineStepStatus(str, Enum):
    """Status of individual pipeline steps.

    Each step in the video pipeline goes through these states.
    Used for progress tracking and error reporting.
    """

    PENDING = "pending"
    """Step not yet started."""

    RUNNING = "running"
    """Step currently executing."""

    COMPLETED = "completed"
    """Step completed successfully."""

    FAILED = "failed"
    """Step failed with error."""

    SKIPPED = "skipped"
    """Step was skipped (optional or not needed)."""

    RETRYING = "retrying"
    """Step failed but retrying."""


# =============================================================================
# VALIDATION STATUS
# =============================================================================

class ValidationStatus(str, Enum):
    """Status of content validation checks.

    Used by AI validation to track quality checks.
    """

    PENDING = "pending"
    """Validation not yet performed."""

    VALIDATING = "validating"
    """Validation in progress."""

    PASSED = "passed"
    """Validation passed all checks."""

    FAILED = "failed"
    """Validation failed one or more checks."""

    NEEDS_REVIEW = "needs_review"
    """Validation uncertain, needs human review."""


# =============================================================================
# API RESPONSE STATUS
# =============================================================================

class APIStatus(str, Enum):
    """Status codes for API responses.

    Standardized status codes across all API interactions.
    """

    SUCCESS = "success"
    """API call completed successfully."""

    ERROR = "error"
    """API call failed with error."""

    RATE_LIMITED = "rate_limited"
    """API call rejected due to rate limiting."""

    TIMEOUT = "timeout"
    """API call timed out."""

    UNAUTHORIZED = "unauthorized"
    """API call failed due to authentication."""

    NOT_FOUND = "not_found"
    """Requested resource not found."""

    INVALID_REQUEST = "invalid_request"
    """Request was malformed or invalid."""


# =============================================================================
# LOG LEVELS
# =============================================================================

class LogLevel(str, Enum):
    """Log severity levels.

    Used by PipelineDisplay for consistent logging.
    """

    DEBUG = "debug"
    """Detailed debugging information."""

    INFO = "info"
    """General informational messages."""

    STEP = "step"
    """Pipeline step progress updates."""

    SUCCESS = "success"
    """Successful operation completion."""

    WARNING = "warning"
    """Warning that doesn't stop execution."""

    ERROR = "error"
    """Error that may affect results."""

    CRITICAL = "critical"
    """Critical error that stops execution."""


# =============================================================================
# CONTENT TYPE
# =============================================================================

class ContentType(str, Enum):
    """Type of content being generated.

    Different content types have different pipelines and outputs.
    """

    CAROUSEL = "carousel"
    """Multi-image carousel post (Instagram)."""

    REEL = "reel"
    """Short-form video (Instagram Reels/TikTok)."""

    STORY = "story"
    """Ephemeral story content."""

    SINGLE_IMAGE = "single_image"
    """Single image post."""


# =============================================================================
# PLATFORM
# =============================================================================

class Platform(str, Enum):
    """Target social media platform.

    Currently only Instagram is fully supported.
    """

    INSTAGRAM = "instagram"
    """Instagram (Reels, Carousels, Stories)."""

    TIKTOK = "tiktok"
    """TikTok (short videos)."""

    YOUTUBE_SHORTS = "youtube_shorts"
    """YouTube Shorts."""

    THREADS = "threads"
    """Threads (text-focused)."""


# =============================================================================
# VIDEO SOURCE
# =============================================================================

class VideoSource(str, Enum):
    """Source for stock video footage.

    Used by VideoSearcher to determine which API to use.
    """

    PEXELS = "pexels"
    """Pexels free stock video API."""

    PIXABAY = "pixabay"
    """Pixabay free stock video (not implemented)."""

    LOCAL = "local"
    """Local video files."""


# =============================================================================
# VOICE PROVIDER
# =============================================================================

class VoiceProvider(str, Enum):
    """Text-to-speech provider.

    Different TTS providers for voice generation.
    """

    RVC = "rvc"
    """RVC (Retrieval-based Voice Conversion) - local, free."""

    OPENAI = "openai"
    """OpenAI TTS API."""

    ELEVENLABS = "elevenlabs"
    """ElevenLabs TTS API."""

    EDGE = "edge"
    """Microsoft Edge TTS (free)."""


# =============================================================================
# AI PROVIDER
# =============================================================================

class AIProvider(str, Enum):
    """AI text generation provider.

    Different providers for text/content generation.
    """

    ZAI = "zai"
    """Z.AI (GLM models) - cheap."""

    GROQ = "groq"
    """Groq (Llama models) - fast, free tier."""

    OPENAI = "openai"
    """OpenAI (GPT models) - high quality."""

    GEMINI = "gemini"
    """Google Gemini - free tier available."""

    LMSTUDIO = "lmstudio"
    """LM Studio local server - free."""

    OLLAMA = "ollama"
    """Ollama local models - free."""


# =============================================================================
# STATUS ICONS (ASCII-compatible)
# =============================================================================
# Used by CLI display for visual feedback

STATUS_ICONS: Final[dict[str, str]] = {
    "pending": "[ ]",
    "running": "[~]",
    "completed": "[OK]",
    "success": "[OK]",
    "failed": "[X]",
    "error": "[X]",
    "warning": "[!]",
    "info": "[i]",
    "step": "[>]",
    "debug": "[.]",
}
"""ASCII icons for status display. Windows console compatible."""


# =============================================================================
# STATUS COLORS (Rich markup)
# =============================================================================

STATUS_COLORS: Final[dict[str, str]] = {
    "pending": "dim",
    "running": "cyan",
    "completed": "green",
    "success": "green",
    "failed": "red",
    "error": "red",
    "warning": "yellow",
    "info": "white",
    "step": "cyan",
    "debug": "dim",
}
"""Rich color names for status display."""
