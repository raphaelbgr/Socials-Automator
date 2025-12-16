"""Global constants package for Socials Automator.

This package centralizes all constants, enums, types, and configuration values
used throughout the project. Import from here for consistency.

PACKAGE STRUCTURE:
-----------------
- video.py    : Video resolution, FPS, subtitle settings
- paths.py    : Directory paths, file patterns, naming conventions
- limits.py   : API limits, content constraints, timeouts
- status.py   : Status enums, state definitions, display icons
- types.py    : Type aliases, TypedDicts, protocols

USAGE EXAMPLES:
--------------
    # Import specific constants
    from socials_automator.constants import VIDEO_WIDTH, VIDEO_HEIGHT

    # Import enums
    from socials_automator.constants import ContentStatus, PipelineStepStatus

    # Import types
    from socials_automator.constants import VideoMetadataDict, AIClient

    # Import path helpers
    from socials_automator.constants import get_temp_dir, get_pexels_cache_dir

AI CONTEXT:
-----------
This package is the single source of truth for all magic numbers and
configuration values. When adding new features, check here first for
existing constants, and add new ones here rather than inline.

Benefits:
- Centralized configuration
- Easy to find and modify values
- Consistent naming conventions
- Full type hints and documentation
"""

# =============================================================================
# VIDEO CONSTANTS
# =============================================================================
from .video import (
    # Resolution
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_ASPECT_RATIO,
    VIDEO_ASPECT_RATIO_FLOAT,
    # Frame rate and timing
    VIDEO_FPS,
    VIDEO_MIN_DURATION_SECONDS,
    VIDEO_MAX_DURATION_SECONDS,
    VIDEO_DEFAULT_DURATION_SECONDS,
    # Codec and format
    VIDEO_CODEC,
    VIDEO_FORMAT,
    AUDIO_CODEC,
    AUDIO_SAMPLE_RATE,
    # Subtitles
    SUBTITLE_FONT_SIZE_DEFAULT,
    SUBTITLE_FONT_SIZE_MIN,
    SUBTITLE_FONT_SIZE_MAX,
    SUBTITLE_FONT_NAME,
    SUBTITLE_FONT_COLOR,
    SUBTITLE_STROKE_COLOR,
    SUBTITLE_STROKE_WIDTH,
    SUBTITLE_HIGHLIGHT_COLOR,
    SUBTITLE_POSITION_Y_PERCENT,
    SUBTITLE_MAX_WORDS_PER_LINE,
    # Clips
    CLIP_MIN_DURATION_SECONDS,
    CLIP_MAX_DURATION_SECONDS,
    CLIP_CROSSFADE_DURATION,
    SEGMENTS_DEFAULT_COUNT,
    SEGMENTS_MIN_COUNT,
    SEGMENTS_MAX_COUNT,
    # Watermark
    WATERMARK_ENABLED,
    WATERMARK_OPACITY,
    WATERMARK_POSITION,
    WATERMARK_MARGIN,
)

# =============================================================================
# PATH CONSTANTS
# =============================================================================
from .paths import (
    # Project structure
    PROJECT_ROOT,
    PROFILES_DIR_NAME,
    CONFIG_DIR_NAME,
    LOGS_DIR_NAME,
    TEMP_DIR_NAME,
    # Content organization
    POSTS_DIR_NAME,
    REELS_DIR_NAME,
    STATUS_GENERATED,
    STATUS_PENDING_POST,
    STATUS_POSTED,
    STATUS_FAILED,
    # File naming
    POST_ID_FORMAT,
    POST_ID_DATE_FORMAT,
    SLIDE_FILENAME_PATTERN,
    CAPTION_FILENAME,
    CAPTION_HASHTAGS_FILENAME,
    METADATA_FILENAME,
    FINAL_VIDEO_FILENAME,
    # Temp files
    TEMP_PREFIX,
    TEMP_VIDEO_PREFIX,
    MOVIEPY_TEMP_AUDIO_SUFFIX,
    MOVIEPY_TEMP_VIDEO_SUFFIX,
    # Cache
    PEXELS_CACHE_DIR_NAME,
    PEXELS_CACHE_SUBDIR,
    PEXELS_CACHE_INDEX_FILENAME,
    # Logs
    LOG_AI_CALLS,
    LOG_INSTAGRAM_API,
    LOG_VIDEO_PIPELINE,
    # Config files
    PROVIDERS_CONFIG_FILENAME,
    NICHES_CONFIG_FILENAME,
    PROFILE_METADATA_FILENAME,
    # Helper functions
    get_project_root,
    get_temp_dir,
    get_logs_dir,
    get_profiles_dir,
    get_config_dir,
    get_pexels_cache_dir,
)

# =============================================================================
# LIMIT CONSTANTS
# =============================================================================
from .limits import (
    # Instagram limits
    INSTAGRAM_CAPTION_MAX_LENGTH,
    INSTAGRAM_HASHTAG_MAX_COUNT,
    INSTAGRAM_CAROUSEL_MIN_SLIDES,
    INSTAGRAM_CAROUSEL_MAX_SLIDES,
    INSTAGRAM_POSTS_PER_DAY_LIMIT,
    INSTAGRAM_REEL_MIN_DURATION,
    INSTAGRAM_REEL_MAX_DURATION,
    INSTAGRAM_REEL_MAX_DURATION_EXTENDED,
    # Pexels limits
    PEXELS_REQUESTS_PER_MONTH,
    PEXELS_REQUESTS_PER_HOUR,
    PEXELS_VIDEOS_PER_SEARCH,
    PEXELS_MIN_VIDEO_DURATION,
    PEXELS_MAX_VIDEO_DURATION,
    # AI limits
    AI_MAX_TOKENS_DEFAULT,
    AI_MAX_RETRIES,
    AI_TIMEOUT_SECONDS,
    AI_TEMPERATURE_DEFAULT,
    # Content limits
    CONTENT_TITLE_MAX_LENGTH,
    CONTENT_SLIDE_TEXT_MAX_LENGTH,
    CONTENT_NARRATION_MAX_LENGTH,
    CONTENT_KEYWORD_MAX_COUNT,
    # Retry settings
    RETRY_MAX_ATTEMPTS,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    # Timeouts
    TIMEOUT_HTTP_CONNECT,
    TIMEOUT_HTTP_READ,
    TIMEOUT_VIDEO_DOWNLOAD,
    TIMEOUT_VIDEO_RENDER,
    # Validation
    VALIDATION_CAPTION_MIN_LENGTH,
    VALIDATION_CAPTION_MAX_LENGTH,
    VALIDATION_CAPTION_MIN_SCORE,
    VALIDATION_MIN_TOOLS_MENTIONED,
    VALIDATION_MAX_GENERIC_PHRASES,
    # Batch limits
    BATCH_GENERATE_MAX_COUNT,
    QUEUE_MAX_PENDING,
    LOOP_MIN_INTERVAL_SECONDS,
    LOOP_DEFAULT_DELAY_SECONDS,
    # File size limits
    FILE_MAX_VIDEO_SIZE_MB,
    FILE_MAX_IMAGE_SIZE_MB,
    FILE_MAX_AUDIO_SIZE_MB,
    # Cache limits
    CACHE_MAX_AGE_DAYS,
    CACHE_MAX_SIZE_GB,
    CACHE_MAX_VIDEOS,
)

# =============================================================================
# STATUS ENUMS
# =============================================================================
from .status import (
    # Main enums
    ContentStatus,
    PipelineStepStatus,
    ValidationStatus,
    APIStatus,
    LogLevel,
    ContentType,
    Platform,
    VideoSource,
    VoiceProvider,
    AIProvider,
    # Display helpers
    STATUS_ICONS,
    STATUS_COLORS,
)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================
from .types import (
    # Basic type aliases
    JSON,
    JSONValue,
    PathLike,
    Keywords,
    Hashtags,
    # Callbacks
    ProgressCallback,
    LogCallback,
    ValidationCallback,
    # Profile types
    ProfileConfig,
    ContentPillar,
    # Video types
    VideoSegmentData,
    VideoClipData,
    VideoMetadataDict,
    # Pexels types
    PexelsVideoFile,
    PexelsVideo,
    PexelsCacheEntry,
    # AI types
    AIScriptResponse,
    AICaptionResponse,
    AIValidationResponse,
    # Protocols
    AIClient,
    CacheProtocol,
    # Result types
    Result,
    ValidationResult,
)


# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # Video
    "VIDEO_WIDTH",
    "VIDEO_HEIGHT",
    "VIDEO_ASPECT_RATIO",
    "VIDEO_ASPECT_RATIO_FLOAT",
    "VIDEO_FPS",
    "VIDEO_MIN_DURATION_SECONDS",
    "VIDEO_MAX_DURATION_SECONDS",
    "VIDEO_DEFAULT_DURATION_SECONDS",
    "VIDEO_CODEC",
    "VIDEO_FORMAT",
    "AUDIO_CODEC",
    "AUDIO_SAMPLE_RATE",
    "SUBTITLE_FONT_SIZE_DEFAULT",
    "SUBTITLE_FONT_SIZE_MIN",
    "SUBTITLE_FONT_SIZE_MAX",
    "SUBTITLE_FONT_NAME",
    "SUBTITLE_FONT_COLOR",
    "SUBTITLE_STROKE_COLOR",
    "SUBTITLE_STROKE_WIDTH",
    "SUBTITLE_HIGHLIGHT_COLOR",
    "SUBTITLE_POSITION_Y_PERCENT",
    "SUBTITLE_MAX_WORDS_PER_LINE",
    "CLIP_MIN_DURATION_SECONDS",
    "CLIP_MAX_DURATION_SECONDS",
    "CLIP_CROSSFADE_DURATION",
    "SEGMENTS_DEFAULT_COUNT",
    "SEGMENTS_MIN_COUNT",
    "SEGMENTS_MAX_COUNT",
    "WATERMARK_ENABLED",
    "WATERMARK_OPACITY",
    "WATERMARK_POSITION",
    "WATERMARK_MARGIN",
    # Paths
    "PROJECT_ROOT",
    "PROFILES_DIR_NAME",
    "CONFIG_DIR_NAME",
    "LOGS_DIR_NAME",
    "TEMP_DIR_NAME",
    "POSTS_DIR_NAME",
    "REELS_DIR_NAME",
    "STATUS_GENERATED",
    "STATUS_PENDING_POST",
    "STATUS_POSTED",
    "STATUS_FAILED",
    "POST_ID_FORMAT",
    "POST_ID_DATE_FORMAT",
    "SLIDE_FILENAME_PATTERN",
    "CAPTION_FILENAME",
    "CAPTION_HASHTAGS_FILENAME",
    "METADATA_FILENAME",
    "FINAL_VIDEO_FILENAME",
    "TEMP_PREFIX",
    "TEMP_VIDEO_PREFIX",
    "MOVIEPY_TEMP_AUDIO_SUFFIX",
    "MOVIEPY_TEMP_VIDEO_SUFFIX",
    "PEXELS_CACHE_DIR_NAME",
    "PEXELS_CACHE_SUBDIR",
    "PEXELS_CACHE_INDEX_FILENAME",
    "LOG_AI_CALLS",
    "LOG_INSTAGRAM_API",
    "LOG_VIDEO_PIPELINE",
    "PROVIDERS_CONFIG_FILENAME",
    "NICHES_CONFIG_FILENAME",
    "PROFILE_METADATA_FILENAME",
    "get_project_root",
    "get_temp_dir",
    "get_logs_dir",
    "get_profiles_dir",
    "get_config_dir",
    "get_pexels_cache_dir",
    # Limits
    "INSTAGRAM_CAPTION_MAX_LENGTH",
    "INSTAGRAM_HASHTAG_MAX_COUNT",
    "INSTAGRAM_CAROUSEL_MIN_SLIDES",
    "INSTAGRAM_CAROUSEL_MAX_SLIDES",
    "INSTAGRAM_POSTS_PER_DAY_LIMIT",
    "INSTAGRAM_REEL_MIN_DURATION",
    "INSTAGRAM_REEL_MAX_DURATION",
    "INSTAGRAM_REEL_MAX_DURATION_EXTENDED",
    "PEXELS_REQUESTS_PER_MONTH",
    "PEXELS_REQUESTS_PER_HOUR",
    "PEXELS_VIDEOS_PER_SEARCH",
    "PEXELS_MIN_VIDEO_DURATION",
    "PEXELS_MAX_VIDEO_DURATION",
    "AI_MAX_TOKENS_DEFAULT",
    "AI_MAX_RETRIES",
    "AI_TIMEOUT_SECONDS",
    "AI_TEMPERATURE_DEFAULT",
    "CONTENT_TITLE_MAX_LENGTH",
    "CONTENT_SLIDE_TEXT_MAX_LENGTH",
    "CONTENT_NARRATION_MAX_LENGTH",
    "CONTENT_KEYWORD_MAX_COUNT",
    "RETRY_MAX_ATTEMPTS",
    "RETRY_BASE_DELAY_SECONDS",
    "RETRY_MAX_DELAY_SECONDS",
    "TIMEOUT_HTTP_CONNECT",
    "TIMEOUT_HTTP_READ",
    "TIMEOUT_VIDEO_DOWNLOAD",
    "TIMEOUT_VIDEO_RENDER",
    "VALIDATION_CAPTION_MIN_LENGTH",
    "VALIDATION_CAPTION_MAX_LENGTH",
    "VALIDATION_CAPTION_MIN_SCORE",
    "VALIDATION_MIN_TOOLS_MENTIONED",
    "VALIDATION_MAX_GENERIC_PHRASES",
    "BATCH_GENERATE_MAX_COUNT",
    "QUEUE_MAX_PENDING",
    "LOOP_MIN_INTERVAL_SECONDS",
    "LOOP_DEFAULT_DELAY_SECONDS",
    "FILE_MAX_VIDEO_SIZE_MB",
    "FILE_MAX_IMAGE_SIZE_MB",
    "FILE_MAX_AUDIO_SIZE_MB",
    "CACHE_MAX_AGE_DAYS",
    "CACHE_MAX_SIZE_GB",
    "CACHE_MAX_VIDEOS",
    # Status enums
    "ContentStatus",
    "PipelineStepStatus",
    "ValidationStatus",
    "APIStatus",
    "LogLevel",
    "ContentType",
    "Platform",
    "VideoSource",
    "VoiceProvider",
    "AIProvider",
    "STATUS_ICONS",
    "STATUS_COLORS",
    # Types
    "JSON",
    "JSONValue",
    "PathLike",
    "Keywords",
    "Hashtags",
    "ProgressCallback",
    "LogCallback",
    "ValidationCallback",
    "ProfileConfig",
    "ContentPillar",
    "VideoSegmentData",
    "VideoClipData",
    "VideoMetadataDict",
    "PexelsVideoFile",
    "PexelsVideo",
    "PexelsCacheEntry",
    "AIScriptResponse",
    "AICaptionResponse",
    "AIValidationResponse",
    "AIClient",
    "CacheProtocol",
    "Result",
    "ValidationResult",
]
