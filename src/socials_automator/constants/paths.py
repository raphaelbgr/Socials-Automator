"""Path-related constants for the Socials Automator.

This module contains all directory paths, file patterns, and naming conventions:
- Project directory structure
- Output folder organization
- File naming patterns
- Temp file locations

AI CONTEXT:
-----------
The project uses a hierarchical folder structure:
  profiles/<name>/posts/YYYY/MM/<status>/<post-id>/
  profiles/<name>/reels/YYYY/MM/<status>/<post-id>/

Status folders: generated -> pending-post -> posted

MODIFICATION GUIDE:
------------------
- Change *_DIR constants to reorganize project structure
- Update *_PATTERN constants for different naming conventions
- TEMP_DIR is critical for MoviePy and other tools that create temp files
"""

from pathlib import Path
from typing import Final

# =============================================================================
# PROJECT ROOT DETECTION
# =============================================================================
# These paths are relative to the project root

def get_project_root() -> Path:
    """Get the project root directory.

    Walks up from this file to find the directory containing 'profiles' or 'src'.
    Falls back to current working directory if not found.

    Returns:
        Path to project root directory.
    """
    current = Path(__file__).resolve().parent

    # Walk up to find project root (contains 'profiles' or 'pyproject.toml')
    for _ in range(10):  # Max 10 levels up
        if (current / "profiles").exists() or (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback to current working directory
    return Path.cwd()


# =============================================================================
# MAIN DIRECTORIES
# =============================================================================

PROJECT_ROOT: Path = get_project_root()
"""Project root directory. Auto-detected from file location."""

PROFILES_DIR_NAME: Final[str] = "profiles"
"""Name of the profiles directory."""

CONFIG_DIR_NAME: Final[str] = "config"
"""Name of the configuration directory."""

LOGS_DIR_NAME: Final[str] = "logs"
"""Name of the logs directory."""

TEMP_DIR_NAME: Final[str] = "temp"
"""Name of the temp directory for intermediate files."""


# =============================================================================
# CONTENT ORGANIZATION
# =============================================================================
# Folder structure for posts and reels

POSTS_DIR_NAME: Final[str] = "posts"
"""Subdirectory name for carousel posts within a profile."""

REELS_DIR_NAME: Final[str] = "reels"
"""Subdirectory name for video reels within a profile."""

# Post status folders (workflow stages)
STATUS_GENERATED: Final[str] = "generated"
"""Folder for newly generated content (not yet reviewed)."""

STATUS_PENDING_POST: Final[str] = "pending-post"
"""Folder for content scheduled/queued for posting."""

STATUS_POSTED: Final[str] = "posted"
"""Folder for content that has been published."""

STATUS_FAILED: Final[str] = "failed"
"""Folder for content that failed to post."""


# =============================================================================
# FILE NAMING PATTERNS
# =============================================================================

POST_ID_FORMAT: Final[str] = "%d-%03d-%s"
"""Post ID format: day-number-slug (e.g., '16-001-ai-tools')."""

POST_ID_DATE_FORMAT: Final[str] = "%Y%m%d-%H%M%S"
"""Date format for post IDs when using timestamps."""

SLIDE_FILENAME_PATTERN: Final[str] = "slide_{:02d}.jpg"
"""Pattern for slide image files (e.g., 'slide_01.jpg')."""

CAPTION_FILENAME: Final[str] = "caption.txt"
"""Filename for caption text (without hashtags)."""

CAPTION_HASHTAGS_FILENAME: Final[str] = "caption+hashtags.txt"
"""Filename for caption with hashtags (used for posting)."""

METADATA_FILENAME: Final[str] = "metadata.json"
"""Filename for post/reel metadata."""

FINAL_VIDEO_FILENAME: Final[str] = "final.mp4"
"""Filename for the final rendered video."""


# =============================================================================
# TEMP FILE SETTINGS
# =============================================================================
# Configuration for temporary files created during processing

TEMP_PREFIX: Final[str] = "socials_"
"""Prefix for temp directories and files."""

TEMP_VIDEO_PREFIX: Final[str] = "video_"
"""Prefix for video processing temp directories."""

# MoviePy temp file settings
MOVIEPY_TEMP_AUDIO_SUFFIX: Final[str] = "_temp_audio.mp3"
"""Suffix for MoviePy temporary audio files."""

MOVIEPY_TEMP_VIDEO_SUFFIX: Final[str] = "_temp_video.mp4"
"""Suffix for MoviePy temporary video files."""


# =============================================================================
# CACHE DIRECTORIES
# =============================================================================

PEXELS_CACHE_DIR_NAME: Final[str] = "pexels"
"""Directory name for Pexels video cache."""

PEXELS_CACHE_SUBDIR: Final[str] = "cache"
"""Subdirectory within pexels for cached videos."""

PEXELS_CACHE_INDEX_FILENAME: Final[str] = "index.json"
"""Filename for Pexels cache index."""

PEXELS_IMAGE_CACHE_SUBDIR: Final[str] = "image-cache"
"""Subdirectory within pexels for cached images."""

PEXELS_IMAGE_CACHE_INDEX_FILENAME: Final[str] = "index.json"
"""Filename for Pexels image cache index."""

# Profile assets
PROFILE_ASSETS_DIR_NAME: Final[str] = "assets"
"""Directory name for profile assets."""

PROFILE_IMAGES_DIR_NAME: Final[str] = "images"
"""Directory name for local image library within assets."""

IMAGE_OVERLAYS_FILENAME: Final[str] = "image_overlays.json"
"""Filename for image overlay data in output directory."""


# =============================================================================
# LOG FILES
# =============================================================================

LOG_AI_CALLS: Final[str] = "ai_calls.log"
"""Log file for AI API request/response logging."""

LOG_INSTAGRAM_API: Final[str] = "instagram_api.log"
"""Log file for Instagram API calls."""

LOG_VIDEO_PIPELINE: Final[str] = "video_pipeline.log"
"""Log file for video generation pipeline."""


# =============================================================================
# CONFIG FILES
# =============================================================================

PROVIDERS_CONFIG_FILENAME: Final[str] = "providers.yaml"
"""Filename for AI provider configuration."""

NICHES_CONFIG_FILENAME: Final[str] = "niches.json"
"""Filename for niche definitions."""

PROFILE_METADATA_FILENAME: Final[str] = "metadata.json"
"""Filename for profile metadata within profile directory."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_temp_dir() -> Path:
    """Get the project temp directory, creating it if needed.

    Returns:
        Path to temp directory.
    """
    temp_dir = PROJECT_ROOT / TEMP_DIR_NAME
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_logs_dir() -> Path:
    """Get the logs directory, creating it if needed.

    Returns:
        Path to logs directory.
    """
    logs_dir = PROJECT_ROOT / LOGS_DIR_NAME
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_profiles_dir() -> Path:
    """Get the profiles directory.

    Returns:
        Path to profiles directory.
    """
    return PROJECT_ROOT / PROFILES_DIR_NAME


def get_config_dir() -> Path:
    """Get the config directory.

    Returns:
        Path to config directory.
    """
    return PROJECT_ROOT / CONFIG_DIR_NAME


def get_pexels_cache_dir() -> Path:
    """Get the Pexels video cache directory, creating it if needed.

    Returns:
        Path to Pexels video cache directory.
    """
    cache_dir = PROJECT_ROOT / PEXELS_CACHE_DIR_NAME / PEXELS_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_pexels_image_cache_dir() -> Path:
    """Get the Pexels image cache directory, creating it if needed.

    Returns:
        Path to Pexels image cache directory.
    """
    cache_dir = PROJECT_ROOT / PEXELS_CACHE_DIR_NAME / PEXELS_IMAGE_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
