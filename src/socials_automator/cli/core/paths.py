"""Path utilities for CLI - pure functions for path manipulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal


def get_profiles_dir(base_dir: Path | None = None) -> Path:
    """Get the profiles directory.

    Pure function.

    Args:
        base_dir: Base directory (defaults to cwd)

    Returns:
        Path to profiles directory
    """
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / "profiles"


def get_profile_path(profile: str, base_dir: Path | None = None) -> Path:
    """Get path to a specific profile directory.

    Pure function.

    Args:
        profile: Profile name
        base_dir: Base directory (defaults to cwd)

    Returns:
        Path to profile directory
    """
    return get_profiles_dir(base_dir) / profile


def get_output_dir(
    profile_path: Path,
    content_type: Literal["posts", "reels"],
    status: Literal["generated", "pending-post", "posted"] = "generated",
    now: datetime | None = None,
) -> Path:
    """Get output directory for content.

    Pure function.

    Args:
        profile_path: Path to profile directory
        content_type: Type of content ('posts' or 'reels')
        status: Content status folder
        now: Current datetime (defaults to now)

    Returns:
        Path to output directory
    """
    if now is None:
        now = datetime.now()

    return (
        profile_path
        / content_type
        / now.strftime("%Y")
        / now.strftime("%m")
        / status
    )


def get_reel_folder_name(
    base_dir: Path,
    topic_slug: str = "reel",
    now: datetime | None = None,
) -> str:
    """Generate folder name for a new reel.

    Pure function.

    Args:
        base_dir: Base directory containing existing reels
        topic_slug: Topic slug for folder name
        now: Current datetime (defaults to now)

    Returns:
        Folder name like '17-003-ai-tips'
    """
    if now is None:
        now = datetime.now()

    day = now.strftime("%d")

    # Count existing reels for today
    existing = list(base_dir.glob(f"{day}-*")) if base_dir.exists() else []
    reel_number = len(existing) + 1

    # Sanitize topic slug
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", topic_slug.lower())[:50].strip("-")

    return f"{day}-{reel_number:03d}-{slug}"


def get_post_folder_name(
    base_dir: Path,
    topic_slug: str = "post",
    now: datetime | None = None,
) -> str:
    """Generate folder name for a new post.

    Pure function.

    Args:
        base_dir: Base directory containing existing posts
        topic_slug: Topic slug for folder name
        now: Current datetime (defaults to now)

    Returns:
        Folder name like '17-003-ai-tips'
    """
    if now is None:
        now = datetime.now()

    day = now.strftime("%d")

    # Count existing posts for today
    existing = list(base_dir.glob(f"{day}-*")) if base_dir.exists() else []
    post_number = len(existing) + 1

    # Sanitize topic slug
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", topic_slug.lower())[:50].strip("-")

    return f"{day}-{post_number:03d}-{slug}"


def generate_post_id(now: datetime | None = None) -> str:
    """Generate a unique post ID.

    Pure function.

    Args:
        now: Current datetime (defaults to now)

    Returns:
        Post ID like '20251217-143052'
    """
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def get_logs_dir(base_dir: Path | None = None) -> Path:
    """Get the logs directory.

    Pure function.

    Args:
        base_dir: Base directory (defaults to cwd)

    Returns:
        Path to logs directory
    """
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / "logs"


def get_config_dir(base_dir: Path | None = None) -> Path:
    """Get the config directory.

    Pure function.

    Args:
        base_dir: Base directory (defaults to cwd)

    Returns:
        Path to config directory
    """
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / "config"


def get_fonts_dir(base_dir: Path | None = None) -> Path:
    """Get the fonts directory.

    Pure function.

    Args:
        base_dir: Base directory (defaults to cwd)

    Returns:
        Path to fonts directory
    """
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / "fonts"
