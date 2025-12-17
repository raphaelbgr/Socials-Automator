"""Stateless service for maintenance operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TokenStatus:
    """Token status information."""

    valid: bool
    expires_at: Optional[str] = None
    error: Optional[str] = None


def init_project_structure(base_dir: Path | None = None) -> List[str]:
    """Initialize project directory structure.

    Args:
        base_dir: Base directory (defaults to cwd)

    Returns:
        List of directories that were created
    """
    if base_dir is None:
        base_dir = Path.cwd()

    dirs = [
        "profiles",
        "config",
        "logs",
        "fonts",
    ]

    created = []
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            created.append(dir_name)

    return created


def check_token() -> TokenStatus:
    """Check Instagram token validity.

    Returns:
        TokenStatus with validity information
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from socials_automator.instagram.models import InstagramConfig
        from socials_automator.instagram import TokenManager

        config = InstagramConfig.from_env()
        token_manager = TokenManager(config)

        info = token_manager.check_token()

        return TokenStatus(
            valid=info.get("valid", False),
            expires_at=info.get("expires_at"),
        )

    except ValueError as e:
        return TokenStatus(valid=False, error=str(e))
    except Exception as e:
        return TokenStatus(valid=False, error=str(e))


def refresh_token() -> Tuple[bool, Optional[str]]:
    """Refresh Instagram token.

    Returns:
        Tuple of (success, new_expiry or error_message)
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from socials_automator.instagram.models import InstagramConfig
        from socials_automator.instagram import TokenManager

        config = InstagramConfig.from_env()
        token_manager = TokenManager(config)

        result = token_manager.refresh_token()

        return True, result.get("expires_at")

    except Exception as e:
        return False, str(e)


def get_profile_status(profile_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Get post and reel counts for a profile.

    Args:
        profile_path: Path to profile directory

    Returns:
        Tuple of (post_counts, reel_counts) dicts
    """
    post_counts = {"generated": 0, "pending": 0, "posted": 0}
    reel_counts = {"generated": 0, "pending": 0, "posted": 0}

    # Count posts
    posts_dir = profile_path / "posts"
    if posts_dir.exists():
        for year_dir in posts_dir.glob("*"):
            if not (year_dir.is_dir() and year_dir.name.isdigit()):
                continue
            for month_dir in year_dir.glob("*"):
                if not (month_dir.is_dir() and month_dir.name.isdigit()):
                    continue

                for status in ["generated", "pending-post", "posted"]:
                    status_dir = month_dir / status
                    if status_dir.exists():
                        count = sum(1 for d in status_dir.iterdir() if d.is_dir())
                        key = "pending" if status == "pending-post" else status
                        post_counts[key] += count

    # Count reels
    reels_dir = profile_path / "reels"
    if reels_dir.exists():
        for year_dir in reels_dir.glob("*"):
            if not (year_dir.is_dir() and year_dir.name.isdigit()):
                continue
            for month_dir in year_dir.glob("*"):
                if not (month_dir.is_dir() and month_dir.name.isdigit()):
                    continue

                for status in ["generated", "pending-post", "posted"]:
                    status_dir = month_dir / status
                    if status_dir.exists():
                        count = sum(1 for d in status_dir.iterdir() if d.is_dir())
                        key = "pending" if status == "pending-post" else status
                        reel_counts[key] += count

    return post_counts, reel_counts


def load_niches(niches_path: Path | None = None) -> List[dict]:
    """Load niches from niches.json.

    Args:
        niches_path: Path to niches.json (defaults to docs/niches.json)

    Returns:
        List of niche dictionaries
    """
    if niches_path is None:
        niches_path = Path.cwd() / "docs" / "niches.json"

    if not niches_path.exists():
        return []

    try:
        with open(niches_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("niches", [])
    except Exception:
        return []


def create_profile(
    name: str,
    handle: str,
    niche: Optional[str],
    profiles_dir: Path | None = None,
) -> Path:
    """Create a new profile.

    Args:
        name: Profile folder name
        handle: Instagram handle
        niche: Niche ID
        profiles_dir: Profiles directory

    Returns:
        Path to created profile
    """
    if profiles_dir is None:
        profiles_dir = Path.cwd() / "profiles"

    profile_path = profiles_dir / name
    profile_path.mkdir(parents=True, exist_ok=True)

    # Create metadata.json
    metadata = {
        "profile": {
            "instagram_handle": handle,
            "niche_id": niche or "",
        },
        "content_strategy": {
            "pillars": [],
            "topics": [],
            "carousel_settings": {
                "min_slides": 3,
                "max_slides": 10,
            },
        },
    }

    metadata_path = profile_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return profile_path


def find_reels_for_artifact_update(
    profile_path: Path,
) -> List[Path]:
    """Find reels that need artifact metadata update.

    Args:
        profile_path: Path to profile directory

    Returns:
        List of reel folder paths
    """
    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    all_reels = []

    for year_dir in reels_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            for status in ["generated", "pending-post", "posted"]:
                status_dir = month_dir / status
                if not status_dir.exists():
                    continue

                for reel_dir in status_dir.glob("*"):
                    if reel_dir.is_dir() and (reel_dir / "metadata.json").exists():
                        all_reels.append(reel_dir)

    return sorted(all_reels)


def update_reel_artifacts(reel_path: Path) -> bool:
    """Update artifact metadata for a single reel.

    Args:
        reel_path: Path to reel folder

    Returns:
        True if updated, False if skipped
    """
    try:
        metadata_path = reel_path / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Check if artifacts need update
        if "artifacts" in metadata and metadata["artifacts"]:
            return False  # Already has artifacts

        # Find artifact files
        artifacts = {}
        if (reel_path / "final.mp4").exists():
            artifacts["video"] = "final.mp4"
        if (reel_path / "thumbnail.jpg").exists():
            artifacts["thumbnail"] = "thumbnail.jpg"
        if (reel_path / "audio.mp3").exists():
            artifacts["audio"] = "audio.mp3"

        if not artifacts:
            return False  # No artifacts to add

        # Update metadata
        metadata["artifacts"] = artifacts

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception:
        return False
