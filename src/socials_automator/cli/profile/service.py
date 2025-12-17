"""Stateless service for profile operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..core.paths import get_profiles_dir
from ..core.types import ProfileConfig


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation for a single reel."""

    path: Path
    action: str  # "generated", "skipped", "failed"
    error: Optional[str] = None


def list_all_profiles(profiles_dir: Path | None = None) -> List[ProfileConfig]:
    """List all available profiles.

    Pure function - returns ProfileConfig objects.

    Args:
        profiles_dir: Profiles directory (defaults to cwd/profiles)

    Returns:
        List of ProfileConfig objects
    """
    if profiles_dir is None:
        profiles_dir = get_profiles_dir()

    if not profiles_dir.exists():
        return []

    profiles = []

    for profile_dir in sorted(profiles_dir.iterdir()):
        if not profile_dir.is_dir():
            continue

        metadata_path = profile_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            profile_info = metadata.get("profile", {})

            profiles.append(ProfileConfig(
                name=profile_dir.name,
                path=profile_dir,
                handle=profile_info.get("instagram_handle", ""),
                niche=profile_info.get("niche_id", ""),
                metadata=metadata,
            ))
        except Exception:
            # Skip invalid profiles
            continue

    return profiles


def find_reels_for_thumbnails(
    profile_path: Path,
    force: bool = False,
) -> List[Path]:
    """Find reels that need thumbnails.

    Args:
        profile_path: Path to profile directory
        force: If True, return all reels (regenerate all)

    Returns:
        List of reel folder paths needing thumbnails
    """
    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    all_reels = []

    for year_dir in reels_dir.glob("*"):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        for month_dir in year_dir.glob("*"):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue

            for status in ["generated", "pending-post", "posted"]:
                status_dir = month_dir / status
                if not status_dir.exists():
                    continue

                for reel_dir in status_dir.glob("*"):
                    if not reel_dir.is_dir():
                        continue

                    # Check if reel has video
                    video_path = reel_dir / "final.mp4"
                    if not video_path.exists():
                        continue

                    # Check if thumbnail exists
                    thumbnail_path = reel_dir / "thumbnail.jpg"
                    if thumbnail_path.exists() and not force:
                        continue

                    all_reels.append(reel_dir)

    return sorted(all_reels)


def generate_thumbnail(
    reel_path: Path,
    font_size: int = 54,
) -> ThumbnailResult:
    """Generate thumbnail for a single reel.

    Args:
        reel_path: Path to reel folder
        font_size: Font size for thumbnail text

    Returns:
        ThumbnailResult with action and any error
    """
    try:
        from socials_automator.video.pipeline.thumbnail_generator import ThumbnailGenerator

        # Load metadata
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return ThumbnailResult(
                path=reel_path,
                action="failed",
                error="No metadata.json found",
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Get topic for thumbnail text
        topic = metadata.get("topic", "")
        if not topic:
            return ThumbnailResult(
                path=reel_path,
                action="failed",
                error="No topic in metadata",
            )

        # Generate thumbnail
        generator = ThumbnailGenerator(
            video_path=reel_path / "final.mp4",
            output_path=reel_path / "thumbnail.jpg",
            topic=topic,
            font_size=font_size,
        )

        generator.generate()

        return ThumbnailResult(
            path=reel_path,
            action="generated",
        )

    except Exception as e:
        return ThumbnailResult(
            path=reel_path,
            action="failed",
            error=str(e),
        )
