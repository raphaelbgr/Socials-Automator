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


def find_posted_reels(profile_path: Path) -> List[Path]:
    """Find all reels in posted folder.

    Args:
        profile_path: Path to profile directory

    Returns:
        List of reel folder paths in posted folders
    """
    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    posted_reels = []

    for year_dir in reels_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            posted_dir = month_dir / "posted"
            if not posted_dir.exists():
                continue

            for reel_dir in posted_dir.glob("*"):
                if reel_dir.is_dir() and (reel_dir / "final.mp4").exists():
                    posted_reels.append(reel_dir)

    return sorted(posted_reels)


def migrate_reel_platform_status(reel_path: Path, dry_run: bool = False) -> str:
    """Add platform_status.instagram to a posted reel's metadata.

    For backwards compatibility, marks existing posted reels as uploaded to Instagram.

    Args:
        reel_path: Path to reel folder
        dry_run: If True, don't actually modify files

    Returns:
        Status string: "updated", "skipped" (already has status), or "error"
    """
    try:
        metadata_path = reel_path / "metadata.json"

        if not metadata_path.exists():
            return "error"

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Check if platform_status already exists
        if "platform_status" in metadata:
            platform_status = metadata["platform_status"]
            # Skip if instagram is already tracked
            if "instagram" in platform_status and platform_status["instagram"].get("uploaded"):
                return "skipped"

        if dry_run:
            return "would_update"

        # Initialize platform_status if needed
        if "platform_status" not in metadata:
            metadata["platform_status"] = {}

        # Try to get upload timestamp from various sources
        uploaded_at = None

        # 1. Check for existing instagram-related metadata
        if "instagram" in metadata:
            ig_data = metadata["instagram"]
            uploaded_at = ig_data.get("published_at") or ig_data.get("uploaded_at")

        # 2. Check for _upload_state (from previous upload logic)
        if not uploaded_at and "_upload_state" in metadata:
            uploaded_at = metadata["_upload_state"].get("uploaded_at")

        # 3. Fall back to file modification time
        if not uploaded_at:
            import os
            from datetime import datetime
            mtime = os.path.getmtime(metadata_path)
            uploaded_at = datetime.fromtimestamp(mtime).isoformat()

        # Add instagram status
        metadata["platform_status"]["instagram"] = {
            "uploaded": True,
            "uploaded_at": uploaded_at,
            "media_id": metadata.get("instagram", {}).get("media_id"),
            "permalink": metadata.get("instagram", {}).get("permalink"),
            "migrated": True,  # Flag to indicate this was auto-migrated
        }

        # Write back
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return "updated"

    except Exception as e:
        return f"error: {e}"


@dataclass
class CleanupResult:
    """Result of cleaning up a single reel."""

    reel_path: Path
    reel_name: str
    status: str  # "cleaned", "skipped", "already_cleaned", "error"
    space_freed_mb: float = 0.0
    video_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CleanupSummary:
    """Summary of cleanup operation."""

    total_reels: int
    cleaned: int
    skipped: int
    already_cleaned: int
    errors: int
    total_space_freed_mb: float
    results: List[CleanupResult]


def find_reels_for_cleanup(
    profile_path: Path,
    older_than_days: Optional[int] = None,
) -> List[Path]:
    """Find posted reels that can be cleaned up.

    Args:
        profile_path: Path to profile directory
        older_than_days: Only include reels older than this many days

    Returns:
        List of reel folder paths eligible for cleanup
    """
    from datetime import datetime, timedelta

    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    eligible_reels = []
    cutoff_date = None
    if older_than_days is not None:
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

    for year_dir in reels_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            posted_dir = month_dir / "posted"
            if not posted_dir.exists():
                continue

            for reel_dir in posted_dir.glob("*"):
                if not reel_dir.is_dir():
                    continue

                # Must have final.mp4 to be eligible (not already cleaned)
                video_path = reel_dir / "final.mp4"
                if not video_path.exists():
                    continue

                # Check age if specified
                if cutoff_date is not None:
                    metadata_path = reel_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, encoding="utf-8") as f:
                                metadata = json.load(f)
                            # Check upload date
                            upload_date_str = None
                            platform_status = metadata.get("platform_status", {})
                            if "instagram" in platform_status:
                                upload_date_str = platform_status["instagram"].get("uploaded_at")

                            if upload_date_str:
                                upload_date = datetime.fromisoformat(upload_date_str.replace("Z", "+00:00"))
                                if upload_date.replace(tzinfo=None) > cutoff_date:
                                    continue  # Too recent
                        except Exception:
                            pass  # Include if we can't parse date

                eligible_reels.append(reel_dir)

    return sorted(eligible_reels)


async def fetch_instagram_video_url(media_id: str, access_token: str) -> Optional[str]:
    """Fetch the video URL from Instagram API.

    Args:
        media_id: Instagram media ID
        access_token: Instagram access token

    Returns:
        Video URL or None if failed
    """
    import aiohttp

    try:
        url = f"https://graph.facebook.com/v21.0/{media_id}"
        params = {
            "fields": "media_url",
            "access_token": access_token,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("media_url")
    except Exception:
        pass

    return None


def cleanup_single_reel(
    reel_path: Path,
    dry_run: bool = False,
    video_url: Optional[str] = None,
) -> CleanupResult:
    """Clean up a single reel by removing video file and updating metadata.

    Args:
        reel_path: Path to reel folder
        dry_run: If True, don't actually delete files
        video_url: Instagram video URL to preserve in metadata

    Returns:
        CleanupResult with status and space freed
    """
    from datetime import datetime

    reel_name = reel_path.name
    video_path = reel_path / "final.mp4"
    metadata_path = reel_path / "metadata.json"
    debug_log_path = reel_path / "debug_log.txt"

    # Check if already cleaned
    if not video_path.exists():
        # Check metadata for cleanup record
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                if metadata.get("cleanup", {}).get("video_deleted"):
                    return CleanupResult(
                        reel_path=reel_path,
                        reel_name=reel_name,
                        status="already_cleaned",
                    )
            except Exception:
                pass

        return CleanupResult(
            reel_path=reel_path,
            reel_name=reel_name,
            status="skipped",
            error="No video file found",
        )

    # Get file size before deletion
    try:
        video_size_bytes = video_path.stat().st_size
        space_freed_mb = video_size_bytes / (1024 * 1024)
    except Exception:
        space_freed_mb = 0.0

    # Also count debug_log.txt if present
    debug_size_mb = 0.0
    if debug_log_path.exists():
        try:
            debug_size_mb = debug_log_path.stat().st_size / (1024 * 1024)
            space_freed_mb += debug_size_mb
        except Exception:
            pass

    if dry_run:
        return CleanupResult(
            reel_path=reel_path,
            reel_name=reel_name,
            status="would_clean",
            space_freed_mb=space_freed_mb,
            video_url=video_url,
        )

    try:
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        # Add video URL to platform status if provided
        if video_url:
            if "platform_status" not in metadata:
                metadata["platform_status"] = {}
            if "instagram" not in metadata["platform_status"]:
                metadata["platform_status"]["instagram"] = {}
            metadata["platform_status"]["instagram"]["video_url"] = video_url

        # Add cleanup record
        metadata["cleanup"] = {
            "cleaned_at": datetime.now().isoformat(),
            "video_deleted": True,
            "space_freed_mb": round(space_freed_mb, 2),
            "files_removed": ["final.mp4"],
        }

        # Also remove debug_log.txt
        if debug_log_path.exists():
            debug_log_path.unlink()
            metadata["cleanup"]["files_removed"].append("debug_log.txt")

        # Delete video file
        video_path.unlink()

        # Save updated metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return CleanupResult(
            reel_path=reel_path,
            reel_name=reel_name,
            status="cleaned",
            space_freed_mb=space_freed_mb,
            video_url=video_url,
        )

    except Exception as e:
        return CleanupResult(
            reel_path=reel_path,
            reel_name=reel_name,
            status="error",
            error=str(e),
        )


async def cleanup_reels(
    profile_path: Path,
    older_than_days: Optional[int] = None,
    dry_run: bool = False,
    fetch_video_urls: bool = True,
    progress_callback=None,
) -> CleanupSummary:
    """Clean up posted reels by removing video files.

    Args:
        profile_path: Path to profile directory
        older_than_days: Only clean reels older than this many days
        dry_run: If True, don't actually delete files
        fetch_video_urls: If True, fetch Instagram video URLs before cleanup
        progress_callback: Optional callback(current, total, reel_name)

    Returns:
        CleanupSummary with results
    """
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Find eligible reels
    reels = find_reels_for_cleanup(profile_path, older_than_days)

    if not reels:
        return CleanupSummary(
            total_reels=0,
            cleaned=0,
            skipped=0,
            already_cleaned=0,
            errors=0,
            total_space_freed_mb=0.0,
            results=[],
        )

    # Get access token for API calls if fetching URLs
    access_token = None
    if fetch_video_urls:
        access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")

    results = []
    total = len(reels)

    for idx, reel_path in enumerate(reels):
        if progress_callback:
            progress_callback(idx + 1, total, reel_path.name)

        # Try to fetch video URL from Instagram API
        video_url = None
        if fetch_video_urls and access_token:
            metadata_path = reel_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                    media_id = metadata.get("platform_status", {}).get("instagram", {}).get("media_id")
                    if media_id:
                        video_url = await fetch_instagram_video_url(media_id, access_token)
                except Exception:
                    pass

        # Clean up the reel
        result = cleanup_single_reel(reel_path, dry_run=dry_run, video_url=video_url)
        results.append(result)

    # Calculate summary
    cleaned = sum(1 for r in results if r.status in ("cleaned", "would_clean"))
    skipped = sum(1 for r in results if r.status == "skipped")
    already_cleaned = sum(1 for r in results if r.status == "already_cleaned")
    errors = sum(1 for r in results if r.status == "error")
    total_space_freed = sum(r.space_freed_mb for r in results)

    return CleanupSummary(
        total_reels=total,
        cleaned=cleaned,
        skipped=skipped,
        already_cleaned=already_cleaned,
        errors=errors,
        total_space_freed_mb=total_space_freed,
        results=results,
    )
