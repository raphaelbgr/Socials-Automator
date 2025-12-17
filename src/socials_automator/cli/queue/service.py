"""Stateless service for queue operations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class QueuedPost:
    """Information about a queued post."""

    path: Path
    status: str  # "generated" or "pending"
    folder: str
    topic: str
    slides: int
    year: str
    month: str


def find_all_queued_posts(profile_path: Path) -> List[QueuedPost]:
    """Find all posts in queue (generated and pending-post folders).

    Args:
        profile_path: Path to profile directory

    Returns:
        List of QueuedPost objects sorted by folder name
    """
    posts_dir = profile_path / "posts"
    all_posts = []

    if not posts_dir.exists():
        return []

    for year_dir in posts_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            # Check generated folder
            generated_dir = month_dir / "generated"
            if generated_dir.exists():
                for post_dir in generated_dir.iterdir():
                    post = _load_post_info(post_dir, "generated", year_dir.name, month_dir.name)
                    if post:
                        all_posts.append(post)

            # Check pending-post folder
            pending_dir = month_dir / "pending-post"
            if pending_dir.exists():
                for post_dir in pending_dir.iterdir():
                    post = _load_post_info(post_dir, "pending", year_dir.name, month_dir.name)
                    if post:
                        all_posts.append(post)

    # Sort by folder name
    all_posts.sort(key=lambda p: (p.year, p.month, p.folder))
    return all_posts


def find_generated_posts(profile_path: Path) -> List[QueuedPost]:
    """Find all posts in generated folders (ready to schedule).

    Args:
        profile_path: Path to profile directory

    Returns:
        List of QueuedPost objects sorted by folder name
    """
    posts_dir = profile_path / "posts"
    generated_posts = []

    if not posts_dir.exists():
        return []

    for year_dir in posts_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            generated_dir = month_dir / "generated"
            if generated_dir.exists():
                for post_dir in generated_dir.iterdir():
                    post = _load_post_info(post_dir, "generated", year_dir.name, month_dir.name)
                    if post:
                        generated_posts.append(post)

    # Sort by folder name
    generated_posts.sort(key=lambda p: (p.year, p.month, p.folder))
    return generated_posts


def schedule_post(post: QueuedPost) -> bool:
    """Move a post from generated to pending-post folder.

    Args:
        post: QueuedPost to schedule

    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate destination path
        pending_dir = post.path.parent.parent / "pending-post"
        pending_dir.mkdir(parents=True, exist_ok=True)

        dest_path = pending_dir / post.folder

        # Move folder
        shutil.move(str(post.path), str(dest_path))
        return True

    except Exception:
        return False


def _load_post_info(
    post_dir: Path,
    status: str,
    year: str,
    month: str,
) -> Optional[QueuedPost]:
    """Load post information from directory.

    Args:
        post_dir: Path to post directory
        status: Post status ("generated" or "pending")
        year: Year string
        month: Month string

    Returns:
        QueuedPost if valid, None otherwise
    """
    if not post_dir.is_dir():
        return None

    metadata_path = post_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        return QueuedPost(
            path=post_dir,
            status=status,
            folder=post_dir.name,
            topic=metadata.get("topic", "Unknown"),
            slides=metadata.get("slide_count", len(list(post_dir.glob("slide_*.jpg")))),
            year=year,
            month=month,
        )
    except Exception:
        return None
