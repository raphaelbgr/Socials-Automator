"""Scanner for posted content metadata.

Scans posted/*/metadata.json files to extract actual published content.
This is the source of truth for what has been posted to social platforms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("history")


@dataclass
class PostedContent:
    """Represents a piece of posted content."""

    topic: str
    folder_name: str
    posted_at: Optional[datetime] = None
    metadata_path: Optional[Path] = None
    content_type: str = "reel"  # reel, post, news

    # Optional metadata fields
    narration: Optional[str] = None
    hook_text: Optional[str] = None
    hook_type: Optional[str] = None
    headlines: list[str] = field(default_factory=list)  # For news


class PostedContentScanner:
    """Scans profile folders for posted content metadata."""

    def __init__(self, profile_path: Path):
        """Initialize scanner for a profile.

        Args:
            profile_path: Path to the profile directory.
        """
        self.profile_path = Path(profile_path)
        self.profile_name = self.profile_path.name

    def scan_reels(self, days: int = 30) -> list[PostedContent]:
        """Scan posted reels for content history.

        Args:
            days: How far back to scan (default 30 days).

        Returns:
            List of PostedContent from posted reels.
        """
        return self._scan_content_type("reels", days)

    def scan_posts(self, days: int = 30) -> list[PostedContent]:
        """Scan posted carousel posts for content history.

        Args:
            days: How far back to scan (default 30 days).

        Returns:
            List of PostedContent from posted carousel posts.
        """
        return self._scan_content_type("posts", days)

    def scan_all(self, days: int = 30) -> list[PostedContent]:
        """Scan all posted content types.

        Args:
            days: How far back to scan.

        Returns:
            Combined list of all posted content.
        """
        all_content = []
        all_content.extend(self.scan_reels(days))
        all_content.extend(self.scan_posts(days))
        return all_content

    def _scan_content_type(
        self,
        content_type: str,
        days: int,
    ) -> list[PostedContent]:
        """Scan a specific content type directory.

        Args:
            content_type: 'reels' or 'posts'.
            days: How far back to scan.

        Returns:
            List of PostedContent found.
        """
        content_dir = self.profile_path / content_type
        if not content_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        results = []

        try:
            # Find all metadata.json files in posted folders
            pattern = "**/posted/**/metadata.json"
            metadata_files = list(content_dir.glob(pattern))

            logger.debug(
                f"POSTED_SCAN | {self.profile_name}/{content_type} | "
                f"found {len(metadata_files)} metadata files"
            )

            for metadata_path in metadata_files:
                try:
                    content = self._parse_metadata(
                        metadata_path,
                        content_type,
                        cutoff,
                    )
                    if content:
                        results.append(content)
                except Exception as e:
                    logger.debug(f"Could not parse {metadata_path}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error scanning {content_type}: {e}")

        logger.info(
            f"POSTED_HISTORY | {self.profile_name}/{content_type} | "
            f"{len(results)} items in last {days} days"
        )

        return results

    def _parse_metadata(
        self,
        metadata_path: Path,
        content_type: str,
        cutoff: datetime,
    ) -> Optional[PostedContent]:
        """Parse a metadata.json file into PostedContent.

        Args:
            metadata_path: Path to metadata.json file.
            content_type: 'reels' or 'posts'.
            cutoff: Datetime cutoff for filtering old content.

        Returns:
            PostedContent if valid and recent, None otherwise.
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Check timestamp
        posted_at = None
        timestamp_str = (
            metadata.get("posted_at")
            or metadata.get("created_at")
            or metadata.get("platform_status", {}).get("instagram", {}).get("uploaded_at")
        )

        if timestamp_str:
            try:
                # Handle various ISO format variations
                timestamp_str = timestamp_str.replace("Z", "+00:00")
                posted_at = datetime.fromisoformat(timestamp_str)

                # Remove timezone for comparison
                if posted_at.tzinfo:
                    posted_at = posted_at.replace(tzinfo=None)

                if posted_at < cutoff:
                    return None  # Too old
            except (ValueError, TypeError):
                pass  # Include if can't parse date

        # Extract topic from various possible fields
        topic = (
            metadata.get("topic")
            or metadata.get("title")
            or metadata.get("script", {}).get("topic")
            or metadata_path.parent.name  # Folder name as fallback
        )

        if not topic:
            return None

        # Build PostedContent
        content = PostedContent(
            topic=topic,
            folder_name=metadata_path.parent.name,
            posted_at=posted_at,
            metadata_path=metadata_path,
            content_type=content_type.rstrip("s"),  # 'reels' -> 'reel'
        )

        # Extract optional fields
        content.narration = metadata.get("narration")
        content.hook_text = metadata.get("hook_text")
        content.hook_type = metadata.get("hook_type")

        # For news content, extract headlines
        if "news_brief" in metadata:
            stories = metadata.get("news_brief", {}).get("stories", [])
            content.headlines = [s.get("headline", "") for s in stories if s.get("headline")]

        return content

    def get_topics(self, days: int = 30) -> list[str]:
        """Get list of topic strings from posted content.

        Convenience method for backward compatibility.

        Args:
            days: How far back to scan.

        Returns:
            List of topic strings.
        """
        content = self.scan_reels(days)
        return [c.topic for c in content if c.topic]

    def get_headlines(self, days: int = 30) -> list[str]:
        """Get list of news headlines from posted content.

        Args:
            days: How far back to scan.

        Returns:
            List of headline strings.
        """
        content = self.scan_reels(days)
        headlines = []
        for c in content:
            headlines.extend(c.headlines)
        return headlines
