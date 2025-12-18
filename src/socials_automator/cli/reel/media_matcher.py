"""Instagram media matcher utility.

Matches local reel folders with Instagram media by timestamp.
Used to recover missing media_id when uploads succeed but response fails.

Usage:
    from socials_automator.cli.reel.media_matcher import InstagramMediaMatcher

    matcher = InstagramMediaMatcher(profile_path, console)
    fixes = await matcher.fix_missing_media_ids(posted_reels)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from rich.console import Console


@dataclass
class MediaMatch:
    """Result of matching a folder to Instagram media."""

    folder_name: str
    media_id: str
    permalink: Optional[str]
    timestamp: datetime
    confidence: str  # "high", "medium", "low"


@dataclass
class MatchCandidate:
    """A potential match between folder and Instagram media."""

    folder_name: str
    folder_path: Path
    folder_upload_time: datetime
    media_id: str
    media_timestamp: datetime
    media_permalink: Optional[str]
    time_diff_seconds: float

    @property
    def confidence(self) -> str:
        """Determine match confidence based on time difference."""
        if self.time_diff_seconds < 60:  # Within 1 minute
            return "high"
        elif self.time_diff_seconds < 300:  # Within 5 minutes
            return "medium"
        else:
            return "low"


class InstagramMediaMatcher:
    """Matches local reel folders with Instagram media.

    Uses timestamp comparison to match folders missing media_id
    with actual Instagram media posts.

    Match Algorithm:
    1. Fetch recent Instagram media (reels only)
    2. For each folder missing media_id:
       - Get uploaded_at timestamp from metadata
       - Find Instagram media within time window
       - Prefer closest timestamp match
    3. Update metadata with matched media_id

    Time Windows:
    - High confidence: < 1 minute difference
    - Medium confidence: < 5 minutes difference
    - Low confidence: < 30 minutes difference (not auto-applied)
    """

    # Maximum time difference to consider a match (seconds)
    MAX_TIME_DIFF_AUTO = 300  # 5 minutes for auto-fix
    MAX_TIME_DIFF_SUGGEST = 1800  # 30 minutes for suggestions

    def __init__(self, profile_path: Path, console: Console):
        """Initialize matcher.

        Args:
            profile_path: Path to profile folder.
            console: Rich console for output.
        """
        self.profile_path = profile_path
        self.console = console
        self._instagram_media: Optional[list[dict]] = None

    async def fix_missing_media_ids(
        self,
        posted_reels: list[Any],  # ReelInfo from duplicates.py
        auto_apply: bool = True,
    ) -> dict[str, dict]:
        """Fix missing media_ids for posted reels.

        Args:
            posted_reels: List of ReelInfo objects from DuplicateResolver.
            auto_apply: If True, automatically update metadata for high/medium confidence matches.

        Returns:
            Dict mapping folder_name to fix info:
            {
                "folder_name": {
                    "media_id": "...",
                    "permalink": "...",
                    "confidence": "high|medium",
                }
            }
        """
        # Find folders missing media_id
        folders_missing = []
        for reel_info in posted_reels:
            ig_status = reel_info.metadata.get("platform_status", {}).get("instagram", {})
            if ig_status.get("uploaded") and not ig_status.get("media_id"):
                folders_missing.append(reel_info)

        if not folders_missing:
            return {}

        # Show header for media matching section
        self.console.print()
        self.console.print(f"  [bold]>>> MEDIA ID RECOVERY[/bold]")
        self.console.print(f"  [yellow]Found {len(folders_missing)} folder(s) missing Instagram media_id[/yellow]")

        # Fetch Instagram media
        self.console.print(f"  [dim]Fetching Instagram media...[/dim]")
        instagram_media = await self._fetch_instagram_media()
        if not instagram_media:
            self.console.print(f"  [red][X] Could not fetch Instagram media[/red]")
            return {}

        # Filter to reels only
        reels_only = [m for m in instagram_media if m.get("media_type") in ("VIDEO", "REELS")]
        self.console.print(f"  [dim]Found {len(reels_only)} reels on Instagram[/dim]")

        if not reels_only:
            self.console.print(f"  [dim]No reels found to match[/dim]")
            return {}

        # Match folders to media
        fixes = {}
        unmatched = []

        for reel_info in folders_missing:
            match = self._find_best_match(reel_info, instagram_media)

            if match and match.confidence in ("high", "medium"):
                if auto_apply:
                    success = self._apply_fix(reel_info.path, match)
                    if success:
                        fixes[reel_info.path.name] = {
                            "media_id": match.media_id,
                            "permalink": match.media_permalink,
                            "confidence": match.confidence,
                        }
                        # Show match result
                        conf_style = "green" if match.confidence == "high" else "yellow"
                        time_diff = int(match.time_diff_seconds)
                        self.console.print(
                            f"  [green][MATCH][/green] {reel_info.path.name[:50]}..."
                        )
                        self.console.print(
                            f"          -> media_id: [cyan]{match.media_id}[/cyan]"
                        )
                        self.console.print(
                            f"          -> confidence: [{conf_style}]{match.confidence}[/{conf_style}] "
                            f"(time diff: {time_diff}s)"
                        )
            elif match:
                # Low confidence - report but don't apply
                unmatched.append((reel_info, match))
            else:
                unmatched.append((reel_info, None))

        # Summary
        self.console.print()
        if fixes:
            self.console.print(f"  [green][OK] Recovered {len(fixes)} media ID(s)[/green]")
        if unmatched:
            low_conf = sum(1 for _, m in unmatched if m is not None)
            no_match = len(unmatched) - low_conf
            if low_conf > 0:
                self.console.print(f"  [dim]{low_conf} low-confidence match(es) skipped[/dim]")
            if no_match > 0:
                self.console.print(f"  [dim]{no_match} folder(s) could not be matched[/dim]")

        return fixes

    async def _fetch_instagram_media(self, limit: int = 100) -> list[dict]:
        """Fetch recent Instagram media.

        Args:
            limit: Maximum number of media items to fetch.

        Returns:
            List of media dicts with id, timestamp, permalink, media_type.
        """
        if self._instagram_media is not None:
            return self._instagram_media

        try:
            from socials_automator.platforms import PlatformRegistry

            # Load Instagram config from profile
            config = PlatformRegistry.load_config("instagram", self.profile_path)

            # Create client directly
            from socials_automator.instagram.client import InstagramAPIConfig, InstagramClient

            api_config = InstagramAPIConfig(
                access_token=config.access_token,
                instagram_user_id=config.user_id,
            )
            client = InstagramClient(api_config)

            # Fetch media
            media = await client.get_recent_media(limit=limit)
            self._instagram_media = media
            return media

        except Exception as e:
            self.console.print(f"  [red]Error fetching Instagram media: {e}[/red]")
            return []

    def _find_best_match(
        self,
        reel_info: Any,
        instagram_media: list[dict],
    ) -> Optional[MatchCandidate]:
        """Find the best matching Instagram media for a folder.

        Args:
            reel_info: ReelInfo object with path and metadata.
            instagram_media: List of Instagram media items.

        Returns:
            Best MatchCandidate if found, None otherwise.
        """
        # Get folder upload time from metadata
        ig_status = reel_info.metadata.get("platform_status", {}).get("instagram", {})
        uploaded_at_str = ig_status.get("uploaded_at")

        if not uploaded_at_str:
            return None

        try:
            # Parse ISO format timestamp
            folder_upload_time = datetime.fromisoformat(
                uploaded_at_str.replace("Z", "+00:00")
            )
            # Remove timezone for comparison (assume local time)
            if folder_upload_time.tzinfo:
                folder_upload_time = folder_upload_time.replace(tzinfo=None)
        except Exception:
            return None

        # Find matching media
        candidates = []
        for media in instagram_media:
            # Only match VIDEO/REELS type
            media_type = media.get("media_type", "")
            if media_type not in ("VIDEO", "REELS"):
                continue

            # Skip if already matched to another folder
            media_id = media.get("id")
            if not media_id:
                continue

            # Parse media timestamp
            media_timestamp_str = media.get("timestamp")
            if not media_timestamp_str:
                continue

            try:
                media_timestamp = datetime.fromisoformat(
                    media_timestamp_str.replace("Z", "+00:00")
                )
                if media_timestamp.tzinfo:
                    media_timestamp = media_timestamp.replace(tzinfo=None)
            except Exception:
                continue

            # Calculate time difference
            time_diff = abs((folder_upload_time - media_timestamp).total_seconds())

            # Only consider if within max window
            if time_diff <= self.MAX_TIME_DIFF_SUGGEST:
                candidates.append(MatchCandidate(
                    folder_name=reel_info.path.name,
                    folder_path=reel_info.path,
                    folder_upload_time=folder_upload_time,
                    media_id=media_id,
                    media_timestamp=media_timestamp,
                    media_permalink=media.get("permalink"),
                    time_diff_seconds=time_diff,
                ))

        if not candidates:
            return None

        # Sort by time difference (closest first)
        candidates.sort(key=lambda c: c.time_diff_seconds)
        return candidates[0]

    def _apply_fix(self, reel_path: Path, match: MatchCandidate) -> bool:
        """Apply fix by updating metadata.json.

        Args:
            reel_path: Path to reel folder.
            match: MatchCandidate with media_id to apply.

        Returns:
            True if updated successfully.
        """
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            # Update platform_status.instagram
            if "platform_status" not in metadata:
                metadata["platform_status"] = {}
            if "instagram" not in metadata["platform_status"]:
                metadata["platform_status"]["instagram"] = {}

            ig_status = metadata["platform_status"]["instagram"]
            ig_status["media_id"] = match.media_id
            if match.media_permalink:
                ig_status["permalink"] = match.media_permalink
            ig_status["matched_by"] = "timestamp"
            ig_status["match_confidence"] = match.confidence
            ig_status["match_time_diff_seconds"] = match.time_diff_seconds

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            return True

        except Exception as e:
            self.console.print(f"  [red]Error updating {reel_path.name}: {e}[/red]")
            return False


async def match_folder_to_instagram(
    folder_path: Path,
    profile_path: Path,
    console: Optional[Console] = None,
) -> Optional[MediaMatch]:
    """Standalone function to match a single folder to Instagram media.

    Convenience function for one-off matching.

    Args:
        folder_path: Path to reel folder.
        profile_path: Path to profile folder.
        console: Optional Rich console.

    Returns:
        MediaMatch if found, None otherwise.
    """
    from rich.console import Console as RichConsole

    if console is None:
        console = RichConsole(quiet=True)

    # Create a minimal reel_info-like object
    class MinimalReelInfo:
        def __init__(self, path: Path):
            self.path = path
            self.metadata = {}
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)

    reel_info = MinimalReelInfo(folder_path)
    matcher = InstagramMediaMatcher(profile_path, console)

    instagram_media = await matcher._fetch_instagram_media()
    if not instagram_media:
        return None

    match = matcher._find_best_match(reel_info, instagram_media)
    if match:
        return MediaMatch(
            folder_name=match.folder_name,
            media_id=match.media_id,
            permalink=match.media_permalink,
            timestamp=match.media_timestamp,
            confidence=match.confidence,
        )

    return None
