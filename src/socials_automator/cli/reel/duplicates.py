"""Duplicate Resolver - Detects and merges duplicate reels.

Handles duplicate detection and resolution:
- Finds duplicates by Instagram media_id or permalink
- Merges metadata keeping latest/most complete data
- Keeps folder with better name (topic slug vs generic)
- Reports all actions for CLI display

Used by upload flow to clean up duplicates before upload.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console

from ..core.console import console as default_console


@dataclass
class ReelInfo:
    """Information about a single reel."""

    path: Path
    folder_name: str
    folder_type: str  # "generated", "pending-post", "posted"
    has_topic_slug: bool
    topic: str
    instagram_media_id: Optional[str]
    instagram_permalink: Optional[str]
    instagram_uploaded: bool
    instagram_uploaded_at: Optional[datetime]
    metadata: Dict[str, Any]
    has_video: bool
    has_caption: bool
    has_thumbnail: bool
    completeness_score: int = 0

    def __post_init__(self):
        # Calculate completeness score
        score = 0
        if self.has_video:
            score += 10
        if self.has_caption:
            score += 5
        if self.has_thumbnail:
            score += 3
        if self.topic and self.topic != "Unknown topic":
            score += 2
        if self.has_topic_slug:
            score += 2
        if self.instagram_uploaded:
            score += 5
        self.completeness_score = score


@dataclass
class DuplicateGroup:
    """A group of duplicate reels."""

    key: str  # media_id or permalink used to identify duplicates
    reels: List[ReelInfo] = field(default_factory=list)

    @property
    def best_reel(self) -> Optional[ReelInfo]:
        """Get the best reel to keep (highest completeness, prefer posted folder)."""
        if not self.reels:
            return None

        # Sort by: posted folder first, then completeness, then has_topic_slug
        sorted_reels = sorted(
            self.reels,
            key=lambda r: (
                r.folder_type == "posted",  # Prefer posted
                r.completeness_score,
                r.has_topic_slug,
                r.instagram_uploaded_at or datetime.min,
            ),
            reverse=True,
        )
        return sorted_reels[0]

    @property
    def duplicates_to_remove(self) -> List[ReelInfo]:
        """Get reels that should be removed (all except best)."""
        best = self.best_reel
        if not best:
            return []
        return [r for r in self.reels if r.path != best.path]


@dataclass
class MergeResult:
    """Result of merging two reels."""

    source_path: Path
    target_path: Path
    success: bool
    fields_merged: List[str] = field(default_factory=list)
    source_deleted: bool = False
    new_folder_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ResolutionResult:
    """Result of resolving all duplicates."""

    total_groups: int
    total_duplicates_removed: int
    merges: List[MergeResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DuplicateResolver:
    """Resolves duplicate reels across folders.

    Example:
        resolver = DuplicateResolver(console)
        groups = resolver.find_duplicates(profile_path)
        for group in groups:
            result = resolver.merge_group(group, dry_run=False)
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize resolver.

        Args:
            console: Rich console for output.
        """
        self.console = console or default_console

    def scan_reels(self, profile_path: Path) -> List[ReelInfo]:
        """Scan all reel folders and collect info.

        Scans generated/, pending-post/, and posted/ folders.

        Args:
            profile_path: Path to profile folder.

        Returns:
            List of ReelInfo for all found reels.
        """
        reels = []
        reels_base = profile_path / "reels"

        if not reels_base.exists():
            return reels

        for year_dir in reels_base.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue

                for folder_type in ["generated", "pending-post", "posted"]:
                    status_dir = month_dir / folder_type
                    if not status_dir.exists():
                        continue

                    for reel_dir in status_dir.iterdir():
                        if not reel_dir.is_dir():
                            continue

                        info = self._get_reel_info(reel_dir, folder_type)
                        if info:
                            reels.append(info)

        return reels

    def _get_reel_info(self, reel_path: Path, folder_type: str) -> Optional[ReelInfo]:
        """Get info about a single reel.

        Args:
            reel_path: Path to reel folder.
            folder_type: Type of folder (generated, pending-post, posted).

        Returns:
            ReelInfo or None if folder is invalid.
        """
        folder_name = reel_path.name

        # Check if folder name has topic slug (e.g., "18-003-topic-slug" vs "18-003-reel")
        parts = folder_name.split("-", 2)
        has_topic_slug = (
            len(parts) >= 3
            and parts[2] != "reel"
            and len(parts[2]) > 4
        )

        # Load metadata
        metadata = {}
        metadata_path = reel_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        # Extract Instagram info
        platform_status = metadata.get("platform_status", {})
        ig_status = platform_status.get("instagram", {})

        instagram_media_id = ig_status.get("media_id")
        instagram_permalink = ig_status.get("permalink")
        instagram_uploaded = ig_status.get("uploaded", False)
        instagram_uploaded_at = None

        if ig_status.get("uploaded_at"):
            try:
                uploaded_str = ig_status["uploaded_at"]
                instagram_uploaded_at = datetime.fromisoformat(uploaded_str.replace("Z", "+00:00"))
            except Exception:
                pass

        # Check files
        has_video = (reel_path / "final.mp4").exists()
        has_caption = (
            (reel_path / "caption+hashtags.txt").exists()
            or (reel_path / "caption.txt").exists()
        )
        has_thumbnail = (
            (reel_path / "thumbnail.jpg").exists()
            or (reel_path / "thumbnail.png").exists()
        )

        topic = metadata.get("topic", "")

        return ReelInfo(
            path=reel_path,
            folder_name=folder_name,
            folder_type=folder_type,
            has_topic_slug=has_topic_slug,
            topic=topic,
            instagram_media_id=instagram_media_id,
            instagram_permalink=instagram_permalink,
            instagram_uploaded=instagram_uploaded,
            instagram_uploaded_at=instagram_uploaded_at,
            metadata=metadata,
            has_video=has_video,
            has_caption=has_caption,
            has_thumbnail=has_thumbnail,
        )

    def find_duplicates(self, profile_path: Path) -> List[DuplicateGroup]:
        """Find duplicate reels by Instagram media_id or permalink.

        Args:
            profile_path: Path to profile folder.

        Returns:
            List of DuplicateGroup containing duplicate reels.
        """
        reels = self.scan_reels(profile_path)

        # Group by media_id first, then by permalink
        by_media_id: Dict[str, List[ReelInfo]] = {}
        by_permalink: Dict[str, List[ReelInfo]] = {}
        no_instagram: List[ReelInfo] = []

        for reel in reels:
            if reel.instagram_media_id:
                key = reel.instagram_media_id
                if key not in by_media_id:
                    by_media_id[key] = []
                by_media_id[key].append(reel)
            elif reel.instagram_permalink:
                key = reel.instagram_permalink
                if key not in by_permalink:
                    by_permalink[key] = []
                by_permalink[key].append(reel)
            else:
                no_instagram.append(reel)

        # Build duplicate groups (only groups with 2+ reels)
        groups = []

        for media_id, reel_list in by_media_id.items():
            if len(reel_list) > 1:
                groups.append(DuplicateGroup(
                    key=f"media_id:{media_id}",
                    reels=reel_list,
                ))

        for permalink, reel_list in by_permalink.items():
            if len(reel_list) > 1:
                # Check if these aren't already in a media_id group
                already_grouped = any(
                    r.instagram_media_id and r.instagram_media_id in by_media_id
                    for r in reel_list
                )
                if not already_grouped:
                    groups.append(DuplicateGroup(
                        key=f"permalink:{permalink}",
                        reels=reel_list,
                    ))

        return groups

    def merge_group(
        self,
        group: DuplicateGroup,
        dry_run: bool = False,
    ) -> List[MergeResult]:
        """Merge a duplicate group, keeping the best reel.

        Args:
            group: DuplicateGroup to merge.
            dry_run: If True, don't actually modify files.

        Returns:
            List of MergeResult for each duplicate removed.
        """
        results = []
        best = group.best_reel

        if not best:
            return results

        for duplicate in group.duplicates_to_remove:
            result = self._merge_reel(
                source=duplicate,
                target=best,
                dry_run=dry_run,
            )
            results.append(result)

        return results

    def _merge_reel(
        self,
        source: ReelInfo,
        target: ReelInfo,
        dry_run: bool = False,
    ) -> MergeResult:
        """Merge source reel into target and delete source.

        Merge strategy:
        - Keep latest/most complete metadata from both
        - Keep target folder, optionally rename if source has better name
        - Delete source folder after merge

        Args:
            source: Reel to merge from (will be deleted).
            target: Reel to merge into (will be kept).
            dry_run: If True, don't actually modify files.

        Returns:
            MergeResult with details of the merge.
        """
        result = MergeResult(
            source_path=source.path,
            target_path=target.path,
            success=True,
        )

        try:
            # Determine best folder name
            new_name = self._get_best_folder_name(source, target)
            if new_name != target.folder_name:
                result.new_folder_name = new_name

            if dry_run:
                result.fields_merged = ["metadata (would merge)"]
                result.source_deleted = True
                return result

            # Merge metadata
            merged_fields = self._merge_metadata(source, target)
            result.fields_merged = merged_fields

            # Copy missing files from source to target
            for filename in ["caption.txt", "caption+hashtags.txt", "thumbnail.jpg", "thumbnail.png"]:
                source_file = source.path / filename
                target_file = target.path / filename
                if source_file.exists() and not target_file.exists():
                    shutil.copy2(source_file, target_file)
                    result.fields_merged.append(f"file:{filename}")

            # Rename target folder if needed
            if new_name and new_name != target.folder_name:
                new_path = target.path.parent / new_name
                if not new_path.exists():
                    target.path.rename(new_path)
                    result.target_path = new_path

            # Delete source folder
            shutil.rmtree(source.path)
            result.source_deleted = True

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _merge_metadata(self, source: ReelInfo, target: ReelInfo) -> List[str]:
        """Merge metadata from source into target.

        Keeps latest data from both, preferring target for conflicts.

        Args:
            source: Source reel info.
            target: Target reel info.

        Returns:
            List of merged field names.
        """
        merged_fields = []
        target_meta = target.metadata.copy()
        source_meta = source.metadata

        # Fields to merge (prefer non-empty values)
        fields_to_check = [
            "topic", "title", "narration", "duration_seconds",
            "segments", "clips_used", "news_brief",
        ]

        for field in fields_to_check:
            source_val = source_meta.get(field)
            target_val = target_meta.get(field)

            # Keep source value if target is empty/missing
            if source_val and not target_val:
                target_meta[field] = source_val
                merged_fields.append(field)

        # Merge platform_status (keep uploaded status from both)
        source_platforms = source_meta.get("platform_status", {})
        target_platforms = target_meta.get("platform_status", {})

        for platform, status in source_platforms.items():
            if platform not in target_platforms:
                target_platforms[platform] = status
                merged_fields.append(f"platform_status.{platform}")
            elif status.get("uploaded") and not target_platforms[platform].get("uploaded"):
                target_platforms[platform] = status
                merged_fields.append(f"platform_status.{platform}")

        target_meta["platform_status"] = target_platforms

        # Merge artifacts status
        source_artifacts = source_meta.get("artifacts", {})
        target_artifacts = target_meta.get("artifacts", {})

        for artifact, status in source_artifacts.items():
            if artifact not in target_artifacts:
                target_artifacts[artifact] = status
            elif status.get("status") == "ok" and target_artifacts[artifact].get("status") != "ok":
                target_artifacts[artifact] = status

        if source_artifacts:
            target_meta["artifacts"] = target_artifacts

        # Save merged metadata
        metadata_path = target.path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(target_meta, f, indent=2, ensure_ascii=False, default=str)

        return merged_fields

    def _get_best_folder_name(self, reel1: ReelInfo, reel2: ReelInfo) -> str:
        """Get the best folder name from two reels.

        Prefers:
        1. Name with topic slug over generic "reel"
        2. Longer/more descriptive name

        Args:
            reel1: First reel.
            reel2: Second reel.

        Returns:
            Best folder name to use.
        """
        # Prefer name with topic slug
        if reel1.has_topic_slug and not reel2.has_topic_slug:
            return reel1.folder_name
        if reel2.has_topic_slug and not reel1.has_topic_slug:
            return reel2.folder_name

        # Both have or both don't have topic slug - prefer longer name
        if len(reel1.folder_name) > len(reel2.folder_name):
            return reel1.folder_name
        return reel2.folder_name

    def resolve_all(
        self,
        profile_path: Path,
        dry_run: bool = False,
    ) -> ResolutionResult:
        """Find and resolve all duplicates.

        Args:
            profile_path: Path to profile folder.
            dry_run: If True, don't actually modify files.

        Returns:
            ResolutionResult with summary.
        """
        groups = self.find_duplicates(profile_path)

        result = ResolutionResult(
            total_groups=len(groups),
            total_duplicates_removed=0,
        )

        for group in groups:
            merge_results = self.merge_group(group, dry_run=dry_run)
            result.merges.extend(merge_results)
            result.total_duplicates_removed += len([m for m in merge_results if m.success])

        return result

    def display_group(self, group: DuplicateGroup) -> None:
        """Display a duplicate group to console."""
        self.console.print(f"  [yellow]Duplicate group:[/yellow] {group.key[:50]}...")

        best = group.best_reel
        for reel in group.reels:
            is_best = reel.path == best.path if best else False
            status = "[green][KEEP][/green]" if is_best else "[red][DELETE][/red]"
            folder_info = f"{reel.folder_type}/{reel.folder_name}"

            self.console.print(f"    {status} {folder_info}")
            self.console.print(f"          [dim]Score: {reel.completeness_score} | Topic: {reel.has_topic_slug}[/dim]")

    def display_merge_result(self, result: MergeResult) -> None:
        """Display merge result to console."""
        if result.success:
            self.console.print(f"    [green][MERGE][/green] {result.source_path.name} -> {result.target_path.name}")
            if result.fields_merged:
                self.console.print(f"             [dim]Merged: {', '.join(result.fields_merged[:5])}[/dim]")
            if result.new_folder_name:
                self.console.print(f"             [dim]Renamed to: {result.new_folder_name}[/dim]")
        else:
            self.console.print(f"    [red][X][/red] Failed to merge: {result.error}")
