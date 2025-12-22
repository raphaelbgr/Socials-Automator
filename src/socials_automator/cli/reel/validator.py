"""Reel Validator - Validates and repairs reel artifacts.

Provides validation and repair functionality for reels:
- Validates required artifacts (video, metadata, captions, thumbnail)
- Repairs missing artifacts using CaptionService
- Reports status for CLI display

Used by the upload flow to ensure reels are complete before upload.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List

from rich.console import Console

from ..core.console import console as default_console


class ReelStatus(Enum):
    """Status of a reel after validation."""

    VALID = "valid"  # All artifacts present and valid
    REPAIRABLE = "repairable"  # Missing artifacts can be regenerated
    INVALID = "invalid"  # Missing video or unrepairable
    ALREADY_POSTED = "already_posted"  # Already uploaded to all platforms


@dataclass
class ArtifactCheck:
    """Result of checking a single artifact."""

    name: str
    path: Path
    exists: bool
    has_content: bool
    required: bool
    can_regenerate: bool = False
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.exists and self.has_content


@dataclass
class ValidationResult:
    """Result of validating a reel."""

    reel_path: Path
    reel_name: str
    status: ReelStatus
    artifacts: List[ArtifactCheck] = field(default_factory=list)
    topic: str = ""
    narration: str = ""
    duration: float = 0.0
    instagram_status: Optional[dict] = None
    error: Optional[str] = None

    @property
    def missing_required(self) -> List[ArtifactCheck]:
        return [a for a in self.artifacts if a.required and not a.is_valid]

    @property
    def missing_optional(self) -> List[ArtifactCheck]:
        return [a for a in self.artifacts if not a.required and not a.is_valid]

    @property
    def can_repair(self) -> bool:
        """Check if all missing required artifacts can be regenerated."""
        return all(a.can_regenerate for a in self.missing_required)


@dataclass
class RepairResult:
    """Result of repairing a reel."""

    reel_path: Path
    success: bool
    repaired: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ReelValidator:
    """Validates and repairs reel artifacts.

    Example:
        validator = ReelValidator(console)
        result = validator.validate(reel_path)
        if result.status == ReelStatus.REPAIRABLE:
            repair_result = await validator.repair(reel_path, profile_path)
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize validator.

        Args:
            console: Rich console for output. Uses default if not provided.
        """
        self.console = console or default_console

    def validate(self, reel_path: Path) -> ValidationResult:
        """Validate a reel folder.

        Checks for required artifacts and determines if reel is valid,
        repairable, or invalid.

        Args:
            reel_path: Path to reel folder.

        Returns:
            ValidationResult with status and artifact details.
        """
        reel_name = reel_path.name
        result = ValidationResult(
            reel_path=reel_path,
            reel_name=reel_name,
            status=ReelStatus.VALID,
        )

        # Load metadata if exists
        metadata = {}
        metadata_path = reel_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                result.topic = metadata.get("topic", "")
                result.narration = metadata.get("narration", "")
                result.duration = metadata.get("duration_seconds", 0)
                result.instagram_status = metadata.get("platform_status", {}).get("instagram")
            except Exception:
                pass

        # Check required artifacts
        artifacts_to_check = [
            ("final.mp4", True, False),  # (name, required, can_regenerate)
            ("metadata.json", True, True),
            ("caption.txt", True, True),
            ("caption+hashtags.txt", True, True),
        ]

        for name, required, can_regen in artifacts_to_check:
            artifact_path = reel_path / name
            exists = artifact_path.exists()
            has_content = False

            if exists:
                if name.endswith(".txt"):
                    try:
                        content = artifact_path.read_text(encoding="utf-8").strip()
                        has_content = len(content) > 10
                    except Exception:
                        has_content = False
                elif name.endswith(".json"):
                    try:
                        with open(artifact_path, encoding="utf-8") as f:
                            data = json.load(f)
                        has_content = bool(data)
                    except Exception:
                        has_content = False
                elif name.endswith(".mp4"):
                    try:
                        has_content = artifact_path.stat().st_size > 1000
                    except Exception:
                        has_content = False
                else:
                    has_content = True

            result.artifacts.append(ArtifactCheck(
                name=name,
                path=artifact_path,
                exists=exists,
                has_content=has_content,
                required=required,
                can_regenerate=can_regen,
            ))

        # Check thumbnail (jpg or png)
        thumbnail_jpg = reel_path / "thumbnail.jpg"
        thumbnail_png = reel_path / "thumbnail.png"
        thumbnail_exists = thumbnail_jpg.exists() or thumbnail_png.exists()
        thumbnail_path = thumbnail_jpg if thumbnail_jpg.exists() else thumbnail_png

        result.artifacts.append(ArtifactCheck(
            name="thumbnail.jpg/png",
            path=thumbnail_path,
            exists=thumbnail_exists,
            has_content=thumbnail_exists,
            required=True,
            can_regenerate=True,  # Can regenerate from video
        ))

        # Determine status
        video_artifact = next((a for a in result.artifacts if a.name == "final.mp4"), None)

        if video_artifact and not video_artifact.is_valid:
            # No video = invalid, cannot repair
            result.status = ReelStatus.INVALID
            result.error = "Missing video file (cannot repair)"
        elif result.missing_required:
            if result.can_repair:
                result.status = ReelStatus.REPAIRABLE
            else:
                result.status = ReelStatus.INVALID
                result.error = "Missing artifacts that cannot be regenerated"
        else:
            # Check if already posted
            if result.instagram_status and result.instagram_status.get("uploaded"):
                result.status = ReelStatus.ALREADY_POSTED
            else:
                result.status = ReelStatus.VALID

        return result

    async def repair(
        self,
        reel_path: Path,
        profile_path: Path,
        dry_run: bool = False,
    ) -> RepairResult:
        """Repair a reel by regenerating missing artifacts.

        Args:
            reel_path: Path to reel folder.
            profile_path: Path to profile folder (for hashtags).
            dry_run: If True, don't actually regenerate.

        Returns:
            RepairResult with list of repaired/failed artifacts.
        """
        from ...services.caption_service import CaptionService

        result = RepairResult(reel_path=reel_path, success=True)

        # First validate to see what's missing
        validation = self.validate(reel_path)

        if validation.status == ReelStatus.INVALID:
            result.success = False
            result.error = validation.error
            return result

        if validation.status == ReelStatus.VALID:
            # Nothing to repair
            return result

        # Load profile metadata for hashtags
        profile_handle = ""
        profile_hashtag = ""
        profile_hashtags = []
        profile_meta_path = profile_path / "metadata.json"
        if profile_meta_path.exists():
            try:
                with open(profile_meta_path, encoding="utf-8") as f:
                    profile_meta = json.load(f)
                profile_handle = profile_meta.get("instagram_handle", "")
                profile_name = profile_meta.get("name", "")
                if profile_name:
                    profile_hashtag = f"#{profile_name.replace('.', '').replace('_', '').title()}"
                profile_hashtags = profile_meta.get("hashtags", [])
            except Exception:
                pass

        # Get topic and narration from metadata
        topic = validation.topic or "Video content"
        narration = validation.narration

        if dry_run:
            for artifact in validation.missing_required:
                result.repaired.append(f"{artifact.name} (would regenerate)")
            return result

        # Regenerate missing artifacts
        caption_service = CaptionService()

        for artifact in validation.missing_required:
            try:
                if artifact.name == "caption.txt":
                    caption_result = await caption_service.generate_caption(
                        topic=topic,
                        narration=narration,
                        profile_handle=profile_handle,
                        profile_hashtag=profile_hashtag,
                    )
                    if caption_result.success:
                        caption_path = reel_path / "caption.txt"
                        caption_path.write_text(caption_result.caption, encoding="utf-8")
                        result.repaired.append("caption.txt")
                    else:
                        result.failed.append(f"caption.txt: {caption_result.error}")

                elif artifact.name == "caption+hashtags.txt":
                    # First ensure caption.txt exists
                    caption_path = reel_path / "caption.txt"
                    if not caption_path.exists():
                        # Generate caption first
                        caption_result = await caption_service.generate_caption(
                            topic=topic,
                            narration=narration,
                            profile_handle=profile_handle,
                            profile_hashtag=profile_hashtag,
                        )
                        if caption_result.success:
                            caption_path.write_text(caption_result.caption, encoding="utf-8")
                            if "caption.txt" not in result.repaired:
                                result.repaired.append("caption.txt")
                        else:
                            result.failed.append(f"caption+hashtags.txt: Caption generation failed")
                            continue

                    caption = caption_path.read_text(encoding="utf-8").strip()

                    # Get hashtags (limit to Instagram max)
                    from socials_automator.hashtag import INSTAGRAM_MAX_HASHTAGS
                    if profile_hashtags:
                        hashtag_str = " ".join(f"#{tag}" if not tag.startswith("#") else tag for tag in profile_hashtags[:INSTAGRAM_MAX_HASHTAGS])
                    else:
                        hashtag_result = await caption_service.generate_hashtags(
                            topic=topic,
                            caption=caption,
                        )
                        if hashtag_result.success:
                            hashtag_str = hashtag_result.hashtag_string
                        else:
                            hashtag_str = "#video #content #viral"

                    full_caption = f"{caption}\n\n{hashtag_str}"
                    output_path = reel_path / "caption+hashtags.txt"
                    output_path.write_text(full_caption, encoding="utf-8")
                    result.repaired.append("caption+hashtags.txt")

                elif artifact.name == "thumbnail.jpg/png":
                    video_path = reel_path / "final.mp4"
                    if video_path.exists():
                        try:
                            from ...video.pipeline.thumbnail_generator import ThumbnailGenerator

                            generator = ThumbnailGenerator(
                                video_path=video_path,
                                output_path=reel_path / "thumbnail.jpg",
                                topic=topic,
                            )
                            generator.generate()
                            result.repaired.append("thumbnail.jpg")
                        except Exception as e:
                            result.failed.append(f"thumbnail: {e}")
                    else:
                        result.failed.append("thumbnail: No video to extract from")

                elif artifact.name == "metadata.json":
                    # Regenerate minimal metadata
                    metadata = {
                        "topic": topic or reel_path.name,
                        "created_at": None,
                    }

                    # Get duration from video
                    video_path = reel_path / "final.mp4"
                    if video_path.exists():
                        try:
                            import subprocess
                            proc = subprocess.run(
                                [
                                    "ffprobe", "-v", "error",
                                    "-show_entries", "format=duration",
                                    "-of", "default=noprint_wrappers=1:nokey=1",
                                    str(video_path)
                                ],
                                capture_output=True,
                                text=True,
                            )
                            if proc.returncode == 0:
                                metadata["duration_seconds"] = float(proc.stdout.strip())
                        except Exception:
                            pass

                    metadata_path = reel_path / "metadata.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                    result.repaired.append("metadata.json")

            except Exception as e:
                result.failed.append(f"{artifact.name}: {e}")

        result.success = len(result.failed) == 0
        return result

    def delete_invalid(self, reel_path: Path, dry_run: bool = False) -> bool:
        """Delete an invalid reel folder.

        Args:
            reel_path: Path to reel folder.
            dry_run: If True, don't actually delete.

        Returns:
            True if deleted (or would be deleted in dry_run).
        """
        if dry_run:
            return True

        try:
            shutil.rmtree(reel_path)
            return True
        except Exception:
            return False

    def display_validation(self, result: ValidationResult, verbose: bool = True) -> None:
        """Display validation result to console.

        Args:
            result: ValidationResult to display.
            verbose: If True, show all artifacts.
        """
        status_style = {
            ReelStatus.VALID: "[green][OK][/green]",
            ReelStatus.REPAIRABLE: "[yellow][REPAIR][/yellow]",
            ReelStatus.INVALID: "[red][INVALID][/red]",
            ReelStatus.ALREADY_POSTED: "[cyan][POSTED][/cyan]",
        }

        status_icon = status_style.get(result.status, "[?]")

        if verbose:
            for artifact in result.artifacts:
                if artifact.is_valid:
                    if artifact.name == "final.mp4":
                        try:
                            size_mb = artifact.path.stat().st_size / (1024 * 1024)
                            self.console.print(f"      [green][OK][/green] {artifact.name} ({size_mb:.1f} MB)")
                        except Exception:
                            self.console.print(f"      [green][OK][/green] {artifact.name}")
                    elif artifact.name.endswith(".txt"):
                        try:
                            chars = len(artifact.path.read_text(encoding="utf-8"))
                            self.console.print(f"      [green][OK][/green] {artifact.name} ({chars} chars)")
                        except Exception:
                            self.console.print(f"      [green][OK][/green] {artifact.name}")
                    else:
                        self.console.print(f"      [green][OK][/green] {artifact.name}")
                elif artifact.required:
                    if artifact.can_regenerate:
                        self.console.print(f"      [yellow][!][/yellow]  {artifact.name} [yellow](can repair)[/yellow]")
                    else:
                        self.console.print(f"      [red][X][/red]  {artifact.name} [red](REQUIRED)[/red]")
                else:
                    self.console.print(f"      [dim][ ][/dim]  {artifact.name} [dim](optional)[/dim]")

    def display_repair(self, result: RepairResult) -> None:
        """Display repair result to console."""
        if result.repaired:
            for name in result.repaired:
                self.console.print(f"      [cyan][REGEN][/cyan] {name}")

        if result.failed:
            for msg in result.failed:
                self.console.print(f"      [red][X][/red] {msg}")
