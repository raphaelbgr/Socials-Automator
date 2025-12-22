"""Artifact validation and regeneration for reels."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..core.console import console


@dataclass
class ArtifactStatus:
    """Status of a single artifact."""

    name: str
    path: Path
    required: bool
    exists: bool
    has_content: bool = True
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if artifact is valid."""
        if self.required and not self.exists:
            return False
        if self.exists and not self.has_content:
            return False
        return True


@dataclass
class AuditResult:
    """Result of artifact audit."""

    reel_path: Path
    artifacts: List[ArtifactStatus] = field(default_factory=list)
    regenerated: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if all required artifacts are valid."""
        return all(a.is_valid for a in self.artifacts if a.required)

    @property
    def missing_required(self) -> List[ArtifactStatus]:
        """Get list of missing required artifacts."""
        return [a for a in self.artifacts if a.required and not a.is_valid]

    @property
    def missing_optional(self) -> List[ArtifactStatus]:
        """Get list of missing optional artifacts."""
        return [a for a in self.artifacts if not a.required and not a.is_valid]


def audit_reel_artifacts(reel_path: Path) -> AuditResult:
    """Audit all artifacts in a reel folder.

    Checks for:
    - final.mp4 (required)
    - metadata.json (required)
    - caption.txt (required, must have content)
    - caption+hashtags.txt (required, must have content)
    - thumbnail.jpg or thumbnail.png (required)

    Args:
        reel_path: Path to reel folder

    Returns:
        AuditResult with status of all artifacts
    """
    result = AuditResult(reel_path=reel_path)

    # Define artifacts to check - ALL are required for upload
    artifacts_to_check = [
        ("final.mp4", True),  # (filename, required)
        ("metadata.json", True),
        ("caption.txt", True),
        ("caption+hashtags.txt", True),  # Required - Instagram needs hashtags
    ]

    for filename, required in artifacts_to_check:
        artifact_path = reel_path / filename
        exists = artifact_path.exists()

        # Check content for text files
        has_content = True
        if exists and filename.endswith(".txt"):
            content = artifact_path.read_text(encoding="utf-8").strip()
            has_content = len(content) > 10  # At least 10 chars

        # Check content for JSON files
        if exists and filename.endswith(".json"):
            try:
                with open(artifact_path, encoding="utf-8") as f:
                    data = json.load(f)
                has_content = bool(data)
            except (json.JSONDecodeError, Exception):
                has_content = False

        result.artifacts.append(ArtifactStatus(
            name=filename,
            path=artifact_path,
            required=required,
            exists=exists,
            has_content=has_content,
        ))

    # Check for thumbnail (jpg or png) - REQUIRED for Instagram cover image
    thumbnail_jpg = reel_path / "thumbnail.jpg"
    thumbnail_png = reel_path / "thumbnail.png"
    thumbnail_exists = thumbnail_jpg.exists() or thumbnail_png.exists()
    thumbnail_path = thumbnail_jpg if thumbnail_jpg.exists() else thumbnail_png

    result.artifacts.append(ArtifactStatus(
        name="thumbnail.jpg/png",
        path=thumbnail_path,
        required=True,  # Required - Instagram cover image
        exists=thumbnail_exists,
        has_content=thumbnail_exists,
    ))

    return result


def get_thumbnail_path(reel_path: Path) -> Optional[Path]:
    """Get the thumbnail path, checking for both jpg and png.

    Args:
        reel_path: Path to reel folder

    Returns:
        Path to thumbnail if exists, None otherwise
    """
    for ext in ["jpg", "png", "jpeg"]:
        path = reel_path / f"thumbnail.{ext}"
        if path.exists():
            return path
    return None


def regenerate_missing_artifacts(
    reel_path: Path,
    audit_result: AuditResult,
    profile_path: Optional[Path] = None,
) -> AuditResult:
    """Attempt to regenerate missing artifacts.

    Can regenerate:
    - caption.txt (from metadata or script)
    - caption+hashtags.txt (from metadata)
    - thumbnail.jpg (from video)
    - metadata.json (partial, from other files)

    Cannot regenerate:
    - final.mp4 (requires full pipeline)

    Args:
        reel_path: Path to reel folder
        audit_result: Previous audit result
        profile_path: Path to profile (for hashtags)

    Returns:
        Updated AuditResult
    """
    for artifact in audit_result.missing_required + audit_result.missing_optional:
        try:
            if artifact.name == "caption.txt":
                if _regenerate_caption(reel_path):
                    audit_result.regenerated.append(artifact.name)
                    artifact.exists = True
                    artifact.has_content = True
                else:
                    audit_result.failed.append(artifact.name)

            elif artifact.name == "caption+hashtags.txt":
                if _regenerate_caption_with_hashtags(reel_path, profile_path):
                    audit_result.regenerated.append(artifact.name)
                    artifact.exists = True
                    artifact.has_content = True
                else:
                    audit_result.failed.append(artifact.name)

            elif artifact.name == "thumbnail.jpg":
                if _regenerate_thumbnail(reel_path):
                    audit_result.regenerated.append(artifact.name)
                    artifact.exists = True
                    artifact.has_content = True
                else:
                    audit_result.failed.append(artifact.name)

            elif artifact.name == "metadata.json":
                if _regenerate_metadata(reel_path):
                    audit_result.regenerated.append(artifact.name)
                    artifact.exists = True
                    artifact.has_content = True
                else:
                    audit_result.failed.append(artifact.name)

            elif artifact.name == "final.mp4":
                # Cannot regenerate video - mark as failed
                artifact.error = "Video cannot be regenerated - requires full pipeline"
                audit_result.failed.append(artifact.name)

        except Exception as e:
            artifact.error = str(e)
            audit_result.failed.append(artifact.name)

    return audit_result


def _regenerate_caption(reel_path: Path, use_ai: bool = True) -> bool:
    """Regenerate caption.txt from metadata, script, or AI.

    Args:
        reel_path: Path to reel folder
        use_ai: If True and simpler methods fail, use AI to generate caption

    Returns:
        True if caption was regenerated successfully
    """
    # Try to get caption from metadata
    metadata_path = reel_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            caption = metadata.get("caption", "")
            if not caption:
                # Try to build from topic and hook
                topic = metadata.get("topic", "")
                hook = metadata.get("hook", "")
                if topic:
                    caption = f"{hook}\n\n{topic}" if hook else topic

            if caption and len(caption) > 10:
                caption_path = reel_path / "caption.txt"
                caption_path.write_text(caption, encoding="utf-8")
                return True

        except Exception:
            pass

    # Try to get from script.json
    script_path = reel_path / "script.json"
    if script_path.exists():
        try:
            with open(script_path, encoding="utf-8") as f:
                script = json.load(f)

            hook = script.get("hook", {}).get("text", "")
            title = script.get("title", "")
            caption = f"{hook}\n\n{title}" if hook else title

            if caption and len(caption) > 10:
                caption_path = reel_path / "caption.txt"
                caption_path.write_text(caption, encoding="utf-8")
                return True

        except Exception:
            pass

    # If simpler methods failed and we have topic/narration, use AI
    if use_ai:
        topic = metadata.get("topic", "")
        narration = metadata.get("narration", "")

        if topic or narration:
            try:
                caption = _generate_caption_with_ai(topic, narration)
                if caption and len(caption) > 10:
                    caption_path = reel_path / "caption.txt"
                    caption_path.write_text(caption, encoding="utf-8")
                    console.print(f"      [cyan][AI][/cyan] Generated new caption ({len(caption)} chars)")
                    return True
            except Exception as e:
                console.print(f"      [dim]AI caption generation failed: {e}[/dim]")

    return False


def _generate_caption_with_ai(topic: str, narration: str = "") -> str:
    """Generate caption using AI based on topic and narration.

    Uses CaptionService for consistent caption generation.

    Args:
        topic: The video topic
        narration: Optional narration text for context

    Returns:
        Generated caption string
    """
    import asyncio
    from socials_automator.services.caption_service import CaptionService

    service = CaptionService()

    # Run async in sync context
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(service.generate_caption(
            topic=topic,
            narration=narration,
        ))
        if result.success:
            return result.caption
        else:
            raise Exception(result.error or "Caption generation failed")
    finally:
        loop.close()


def _regenerate_caption_with_hashtags(
    reel_path: Path,
    profile_path: Optional[Path] = None,
) -> bool:
    """Regenerate caption+hashtags.txt from caption and profile hashtags.

    If caption.txt doesn't exist or is empty, uses AI to generate one.
    If no hashtags are configured, uses AI to generate relevant hashtags.
    """
    # First ensure caption.txt exists
    caption_path = reel_path / "caption.txt"
    if not caption_path.exists() or caption_path.stat().st_size < 10:
        if not _regenerate_caption(reel_path, use_ai=True):
            return False

    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption or len(caption) < 10:
        return False

    # Get topic from metadata for hashtag generation
    topic = ""
    metadata_path = reel_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            topic = metadata.get("topic", "")
        except Exception:
            pass

    # Get hashtags from profile or metadata
    hashtags = []

    # Try profile metadata
    if profile_path:
        profile_meta_path = profile_path / "metadata.json"
        if profile_meta_path.exists():
            try:
                with open(profile_meta_path, encoding="utf-8") as f:
                    profile_meta = json.load(f)
                hashtags = profile_meta.get("hashtags", [])
            except Exception:
                pass

    # Try reel metadata
    if not hashtags:
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                hashtags = metadata.get("hashtags", [])
            except Exception:
                pass

    # If still no hashtags and we have topic, generate with AI
    if not hashtags and topic:
        try:
            hashtags = _generate_hashtags_with_ai(topic, caption)
            if hashtags:
                console.print(f"      [cyan][AI][/cyan] Generated {len(hashtags)} hashtags")
        except Exception as e:
            console.print(f"      [dim]AI hashtag generation failed: {e}[/dim]")

    # Build caption with hashtags (limit to Instagram max)
    from socials_automator.hashtag import INSTAGRAM_MAX_HASHTAGS
    if hashtags:
        hashtag_str = " ".join(f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags[:INSTAGRAM_MAX_HASHTAGS])
        full_caption = f"{caption}\n\n{hashtag_str}"
    else:
        full_caption = caption

    # Write file
    output_path = reel_path / "caption+hashtags.txt"
    output_path.write_text(full_caption, encoding="utf-8")
    return True


def _generate_hashtags_with_ai(topic: str, caption: str = "") -> list[str]:
    """Generate relevant hashtags using AI.

    Uses CaptionService for consistent hashtag generation.

    Args:
        topic: The video topic
        caption: Optional caption for more context

    Returns:
        List of hashtag strings (without # prefix)
    """
    import asyncio
    from socials_automator.services.caption_service import CaptionService

    service = CaptionService()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(service.generate_hashtags(
            topic=topic,
            caption=caption,
        ))
        if result.success:
            return result.hashtags
        else:
            return []
    finally:
        loop.close()


def _regenerate_thumbnail(reel_path: Path) -> bool:
    """Regenerate thumbnail from video."""
    video_path = reel_path / "final.mp4"
    if not video_path.exists():
        return False

    # Get topic from metadata for thumbnail text
    topic = ""
    metadata_path = reel_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            topic = metadata.get("topic", "")
        except Exception:
            pass

    try:
        from socials_automator.video.pipeline.thumbnail_generator import ThumbnailGenerator

        generator = ThumbnailGenerator(
            video_path=video_path,
            output_path=reel_path / "thumbnail.jpg",
            topic=topic,
        )
        generator.generate()
        return True

    except Exception:
        return False


def _regenerate_metadata(reel_path: Path) -> bool:
    """Regenerate minimal metadata.json from available files."""
    metadata = {}

    # Try to extract info from folder name
    folder_name = reel_path.name
    # Format: DD-NNN-topic-slug
    parts = folder_name.split("-", 2)
    if len(parts) >= 3:
        topic_slug = parts[2].replace("-", " ").title()
        metadata["topic"] = topic_slug

    # Try to get caption
    caption_path = reel_path / "caption.txt"
    if caption_path.exists():
        metadata["caption"] = caption_path.read_text(encoding="utf-8").strip()

    # Try to get video duration
    video_path = reel_path / "final.mp4"
    if video_path.exists():
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                metadata["duration_seconds"] = float(result.stdout.strip())
        except Exception:
            pass

    if metadata:
        metadata_path = reel_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return True

    return False


def show_audit_result(audit_result: AuditResult, verbose: bool = False) -> None:
    """Display audit result to console."""
    if audit_result.is_valid and not audit_result.regenerated:
        if verbose:
            console.print(f"  [green][OK] All artifacts valid[/green]")
        return

    # Show status
    for artifact in audit_result.artifacts:
        if artifact.is_valid:
            if verbose:
                console.print(f"    [green][OK][/green] {artifact.name}")
        elif artifact.name in audit_result.regenerated:
            console.print(f"    [yellow][REGEN][/yellow] {artifact.name}")
        elif artifact.required:
            console.print(f"    [red][MISSING][/red] {artifact.name} (required)")
        else:
            console.print(f"    [dim][MISSING][/dim] {artifact.name} (optional)")

    # Summary
    if audit_result.regenerated:
        console.print(f"  [yellow]Regenerated {len(audit_result.regenerated)} artifact(s)[/yellow]")

    if audit_result.failed:
        for name in audit_result.failed:
            artifact = next((a for a in audit_result.artifacts if a.name == name), None)
            if artifact and artifact.error:
                console.print(f"  [red][X] {name}: {artifact.error}[/red]")
