"""Stateless service for reel generation and upload."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.types import Result, Success, Failure, GenerationResult
from ..core.paths import get_output_dir, get_reel_folder_name, generate_post_id
from .params import ReelGenerationParams, ReelUploadParams


class ReelGeneratorService:
    """Stateless service for reel generation.

    All state is passed via params - no instance state.
    """

    async def generate(self, params: ReelGenerationParams) -> Result[GenerationResult]:
        """Generate a single reel.

        Stateless - all configuration passed via params.

        Args:
            params: Immutable generation parameters

        Returns:
            Result containing GenerationResult or Failure
        """
        from socials_automator.video.pipeline import VideoPipeline, setup_logging

        setup_logging()

        # Build progress callback
        def progress_callback(stage: str, progress: float, message: str) -> None:
            pass  # Display handled by pipeline's internal display

        # Create pipeline with params
        pipeline = VideoPipeline(
            voice=params.voice,
            voice_rate=params.voice_rate,
            voice_pitch=params.voice_pitch,
            text_ai=params.text_ai,
            video_matcher=params.video_matcher,
            subtitle_size=params.subtitle_size,
            subtitle_font=params.font,
            target_duration=params.target_duration,
            progress_callback=progress_callback,
            gpu_accelerate=params.gpu_accelerate,
            gpu_index=params.gpu_index,
        )

        try:
            # Determine output directory
            output_dir = params.output_dir
            if output_dir is None:
                output_dir = self._create_output_dir(params)

            # Generate post ID
            post_id = generate_post_id()

            # Run pipeline
            video_path = await pipeline.generate(
                profile_path=params.profile_path,
                output_dir=output_dir,
                post_id=post_id,
            )

            if video_path is None or not video_path.exists():
                return Failure("Pipeline completed but no video was generated")

            # Rename folder with topic slug if applicable
            video_path = self._rename_with_topic_slug(video_path, params)

            # Audit and regenerate missing artifacts
            reel_folder = video_path.parent if video_path.is_file() else video_path
            audit_result = self._audit_and_fix_artifacts(reel_folder, params.profile_path)

            if not audit_result.is_valid:
                missing = [a.name for a in audit_result.missing_required]
                return Failure(f"Missing required artifacts: {', '.join(missing)}")

            # Get actual duration from metadata
            duration = self._get_video_duration(video_path)

            return Success(GenerationResult(
                success=True,
                output_path=video_path,
                duration_seconds=duration,
                metadata={
                    "post_id": post_id,
                    "regenerated_artifacts": audit_result.regenerated,
                },
            ))

        except Exception as e:
            return Failure(str(e))

    async def dry_run(self, params: ReelGenerationParams) -> Result[dict]:
        """Run first few pipeline steps without full generation.

        Args:
            params: Immutable generation parameters

        Returns:
            Result containing step results or Failure
        """
        from socials_automator.video.pipeline import (
            VideoPipeline,
            setup_logging,
            ProfileMetadata,
            PipelineContext,
        )
        import tempfile

        setup_logging()

        pipeline = VideoPipeline(
            voice=params.voice,
            voice_rate=params.voice_rate,
            voice_pitch=params.voice_pitch,
            text_ai=params.text_ai,
            video_matcher=params.video_matcher,
            subtitle_size=params.subtitle_size,
            subtitle_font=params.font,
            target_duration=params.target_duration,
            gpu_accelerate=params.gpu_accelerate,
            gpu_index=params.gpu_index,
        )

        try:
            profile = ProfileMetadata.from_file(params.profile_path / "metadata.json")
            temp_dir = Path(tempfile.mkdtemp())

            context = PipelineContext(
                profile=profile,
                post_id="dry-run-test",
                output_dir=temp_dir / "output",
                temp_dir=temp_dir,
            )

            results = {}

            # Step 1: Topic Selection
            context = await pipeline.steps[0].execute(context)
            results["topic"] = {
                "topic": context.topic.topic,
                "pillar": context.topic.pillar_name,
            }

            # Step 2: Research
            context = await pipeline.steps[1].execute(context)
            results["research"] = {
                "key_points": len(context.research.key_points),
            }

            # Step 3: Script Planning
            context = await pipeline.steps[2].execute(context)
            results["script"] = {
                "title": context.script.title,
                "segments": len(context.script.segments),
                "duration": context.script.total_duration,
            }

            return Success(results)

        except Exception as e:
            return Failure(str(e))

    def _create_output_dir(self, params: ReelGenerationParams) -> Path:
        """Create output directory for reel."""
        now = datetime.now()
        base_dir = get_output_dir(params.profile_path, "reels", "generated", now)
        base_dir.mkdir(parents=True, exist_ok=True)

        folder_name = get_reel_folder_name(base_dir, "reel", now)
        output_dir = base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _rename_with_topic_slug(self, video_path: Path, params: ReelGenerationParams) -> Path:
        """Rename output folder with topic slug from metadata."""
        import re
        import shutil

        if params.output_dir is not None:
            # Don't rename if custom output dir was specified
            return video_path

        metadata_path = video_path.parent / "metadata.json"
        if not metadata_path.exists():
            return video_path

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            topic = metadata.get("topic", "")
            if not topic:
                return video_path

            # Create slug from topic
            slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50].strip("-")

            # Get current folder parts
            current_folder = video_path.parent
            folder_name = current_folder.name

            # Extract day and number from current folder (e.g., "17-003-reel")
            match = re.match(r"^(\d{2}-\d{3})-", folder_name)
            if not match:
                return video_path

            prefix = match.group(1)
            new_folder_name = f"{prefix}-{slug}"
            new_folder_path = current_folder.parent / new_folder_name

            if current_folder != new_folder_path:
                shutil.move(str(current_folder), str(new_folder_path))
                return new_folder_path / video_path.name

        except Exception:
            pass  # Keep original path on any error

        return video_path

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration from metadata."""
        metadata_path = video_path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                return float(metadata.get("duration_seconds", 60))
            except Exception:
                pass
        return 60.0

    def _audit_and_fix_artifacts(self, reel_path: Path, profile_path: Path):
        """Audit artifacts and regenerate missing ones."""
        from .artifacts import (
            audit_reel_artifacts,
            regenerate_missing_artifacts,
            show_audit_result,
        )
        from ..core.console import console

        console.print("\n[bold]Artifact Audit[/bold]")

        # Run audit
        audit_result = audit_reel_artifacts(reel_path)

        # Regenerate missing artifacts
        if audit_result.missing_required or audit_result.missing_optional:
            console.print("  [yellow]Regenerating missing artifacts...[/yellow]")
            audit_result = regenerate_missing_artifacts(
                reel_path, audit_result, profile_path
            )

        # Show results
        show_audit_result(audit_result, verbose=True)

        return audit_result


class ReelUploaderService:
    """Stateless service for reel upload to Instagram."""

    async def upload_all(self, params: ReelUploadParams) -> Result[list]:
        """Upload all pending reels.

        Args:
            params: Immutable upload parameters

        Returns:
            Result containing list of upload results or Failure
        """
        # Find pending reels
        pending = self._find_pending_reels(params)
        if not pending:
            return Failure("No pending reels found")

        # Limit to one if requested
        if params.post_one:
            pending = pending[:1]

        total = len(pending)

        # Upload each reel
        results = []
        for index, reel_path in enumerate(pending, 1):
            result = await self._upload_single(reel_path, params, index, total)
            results.append(result)

        return Success(results)

    async def upload_single(self, params: ReelUploadParams) -> Result[dict]:
        """Upload a specific reel by ID.

        Args:
            params: Immutable upload parameters (must have reel_id)

        Returns:
            Result containing upload result or Failure
        """
        if not params.reel_id:
            return Failure("No reel ID specified")

        reel_path = self._find_reel_by_id(params)
        if reel_path is None:
            return Failure(f"Reel not found: {params.reel_id}")

        result = await self._upload_single(reel_path, params, 1, 1)
        return Success(result)

    def _find_pending_reels(self, params: ReelUploadParams) -> list[Path]:
        """Find all pending reels for profile."""
        pending_paths = []

        # Check both generated and pending-post folders
        for status in ["generated", "pending-post"]:
            base_dir = params.profile_path / "reels"
            if not base_dir.exists():
                continue

            for year_dir in sorted(base_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                for month_dir in sorted(year_dir.iterdir()):
                    if not month_dir.is_dir():
                        continue
                    status_dir = month_dir / status
                    if not status_dir.exists():
                        continue
                    for reel_dir in sorted(status_dir.iterdir()):
                        if reel_dir.is_dir() and (reel_dir / "final.mp4").exists():
                            pending_paths.append(reel_dir)

        return pending_paths

    def _find_reel_by_id(self, params: ReelUploadParams) -> Optional[Path]:
        """Find a specific reel by ID."""
        all_reels = self._find_pending_reels(params)
        for reel_path in all_reels:
            if params.reel_id in reel_path.name:
                return reel_path
        return None

    async def _upload_single(
        self,
        reel_path: Path,
        params: ReelUploadParams,
        index: int = 1,
        total: int = 1,
    ) -> dict:
        """Upload a single reel to Instagram."""
        import shutil
        from datetime import datetime

        from socials_automator.instagram.models import InstagramConfig, InstagramProgress
        from socials_automator.instagram.client import InstagramClient
        from socials_automator.instagram.uploader import CloudinaryUploader
        from ..core.console import console

        def log(message: str, style: str = "dim") -> None:
            """Print timestamped log message."""
            ts = datetime.now().strftime("%H:%M:%S")
            console.print(f"  [{style}]{ts}[/{style}] {message}")

        result = {
            "path": reel_path,
            "success": False,
            "error": None,
        }

        if params.dry_run:
            result["success"] = True
            result["dry_run"] = True
            return result

        try:
            # Load Instagram config
            config = InstagramConfig.from_env()

            # Load reel metadata early for display
            metadata_path = reel_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Get video path for size display
            video_path = reel_path / "final.mp4"

            # Get reel info for header
            topic = metadata.get("topic", "Unknown topic")
            duration = metadata.get("duration_seconds", 0)
            file_size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0

            # Get folder location (e.g., "2025/12/pending-post")
            folder_parts = reel_path.parts
            try:
                reels_idx = folder_parts.index("reels")
                folder_location = "/".join(folder_parts[reels_idx + 1:])
            except ValueError:
                folder_location = str(reel_path)

            # Show reel info header
            console.print()
            console.print(f"  [bold white on blue] {index}/{total} [/bold white on blue] [bold cyan]{reel_path.name}[/bold cyan]")
            console.print(f"  [dim]Folder:[/dim] {folder_location}")
            console.print(f"  [dim]Topic:[/dim] {topic[:70]}{'...' if len(topic) > 70 else ''}")
            console.print(f"  [dim]Size:[/dim] {file_size_mb:.1f} MB | [dim]Duration:[/dim] {duration:.0f}s")
            console.print()

            # Audit and fix artifacts before upload
            from .artifacts import (
                audit_reel_artifacts,
                regenerate_missing_artifacts,
                show_audit_result,
            )

            log("[yellow]Auditing artifacts...[/yellow]")
            audit_result = audit_reel_artifacts(reel_path)

            # Regenerate missing artifacts if possible
            if audit_result.missing_required or audit_result.missing_optional:
                log("[yellow]Regenerating missing artifacts...[/yellow]")
                audit_result = regenerate_missing_artifacts(
                    reel_path, audit_result, params.profile_path
                )
                show_audit_result(audit_result)

            # Check if we can proceed
            if not audit_result.is_valid:
                missing = [a.name for a in audit_result.missing_required]
                result["error"] = f"Missing required artifacts: {', '.join(missing)}"
                console.print(f"  [red][X] Cannot upload - missing: {', '.join(missing)}[/red]")
                return result

            if audit_result.regenerated:
                log(f"[green][OK] Regenerated: {', '.join(audit_result.regenerated)}[/green]")
            else:
                log("[green][OK] All artifacts valid[/green]")

            # Get caption (prefer caption.txt, fallback to metadata)
            caption_path = reel_path / "caption.txt"
            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8")
            else:
                caption = metadata.get("caption", "")

            if not video_path.exists():
                result["error"] = "Video file not found"
                log(f"[red][X] Video file not found[/red]", "red")
                return result

            # Progress callback for Instagram client
            async def progress_callback(progress: InstagramProgress) -> None:
                log(f"[cyan]{progress.current_step}[/cyan]")

            # Step 1: Upload video to Cloudinary
            log("[yellow]Step 1/4:[/yellow] Uploading video to Cloudinary...")
            uploader = CloudinaryUploader(config)
            video_url = await uploader.upload_video_async(video_path)
            log("[green][OK] Video uploaded[/green]")

            # Step 2: Upload thumbnail if exists
            thumbnail_path = reel_path / "thumbnail.jpg"
            cover_url = None
            if thumbnail_path.exists():
                log("[yellow]Step 2/4:[/yellow] Uploading thumbnail...")
                cover_url = uploader.upload_image(thumbnail_path)
                log("[green][OK] Thumbnail uploaded[/green]")
            else:
                log("[dim]Step 2/4:[/dim] No thumbnail found, skipping")

            # Step 3: Create reel container and wait for processing
            log("[yellow]Step 3/4:[/yellow] Creating Instagram reel container...")
            client = InstagramClient(config, progress_callback=progress_callback)
            publish_result = await client.publish_reel(
                video_url=video_url,
                caption=caption,
                cover_url=cover_url,
            )

            # Step 4: Cleanup
            log("[yellow]Step 4/4:[/yellow] Cleaning up Cloudinary files...")
            await uploader.cleanup_async()
            log("[green][OK] Cleanup complete[/green]")

            if publish_result.success:
                result["success"] = True
                result["media_id"] = publish_result.media_id
                result["permalink"] = publish_result.permalink

                console.print()
                console.print(f"  [bold green][OK] Published successfully![/bold green]")
                if publish_result.permalink:
                    console.print(f"  [cyan]{publish_result.permalink}[/cyan]")
                console.print()

                # Move to posted folder
                self._move_to_posted(reel_path, params.profile_path)
                log("[dim]Moved to posted/ folder[/dim]")
            else:
                result["error"] = publish_result.error_message
                console.print()
                console.print(f"  [bold red][X] Upload failed[/bold red]")
                console.print(f"  [red]{publish_result.error_message}[/red]")
                console.print()

        except Exception as e:
            result["error"] = str(e)
            console.print()
            console.print(f"  [bold red][X] Error[/bold red]")
            console.print(f"  [red]{e}[/red]")
            console.print()

        return result

    def _move_to_posted(self, reel_path: Path, profile_path: Path) -> None:
        """Move reel folder to posted status."""
        import shutil

        # Determine new path by replacing status folder
        parts = list(reel_path.parts)
        for i, part in enumerate(parts):
            if part in ["generated", "pending-post"]:
                parts[i] = "posted"
                break

        new_path = Path(*parts)

        # Create parent directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move folder
        if reel_path.exists() and not new_path.exists():
            shutil.move(str(reel_path), str(new_path))
