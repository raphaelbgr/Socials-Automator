"""Stateless service for reel generation and upload."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.types import Result, Success, Failure, GenerationResult
from ..core.paths import get_output_dir, get_reel_folder_name, generate_post_id
from .params import ReelGenerationParams, ReelUploadParams


def get_platform_status(metadata: dict) -> dict[str, dict[str, Any]]:
    """Get platform_status from metadata, initializing if needed.

    Returns dict like:
        {
            "instagram": {"uploaded": True, "uploaded_at": "...", "media_id": "..."},
            "tiktok": {"uploaded": False}
        }
    """
    return metadata.get("platform_status", {})


def is_platform_uploaded(metadata: dict, platform: str) -> bool:
    """Check if a platform has already been uploaded to."""
    status = get_platform_status(metadata)
    return status.get(platform, {}).get("uploaded", False)


def update_platform_status(
    metadata_path: Path,
    platform: str,
    success: bool,
    media_id: Optional[str] = None,
    permalink: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update platform_status in metadata file after upload attempt.

    Args:
        metadata_path: Path to metadata.json
        platform: Platform name (e.g., "instagram", "tiktok")
        success: Whether upload succeeded
        media_id: Media ID from platform (if success)
        permalink: URL to posted content (if success)
        error: Error message (if failed)
    """
    # Load existing metadata
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Initialize platform_status if needed
    if "platform_status" not in metadata:
        metadata["platform_status"] = {}

    # Update this platform's status
    now = datetime.now().isoformat()

    if success:
        metadata["platform_status"][platform] = {
            "uploaded": True,
            "uploaded_at": now,
            "media_id": media_id,
            "permalink": permalink,
        }
    else:
        metadata["platform_status"][platform] = {
            "uploaded": False,
            "last_attempt": now,
            "error": error,
        }

    # Write back
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


class ReelGeneratorService:
    """Stateless service for reel generation.

    All state is passed via params - no instance state.
    """

    async def generate(self, params: ReelGenerationParams) -> Result[GenerationResult]:
        """Generate a single reel.

        Stateless - all configuration passed via params.
        Automatically uses NewsPipeline for news profiles.

        Args:
            params: Immutable generation parameters

        Returns:
            Result containing GenerationResult or Failure
        """
        from socials_automator.video.pipeline import VideoPipeline, setup_logging
        from ..core.console import console

        setup_logging()

        # Build progress callback
        def progress_callback(stage: str, progress: float, message: str) -> None:
            pass  # Display handled by pipeline's internal display

        # Choose pipeline based on profile type
        if params.is_news_profile:
            pipeline = self._create_news_pipeline(params, progress_callback)
        else:
            pipeline = self._create_standard_pipeline(params, progress_callback)

        # Track output directory for cleanup on failure
        output_dir = None
        output_dir_created = False

        try:
            # Determine output directory
            output_dir = params.output_dir
            if output_dir is None:
                output_dir = self._create_output_dir(params)
                output_dir_created = True

            # Generate post ID
            post_id = generate_post_id()

            # Run pipeline
            video_path = await pipeline.generate(
                profile_path=params.profile_path,
                output_dir=output_dir,
                post_id=post_id,
            )

            if video_path is None or not video_path.exists():
                # Cleanup on failure
                self._cleanup_failed_output(output_dir, output_dir_created, console)
                return Failure("Pipeline completed but no video was generated")

            # Rename folder with topic slug if applicable
            video_path = self._rename_with_topic_slug(video_path, params)

            # Audit and regenerate missing artifacts
            reel_folder = video_path.parent if video_path.is_file() else video_path
            audit_result = self._audit_and_fix_artifacts(reel_folder, params.profile_path)

            if not audit_result.is_valid:
                missing = [a.name for a in audit_result.missing_required]
                # Cleanup on failure
                self._cleanup_failed_output(reel_folder, output_dir_created, console)
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
            # Cleanup on exception
            if output_dir:
                self._cleanup_failed_output(output_dir, output_dir_created, console)
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
            profile_path=params.profile_path,
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

    def _create_standard_pipeline(self, params: ReelGenerationParams, progress_callback):
        """Create standard video pipeline."""
        from socials_automator.video.pipeline import VideoPipeline

        return VideoPipeline(
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
            profile_path=params.profile_path,
        )

    def _create_news_pipeline(self, params: ReelGenerationParams, progress_callback):
        """Create news pipeline for news profiles."""
        from socials_automator.video.pipeline.news_orchestrator import NewsPipeline
        from socials_automator.news.models import NewsEdition

        # Parse edition if provided
        edition = None
        if params.news_edition:
            try:
                edition = NewsEdition(params.news_edition)
            except ValueError:
                pass  # Will auto-detect from time

        return NewsPipeline(
            voice=params.voice,
            voice_rate=params.voice_rate,
            voice_pitch=params.voice_pitch,
            text_ai=params.text_ai,
            subtitle_size=params.subtitle_size,
            subtitle_font=params.font,
            target_duration=params.target_duration,
            progress_callback=progress_callback,
            gpu_accelerate=params.gpu_accelerate,
            gpu_index=params.gpu_index,
            # News-specific
            edition=edition,
            story_count=params.news_story_count,
            max_news_age_hours=params.news_max_age_hours,
            profile_name=params.profile,  # For theme history tracking
            profile_path=params.profile_path,  # For profile-scoped data storage
        )

    def _create_output_dir(self, params: ReelGenerationParams) -> Path:
        """Create output directory for reel."""
        now = datetime.now()
        base_dir = get_output_dir(params.profile_path, "reels", "generated", now)
        base_dir.mkdir(parents=True, exist_ok=True)

        folder_name = get_reel_folder_name(base_dir, "reel", now)
        output_dir = base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _cleanup_failed_output(self, output_dir: Path, was_created: bool, console) -> None:
        """Clean up output directory after failed generation.

        Args:
            output_dir: Path to output directory to clean up.
            was_created: Whether the directory was created by this service.
            console: Rich console for status display.
        """
        import shutil

        if not output_dir or not output_dir.exists():
            return

        # Only cleanup if we created the directory
        if not was_created:
            return

        try:
            folder_name = output_dir.name
            shutil.rmtree(output_dir)
            console.print()
            console.print(f"[dim]{'-' * 60}[/dim]")
            console.print(f"[bold yellow]>>> CLEANUP[/bold yellow]")
            console.print(f"[yellow][!][/yellow] Removed failed output folder: [cyan]{folder_name}[/cyan]")
            console.print(f"[dim]{'-' * 60}[/dim]")
        except Exception as e:
            console.print(f"[red][X] Failed to cleanup: {e}[/red]")

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
    """Stateless service for reel upload to multiple platforms."""

    async def upload_all(self, params: ReelUploadParams) -> Result[list]:
        """Upload all pending reels to specified platforms.

        Includes pre-flight steps:
        1. Scan all folders for reels
        2. Detect and merge duplicates
        3. Validate and repair reels
        4. Clean up invalid reels
        5. Upload valid reels
        6. Move already-uploaded to /posted

        Args:
            params: Immutable upload parameters

        Returns:
            Result containing list of upload results or Failure
        """
        from ..core.console import console

        # =================================================================
        # PRE-FLIGHT: Scan, dedupe, validate, repair
        # =================================================================
        preflight_result = await self._run_preflight(params)

        if not preflight_result["reels_to_upload"]:
            if preflight_result["already_posted"]:
                # All reels were already posted - still run postflight verification
                console.print()
                console.print("[dim]All reels already uploaded to requested platforms.[/dim]")
                await self._run_postflight(params)
                return Success([])
            return Failure("No valid reels found after pre-flight checks")

        # Get the validated reels to upload
        pending = preflight_result["reels_to_upload"]

        # Limit to one if requested
        if params.post_one:
            pending = pending[:1]

        total = len(pending)

        # =================================================================
        # UPLOAD: Process each reel
        # =================================================================
        console.print()
        console.print(f"[bold]>>> UPLOAD[/bold]")

        results = []
        for index, reel_path in enumerate(pending, 1):
            result = await self._upload_single(reel_path, params, index, total)
            results.append(result)

        # =================================================================
        # POST-FLIGHT: Verify posted folders
        # =================================================================
        await self._run_postflight(params)

        return Success(results)

    async def _run_preflight(self, params: ReelUploadParams) -> dict:
        """Run pre-flight checks: scan, dedupe, validate, repair.

        When params.reel_id is set, runs a targeted preflight for just that reel.
        Otherwise runs full preflight scan on all folders.

        Args:
            params: Upload parameters.

        Returns:
            Dict with:
                - reels_to_upload: List of valid reel paths ready for upload
                - already_posted: List of reels moved to /posted (already uploaded)
                - repaired: List of repaired reel paths
                - deleted: List of deleted invalid reel paths
                - duplicates_merged: Number of duplicates merged
        """
        from ..core.console import console
        from .validator import ReelValidator, ReelStatus
        from .duplicates import DuplicateResolver

        result = {
            "reels_to_upload": [],
            "already_posted": [],
            "repaired": [],
            "deleted": [],
            "renamed": [],
            "duplicates_merged": 0,
        }

        validator = ReelValidator(console)
        resolver = DuplicateResolver(console)

        # =================================================================
        # TARGETED MODE: When reel_id is specified, only process that reel
        # =================================================================
        if params.reel_id:
            return await self._run_preflight_single(params, validator)

        # =================================================================
        # FULL MODE: Scan all folders
        # =================================================================
        console.print()
        console.print(f"[bold]>>> PRE-FLIGHT SCAN[/bold]")
        console.print(f"  Scanning generated/pending-post/posted folders...")

        all_reels = resolver.scan_reels(params.profile_path)
        console.print(f"  Found: [cyan]{len(all_reels)}[/cyan] reels across folders")

        if not all_reels:
            return result

        # =================================================================
        # Step 2: Detect and merge duplicates
        # =================================================================
        console.print()
        console.print(f"[bold]>>> DUPLICATE DETECTION[/bold]")

        duplicate_groups = resolver.find_duplicates(params.profile_path)

        if duplicate_groups:
            console.print(f"  [yellow][!] Found {len(duplicate_groups)} duplicate group(s)[/yellow]")
            console.print()

            for group in duplicate_groups:
                resolver.display_group(group)

                if not params.dry_run:
                    merge_results = resolver.merge_group(group, dry_run=False)
                    for mr in merge_results:
                        resolver.display_merge_result(mr)
                        if mr.success:
                            result["duplicates_merged"] += 1
                else:
                    console.print(f"    [dim](dry-run: would merge)[/dim]")

                console.print()
        else:
            console.print(f"  [green][OK][/green] No duplicates found")

        # =================================================================
        # Step 3: Normalize folder names
        # =================================================================
        console.print()
        console.print(f"[bold]>>> FOLDER NORMALIZATION[/bold]")

        # Scan ALL reel folders (not just pending) for normalization
        all_reel_paths = [r.path for r in resolver.scan_reels(params.profile_path)]
        renamed_count = 0

        for reel_path in all_reel_paths:
            if self._needs_folder_rename(reel_path.name):
                old_name = reel_path.name
                if not params.dry_run:
                    new_path = self._normalize_folder_name(reel_path, dry_run=False)
                    if new_path:
                        console.print(f"  [yellow][RENAME][/yellow] {old_name}")
                        console.print(f"       -> {new_path.name}")
                        renamed_count += 1
                        result["renamed"].append((old_name, new_path.name))
                else:
                    new_path = self._normalize_folder_name(reel_path, dry_run=True)
                    if new_path:
                        console.print(f"  [dim](dry-run: would rename {old_name} -> {new_path.name})[/dim]")
                        renamed_count += 1

        if renamed_count == 0:
            console.print(f"  [green][OK][/green] All folder names valid")

        # =================================================================
        # Step 4: Validate and repair reels
        # =================================================================
        console.print()
        console.print(f"[bold]>>> VALIDATION[/bold]")

        # Re-scan after normalization
        pending_reels = self._find_pending_reels(params)

        for reel_path in pending_reels:
            validation = validator.validate(reel_path)

            # Show reel info
            console.print()
            console.print(f"  [bold]{reel_path.name}[/bold]")

            if validation.status == ReelStatus.VALID:
                console.print(f"    [green][OK][/green] All artifacts valid")
                result["reels_to_upload"].append(reel_path)

            elif validation.status == ReelStatus.ALREADY_POSTED:
                console.print(f"    [cyan][POSTED][/cyan] Already uploaded to all platforms")

                # Move to /posted if not already there
                if not self._is_in_posted(reel_path):
                    if not params.dry_run:
                        new_path = self._move_to_posted(reel_path, params.profile_path)
                        if new_path:
                            console.print(f"    [dim]Moved to posted/[/dim]")
                            result["already_posted"].append(new_path)
                    else:
                        console.print(f"    [dim](dry-run: would move to posted/)[/dim]")
                        result["already_posted"].append(reel_path)
                else:
                    result["already_posted"].append(reel_path)

            elif validation.status == ReelStatus.REPAIRABLE:
                console.print(f"    [yellow][REPAIR][/yellow] Missing artifacts - attempting repair...")
                validator.display_validation(validation, verbose=True)

                if not params.dry_run:
                    repair_result = await validator.repair(
                        reel_path, params.profile_path, dry_run=False
                    )
                    validator.display_repair(repair_result)

                    if repair_result.success:
                        console.print(f"    [green][OK][/green] Repair successful")
                        result["repaired"].append(reel_path)
                        result["reels_to_upload"].append(reel_path)
                    else:
                        console.print(f"    [red][X][/red] Repair failed: {repair_result.error}")
                else:
                    console.print(f"    [dim](dry-run: would attempt repair)[/dim]")

            elif validation.status == ReelStatus.INVALID:
                console.print(f"    [red][INVALID][/red] {validation.error}")
                validator.display_validation(validation, verbose=True)

                # Delete invalid reel
                if not params.dry_run:
                    if validator.delete_invalid(reel_path, dry_run=False):
                        console.print(f"    [yellow][DELETE][/yellow] Removed invalid reel folder")
                        result["deleted"].append(reel_path)
                else:
                    console.print(f"    [dim](dry-run: would delete)[/dim]")
                    result["deleted"].append(reel_path)

        # =================================================================
        # Summary
        # =================================================================
        console.print()
        console.print(f"[bold]>>> PRE-FLIGHT SUMMARY[/bold]")
        console.print(f"  Ready to upload: [green]{len(result['reels_to_upload'])}[/green]")

        # Count reels already in posted/ folder (for display and flow control)
        posted_count = len([r for r in all_reels if r.folder_type == "posted"])
        if result["already_posted"]:
            console.print(f"  Already posted:  [cyan]{len(result['already_posted'])}[/cyan]")
        elif posted_count > 0 and not result["reels_to_upload"]:
            # All reels already in posted/ folder
            console.print(f"  In posted/:      [cyan]{posted_count}[/cyan]")
            result["already_posted"] = [r.path for r in all_reels if r.folder_type == "posted"]

        if result["repaired"]:
            console.print(f"  Repaired:        [yellow]{len(result['repaired'])}[/yellow]")
        if result["renamed"]:
            console.print(f"  Renamed:         [yellow]{len(result['renamed'])}[/yellow]")
        if result["deleted"]:
            console.print(f"  Deleted:         [red]{len(result['deleted'])}[/red]")
        if result["duplicates_merged"]:
            console.print(f"  Duplicates:      [dim]{result['duplicates_merged']} merged[/dim]")

        return result

    async def _run_postflight(self, params: ReelUploadParams) -> dict:
        """Run post-flight verification on posted folders.

        Checks all folders in posted/ to ensure:
        - Instagram metadata present (media_id, permalink, uploaded_at)
        - Folder name is normalized (no generic names, no timestamps)
        - All required artifacts present

        Auto-fixes issues when possible:
        - Matches missing media_id by fetching Instagram media and comparing timestamps
        - Extracts topic from script.json or news_brief if metadata has generic topic
        - Renames folders to proper format
        - Regenerates missing artifacts

        Args:
            params: Upload parameters.

        Returns:
            Dict with:
                - verified: Number of folders verified OK
                - issues_found: List of issues found
                - fixed: List of issues that were fixed
        """
        from ..core.console import console
        from .validator import ReelValidator, ReelStatus
        from .duplicates import DuplicateResolver
        from .media_matcher import InstagramMediaMatcher

        result = {
            "verified": 0,
            "issues_found": [],
            "fixed": [],
        }

        resolver = DuplicateResolver(console)

        console.print()
        console.print(f"[bold]>>> POST-FLIGHT VERIFICATION[/bold]")
        console.print(f"  Checking posted/ folders...")

        # Get all posted reels
        posted_reels = [
            r for r in resolver.scan_reels(params.profile_path)
            if r.folder_type == "posted"
        ]

        if not posted_reels:
            console.print(f"  [dim]No posted reels to verify[/dim]")
            return result

        # =================================================================
        # Step 1: Auto-fix missing media_ids using Instagram API
        # =================================================================
        media_id_fixes = {}
        if not params.dry_run:
            matcher = InstagramMediaMatcher(params.profile_path, console)
            media_id_fixes = await matcher.fix_missing_media_ids(posted_reels)

        issues_count = 0

        for reel_info in posted_reels:
            reel_path = reel_info.path
            folder_name = reel_path.name
            issues = []
            fixed_items = []

            # Check 1: Instagram metadata present
            ig_status = reel_info.metadata.get("platform_status", {}).get("instagram", {})

            # Check if we just fixed this folder's media_id (already reported by matcher)
            if folder_name in media_id_fixes:
                result["fixed"].append(f"Recovered media_id: {folder_name}")
                # Reload ig_status after fix
                try:
                    with open(reel_path / "metadata.json", encoding="utf-8") as f:
                        updated_meta = json.load(f)
                    ig_status = updated_meta.get("platform_status", {}).get("instagram", {})
                except Exception:
                    pass

            if not ig_status.get("uploaded"):
                issues.append("missing Instagram upload status")
            elif not ig_status.get("media_id"):
                issues.append("missing Instagram media_id")
            elif not ig_status.get("permalink"):
                issues.append("missing Instagram permalink")

            # Check 2: Folder name normalized
            if self._needs_folder_rename(folder_name):
                old_name = folder_name

                # First, try to fix generic topic in metadata
                if not params.dry_run:
                    better_topic = self._extract_better_topic(reel_path, reel_info.metadata)
                    if better_topic:
                        self._update_metadata_topic(reel_path, better_topic)
                        fixed_items.append(f"updated topic: {better_topic[:30]}...")

                    # Now try to rename with updated metadata
                    new_path = self._normalize_folder_name(reel_path, dry_run=False)
                    if new_path:
                        fixed_items.append(f"renamed: {old_name} -> {new_path.name}")
                        result["fixed"].append(f"Renamed {old_name}")
                        reel_path = new_path  # Update path for artifact check
                    elif not better_topic:
                        # Couldn't fix - truly generic topic
                        issues.append(f"folder has generic name (topic: {reel_info.topic or 'unknown'})")
                else:
                    # Dry run - just report what would be done
                    better_topic = self._extract_better_topic(reel_path, reel_info.metadata)
                    if better_topic:
                        issues.append(f"would update topic to: {better_topic[:30]}...")
                    new_path = self._normalize_folder_name(reel_path, dry_run=True)
                    if new_path:
                        issues.append(f"would rename: {old_name} -> {new_path.name}")
                    elif not better_topic:
                        issues.append(f"folder has generic name (topic: {reel_info.topic or 'unknown'})")

            # Check 3: Required artifacts present
            required_artifacts = ["final.mp4", "metadata.json", "caption.txt", "caption+hashtags.txt"]
            missing_artifacts = []
            for artifact in required_artifacts:
                artifact_path = reel_path / artifact
                if not artifact_path.exists():
                    missing_artifacts.append(artifact)
                elif artifact.endswith(".txt"):
                    try:
                        content = artifact_path.read_text(encoding="utf-8").strip()
                        if len(content) < 10:
                            missing_artifacts.append(f"{artifact} (empty)")
                    except Exception:
                        missing_artifacts.append(f"{artifact} (unreadable)")

            # Try to regenerate missing artifacts
            if missing_artifacts and not params.dry_run:
                from .artifacts import audit_reel_artifacts, regenerate_missing_artifacts
                audit = audit_reel_artifacts(reel_path)
                if audit.missing_required or audit.missing_optional:
                    audit = regenerate_missing_artifacts(reel_path, audit, params.profile_path)
                    if audit.regenerated:
                        for name in audit.regenerated:
                            fixed_items.append(f"regenerated {name}")
                            result["fixed"].append(f"Regenerated {name}")
                        # Re-check what's still missing
                        missing_artifacts = [a.name for a in audit.missing_required]

            if missing_artifacts:
                for artifact in missing_artifacts:
                    issues.append(f"missing {artifact}")

            # Check thumbnail (jpg or png)
            has_thumbnail = (reel_path / "thumbnail.jpg").exists() or (reel_path / "thumbnail.png").exists()
            if not has_thumbnail:
                issues.append("missing thumbnail")

            # Report results for this reel
            if fixed_items:
                console.print(f"  [green][FIX][/green] {folder_name}:")
                for item in fixed_items:
                    console.print(f"        [cyan]-> {item}[/cyan]")

            if issues:
                issues_count += 1
                result["issues_found"].extend([f"{folder_name}: {issue}" for issue in issues])
                if not fixed_items:  # Only show header if we didn't already show FIX
                    if len(issues) <= 2:
                        console.print(f"  [yellow][!][/yellow] {folder_name}: {', '.join(issues)}")
                    else:
                        console.print(f"  [yellow][!][/yellow] {folder_name}: {len(issues)} issues")
                        for issue in issues:
                            console.print(f"        - {issue}")
                else:
                    # Show remaining issues after fixes
                    for issue in issues:
                        console.print(f"        [yellow]! {issue}[/yellow]")
            else:
                # No issues (or all fixed)
                result["verified"] += 1
                if fixed_items:
                    console.print(f"        [green][OK] All issues fixed[/green]")

        # Summary
        console.print()
        if issues_count == 0 and not result["fixed"]:
            console.print(f"  [green][OK][/green] All {result['verified']} posted reels verified")
        else:
            console.print(f"  Verified: [green]{result['verified']}[/green] | Issues: [yellow]{issues_count}[/yellow]")
            if result["fixed"]:
                console.print(f"  Fixed: [cyan]{len(result['fixed'])}[/cyan]")

        return result

    def _extract_better_topic(self, reel_path: Path, metadata: dict) -> Optional[str]:
        """Extract a better topic from script.json, news_brief, or caption.

        Used when metadata has a generic topic like "Reel".

        Args:
            reel_path: Path to reel folder.
            metadata: Current metadata dict.

        Returns:
            Better topic string if found, None otherwise.
        """
        import re

        current_topic = metadata.get("topic", "")

        # Skip if topic is already good (not generic)
        if current_topic and len(current_topic) > 10 and current_topic.lower() != "reel":
            return None

        # Generic phrases to skip
        generic_phrases = [
            "reel", "video", "follow for more", "sneak peek", "behind the scenes",
            "check this out", "watch this", "here's", "ever wonder",
        ]

        def is_generic(text: str) -> bool:
            text_lower = text.lower()
            return any(phrase in text_lower for phrase in generic_phrases)

        # Try script.json first
        script_path = reel_path / "script.json"
        if script_path.exists():
            try:
                with open(script_path, encoding="utf-8") as f:
                    script = json.load(f)
                title = script.get("title", "")
                if title and len(title) > 10 and not is_generic(title):
                    return title
                # Try hook text
                hook = script.get("hook", "")
                if hook and len(hook) > 15 and not is_generic(hook):
                    return hook[:80]
            except Exception:
                pass

        # Try news_brief from metadata
        news_brief = metadata.get("news_brief", {})
        if news_brief:
            theme = news_brief.get("theme", "")
            if theme and len(theme) > 10 and not is_generic(theme):
                return theme
            # Try edition display name + date
            edition = news_brief.get("edition_display", "")
            date = news_brief.get("date", "")
            if edition and date:
                return f"{edition} - {date}"

        # Try to extract from narration (first sentence)
        narration = metadata.get("narration", "")
        if narration:
            # Get first sentence (up to first period or 100 chars)
            first_sentence = narration.split(".")[0][:100].strip()
            if len(first_sentence) > 20 and not is_generic(first_sentence):
                return first_sentence

        # Try caption.txt file
        caption_path = reel_path / "caption.txt"
        if caption_path.exists():
            try:
                caption = caption_path.read_text(encoding="utf-8").strip()
                # Get first line or sentence
                first_line = caption.split("\n")[0].split(".")[0][:80].strip()
                if len(first_line) > 15 and not is_generic(first_line):
                    # Clean up: remove hashtags and mentions
                    first_line = re.sub(r"[#@]\w+", "", first_line).strip()
                    if len(first_line) > 15:
                        return first_line
            except Exception:
                pass

        # Fallback: use upload date from Instagram metadata
        ig_status = metadata.get("platform_status", {}).get("instagram", {})
        uploaded_at = ig_status.get("uploaded_at", "")
        if uploaded_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
                # Create a date-based topic like "Posted Dec 18 2025 15:03"
                return f"Posted {dt.strftime('%b %d %Y %H:%M')}"
            except Exception:
                pass

        return None

    def _update_metadata_topic(self, reel_path: Path, new_topic: str) -> bool:
        """Update the topic in metadata.json.

        Args:
            reel_path: Path to reel folder.
            new_topic: New topic string.

        Returns:
            True if updated, False otherwise.
        """
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            metadata["topic"] = new_topic

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            return True
        except Exception:
            return False

    async def upload_single(self, params: ReelUploadParams) -> Result[dict]:
        """Upload a specific reel by ID.

        Delegates to upload_all() with reel_id filter for consistent behavior.
        The preflight will run in targeted mode (single reel validation).

        Args:
            params: Immutable upload parameters (must have reel_id)

        Returns:
            Result containing upload result or Failure
        """
        if not params.reel_id:
            return Failure("No reel ID specified")

        # Delegate to upload_all - it handles reel_id filtering in preflight
        result = await self.upload_all(params)

        # Convert list result to single result
        if isinstance(result, Failure):
            return result

        results = result.value
        if not results:
            return Success({"path": None, "success": True, "skipped": True, "already_posted": True})

        return Success(results[0])

    async def _run_preflight_single(self, params: ReelUploadParams, validator) -> dict:
        """Run targeted preflight for a single reel.

        Skips full scan/dedupe - only validates the specific reel.
        Used when params.reel_id is set.

        Args:
            params: Upload parameters with reel_id set.
            validator: ReelValidator instance.

        Returns:
            Preflight result dict.
        """
        from ..core.console import console
        from .validator import ReelStatus

        result = {
            "reels_to_upload": [],
            "already_posted": [],
            "repaired": [],
            "deleted": [],
            "renamed": [],
            "duplicates_merged": 0,
        }

        # Find the specific reel
        reel_path = self._find_reel_by_id(params)
        if reel_path is None:
            console.print()
            console.print(f"[bold]>>> PRE-FLIGHT[/bold]")
            console.print(f"  [red][X][/red] Reel not found: {params.reel_id}")
            return result

        console.print()
        console.print(f"[bold]>>> PRE-FLIGHT[/bold]")
        console.print(f"  Target: [cyan]{reel_path.name}[/cyan]")

        # Check folder name normalization
        if self._needs_folder_rename(reel_path.name):
            old_name = reel_path.name
            if not params.dry_run:
                new_path = self._normalize_folder_name(reel_path, dry_run=False)
                if new_path:
                    console.print(f"  [yellow][RENAME][/yellow] {old_name} -> {new_path.name}")
                    reel_path = new_path
                    result["renamed"].append((old_name, new_path.name))
            else:
                new_path = self._normalize_folder_name(reel_path, dry_run=True)
                if new_path:
                    console.print(f"  [dim](dry-run: would rename {old_name} -> {new_path.name})[/dim]")

        # Validate the reel
        validation = validator.validate(reel_path)

        if validation.status == ReelStatus.VALID:
            console.print(f"  [green][OK][/green] Reel validated")
            result["reels_to_upload"].append(reel_path)

        elif validation.status == ReelStatus.ALREADY_POSTED:
            console.print(f"  [cyan][POSTED][/cyan] Already uploaded to all platforms")
            result["already_posted"].append(reel_path)

        elif validation.status == ReelStatus.REPAIRABLE:
            console.print(f"  [yellow][REPAIR][/yellow] Attempting to fix missing artifacts...")
            validator.display_validation(validation, verbose=True)

            if not params.dry_run:
                repair_result = await validator.repair(reel_path, params.profile_path, dry_run=False)
                validator.display_repair(repair_result)

                if repair_result.success:
                    console.print(f"  [green][OK][/green] Repair successful")
                    result["repaired"].append(reel_path)
                    result["reels_to_upload"].append(reel_path)
                else:
                    console.print(f"  [red][X][/red] Repair failed: {repair_result.error}")
            else:
                console.print(f"  [dim](dry-run: would attempt repair)[/dim]")

        elif validation.status == ReelStatus.INVALID:
            console.print(f"  [red][INVALID][/red] {validation.error}")
            validator.display_validation(validation, verbose=True)
            result["deleted"].append(reel_path)

        return result

    def _find_pending_reels(self, params: ReelUploadParams) -> list[Path]:
        """Find all reels that need uploading to requested platforms.

        Searches generated, pending-post, AND posted folders.
        Reels in posted folder are included if they haven't been uploaded
        to all requested platforms yet.
        """
        pending_paths = []

        # Check generated, pending-post, and posted folders
        for status in ["generated", "pending-post", "posted"]:
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
                        if not reel_dir.is_dir():
                            continue
                        if not (reel_dir / "final.mp4").exists():
                            continue

                        # For posted folder, only include if missing requested platforms
                        if status == "posted":
                            if self._needs_platform_upload(reel_dir, params.platforms):
                                pending_paths.append(reel_dir)
                        else:
                            pending_paths.append(reel_dir)

        return pending_paths

    def _needs_platform_upload(self, reel_path: Path, platforms: tuple[str, ...]) -> bool:
        """Check if reel needs upload to any of the requested platforms."""
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return True  # No metadata = needs upload

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return True

        # Check if any requested platform is not yet uploaded
        for platform in platforms:
            if not is_platform_uploaded(metadata, platform):
                return True

        return False

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
        """Upload a single reel to all specified platforms."""
        from datetime import datetime
        from socials_automator.platforms import PlatformRegistry
        from ..core.console import console

        def log(message: str, style: str = "dim") -> None:
            """Print timestamped log message."""
            ts = datetime.now().strftime("%H:%M:%S")
            console.print(f"  [{style}]{ts}[/{style}] {message}")

        # Base result structure
        result = {
            "path": reel_path,
            "success": False,
            "error": None,
            "platforms": {},
        }

        if params.dry_run:
            result["success"] = True
            result["dry_run"] = True
            result["platforms"] = {p: {"success": True, "dry_run": True} for p in params.platforms}
            return result

        try:
            # =================================================================
            # STEP: Move from generated to pending-post
            # =================================================================
            if self._is_in_generated(reel_path):
                log("[dim]Moving to pending-post/...[/dim]")
                new_path = self._move_to_pending_post(reel_path)
                if new_path:
                    reel_path = new_path
                    result["path"] = new_path
                    log("[green][OK] Moved to pending-post/[/green]")

            # =================================================================
            # STEP: Load metadata and validate
            # =================================================================
            metadata_path = reel_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Get video path for size display
            video_path = reel_path / "final.mp4"

            # Get thumbnail (check for jpg, png, jpeg)
            from .artifacts import get_thumbnail_path
            thumbnail_path = get_thumbnail_path(reel_path)

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
            console.print(f"  [dim]Platforms:[/dim] {', '.join(params.platforms)}")
            console.print()

            # Pre-flight validation checklist
            from .artifacts import (
                audit_reel_artifacts,
                regenerate_missing_artifacts,
                get_thumbnail_path,
            )

            console.print("    [bold]Pre-flight Checklist:[/bold]")
            audit_result = audit_reel_artifacts(reel_path)

            # Show checklist items
            for artifact in audit_result.artifacts:
                if artifact.exists and artifact.has_content:
                    # Show file size for video
                    if artifact.name == "final.mp4":
                        size_mb = artifact.path.stat().st_size / (1024 * 1024)
                        console.print(f"      [green][OK][/green] {artifact.name} ({size_mb:.1f} MB)")
                    elif artifact.name.endswith(".txt"):
                        char_count = len(artifact.path.read_text(encoding="utf-8"))
                        console.print(f"      [green][OK][/green] {artifact.name} ({char_count} chars)")
                    else:
                        console.print(f"      [green][OK][/green] {artifact.name}")
                elif artifact.required:
                    console.print(f"      [red][X][/red]  {artifact.name} [red](REQUIRED)[/red]")
                else:
                    console.print(f"      [dim][ ][/dim]  {artifact.name} [dim](optional)[/dim]")

            console.print()

            # Regenerate missing artifacts if possible
            if audit_result.missing_required or audit_result.missing_optional:
                log("[yellow]Attempting to regenerate missing artifacts...[/yellow]")
                audit_result = regenerate_missing_artifacts(
                    reel_path, audit_result, params.profile_path
                )

                if audit_result.regenerated:
                    for name in audit_result.regenerated:
                        console.print(f"      [yellow][REGEN][/yellow] {name}")
                    console.print()

            # Check if we can proceed
            if not audit_result.is_valid:
                missing = [a.name for a in audit_result.missing_required]
                result["error"] = f"Missing required artifacts: {', '.join(missing)}"
                console.print(f"  [bold red][X] Cannot upload - missing required files:[/bold red]")
                for name in missing:
                    console.print(f"      [red]- {name}[/red]")
                return result

            log("[green][OK] Pre-flight check passed[/green]")

            # Get caption (prefer caption+hashtags.txt, fallback to caption.txt, then metadata)
            caption_hashtags_path = reel_path / "caption+hashtags.txt"
            caption_path = reel_path / "caption.txt"
            if caption_hashtags_path.exists():
                caption = caption_hashtags_path.read_text(encoding="utf-8").strip()
            elif caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8").strip()
            else:
                caption = metadata.get("caption", "").strip()

            # If caption is empty/too short, attempt to regenerate
            if len(caption) < 10:
                log("[yellow]Caption is empty or too short, attempting to regenerate...[/yellow]")
                from .artifacts import _regenerate_caption_with_hashtags, _regenerate_caption

                # Try to regenerate caption+hashtags.txt first (includes AI generation)
                if _regenerate_caption_with_hashtags(reel_path, params.profile_path):
                    log("[green][REGEN] Regenerated caption+hashtags.txt[/green]")
                    caption = caption_hashtags_path.read_text(encoding="utf-8").strip()
                # Fallback to regenerating just caption.txt
                elif _regenerate_caption(reel_path, use_ai=True):
                    log("[green][REGEN] Regenerated caption.txt[/green]")
                    caption = caption_path.read_text(encoding="utf-8").strip()

            # Validate caption is not empty (minimum 10 characters)
            if len(caption) < 10:
                result["error"] = "Caption is empty or too short (< 10 chars)"
                console.print(f"  [bold red][X] Caption is empty or too short[/bold red]")
                console.print(f"  [dim]Caption should have at least 10 characters[/dim]")
                console.print(f"  [dim]Could not regenerate from metadata/script/AI[/dim]")
                console.print(f"  [dim]Found: '{caption[:50]}...' ({len(caption)} chars)[/dim]" if caption else "  [dim]Found: (empty)[/dim]")
                return result

            # Hashtag validation step - trim to Instagram limit (5 max)
            from socials_automator.hashtag import validate_hashtags_in_caption, INSTAGRAM_MAX_HASHTAGS
            hashtag_result = validate_hashtags_in_caption(caption, auto_trim=True)

            log(f"[cyan]Hashtag validation:[/cyan]")
            if hashtag_result.was_trimmed:
                log(f"  [yellow][!] Trimmed hashtags: {hashtag_result.original_count} -> {hashtag_result.final_count}[/yellow]")
                log(f"  [dim]Removed: {', '.join(hashtag_result.removed_hashtags)}[/dim]")
                # Update caption with trimmed version
                caption = hashtag_result.caption_after
                # Also update the file for consistency
                if caption_hashtags_path.exists():
                    caption_hashtags_path.write_text(caption, encoding="utf-8")
                    log(f"  [green][OK] Updated caption+hashtags.txt[/green]")
            elif hashtag_result.original_count == 0:
                log(f"  [dim]No hashtags in caption[/dim]")
            else:
                log(f"  [green][OK] Hashtags: {hashtag_result.original_count}/{INSTAGRAM_MAX_HASHTAGS}[/green]")

            if not video_path.exists():
                result["error"] = "Video file not found"
                log(f"[red][X] Video file not found[/red]", "red")
                return result

            # Upload to each platform
            any_success = False
            any_new_upload = False

            for platform_name in params.platforms:
                console.print()
                console.print(f"  [bold magenta]>>> {platform_name.upper()}[/bold magenta]")

                # Check if already uploaded to this platform
                if is_platform_uploaded(metadata, platform_name):
                    existing_status = get_platform_status(metadata).get(platform_name, {})
                    console.print(f"  [dim]Already uploaded on {existing_status.get('uploaded_at', 'unknown date')}[/dim]")
                    if existing_status.get("permalink"):
                        console.print(f"  [dim]{existing_status['permalink']}[/dim]")
                    result["platforms"][platform_name] = {
                        "success": True,
                        "skipped": True,
                        "already_uploaded": True,
                    }
                    any_success = True
                    continue

                # Upload to this platform (with scoped Cloudinary folder)
                any_new_upload = True
                platform_result = await self._upload_to_platform(
                    platform_name=platform_name,
                    video_path=video_path,
                    thumbnail_path=thumbnail_path,  # Already None if not found
                    caption=caption,
                    profile_path=params.profile_path,
                    log_fn=log,
                    reel_folder_name=reel_path.name,  # For scoped Cloudinary uploads
                )

                # If upload failed and we have hashtags, try fallback without hashtags
                if not platform_result.get("success") and hashtag_result.original_count > 0:
                    error_msg = platform_result.get("error", "").lower()
                    # Check if error might be caption/hashtag related
                    if any(hint in error_msg for hint in ["caption", "hashtag", "invalid", "param", "encoding", "character"]):
                        from socials_automator.hashtag import remove_hashtags_from_caption
                        caption_no_hashtags = remove_hashtags_from_caption(caption)
                        log(f"[yellow][!] Upload failed, retrying without hashtags...[/yellow]")
                        log(f"[dim]Removed {hashtag_result.original_count} hashtags from caption[/dim]")

                        # Retry upload without hashtags
                        platform_result = await self._upload_to_platform(
                            platform_name=platform_name,
                            video_path=video_path,
                            thumbnail_path=thumbnail_path,
                            caption=caption_no_hashtags,
                            profile_path=params.profile_path,
                            log_fn=log,
                            reel_folder_name=reel_path.name,
                        )

                        if platform_result.get("success"):
                            log(f"[green][OK] Fallback succeeded (no hashtags)[/green]")
                            platform_result["fallback_used"] = True
                            platform_result["original_hashtag_count"] = hashtag_result.original_count

                result["platforms"][platform_name] = platform_result

                # Update platform_status in metadata
                update_platform_status(
                    metadata_path=metadata_path,
                    platform=platform_name,
                    success=platform_result.get("success", False),
                    media_id=platform_result.get("media_id"),
                    permalink=platform_result.get("permalink"),
                    error=platform_result.get("error"),
                )

                if platform_result.get("success"):
                    any_success = True
                    console.print(f"  [bold green][OK] {platform_name}: Published![/bold green]")
                    if platform_result.get("permalink"):
                        console.print(f"  [cyan]{platform_result['permalink']}[/cyan]")
                    if platform_result.get("fallback_used"):
                        console.print(f"  [dim](uploaded without hashtags due to error)[/dim]")
                else:
                    console.print(f"  [bold red][X] {platform_name}: Failed[/bold red]")
                    if platform_result.get("error"):
                        console.print(f"  [red]{platform_result['error']}[/red]")

            # Set overall success if at least one platform succeeded
            result["success"] = any_success

            # Move to posted folder if ANY new upload succeeded and not already in posted
            if any_new_upload and any_success and not self._is_in_posted(reel_path):
                console.print()
                new_path = self._move_to_posted(reel_path, params.profile_path)
                log("[dim]Moved to posted/ folder[/dim]")
                # Update result path to new location
                if new_path:
                    result["path"] = new_path

        except Exception as e:
            result["error"] = str(e)
            console.print()
            console.print(f"  [bold red][X] Error[/bold red]")
            console.print(f"  [red]{e}[/red]")
            console.print()

        return result

    async def _upload_to_platform(
        self,
        platform_name: str,
        video_path: Path,
        thumbnail_path: Optional[Path],
        caption: str,
        profile_path: Path,
        log_fn,
        reel_folder_name: Optional[str] = None,
    ) -> dict:
        """Upload to a specific platform using the platform adapter."""
        from socials_automator.platforms import PlatformRegistry
        from ..core.console import console

        result = {
            "success": False,
            "error": None,
            "media_id": None,
            "permalink": None,
        }

        try:
            # Load platform config from profile
            log_fn("[yellow]Loading platform config...[/yellow]")
            config = PlatformRegistry.load_config(platform_name, profile_path)

            # Show config details for debugging
            if platform_name == "instagram" and hasattr(config, "user_id"):
                token = getattr(config, "access_token", "")
                user_id = getattr(config, "user_id", "")
                token_prefix = token[:20] + "..." if len(token) > 20 else token

                # Determine token type from prefix
                if token.startswith("EAA"):
                    token_type = "Facebook Login (EAA...)"
                elif token.startswith("IG"):
                    token_type = "Instagram Login (IG...)"
                else:
                    token_type = "Unknown"

                console.print()
                console.print("    [bold]Credentials:[/bold]")
                console.print(f"      User ID:    [cyan]{user_id}[/cyan]")
                console.print(f"      Token:      [dim]{token_prefix}[/dim]")
                console.print(f"      Token type: [dim]{token_type}[/dim]")
                console.print()

                # Validate credentials and show account info
                log_fn("[yellow]Validating credentials...[/yellow]")
                try:
                    publisher = PlatformRegistry.get_publisher(platform_name, config)
                    is_valid, message = await publisher.check_credentials()
                    if is_valid:
                        log_fn(f"[green][OK] {message}[/green]")
                    else:
                        log_fn(f"[red][X] {message}[/red]")
                        result["error"] = message
                        return result
                except Exception as e:
                    log_fn(f"[red][X] Credential check failed: {e}[/red]")
                    result["error"] = f"Credential check failed: {e}"
                    return result

            # Create progress callback with detailed step tracking
            current_stage = {"name": "", "start_time": None}

            async def progress_callback(stage: str, progress: float, message: str) -> None:
                from datetime import datetime

                # Track stage changes for timing
                if stage != current_stage["name"]:
                    if current_stage["start_time"] is not None:
                        elapsed = (datetime.now() - current_stage["start_time"]).total_seconds()
                        if elapsed > 1:
                            log_fn(f"[dim]  ({elapsed:.1f}s)[/dim]")
                    current_stage["name"] = stage
                    current_stage["start_time"] = datetime.now()

                # Format progress message with percentage
                if progress > 0 and progress < 100:
                    log_fn(f"[cyan]{message}[/cyan] [dim]({progress:.0f}%)[/dim]")
                else:
                    log_fn(f"[cyan]{message}[/cyan]")

            # Get publisher with progress callback
            publisher = PlatformRegistry.get_publisher(
                platform_name, config, progress_callback=progress_callback
            )

            # Show upload start
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            log_fn(f"[yellow]Starting upload ({file_size_mb:.1f} MB)...[/yellow]")

            # Extract profile name from profile_path for scoped uploads
            profile_name = profile_path.name if profile_path else None

            # Publish with scoped Cloudinary folder (profile/reel_folder)
            publish_result = await publisher.publish_reel(
                video_path=video_path,
                caption=caption,
                thumbnail_path=thumbnail_path,
                profile=profile_name,
                post_id=reel_folder_name,
            )

            if publish_result.success:
                result["success"] = True
                result["media_id"] = publish_result.media_id
                result["permalink"] = publish_result.permalink

                # Show additional details from publish result
                if hasattr(publish_result, "video_url") and publish_result.video_url:
                    log_fn(f"[dim]Cloudinary: {publish_result.video_url[:60]}...[/dim]")
            else:
                result["error"] = publish_result.error

        except Exception as e:
            result["error"] = str(e)

        return result

    def _is_in_posted(self, reel_path: Path) -> bool:
        """Check if reel is already in posted folder."""
        return "posted" in reel_path.parts

    def _is_in_generated(self, reel_path: Path) -> bool:
        """Check if reel is in generated folder."""
        return "generated" in reel_path.parts

    def _move_to_pending_post(self, reel_path: Path) -> Optional[Path]:
        """Move reel folder from generated to pending-post.

        Returns:
            New path if moved, None if not in generated or move failed.
        """
        import shutil

        if not self._is_in_generated(reel_path):
            return None

        # Determine new path by replacing "generated" with "pending-post"
        parts = list(reel_path.parts)
        for i, part in enumerate(parts):
            if part == "generated":
                parts[i] = "pending-post"
                break

        new_path = Path(*parts)

        # Create parent directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move folder
        if reel_path.exists() and not new_path.exists():
            shutil.move(str(reel_path), str(new_path))
            return new_path

        return None

    def _move_to_posted(self, reel_path: Path, profile_path: Path) -> Optional[Path]:
        """Move reel folder to posted status.

        Handles name conflicts by appending a unique suffix when a folder
        with the same name already exists in posted/.

        Returns:
            New path if moved, None if already in posted or move failed.
        """
        import shutil

        if self._is_in_posted(reel_path):
            return None

        # Determine new path by replacing status folder
        parts = list(reel_path.parts)
        for i, part in enumerate(parts):
            if part in ["generated", "pending-post"]:
                parts[i] = "posted"
                break

        new_path = Path(*parts)

        # Create parent directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if not reel_path.exists():
            return None

        # Handle name conflict: if target exists, append unique suffix
        if new_path.exists():
            # Try to get post_id from metadata for uniqueness
            metadata_path = reel_path / "metadata.json"
            suffix = ""
            if metadata_path.exists():
                try:
                    import json
                    with open(metadata_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    # Use last 8 chars of post_id or media_id as suffix
                    post_id = meta.get("post_id", "")
                    if post_id:
                        suffix = f"-{post_id[-8:]}"
                    else:
                        ig_status = meta.get("platform_status", {}).get("instagram", {})
                        media_id = ig_status.get("media_id", "")
                        if media_id:
                            suffix = f"-{media_id[-8:]}"
                except Exception:
                    pass

            # If no suffix from metadata, use timestamp
            if not suffix:
                from datetime import datetime
                suffix = f"-{datetime.now().strftime('%H%M%S')}"

            # Create new unique path
            new_name = f"{reel_path.name}{suffix}"
            new_path = new_path.parent / new_name

        # Move folder
        shutil.move(str(reel_path), str(new_path))
        return new_path

    def _needs_folder_rename(self, folder_name: str) -> bool:
        """Check if folder name needs normalization.

        Returns True if:
        - Folder ends with "-reel" (generic name)
        - Folder has timestamp suffix like "-13-28" (HH-MM pattern at end)
        - Folder name is too short (missing topic slug)
        """
        import re

        # Check for DD-NNN prefix
        match = re.match(r"^(\d{2}-\d{3})-(.+)$", folder_name)
        if not match:
            return False

        suffix = match.group(2)

        # Generic "reel" name
        if suffix == "reel":
            return True

        # Timestamp suffix pattern (ends with -HH-MM or -HH-MM-SS)
        if re.search(r"-\d{2}-\d{2}(-\d{2})?$", suffix):
            return True

        # Too short (likely missing proper topic)
        if len(suffix) < 5:
            return True

        return False

    def _normalize_folder_name(
        self,
        reel_path: Path,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """Normalize folder name to proper DD-NNN-topic-slug format.

        Reads topic from metadata and renames folder to match.
        Handles conflicts by appending unique suffix from post_id.

        Args:
            reel_path: Path to reel folder.
            dry_run: If True, don't actually rename.

        Returns:
            New path if renamed, None if not needed or failed.
        """
        import re
        import shutil

        folder_name = reel_path.name

        # Check if rename is needed
        if not self._needs_folder_rename(folder_name):
            return None

        # Extract DD-NNN prefix
        match = re.match(r"^(\d{2}-\d{3})-", folder_name)
        if not match:
            return None

        prefix = match.group(1)

        # Read metadata to get topic
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            topic = metadata.get("topic", "")
            if not topic or len(topic) < 3:
                return None

            # Clean topic: remove parenthesized timestamps like (13:28) or (HH:MM:SS)
            topic = re.sub(r"\s*\(\d{1,2}:\d{2}(:\d{2})?\)\s*$", "", topic)
            # Also remove trailing timestamps without parens like "- 13:28"
            topic = re.sub(r"\s*[-:]\s*\d{1,2}:\d{2}(:\d{2})?\s*$", "", topic)

            if not topic or len(topic) < 3:
                return None

            # Create slug from cleaned topic
            slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50].strip("-")

            if not slug or slug == "reel":
                return None

            # Build new folder name
            new_folder_name = f"{prefix}-{slug}"
            new_path = reel_path.parent / new_folder_name

            # Skip if same name
            if new_folder_name == folder_name:
                return None

            if dry_run:
                return new_path

            # Handle name conflict
            if new_path.exists():
                # Append unique suffix from post_id or media_id
                post_id = metadata.get("post_id", "")
                if post_id:
                    suffix = f"-{post_id[-8:]}"
                else:
                    ig_status = metadata.get("platform_status", {}).get("instagram", {})
                    media_id = ig_status.get("media_id", "")
                    if media_id:
                        suffix = f"-{media_id[-8:]}"
                    else:
                        from datetime import datetime
                        suffix = f"-{datetime.now().strftime('%H%M%S')}"

                new_folder_name = f"{prefix}-{slug}{suffix}"
                new_path = reel_path.parent / new_folder_name

            # Rename folder
            shutil.move(str(reel_path), str(new_path))
            return new_path

        except Exception:
            return None
