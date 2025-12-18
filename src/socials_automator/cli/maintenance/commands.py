"""Maintenance CLI commands - utility and maintenance operations."""

from __future__ import annotations

import json
from typing import Optional

import typer

from ..core.console import console
from ..core.paths import get_profile_path, get_profiles_dir
from .display import (
    show_artifacts_update_result,
    show_cleanup_header,
    show_cleanup_progress,
    show_cleanup_result,
    show_init_result,
    show_niches_table,
    show_platform_migration_result,
    show_profile_created,
    show_profile_status,
    show_token_refreshed,
    show_token_status,
)
from .service import (
    check_token,
    cleanup_reels,
    create_profile,
    find_posted_reels,
    find_reels_for_artifact_update,
    find_reels_for_cleanup,
    get_profile_status,
    init_project_structure,
    load_niches,
    migrate_reel_platform_status,
    refresh_token,
    update_reel_artifacts,
)


def init() -> None:
    """Initialize project structure."""
    created = init_project_structure()
    show_init_result(console, created)


def token(
    check: bool = typer.Option(False, "--check", "-c", help="Check token validity"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Refresh the token"),
    exchange: bool = typer.Option(False, "--exchange", "-e", help="Exchange short-lived for long-lived token"),
) -> None:
    """Manage Instagram API token.

    Use --check to verify token validity.
    Use --refresh to refresh an expiring token.
    Use --exchange to exchange a short-lived token for a long-lived one.
    """
    if check:
        status = check_token()
        show_token_status(console, status.valid, status.expires_at, status.error)
        if not status.valid:
            raise typer.Exit(1)
        return

    if refresh:
        success, result = refresh_token()
        if success:
            show_token_refreshed(console, result)
        else:
            console.print(f"[red]Failed to refresh token: {result}[/red]")
            raise typer.Exit(1)
        return

    if exchange:
        # Token exchange logic
        console.print("[yellow]Token exchange not implemented in refactored CLI.[/yellow]")
        console.print("[dim]Use the original CLI for token exchange.[/dim]")
        raise typer.Exit(1)

    # Default: show status
    status = check_token()
    show_token_status(console, status.valid, status.expires_at, status.error)


def status(
    profile: str = typer.Argument(..., help="Profile name"),
) -> None:
    """Show profile status and recent content."""
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Load profile metadata
    metadata_path = profile_path / "metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]Profile metadata not found: {profile}[/red]")
        raise typer.Exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    profile_info = metadata.get("profile", {})
    handle = profile_info.get("instagram_handle", "")
    niche = profile_info.get("niche_id", "")

    # Get counts
    post_counts, reel_counts = get_profile_status(profile_path)

    show_profile_status(console, profile, handle, niche, post_counts, reel_counts)


def new_profile(
    name: str = typer.Argument(..., help="Profile folder name"),
    handle: str = typer.Option(..., "--handle", "-h", help="Instagram handle"),
    niche: Optional[str] = typer.Option(None, "--niche", "-n", help="Niche ID from niches.json"),
) -> None:
    """Create a new profile.

    Creates the profile directory structure with initial metadata.json.
    """
    profiles_dir = get_profiles_dir()
    profile_path = profiles_dir / name

    if profile_path.exists():
        console.print(f"[red]Profile already exists: {name}[/red]")
        raise typer.Exit(1)

    created_path = create_profile(name, handle, niche)
    show_profile_created(console, name, str(created_path))


def list_niches() -> None:
    """List available niches from niches.json."""
    niches = load_niches()
    show_niches_table(console, niches)


def update_artifacts(
    profile: str = typer.Argument(..., help="Profile name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
) -> None:
    """Update artifact metadata for all existing reels.

    Scans all reel folders and adds artifact references to metadata.json
    if they don't already exist.
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find reels
    reels = find_reels_for_artifact_update(profile_path)

    if not reels:
        console.print("[yellow]No reels found.[/yellow]")
        return

    console.print(f"\nFound {len(reels)} reel(s):\n")

    updated = 0
    skipped = 0

    for reel_path in reels:
        reel_name = reel_path.name

        if dry_run:
            console.print(f"  [dim]{reel_name}[/dim]: [yellow]would check[/yellow]")
            # Check if would update
            if update_reel_artifacts(reel_path):
                updated += 1
            else:
                skipped += 1
            continue

        if update_reel_artifacts(reel_path):
            console.print(f"  [dim]{reel_name}[/dim]: [green]updated[/green]")
            updated += 1
        else:
            console.print(f"  [dim]{reel_name}[/dim]: [dim]skipped[/dim]")
            skipped += 1

    console.print()
    show_artifacts_update_result(console, updated, skipped, dry_run)


def migrate_platform_status(
    profile: str = typer.Argument(..., help="Profile name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
) -> None:
    """Migrate existing posted reels to new platform_status format.

    Adds platform_status.instagram to all reels in posted folders,
    marking them as already uploaded to Instagram. This enables
    uploading existing reels to additional platforms like TikTok.
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find all posted reels
    reels = find_posted_reels(profile_path)

    if not reels:
        console.print("[yellow]No posted reels found.[/yellow]")
        return

    console.print(f"\nFound {len(reels)} posted reel(s):\n")

    updated = 0
    skipped = 0
    errors = 0

    for reel_path in reels:
        reel_name = reel_path.name

        result = migrate_reel_platform_status(reel_path, dry_run=dry_run)

        if result == "updated" or result == "would_update":
            style = "yellow" if dry_run else "green"
            action = "would update" if dry_run else "updated"
            console.print(f"  [dim]{reel_name}[/dim]: [{style}]{action}[/{style}]")
            updated += 1
        elif result == "skipped":
            console.print(f"  [dim]{reel_name}[/dim]: [dim]already migrated[/dim]")
            skipped += 1
        else:
            console.print(f"  [dim]{reel_name}[/dim]: [red]{result}[/red]")
            errors += 1

    console.print()
    show_platform_migration_result(console, updated, skipped, errors, dry_run)


def cleanup_posted_reels(
    profile: str = typer.Argument(..., help="Profile name"),
    older_than: Optional[int] = typer.Option(None, "--older-than", "-o", help="Only clean reels older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without deleting"),
    no_fetch_urls: bool = typer.Option(False, "--no-fetch-urls", help="Skip fetching Instagram video URLs"),
) -> None:
    """Clean up posted reels by removing video files.

    Removes final.mp4 and debug_log.txt from posted reels to free disk space.
    Keeps metadata.json, thumbnail.jpg, caption.txt, and caption+hashtags.txt.

    Before deletion, fetches the Instagram video URL and saves it to metadata.
    Use --no-fetch-urls to skip this step.

    Examples:
        # Preview what would be cleaned (no deletion)
        cleanup-reels ai.for.mortals --dry-run

        # Clean all posted reels
        cleanup-reels ai.for.mortals

        # Clean only reels older than 7 days
        cleanup-reels ai.for.mortals --older-than 7
    """
    import asyncio

    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find eligible reels first
    reels = find_reels_for_cleanup(profile_path, older_than)

    if not reels:
        console.print("[yellow]No posted reels found to clean up.[/yellow]")
        if older_than:
            console.print(f"[dim]Try without --older-than to see all posted reels.[/dim]")
        return

    # Show header
    show_cleanup_header(console, profile, len(reels), older_than, dry_run)

    # Progress callback for display
    results_cache = []

    def progress_callback(current: int, total: int, reel_name: str) -> None:
        # We'll show progress after each result in the main loop
        pass

    # Run cleanup
    summary = asyncio.run(cleanup_reels(
        profile_path=profile_path,
        older_than_days=older_than,
        dry_run=dry_run,
        fetch_video_urls=not no_fetch_urls,
        progress_callback=progress_callback,
    ))

    # Show individual results
    for idx, result in enumerate(summary.results, 1):
        show_cleanup_progress(
            console,
            current=idx,
            total=summary.total_reels,
            reel_name=result.reel_name,
            status=result.status,
            space_mb=result.space_freed_mb,
            video_url=result.video_url,
        )

    # Show summary
    show_cleanup_result(
        console,
        cleaned=summary.cleaned,
        skipped=summary.skipped,
        already_cleaned=summary.already_cleaned,
        errors=summary.errors,
        total_space_mb=summary.total_space_freed_mb,
        dry_run=dry_run,
    )

    if summary.errors > 0:
        raise typer.Exit(1)
