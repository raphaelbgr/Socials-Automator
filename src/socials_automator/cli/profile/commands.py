"""Profile CLI commands - thin wrappers orchestrating display and service."""

from __future__ import annotations

from typing import Optional

import typer

from ..core.console import console
from ..core.paths import get_profile_path
from .display import (
    show_no_reels_found,
    show_profiles_table,
    show_thumbnail_config,
    show_thumbnail_progress,
    show_thumbnail_results,
)
from .service import (
    find_reels_for_thumbnails,
    generate_thumbnail,
    list_all_profiles,
)


def list_profiles() -> None:
    """List all available profiles."""
    profiles = list_all_profiles()
    show_profiles_table(console, profiles)


def fix_thumbnails(
    profile: str = typer.Argument(..., help="Profile name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
    force: bool = typer.Option(False, "--force", help="Regenerate ALL thumbnails"),
    font_size: int = typer.Option(54, "--font-size", "-s", help="Font size in pixels"),
) -> None:
    """Generate missing thumbnails for existing reels.

    Scans all reel folders (generated, pending-post, posted) and generates
    thumbnails for any reel that doesn't have one.

    Use --force to regenerate ALL thumbnails.
    Use --font-size to customize text size.
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Display configuration
    show_thumbnail_config(console, profile, dry_run, force, font_size)

    # Find reels needing thumbnails
    reels = find_reels_for_thumbnails(profile_path, force=force)

    if not reels:
        if force:
            show_no_reels_found(console)
        else:
            console.print("[green]All reels have thumbnails.[/green]")
        return

    console.print(f"\nFound {len(reels)} reel(s) needing thumbnails:\n")

    # Process each reel
    generated = 0
    skipped = 0
    failed = 0

    for i, reel_path in enumerate(reels, 1):
        reel_name = reel_path.name

        if dry_run:
            show_thumbnail_progress(console, i, len(reels), reel_name, "[yellow]would generate[/yellow]")
            generated += 1
            continue

        # Generate thumbnail
        result = generate_thumbnail(reel_path, font_size=font_size)

        if result.action == "generated":
            show_thumbnail_progress(console, i, len(reels), reel_name, "[green]generated[/green]")
            generated += 1
        elif result.action == "skipped":
            show_thumbnail_progress(console, i, len(reels), reel_name, "[dim]skipped[/dim]")
            skipped += 1
        else:
            show_thumbnail_progress(
                console, i, len(reels), reel_name,
                f"[red]failed: {result.error}[/red]"
            )
            failed += 1

    # Show summary
    console.print()
    show_thumbnail_results(console, generated, skipped, failed, dry_run)
