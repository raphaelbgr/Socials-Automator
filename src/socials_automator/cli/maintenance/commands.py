"""Maintenance CLI commands - utility and maintenance operations."""

from __future__ import annotations

import json
from typing import Optional

import typer

from ..core.console import console
from ..core.paths import get_profile_path, get_profiles_dir
from .display import (
    show_artifacts_update_result,
    show_init_result,
    show_niches_table,
    show_profile_created,
    show_profile_status,
    show_token_refreshed,
    show_token_status,
)
from .service import (
    check_token,
    create_profile,
    find_reels_for_artifact_update,
    get_profile_status,
    init_project_structure,
    load_niches,
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
