"""Display functions for profile commands - pure functions for Rich output."""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.types import ProfileConfig


def show_profiles_table(console: Console, profiles: List[ProfileConfig]) -> None:
    """Display table of available profiles."""
    if not profiles:
        console.print("[yellow]No profiles found.[/yellow]")
        return

    table = Table(title="Available Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Handle", style="green")
    table.add_column("Niche", style="yellow")

    for profile in profiles:
        table.add_row(
            profile.name,
            profile.handle,
            profile.niche,
        )

    console.print(table)


def show_thumbnail_config(
    console: Console,
    profile: str,
    dry_run: bool,
    force: bool,
    font_size: int,
) -> None:
    """Display thumbnail fix configuration."""
    mode = "Force regenerate ALL" if force else "Generate missing only"

    console.print(Panel(
        f"Fixing thumbnails for [cyan]{profile}[/cyan]\n"
        f"Mode: [yellow]{mode}[/yellow]\n"
        f"Font Size: [yellow]{font_size}px[/yellow]\n"
        f"Dry Run: [{'yellow' if dry_run else 'dim'}]{dry_run}[/]",
        title="Thumbnail Fix",
    ))


def show_thumbnail_results(
    console: Console,
    generated: int,
    skipped: int,
    failed: int,
    dry_run: bool,
) -> None:
    """Display thumbnail generation results."""
    if dry_run:
        console.print(Panel(
            f"[yellow]Dry run complete[/yellow]\n\n"
            f"Would generate: [green]{generated}[/green]\n"
            f"Would skip: [dim]{skipped}[/dim]",
            title="Dry Run Results",
            border_style="yellow",
        ))
    else:
        if failed == 0:
            console.print(Panel(
                f"[bold green]Thumbnails generated successfully![/bold green]\n\n"
                f"Generated: [green]{generated}[/green]\n"
                f"Skipped: [dim]{skipped}[/dim]",
                title="Complete",
                border_style="green",
            ))
        else:
            console.print(Panel(
                f"[yellow]Completed with errors[/yellow]\n\n"
                f"Generated: [green]{generated}[/green]\n"
                f"Skipped: [dim]{skipped}[/dim]\n"
                f"Failed: [red]{failed}[/red]",
                title="Complete",
                border_style="yellow",
            ))


def show_thumbnail_progress(
    console: Console,
    current: int,
    total: int,
    reel_name: str,
    action: str,
) -> None:
    """Display thumbnail generation progress."""
    console.print(f"  [{current}/{total}] {reel_name}: {action}")


def show_no_reels_found(console: Console) -> None:
    """Display message when no reels are found."""
    console.print("[yellow]No reels directory found.[/yellow]")
