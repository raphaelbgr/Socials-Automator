"""Display functions for maintenance commands - pure functions for Rich output."""

from __future__ import annotations

from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def show_init_result(console: Console, dirs_created: List[str]) -> None:
    """Display initialization result."""
    if dirs_created:
        console.print(Panel(
            "[bold green]Project structure initialized![/bold green]\n\n"
            + "\n".join(f"  [green]+[/green] {d}" for d in dirs_created),
            title="Init Complete",
            border_style="green",
        ))
    else:
        console.print("[dim]All directories already exist.[/dim]")


def show_token_status(
    console: Console,
    valid: bool,
    expires_at: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Display token status."""
    if error:
        console.print(Panel(
            f"[red]Token Error[/red]\n\n{error}",
            title="Token Status",
            border_style="red",
        ))
    elif valid:
        expiry_info = f"\n[dim]Expires: {expires_at}[/dim]" if expires_at else ""
        console.print(Panel(
            f"[bold green]Token is valid[/bold green]{expiry_info}",
            title="Token Status",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]Token is invalid or expired[/yellow]\n\n"
            "Use --refresh to refresh the token.",
            title="Token Status",
            border_style="yellow",
        ))


def show_token_refreshed(console: Console, new_expiry: Optional[str] = None) -> None:
    """Display token refresh result."""
    expiry_info = f"\n[dim]New expiry: {new_expiry}[/dim]" if new_expiry else ""
    console.print(Panel(
        f"[bold green]Token refreshed successfully![/bold green]{expiry_info}",
        title="Token Refresh",
        border_style="green",
    ))


def show_profile_status(
    console: Console,
    profile: str,
    handle: str,
    niche: str,
    post_counts: dict,
    reel_counts: dict,
) -> None:
    """Display profile status."""
    console.print(Panel(
        f"[bold]{profile}[/bold]\n"
        f"Handle: [cyan]@{handle}[/cyan]\n"
        f"Niche: [yellow]{niche}[/yellow]\n\n"
        f"[bold]Posts:[/bold]\n"
        f"  Generated: {post_counts.get('generated', 0)}\n"
        f"  Pending: {post_counts.get('pending', 0)}\n"
        f"  Posted: {post_counts.get('posted', 0)}\n\n"
        f"[bold]Reels:[/bold]\n"
        f"  Generated: {reel_counts.get('generated', 0)}\n"
        f"  Pending: {reel_counts.get('pending', 0)}\n"
        f"  Posted: {reel_counts.get('posted', 0)}",
        title="Profile Status",
    ))


def show_niches_table(console: Console, niches: List[dict]) -> None:
    """Display table of available niches."""
    if not niches:
        console.print("[yellow]No niches found.[/yellow]")
        return

    table = Table(title="Available Niches")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")

    for niche in niches:
        table.add_row(
            niche.get("id", ""),
            niche.get("name", ""),
            niche.get("description", "")[:50] + ("..." if len(niche.get("description", "")) > 50 else ""),
        )

    console.print(table)


def show_profile_created(console: Console, name: str, path: str) -> None:
    """Display profile creation result."""
    console.print(Panel(
        f"[bold green]Profile created successfully![/bold green]\n\n"
        f"Name: [cyan]{name}[/cyan]\n"
        f"Path: [dim]{path}[/dim]",
        title="New Profile",
        border_style="green",
    ))


def show_artifacts_update_result(
    console: Console,
    updated: int,
    skipped: int,
    dry_run: bool,
) -> None:
    """Display artifacts update result."""
    if dry_run:
        console.print(Panel(
            f"[yellow]Dry run complete[/yellow]\n\n"
            f"Would update: [green]{updated}[/green]\n"
            f"Would skip: [dim]{skipped}[/dim]",
            title="Dry Run Results",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"[bold green]Artifacts updated![/bold green]\n\n"
            f"Updated: [green]{updated}[/green]\n"
            f"Skipped: [dim]{skipped}[/dim]",
            title="Complete",
            border_style="green",
        ))


def show_platform_migration_result(
    console: Console,
    updated: int,
    skipped: int,
    errors: int,
    dry_run: bool,
) -> None:
    """Display platform status migration result."""
    if dry_run:
        console.print(Panel(
            f"[yellow]Dry run complete[/yellow]\n\n"
            f"Would update: [green]{updated}[/green]\n"
            f"Already migrated: [dim]{skipped}[/dim]\n"
            f"Errors: [{'red' if errors else 'dim'}]{errors}[/]",
            title="Migration Preview",
            border_style="yellow",
        ))
    else:
        border = "green" if errors == 0 else "yellow"
        console.print(Panel(
            f"[bold green]Migration complete![/bold green]\n\n"
            f"Updated: [green]{updated}[/green]\n"
            f"Already migrated: [dim]{skipped}[/dim]\n"
            f"Errors: [{'red' if errors else 'dim'}]{errors}[/]",
            title="Platform Status Migration",
            border_style=border,
        ))


def show_cleanup_header(
    console: Console,
    profile: str,
    total_reels: int,
    older_than_days: Optional[int],
    dry_run: bool,
) -> None:
    """Display cleanup operation header."""
    mode = "[yellow]DRY RUN[/yellow] - " if dry_run else ""
    age_info = f" older than {older_than_days} days" if older_than_days else ""

    console.print()
    console.print(Panel(
        f"{mode}[bold]Cleanup Reels[/bold]\n\n"
        f"Profile: [cyan]{profile}[/cyan]\n"
        f"Reels to process: [white]{total_reels}[/white]{age_info}",
        title=">>> CLEANUP",
        border_style="cyan",
    ))
    console.print()


def show_cleanup_progress(
    console: Console,
    current: int,
    total: int,
    reel_name: str,
    status: str,
    space_mb: float,
    video_url: Optional[str] = None,
) -> None:
    """Display cleanup progress for a single reel."""
    status_icon = {
        "cleaned": "[green][OK][/green]",
        "would_clean": "[yellow][DRY][/yellow]",
        "skipped": "[dim][--][/dim]",
        "already_cleaned": "[dim][OK][/dim]",
        "error": "[red][X][/red]",
    }.get(status, "[?]")

    space_info = f" ({space_mb:.1f} MB)" if space_mb > 0 else ""
    url_info = f" [dim]+URL[/dim]" if video_url else ""

    console.print(f"  [{current:3d}/{total}] {status_icon} {reel_name[:50]}{space_info}{url_info}")


def show_cleanup_result(
    console: Console,
    cleaned: int,
    skipped: int,
    already_cleaned: int,
    errors: int,
    total_space_mb: float,
    dry_run: bool,
) -> None:
    """Display cleanup operation result."""
    if dry_run:
        console.print()
        console.print(Panel(
            f"[yellow]Dry run complete[/yellow]\n\n"
            f"Would clean: [green]{cleaned}[/green] reels\n"
            f"Already cleaned: [dim]{already_cleaned}[/dim]\n"
            f"Skipped: [dim]{skipped}[/dim]\n"
            f"Errors: [{'red' if errors else 'dim'}]{errors}[/]\n\n"
            f"Space to free: [bold cyan]{total_space_mb:.1f} MB[/bold cyan] ({total_space_mb/1024:.2f} GB)",
            title="Dry Run Results",
            border_style="yellow",
        ))
    else:
        border = "green" if errors == 0 else "yellow"
        console.print()
        console.print(Panel(
            f"[bold green]Cleanup complete![/bold green]\n\n"
            f"Cleaned: [green]{cleaned}[/green] reels\n"
            f"Already cleaned: [dim]{already_cleaned}[/dim]\n"
            f"Skipped: [dim]{skipped}[/dim]\n"
            f"Errors: [{'red' if errors else 'dim'}]{errors}[/]\n\n"
            f"Space freed: [bold cyan]{total_space_mb:.1f} MB[/bold cyan] ({total_space_mb/1024:.2f} GB)",
            title="Cleanup Complete",
            border_style=border,
        ))
