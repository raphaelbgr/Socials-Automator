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
