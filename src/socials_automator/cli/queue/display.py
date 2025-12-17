"""Display functions for queue commands - pure functions for Rich output."""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def show_queue_table(console: Console, posts: List[dict]) -> None:
    """Display table of queued posts."""
    if not posts:
        console.print("[yellow]No posts in queue.[/yellow]")
        console.print("\n[dim]Generate posts first:[/dim]")
        console.print("  [cyan]python -m socials_automator.cli generate-post <profile> --topic '...'[/cyan]")
        return

    table = Table(title="Post Queue")
    table.add_column("#", style="dim")
    table.add_column("Status", style="yellow")
    table.add_column("Folder", style="cyan")
    table.add_column("Topic", style="white")
    table.add_column("Slides", style="green")
    table.add_column("Date", style="dim")

    for i, post in enumerate(posts, 1):
        status_style = "yellow" if post.get("status") == "pending" else "dim"
        table.add_row(
            str(i),
            f"[{status_style}]{post.get('status', 'unknown')}[/{status_style}]",
            post.get("folder", ""),
            post.get("topic", "Unknown")[:35] + ("..." if len(post.get("topic", "")) > 35 else ""),
            str(post.get("slides", 0)),
            f"{post.get('year', '')}/{post.get('month', '')}",
        )

    console.print(table)

    # Summary
    generated = sum(1 for p in posts if p.get("status") == "generated")
    pending = sum(1 for p in posts if p.get("status") == "pending")

    console.print(f"\n[bold]Total:[/] {len(posts)} posts")
    console.print(f"  Generated: [dim]{generated}[/dim]")
    console.print(f"  Pending: [yellow]{pending}[/yellow]")


def show_generated_posts_list(console: Console, posts: List[dict]) -> None:
    """Display list of generated posts for scheduling."""
    console.print(f"\n[bold]Found {len(posts)} generated post(s):[/bold]")

    for i, post in enumerate(posts, 1):
        topic = post.get("topic", "Unknown")[:40]
        console.print(f"  {i}. [{post.get('folder', '')}] {topic}")


def show_schedule_prompt(console: Console) -> None:
    """Display scheduling prompt."""
    console.print("\n[yellow]Enter post number to schedule (or 'all' for all):[/yellow]")


def show_schedule_result(
    console: Console,
    scheduled: int,
    skipped: int,
    total: int,
) -> None:
    """Display scheduling result."""
    if scheduled == 0:
        console.print("[yellow]No posts were scheduled.[/yellow]")
    elif scheduled == total:
        console.print(Panel(
            f"[bold green]Scheduled {scheduled} post(s) successfully![/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[yellow]Scheduled {scheduled} post(s), {skipped} skipped[/yellow]",
            border_style="yellow",
        ))


def show_schedule_action(console: Console, folder: str, from_path: str, to_path: str) -> None:
    """Display scheduling action for a single post."""
    console.print(f"  [green]Scheduled:[/green] {folder}")


def show_no_posts_found(console: Console) -> None:
    """Display message when no posts are found."""
    console.print("[yellow]No posts in generated folders.[/yellow]")
    console.print("\n[dim]Generate posts first:[/dim]")
    console.print("  [cyan]python -m socials_automator.cli generate-post <profile> --topic '...'[/cyan]")
