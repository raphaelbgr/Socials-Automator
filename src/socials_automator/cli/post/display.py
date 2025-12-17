"""Display functions for post commands - pure functions for Rich output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .params import PostGenerationParams, PostUploadParams


def show_post_config(console: Console, params: PostGenerationParams) -> None:
    """Display post generation configuration panel."""
    # Format slides info
    if params.slides:
        slides_info = f"{params.slides} slides"
    else:
        slides_info = f"AI decides ({params.min_slides}-{params.max_slides} slides)"

    # Format loop info
    loop_info = ""
    if params.loop_seconds:
        minutes = params.loop_seconds // 60
        if minutes > 0:
            loop_info = f"\nLoop: [yellow]Every {minutes}m (Ctrl+C to stop)[/yellow]"
        else:
            loop_info = f"\nLoop: [yellow]Every {params.loop_seconds}s (Ctrl+C to stop)[/yellow]"

    # Format AI tools info
    ai_tools_info = "Enabled" if params.ai_tools else "Disabled"
    ai_tools_style = "green" if params.ai_tools else "dim"

    console.print(Panel(
        f"Generating carousel post for [cyan]{params.profile}[/cyan]\n"
        f"Topic: [green]{params.topic or 'Auto-generated'}[/green]\n"
        f"Pillar: [yellow]{params.pillar or 'Auto-selected'}[/yellow]\n"
        f"Slides: [yellow]{slides_info}[/yellow]\n"
        f"Text AI: [yellow]{params.text_ai or 'default'}[/yellow]\n"
        f"Image AI: [yellow]{params.image_ai or 'default'}[/yellow]\n"
        f"Count: [yellow]{params.count}[/yellow]\n"
        f"AI Research: [{ai_tools_style}]{ai_tools_info}[/{ai_tools_style}]"
        f"{loop_info}",
        title="Carousel Post Generation",
    ))


def show_post_result(
    console: Console,
    output_path: Path,
    slide_count: int,
    topic: str,
    post_number: Optional[int] = None,
    total_posts: Optional[int] = None,
) -> None:
    """Display successful generation result."""
    # Build title with post progress
    if post_number is not None and total_posts is not None:
        title = f"Complete (Post #{post_number}/{total_posts})"
    elif post_number is not None:
        title = f"Complete (Post #{post_number})"
    else:
        title = "Complete"

    console.print(Panel(
        f"[bold green]Carousel generated successfully![/bold green]\n\n"
        f"[bold]Topic:[/] {topic}\n"
        f"[bold]Output:[/] {output_path}\n"
        f"[bold]Slides:[/] {slide_count}",
        title=title,
        border_style="green",
    ))


def show_post_error(console: Console, error: str, details: Optional[dict] = None) -> None:
    """Display generation error."""
    console.print(f"\n[red]Error: {error}[/red]")
    if details:
        for key, value in details.items():
            console.print(f"  [dim]{key}:[/dim] [yellow]{value}[/yellow]")


def show_upload_config(console: Console, params: PostUploadParams) -> None:
    """Display upload configuration."""
    mode = "Single post" if params.post_one else "All pending posts"
    if params.post_id:
        mode = f"Specific post: {params.post_id}"

    console.print(Panel(
        f"Uploading posts for [cyan]{params.profile}[/cyan]\n"
        f"Mode: [yellow]{mode}[/yellow]\n"
        f"Dry Run: [{'yellow' if params.dry_run else 'dim'}]{params.dry_run}[/]",
        title="Carousel Upload",
    ))


def show_upload_result(
    console: Console,
    success_count: int,
    failed_count: int,
    results: list,
) -> None:
    """Display upload results summary."""
    if success_count > 0 and failed_count == 0:
        console.print(Panel(
            f"[bold green]Successfully uploaded {success_count} post(s)![/bold green]",
            border_style="green",
        ))
    elif success_count > 0 and failed_count > 0:
        console.print(Panel(
            f"[yellow]Uploaded {success_count} post(s), {failed_count} failed[/yellow]",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"[red]All {failed_count} post(s) failed to upload[/red]",
            border_style="red",
        ))


def show_pending_posts_table(console: Console, posts: list) -> None:
    """Display table of pending posts."""
    if not posts:
        console.print("[yellow]No pending posts found.[/yellow]")
        return

    table = Table(title="Pending Posts")
    table.add_column("Folder", style="cyan")
    table.add_column("Topic", style="white")
    table.add_column("Slides", style="yellow")
    table.add_column("Created", style="dim")

    for post in posts:
        table.add_row(
            post.get("folder", ""),
            post.get("topic", "Unknown")[:40],
            str(post.get("slides", 0)),
            post.get("created", ""),
        )

    console.print(table)
    console.print(f"\nTotal: {len(posts)} post(s)")


def show_loop_progress(
    console: Console,
    post_count: int,
    loop_seconds: int,
) -> None:
    """Display loop progress info."""
    minutes = loop_seconds // 60
    if minutes > 0:
        console.print(
            f"\n[dim]Next post in {minutes} minutes... (Ctrl+C to stop)[/dim]"
        )
    else:
        console.print(
            f"\n[dim]Next post in {loop_seconds} seconds... (Ctrl+C to stop)[/dim]"
        )


def show_loop_stopped(console: Console, total: int) -> None:
    """Display loop stopped message."""
    console.print(f"\n[yellow]Loop stopped. Generated {total} post(s).[/yellow]")
