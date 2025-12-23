"""Display functions for reel commands - pure functions for Rich output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.parsers import format_duration
from .params import ReelGenerationParams, ReelUploadParams


def show_reel_config(console: Console, params: ReelGenerationParams) -> None:
    """Display reel generation configuration panel."""
    # Format GPU info
    gpu_info = "Disabled"
    if params.gpu_accelerate:
        if params.gpu_index is not None:
            gpu_info = f"Enabled (GPU {params.gpu_index})"
        else:
            gpu_info = "Enabled (auto-select)"

    # Format voice info
    voice_info = params.voice
    if params.voice_rate != "+0%" or params.voice_pitch != "+0Hz":
        voice_info += f" (rate={params.voice_rate}, pitch={params.voice_pitch})"

    # Format loop info
    loop_info = ""
    if params.loop:
        if params.loop_count:
            loop_info = f"\nLoop: [yellow]{params.loop_count} videos[/yellow]"
        else:
            loop_info = "\nLoop: [yellow]Infinite (Ctrl+C to stop)[/yellow]"
        if params.loop_each:
            interval_str = format_duration(params.loop_each)
            loop_info += f"\nInterval: [yellow]{interval_str}[/yellow]"

    gpu_style = "green" if params.gpu_accelerate else "dim"

    # Format overlay images info
    overlay_style = "green" if params.overlay_images else "dim"
    if params.overlay_images:
        tor_info = " via Tor" if params.use_tor else ""
        dim_info = f", dim={params.blur}" if params.blur else ""
        smart_info = f", smart-pick={params.smart_pick_count}" if params.smart_pick else ""
        overlay_info = f"Enabled ({params.image_provider}{tor_info}{dim_info}{smart_info})"
    else:
        overlay_info = "Disabled"

    # Format news info
    news_info = ""
    if params.is_news_profile:
        edition = params.news_edition or "auto"
        if params.news_story_count:
            stories = str(params.news_story_count)
        else:
            # Calculate what auto will produce based on duration
            from socials_automator.news.curator import calculate_stories_for_duration
            auto_count = calculate_stories_for_duration(params.target_duration)
            stories = f"auto (~{auto_count} for {format_duration(params.target_duration)})"
        news_info = (
            f"\n[bold magenta]News Mode:[/bold magenta] Enabled\n"
            f"Edition: [magenta]{edition}[/magenta]\n"
            f"Stories: [magenta]{stories}[/magenta]\n"
            f"Max Age: [magenta]{params.news_max_age_hours}h[/magenta]"
        )

    # Choose title based on mode
    title = "News Briefing Generation" if params.is_news_profile else "Video Reel Generation"

    console.print(Panel(
        f"Generating video reel for [cyan]{params.profile}[/cyan]\n"
        f"Text AI: [yellow]{params.text_ai or 'default'}[/yellow]\n"
        f"Video Matcher: [yellow]{params.video_matcher}[/yellow]\n"
        f"Voice: [yellow]{voice_info}[/yellow]\n"
        f"Subtitle Size: [yellow]{params.subtitle_size}px[/yellow]\n"
        f"Font: [yellow]{params.font}[/yellow]\n"
        f"Target Length: [yellow]{format_duration(params.target_duration)}[/yellow]\n"
        f"GPU Acceleration: [{gpu_style}]{gpu_info}[/{gpu_style}]\n"
        f"Image Overlays: [{overlay_style}]{overlay_info}[/{overlay_style}]\n"
        f"Topic: [green]{params.topic or 'Auto-generated'}[/green]"
        f"{loop_info}"
        f"{news_info}",
        title=title,
    ))


def show_reel_result(
    console: Console,
    video_path: Path,
    duration: int,
    video_count: Optional[int] = None,
    loop_limit: Optional[int] = None,
) -> None:
    """Display successful generation result."""
    # Build title with loop progress
    if video_count is not None:
        if loop_limit:
            title = f"Complete (Video #{video_count}/{loop_limit})"
        else:
            title = f"Complete (Video #{video_count})"
    else:
        title = "Complete"

    console.print(Panel(
        f"[bold green]Video generated successfully![/bold green]\n\n"
        f"[bold]Output:[/] {video_path}\n"
        f"[bold]Duration:[/] {duration} seconds\n"
        f"[bold]Resolution:[/] 1080x1920 (9:16)",
        title=title,
        border_style="green",
    ))


def show_reel_error(console: Console, error: str, details: Optional[dict] = None) -> None:
    """Display generation error."""
    console.print(f"\n[red]Error: {error}[/red]")
    if details:
        for key, value in details.items():
            console.print(f"  [dim]{key}:[/dim] [yellow]{value}[/yellow]")


def show_upload_config(console: Console, params: ReelUploadParams) -> None:
    """Display upload configuration."""
    mode = "Single reel" if params.post_one else "All pending reels"
    if params.reel_id:
        mode = f"Specific reel: {params.reel_id}"

    # Format platforms list
    platforms_str = ", ".join(params.platforms) if params.platforms else "instagram"

    console.print(Panel(
        f"Uploading reels for [cyan]{params.profile}[/cyan]\n"
        f"Platforms: [magenta]{platforms_str}[/magenta]\n"
        f"Mode: [yellow]{mode}[/yellow]\n"
        f"Dry Run: [{'yellow' if params.dry_run else 'dim'}]{params.dry_run}[/]",
        title="Reel Upload",
    ))


def show_upload_result(
    console: Console,
    success_count: int,
    failed_count: int,
    results: list,
) -> None:
    """Display upload results summary."""
    # Build summary content
    lines = []

    if success_count > 0:
        lines.append(f"[green][OK] Uploaded: {success_count} reel(s)[/green]")

    if failed_count > 0:
        lines.append(f"[red][X] Failed: {failed_count} reel(s)[/red]")

        # Show failure details
        for r in results:
            if not r.get("success") and r.get("error"):
                reel_name = r.get("path", "").name if hasattr(r.get("path", ""), "name") else str(r.get("path", ""))
                lines.append(f"    [dim]{reel_name}:[/dim] [red]{r.get('error')}[/red]")

    # Determine border style
    if failed_count == 0:
        border_style = "green"
        title = "Upload Complete"
    elif success_count > 0:
        border_style = "yellow"
        title = "Upload Partial"
    else:
        border_style = "red"
        title = "Upload Failed"

    console.print(Panel(
        "\n".join(lines),
        title=title,
        border_style=border_style,
    ))


def show_pending_reels_table(console: Console, reels: list) -> None:
    """Display table of pending reels."""
    if not reels:
        console.print("[yellow]No pending reels found.[/yellow]")
        return

    table = Table(title="Pending Reels")
    table.add_column("Folder", style="cyan")
    table.add_column("Topic", style="white")
    table.add_column("Duration", style="yellow")
    table.add_column("Created", style="dim")

    for reel in reels:
        table.add_row(
            reel.get("folder", ""),
            reel.get("topic", "Unknown")[:40],
            f"{reel.get('duration', 0)}s",
            reel.get("created", ""),
        )

    console.print(table)
    console.print(f"\nTotal: {len(reels)} reel(s)")


def show_loop_progress(
    console: Console,
    video_count: int,
    loop_limit: Optional[int],
    wait_seconds: int = 3,
) -> None:
    """Display loop progress info."""
    wait_str = format_duration(wait_seconds) if wait_seconds >= 60 else f"{wait_seconds} seconds"
    if loop_limit:
        remaining = loop_limit - video_count
        console.print(
            f"\n[dim]Starting next video in {wait_str}... "
            f"({remaining} remaining) (Ctrl+C to stop)[/dim]"
        )
    else:
        console.print(
            f"\n[dim]Starting next video in {wait_str}... (Ctrl+C to stop)[/dim]"
        )


def show_loop_complete(console: Console, total: int) -> None:
    """Display loop completion message."""
    console.print(f"\n[bold green]Completed all {total} videos![/bold green]")
