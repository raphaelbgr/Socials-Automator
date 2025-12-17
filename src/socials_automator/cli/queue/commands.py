"""Queue CLI commands - thin wrappers orchestrating display and service."""

from __future__ import annotations

import typer

from ..core.console import console
from ..core.paths import get_profile_path
from .display import (
    show_generated_posts_list,
    show_no_posts_found,
    show_queue_table,
    show_schedule_action,
    show_schedule_result,
)
from .service import (
    find_all_queued_posts,
    find_generated_posts,
    schedule_post,
)


def queue(
    profile: str = typer.Argument(..., help="Profile name"),
) -> None:
    """List all posts in the publishing queue.

    Shows posts from both 'generated' and 'pending-post' folders,
    sorted by timestamp (oldest first).
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find all queued posts
    posts = find_all_queued_posts(profile_path)

    # Convert to display format
    display_posts = [
        {
            "folder": post.folder,
            "topic": post.topic,
            "slides": post.slides,
            "status": post.status,
            "year": post.year,
            "month": post.month,
        }
        for post in posts
    ]

    show_queue_table(console, display_posts)


def schedule(
    profile: str = typer.Argument(..., help="Profile name"),
    all_posts: bool = typer.Option(False, "--all", "-a", help="Schedule all generated posts"),
) -> None:
    """Move generated posts to pending-post queue.

    Workflow:
        1. generate-post: Creates posts in posts/YYYY/MM/generated/
        2. schedule: Moves to posts/YYYY/MM/pending-post/ (this command)
        3. upload-post: Publishes to Instagram, moves to posts/YYYY/MM/posted/
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find generated posts
    generated = find_generated_posts(profile_path)

    if not generated:
        show_no_posts_found(console)
        raise typer.Exit(1)

    # Display list
    display_posts = [
        {"folder": post.folder, "topic": post.topic}
        for post in generated
    ]
    show_generated_posts_list(console, display_posts)

    # Determine which posts to schedule
    if all_posts:
        posts_to_schedule = generated
    else:
        # Interactive selection
        console.print("\n[yellow]Enter post number to schedule (or 'all' for all):[/yellow]")
        selection = input("> ").strip().lower()

        if selection == "all":
            posts_to_schedule = generated
        else:
            try:
                index = int(selection) - 1
                if 0 <= index < len(generated):
                    posts_to_schedule = [generated[index]]
                else:
                    console.print("[red]Invalid selection.[/red]")
                    raise typer.Exit(1)
            except ValueError:
                console.print("[red]Invalid input. Enter a number or 'all'.[/red]")
                raise typer.Exit(1)

    # Schedule posts
    scheduled = 0
    skipped = 0

    for post in posts_to_schedule:
        if schedule_post(post):
            show_schedule_action(console, post.folder, str(post.path), "")
            scheduled += 1
        else:
            console.print(f"  [red]Failed:[/red] {post.folder}")
            skipped += 1

    # Show result
    show_schedule_result(console, scheduled, skipped, len(posts_to_schedule))
