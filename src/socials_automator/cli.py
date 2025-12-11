"""Command-line interface for Socials Automator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich import box

from .content import ContentGenerator
from .content.models import GenerationProgress

# Create Typer app
app = typer.Typer(
    name="socials",
    help="AI-powered Instagram carousel content generator",
    add_completion=False,
)

console = Console()


class VerboseProgressDisplay:
    """Rich display for verbose progress tracking."""

    def __init__(self):
        self.events: list[dict] = []
        self.current_status = "Initializing..."
        self.stats = {
            "text_calls": 0,
            "image_calls": 0,
            "total_cost": 0.0,
            "providers_used": set(),
            "text_provider": None,
            "image_provider": None,
        }

    def add_event(self, progress: GenerationProgress):
        """Add a progress event."""
        event = {
            "step": progress.current_step,
            "event_type": progress.event_type,
            "provider": progress.provider,
            "model": progress.model,
            "prompt": progress.prompt_preview,
            "response": progress.response_preview,
            "duration": progress.duration_seconds,
            "cost": progress.cost_usd,
        }
        self.events.append(event)
        self.current_status = progress.current_step

        # Update stats from progress totals (more reliable)
        self.stats["text_calls"] = progress.total_text_calls
        self.stats["image_calls"] = progress.total_image_calls
        self.stats["total_cost"] = progress.total_cost_usd

        # Track providers (persist values)
        if progress.text_provider:
            self.stats["text_provider"] = progress.text_provider
            self.stats["text_model"] = progress.text_model
            self.stats["providers_used"].add(progress.text_provider)
        if progress.image_provider:
            self.stats["image_provider"] = progress.image_provider
            self.stats["image_model"] = progress.image_model
            self.stats["providers_used"].add(progress.image_provider)

    def render(self, progress: GenerationProgress) -> Panel:
        """Render the progress display."""
        # Build content
        lines = []

        # Status header
        status_color = {
            "planning": "yellow",
            "generating": "cyan",
            "completed": "green",
            "error": "red",
        }.get(progress.status, "white")

        lines.append(f"[bold {status_color}]Status:[/] {progress.status.upper()}")
        lines.append(f"[bold]Step:[/] {progress.current_step}")
        lines.append(f"[bold]Progress:[/] {progress.completed_steps}/{progress.total_steps} ({progress.progress_percent:.0f}%)")
        lines.append("")

        # Current slide info
        if progress.total_slides > 0:
            lines.append(f"[bold cyan]Slide:[/] {progress.current_slide}/{progress.total_slides}")
            lines.append("")

        # Text AI Activity - use persisted stats if current progress doesn't have it
        text_provider = progress.text_provider or self.stats.get("text_provider")
        text_model = progress.text_model or self.stats.get("text_model")
        if text_provider or progress.event_type.startswith("text_"):
            lines.append("[bold magenta]Text AI:[/]")
            if text_provider:
                lines.append(f"  Provider: [green]{text_provider}[/]")
            if text_model:
                lines.append(f"  Model: [blue]{text_model}[/]")
            if progress.text_prompt_preview:
                lines.append(f"  [dim]Prompt:[/] {progress.text_prompt_preview[:80]}...")
            if progress.text_failed_providers:
                failed = " | ".join(progress.text_failed_providers)
                lines.append(f"  [dim]Previous attempts: {failed}[/]")
            lines.append("")

        # Image AI Activity - use persisted stats if current progress doesn't have it
        image_provider = progress.image_provider or self.stats.get("image_provider")
        image_model = progress.image_model or self.stats.get("image_model")
        if image_provider or progress.event_type.startswith("image_"):
            lines.append("[bold magenta]Image AI:[/]")
            if image_provider:
                lines.append(f"  Provider: [green]{image_provider}[/]")
            if image_model:
                lines.append(f"  Model: [blue]{image_model}[/]")
            if progress.image_prompt_preview:
                lines.append(f"  [dim]Prompt:[/] {progress.image_prompt_preview[:80]}...")
            if progress.image_failed_providers:
                failed = " | ".join(progress.image_failed_providers)
                lines.append(f"  [dim]Previous attempts: {failed}[/]")
            lines.append("")

        # Stats
        lines.append("[bold]Session Stats:[/]")
        lines.append(f"  Text API calls: {progress.total_text_calls}")
        lines.append(f"  Image API calls: {progress.total_image_calls}")
        lines.append(f"  Total cost: ${progress.total_cost_usd:.4f}")

        return Panel(
            "\n".join(lines),
            title=f"[bold]Post: {progress.post_id}[/]",
            border_style="blue",
            box=box.ROUNDED,
        )


def get_profiles_dir() -> Path:
    """Get the profiles directory."""
    return Path.cwd() / "profiles"


def get_profile_path(profile: str) -> Path:
    """Get path to a profile directory."""
    return get_profiles_dir() / profile


def load_profile_config(profile_path: Path) -> dict:
    """Load profile configuration."""
    metadata_path = profile_path / "metadata.json"
    if not metadata_path.exists():
        raise typer.BadParameter(f"Profile not found: {profile_path.name}")

    with open(metadata_path) as f:
        return json.load(f)


@app.command()
def list_profiles():
    """List all available profiles."""
    profiles_dir = get_profiles_dir()

    if not profiles_dir.exists():
        console.print("[yellow]No profiles directory found.[/yellow]")
        return

    profiles = [
        d.name for d in profiles_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]

    if not profiles:
        console.print("[yellow]No profiles found.[/yellow]")
        return

    table = Table(title="Available Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Handle", style="green")
    table.add_column("Niche", style="yellow")

    for profile_name in profiles:
        config = load_profile_config(get_profile_path(profile_name))
        profile_info = config.get("profile", {})
        table.add_row(
            profile_name,
            profile_info.get("instagram_handle", ""),
            profile_info.get("niche_id", ""),
        )

    console.print(table)


@app.command()
def generate(
    profile: str = typer.Argument(..., help="Profile name to generate for"),
    topic: str = typer.Option(None, "--topic", "-t", help="Topic for the post"),
    pillar: str = typer.Option(None, "--pillar", "-p", help="Content pillar"),
    count: int = typer.Option(1, "--count", "-n", help="Number of posts to generate"),
    slides: int = typer.Option(None, "--slides", "-s", help="Number of slides (default: AI decides)"),
    min_slides: int = typer.Option(3, "--min-slides", help="Minimum slides when AI decides"),
    max_slides: int = typer.Option(10, "--max-slides", help="Maximum slides when AI decides"),
):
    """Generate carousel posts for a profile.

    By default, the AI decides the optimal number of slides (3-10) based on
    the topic content. Use --slides to force a specific count.
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    config = load_profile_config(profile_path)

    # Get slide settings from profile if not overridden
    carousel_settings = config.get("content_strategy", {}).get("carousel_settings", {})
    profile_min = carousel_settings.get("min_slides", 3)
    profile_max = carousel_settings.get("max_slides", 10)

    # Use profile defaults if not specified on CLI
    effective_min = min_slides if min_slides != 3 else profile_min
    effective_max = max_slides if max_slides != 10 else profile_max

    slides_info = f"{slides} slides" if slides else f"AI decides ({effective_min}-{effective_max} slides)"
    console.print(Panel(
        f"Generating {count} post(s) for [cyan]{profile}[/cyan]\n"
        f"Slide count: [yellow]{slides_info}[/yellow]",
        title="Socials Automator",
    ))

    # Run async generation
    asyncio.run(_generate_posts(
        profile_path=profile_path,
        config=config,
        topic=topic,
        pillar=pillar,
        count=count,
        slides=slides,
        min_slides=min_slides,
        max_slides=max_slides,
    ))


async def _generate_posts(
    profile_path: Path,
    config: dict,
    topic: str | None,
    pillar: str | None,
    count: int,
    slides: int | None,
    min_slides: int = 3,
    max_slides: int = 10,
    verbose: bool = True,
):
    """Async post generation with progress display."""
    display = VerboseProgressDisplay()
    last_progress: GenerationProgress | None = None

    async def progress_callback(progress: GenerationProgress):
        nonlocal last_progress
        last_progress = progress
        display.add_event(progress)

    generator = ContentGenerator(
        profile_path=profile_path,
        profile_config=config,
        progress_callback=progress_callback,
    )

    # Get default content pillar if not specified
    if pillar is None:
        pillars = config.get("content_strategy", {}).get("content_pillars", [])
        pillar = pillars[0]["id"] if pillars else "general"

    if verbose:
        # Verbose mode with live updating panel
        post = None
        output_path = None
        posts = []  # For multi-post generation

        with Live(console=console, refresh_per_second=4) as live:
            if topic:
                # Generate single post with specified topic
                console.print(f"\n[bold cyan]Starting generation:[/] {topic}\n")

                async def update_display():
                    while True:
                        if last_progress:
                            live.update(display.render(last_progress))
                        await asyncio.sleep(0.25)

                # Run generation with display updates
                display_task = asyncio.create_task(update_display())

                try:
                    post = await generator.generate_post(
                        topic=topic,
                        content_pillar=pillar,
                        target_slides=slides,
                        min_slides=min_slides,
                        max_slides=max_slides,
                    )
                    output_path = await generator.save_post(post)
                finally:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass

                # Final update
                if last_progress:
                    live.update(display.render(last_progress))

            else:
                # No topic - use research to find topics
                console.print(f"\n[bold cyan]Researching trending topics and generating {count} post(s)...[/]\n")

                async def update_display():
                    while True:
                        if last_progress:
                            live.update(display.render(last_progress))
                        await asyncio.sleep(0.25)

                display_task = asyncio.create_task(update_display())

                try:
                    posts = await generator.generate_daily_posts(count=count)
                finally:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass

        # Summary output (outside Live context for clean display)
        if posts:
            # Show summary for multiple posts
            console.print(f"\n[green]Generated {len(posts)} post(s)[/green]")
            for p in posts:
                console.print(f"  - {p.topic} ({p.slides_count} slides)")
            # Show output path for last post
            if posts:
                last_post_path = generator._get_output_path(posts[-1])
                console.print(f"\n[dim]Output: {last_post_path.parent}[/dim]")
        elif post and output_path:
            # Summary for single post
            console.print("\n")
            _print_generation_summary(post, output_path, display.stats)

    else:
        # Simple progress bar mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            if topic:
                task = progress.add_task(f"Generating: {topic[:40]}...", total=100)

                post = await generator.generate_post(
                    topic=topic,
                    content_pillar=pillar,
                    target_slides=slides,
                    min_slides=min_slides,
                    max_slides=max_slides,
                )
                output_path = await generator.save_post(post)
                progress.update(task, completed=100)

                console.print(f"\n[green]Post saved to:[/green] {output_path}")

            else:
                task = progress.add_task(f"Generating {count} posts...", total=count * 100)
                posts = await generator.generate_daily_posts(count=count)
                progress.update(task, completed=count * 100)

                console.print(f"\n[green]Generated {len(posts)} post(s)[/green]")
                for post in posts:
                    console.print(f"  - {post.topic} ({post.slides_count} slides)")


def _print_generation_summary(post, output_path: Path, stats: dict):
    """Print a summary after generation."""
    # Create summary table
    table = Table(title="Generation Complete", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Post ID", post.id)
    table.add_row("Topic", post.topic[:50] + "..." if len(post.topic) > 50 else post.topic)
    table.add_row("Slides", str(post.slides_count))
    table.add_row("Hook Type", post.hook_type.value)
    table.add_row("Text Provider", post.text_provider or "N/A")
    table.add_row("Image Provider", post.image_provider or "N/A")
    table.add_row("Generation Time", f"{post.generation_time_seconds:.1f}s" if post.generation_time_seconds else "N/A")
    table.add_row("Total Cost", f"${stats['total_cost']:.4f}")
    table.add_row("Output Path", str(output_path))

    console.print(table)

    # Show hook preview
    console.print(Panel(
        f"[bold]{post.hook_text}[/bold]",
        title="Hook Preview",
        border_style="magenta",
    ))

    # Show slide headings (using ASCII-safe markers for Windows compatibility)
    console.print("\n[bold]Slide Content:[/]")
    for slide in post.slides:
        marker = {"hook": "[H]", "content": "[C]", "cta": "[>]"}.get(slide.slide_type.value, "[*]")
        console.print(f"  {marker} Slide {slide.number}: {slide.heading[:60]}{'...' if len(slide.heading) > 60 else ''}")


@app.command()
def new_profile(
    name: str = typer.Argument(..., help="Profile folder name"),
    handle: str = typer.Option(..., "--handle", "-h", help="Instagram handle"),
    niche: str = typer.Option(None, "--niche", "-n", help="Niche ID from niches.json"),
):
    """Create a new profile."""
    from datetime import datetime

    profile_path = get_profile_path(name)

    if profile_path.exists():
        console.print(f"[red]Profile already exists: {name}[/red]")
        raise typer.Exit(1)

    # Load niche data if specified
    niche_config = {}
    if niche:
        niches_path = Path.cwd() / "docs" / "niches.json"
        if niches_path.exists():
            with open(niches_path) as f:
                niches_data = json.load(f)
            for n in niches_data.get("niches", []):
                if n["id"] == niche:
                    niche_config = n
                    break

    # Create profile structure
    profile_path.mkdir(parents=True)
    (profile_path / "brand" / "fonts").mkdir(parents=True)
    (profile_path / "posts").mkdir()
    (profile_path / "knowledge").mkdir()

    # Create metadata
    metadata = {
        "version": "1.0.0",
        "profile": {
            "id": name.replace(".", "-"),
            "name": name,
            "display_name": handle.replace("@", "").replace(".", " ").title(),
            "instagram_handle": handle if handle.startswith("@") else f"@{handle}",
            "niche_id": niche or "general",
            "tagline": niche_config.get("description", ""),
            "bio": "",
            "description": niche_config.get("description", ""),
            "language": "en",
        },
        "content_strategy": {
            "posts_per_day": 3,
            "content_pillars": niche_config.get("content_pillars", []),
        },
        "hashtag_strategy": {
            "hashtag_sets": niche_config.get("hashtags", {}),
        },
        "research_sources": {
            "subreddits": niche_config.get("research_sources", {}).get("subreddits", []),
            "keywords": niche_config.get("research_sources", {}).get("keywords", []),
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
        },
    }

    with open(profile_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]Created profile:[/green] {profile_path}")
    console.print(f"  Handle: {metadata['profile']['instagram_handle']}")
    console.print(f"  Niche: {niche or 'general'}")
    console.print(f"\nEdit [cyan]{profile_path / 'metadata.json'}[/cyan] to customize.")


@app.command()
def list_niches():
    """List available niches from niches.json."""
    niches_path = Path.cwd() / "docs" / "niches.json"

    if not niches_path.exists():
        console.print("[yellow]niches.json not found in docs/[/yellow]")
        return

    with open(niches_path) as f:
        data = json.load(f)

    table = Table(title="Available Niches")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Tier", style="yellow")
    table.add_column("Competition", style="red")

    for niche in data.get("niches", []):
        table.add_row(
            niche.get("id", ""),
            niche.get("name", ""),
            str(niche.get("tier", "")),
            niche.get("competition", ""),
        )

    console.print(table)


@app.command()
def status(
    profile: str = typer.Argument(..., help="Profile name"),
):
    """Show profile status and recent posts."""
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    config = load_profile_config(profile_path)
    profile_info = config.get("profile", {})

    # Profile info
    console.print(Panel(
        f"[cyan]{profile_info.get('instagram_handle', profile)}[/cyan]\n"
        f"{profile_info.get('tagline', '')}\n\n"
        f"Niche: {profile_info.get('niche_id', 'general')}",
        title=profile_info.get("display_name", profile),
    ))

    # Load knowledge store for stats
    from .knowledge import KnowledgeStore
    store = KnowledgeStore(profile_path)

    recent_posts = store.get_recent_posts(days=7)

    if recent_posts:
        table = Table(title="Recent Posts (Last 7 Days)")
        table.add_column("Date", style="cyan")
        table.add_column("Topic", style="green")
        table.add_column("Slides", style="yellow")

        for post in recent_posts[-10:]:
            table.add_row(
                post.date,
                post.topic[:50] + "..." if len(post.topic) > 50 else post.topic,
                str(post.slides_count),
            )

        console.print(table)
    else:
        console.print("[yellow]No posts generated yet.[/yellow]")

    # Keyword stats
    freq = store.get_keyword_frequency()
    if freq:
        top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        console.print("\n[bold]Top Keywords:[/bold]")
        for kw, count in top_keywords:
            console.print(f"  {kw}: {count}")


@app.command()
def init():
    """Initialize project structure."""
    dirs = [
        "profiles",
        "config",
        "docs",
        "assets/fonts",
    ]

    for dir_path in dirs:
        path = Path.cwd() / dir_path
        path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created:[/green] {dir_path}/")

    # Create .env.example if not exists
    env_example = Path.cwd() / ".env.example"
    if not env_example.exists():
        env_example.write_text("""# Socials Automator - Environment Variables
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
FAL_API_KEY=...
REPLICATE_API_TOKEN=r8_...
""")
        console.print("[green]Created:[/green] .env.example")

    console.print("\n[bold green]Project initialized![/bold green]")
    console.print("Next steps:")
    console.print("  1. Copy .env.example to .env and add your API keys")
    console.print("  2. Run: socials new-profile my-profile --handle @myhandle")
    console.print("  3. Run: socials generate my-profile --topic 'Your topic'")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
