"""Command-line interface for Socials Automator."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich import box

from .content import ContentOrchestrator
from .content.models import GenerationProgress
from .cli_display import ContentGenerationDisplay, InstagramPostingDisplay

# Configure logging to suppress console output - all logs go to files only
# This prevents WARNING/INFO messages from polluting the Rich CLI display

# Remove any default console handlers from root logger
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(logging.CRITICAL)  # Suppress root logger output

# Suppress loggers that might print to console
for logger_name in ["httpx", "httpcore", "urllib3", "asyncio"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

# Setup ai_calls logger with FileHandler for full AI request/response logging
_log_dir = Path(__file__).parent.parent.parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

ai_calls_logger = logging.getLogger("ai_calls")
ai_calls_logger.setLevel(logging.DEBUG)
ai_calls_logger.propagate = False
ai_calls_logger.handlers = []  # Clear any existing handlers
_ai_file_handler = logging.FileHandler(_log_dir / "ai_calls.log", encoding="utf-8")
_ai_file_handler.setLevel(logging.DEBUG)
_ai_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
ai_calls_logger.addHandler(_ai_file_handler)

# Ensure instagram_api logger doesn't propagate to console
instagram_logger = logging.getLogger("instagram_api")
instagram_logger.propagate = False

# Create Typer app
app = typer.Typer(
    name="socials",
    help="AI-powered Instagram carousel content generator",
    add_completion=False,
)

console = Console()


def parse_interval(interval: str) -> int:
    """Parse interval string like '5m', '1h', '30s' to seconds."""
    import re
    match = re.match(r'^(\d+)(s|m|h)?$', interval.lower().strip())
    if not match:
        raise ValueError(f"Invalid interval format: {interval}. Use format like 5m, 1h, 30s")

    value = int(match.group(1))
    unit = match.group(2) or 's'

    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    return value


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

    with open(metadata_path, encoding="utf-8") as f:
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
    post_after: bool = typer.Option(False, "--post", help="Publish to Instagram after generating"),
    auto_retry: bool = typer.Option(False, "--auto-retry", help="Retry indefinitely until valid content"),
    text_ai: str = typer.Option(None, "--text-ai", help="Text AI provider (zai, groq, gemini, openai, lmstudio, ollama)"),
    image_ai: str = typer.Option(None, "--image-ai", help="Image AI provider (dalle, fal_flux, comfy)"),
    loop_each: str = typer.Option(None, "--loop-each", help="Loop interval (e.g., 5m, 1h, 30s)"),
    ai_tools: bool = typer.Option(False, "--ai-tools", help="Enable AI tool calling (AI decides when to search)"),
):
    """Generate carousel posts for a profile.

    By default, the AI decides the optimal number of slides (3-10) based on
    the topic content. Use --slides to force a specific count.

    Use --post to automatically publish to Instagram after generation.
    Use --auto-retry to keep retrying until valid content is generated.
    Use --text-ai to select text provider (zai, groq, gemini, openai, lmstudio, ollama).
    Use --image-ai to select image provider (dalle, fal_flux, comfy).
    Use --loop-each to continuously generate posts (e.g., --loop-each 5m).
    Use --ai-tools to enable AI-driven research (AI decides when and what to search).
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

    # Parse loop interval if provided
    loop_seconds = None
    if loop_each:
        try:
            loop_seconds = parse_interval(loop_each)
            console.print(Panel(
                f"Generating post(s) for [cyan]{profile}[/cyan]\n"
                f"Slide count: [yellow]{slides_info}[/yellow]\n"
                f"Loop: [green]Every {loop_each}[/green] (Ctrl+C to stop)",
                title="Socials Automator - Loop Mode",
            ))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(Panel(
            f"Generating {count} post(s) for [cyan]{profile}[/cyan]\n"
            f"Slide count: [yellow]{slides_info}[/yellow]",
            title="Socials Automator",
        ))

    # Show provider override info
    if text_ai:
        console.print(f"[dim]Text AI: {text_ai}[/dim]")
    if image_ai:
        console.print(f"[dim]Image AI: {image_ai}[/dim]")
    if ai_tools:
        console.print(f"[dim]AI Tools: enabled (AI decides when to search)[/dim]")

    # Run generation (with optional loop)
    if loop_seconds:
        # Loop mode
        import time
        iteration = 0
        try:
            while True:
                iteration += 1
                console.print(f"\n[bold cyan]--- Iteration {iteration} ---[/bold cyan]")
                asyncio.run(_generate_posts(
                    profile_path=profile_path,
                    config=config,
                    topic=None,  # Always auto-generate topic in loop mode
                    pillar=pillar,
                    count=1,  # One post per iteration in loop mode
                    slides=slides,
                    min_slides=min_slides,
                    max_slides=max_slides,
                    post_after=post_after,
                    auto_retry=auto_retry,
                    text_ai=text_ai,
                    image_ai=image_ai,
                    ai_tools=ai_tools,
                ))
                console.print(f"\n[dim]Waiting {loop_each} until next post... (Ctrl+C to stop)[/dim]")
                time.sleep(loop_seconds)
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Loop stopped after {iteration} iteration(s)[/yellow]")
    else:
        # Single run
        asyncio.run(_generate_posts(
            profile_path=profile_path,
            config=config,
            topic=topic,
            pillar=pillar,
            count=count,
            slides=slides,
            min_slides=min_slides,
            max_slides=max_slides,
            post_after=post_after,
            auto_retry=auto_retry,
            text_ai=text_ai,
            image_ai=image_ai,
            ai_tools=ai_tools,
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
    post_after: bool = False,
    auto_retry: bool = False,
    text_ai: str | None = None,
    image_ai: str | None = None,
    ai_tools: bool = False,
):
    """Async post generation with progress display."""
    display = ContentGenerationDisplay()
    last_progress: GenerationProgress | None = None

    async def progress_callback(progress: GenerationProgress | dict):
        nonlocal last_progress
        # Only update last_progress for GenerationProgress objects
        # (dict events are tool call events that shouldn't replace progress)
        if not isinstance(progress, dict):
            last_progress = progress
        display.add_event(progress)

    generator = ContentOrchestrator(
        profile_path=profile_path,
        profile_config=config,
        progress_callback=progress_callback,
        auto_retry=auto_retry,
        text_provider_override=text_ai,
        image_provider_override=image_ai,
        ai_tools=ai_tools,
    )

    # Get default content pillar if not specified
    if pillar is None:
        pillars = config.get("content_strategy", {}).get("content_pillars", [])
        pillar = pillars[0]["id"] if pillars else "general"

    if verbose:
        # Simple line-by-line logging (no Live display)
        post = None
        output_path = None
        posts = []  # For multi-post generation

        if topic:
            # Generate single post with specified topic
            console.print(f"\n[bold cyan]Starting generation:[/] {topic}\n")

            try:
                post = await generator.generate_post(
                    topic=topic,
                    content_pillar=pillar,
                    target_slides=slides,
                    min_slides=min_slides,
                    max_slides=max_slides,
                )
                output_path = await generator.save_post(post)
            except RuntimeError as e:
                console.print(f"\n[red]Generation failed:[/] {e}")
                console.print("[dim]Check logs/ai_calls.log for details[/dim]")
                return

        else:
            # No topic - use research to find topics
            console.print(f"\n[bold cyan]Researching trending topics and generating {count} post(s)...[/]\n")

            try:
                posts = await generator.generate_daily_posts(count=count)
            except RuntimeError as e:
                console.print(f"\n[red]Generation failed:[/red] {e}")
                console.print("[dim]Check logs/ai_calls.log for details[/dim]")
                return

        # Summary output (outside Live context for clean display)
        if posts:
            # Show summary for multiple posts
            console.print(f"\n[green]Generated {len(posts)} post(s)[/green]")
            for p in posts:
                console.print(f"  - {p.topic} ({p.slides_count} slides)")
            # Show output path and Instagram-ready content for last post
            if posts:
                last_post = posts[-1]
                last_post_path = generator._get_output_path(last_post)
                _print_instagram_ready(last_post, last_post_path)
        elif post and output_path:
            # Summary for single post
            console.print("\n")
            _print_generation_summary(post, output_path, display.stats)
            _print_instagram_ready(post, output_path)

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

    # Auto-post to Instagram if requested
    # Determine the path to post (either from single post or first of batch)
    post_path_to_use = output_path
    if not post_path_to_use and posts:
        # Auto-generated topic - use the first generated post
        post_path_to_use = generator._get_output_path(posts[0])

    if post_after and post_path_to_use:
        import shutil
        from dotenv import load_dotenv
        load_dotenv()

        try:
            from .instagram.models import InstagramConfig
            ig_config = InstagramConfig.from_env()
        except ValueError as e:
            console.print(f"\n[red]Cannot post: {e}[/red]")
            console.print("[yellow]Post saved in generated/ folder. Set up Instagram credentials to post.[/yellow]")
            return

        # Move from generated/ to pending-post/
        generated_path = post_path_to_use
        pending_path = generated_path.parent.parent / "pending-post" / generated_path.name
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove existing destination to avoid nested folders
        if pending_path.exists():
            shutil.rmtree(str(pending_path))
        shutil.move(str(generated_path), str(pending_path))
        console.print(f"\n[dim]Moved to pending-post: {pending_path.name}[/dim]")

        # Now post to Instagram (post ALL pending posts)
        console.print("\n[bold cyan]Publishing to Instagram...[/bold cyan]")
        await _post_to_instagram(
            profile_path=profile_path,
            post_id=None,
            config=ig_config,
            dry_run=False,
            post_all=True,  # Post all pending posts, not just the newly generated one
        )


def _print_instagram_ready(post, output_path: Path):
    """Print Instagram-ready caption and hashtags for easy copy-paste."""
    import re

    def safe_print(text: str) -> str:
        """Remove characters that can't be encoded in Windows console."""
        # Remove emojis and other non-ASCII characters for console display
        # but keep the original in the file
        try:
            return text.encode('cp1252', errors='ignore').decode('cp1252')
        except Exception:
            # Fallback: remove all non-ASCII
            return re.sub(r'[^\x00-\x7F]+', '', text)

    # File destination
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]Output:[/bold] {output_path}")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    # Build captions
    caption_text = post.caption if post.caption else post.hook_text
    hashtags_str = " ".join(f"#{tag.lstrip('#')}" for tag in post.hashtags) if post.hashtags else ""
    full_caption = f"{caption_text}\n\n{hashtags_str}" if hashtags_str else caption_text

    # Show Threads-ready caption (caption.txt)
    threads_char_count = len(caption_text)
    threads_ok = threads_char_count <= 500
    threads_status = "[green]OK[/green]" if threads_ok else "[yellow]>500[/yellow]"

    console.print(f"\n[bold magenta]Threads Caption ({threads_char_count} chars - {threads_status}):[/bold magenta]")
    console.print(f"[dim]File: caption.txt[/dim]")
    console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")
    console.print(safe_print(caption_text))
    console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")

    # Show full Instagram caption (caption+hashtags.txt)
    full_char_count = len(full_caption)
    console.print(f"\n[bold yellow]Instagram Caption ({full_char_count} chars):[/bold yellow]")
    console.print(f"[dim]File: caption+hashtags.txt (used for posting)[/dim]")
    console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")
    console.print(safe_print(caption_text))
    if hashtags_str:
        console.print(f"\n{safe_print(hashtags_str)}")
    console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")


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
            with open(niches_path, encoding="utf-8") as f:
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

    with open(profile_path / "metadata.json", "w", encoding="utf-8") as f:
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

    with open(niches_path, encoding="utf-8") as f:
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
def schedule(
    profile: str = typer.Argument(..., help="Profile name"),
    all_posts: bool = typer.Option(False, "--all", "-a", help="Schedule all generated posts"),
):
    """Move generated posts to pending-post queue.

    Workflow:
        1. generate: Creates posts in posts/YYYY/MM/generated/
        2. schedule: Moves to posts/YYYY/MM/pending-post/ (this command)
        3. post: Publishes to Instagram, moves to posts/YYYY/MM/posted/

    Examples:
        socials schedule ai.for.mortals         # Schedule one post interactively
        socials schedule ai.for.mortals --all   # Schedule all generated posts
    """
    import shutil

    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Find posts in generated folders
    posts_dir = profile_path / "posts"
    generated_posts = []

    for year_dir in posts_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    generated_dir = month_dir / "generated"
                    if generated_dir.exists():
                        for post_dir in generated_dir.iterdir():
                            if post_dir.is_dir() and (post_dir / "metadata.json").exists():
                                generated_posts.append(post_dir)

    if not generated_posts:
        console.print("[yellow]No posts in generated folders.[/yellow]")
        console.print("\n[dim]Generate posts first:[/dim]")
        console.print("  [cyan]python -m socials_automator.cli generate <profile> --topic '...'[/cyan]")
        raise typer.Exit(1)

    # Sort by folder name
    generated_posts.sort(key=lambda p: p.name)

    console.print(f"\n[bold]Found {len(generated_posts)} generated post(s):[/bold]")
    for i, post_path in enumerate(generated_posts, 1):
        # Load metadata for topic
        with open(post_path / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        topic = meta.get("post", {}).get("topic", post_path.name)
        console.print(f"  {i}. [cyan]{post_path.name}[/cyan]")
        console.print(f"     {topic[:60]}{'...' if len(topic) > 60 else ''}")

    if all_posts:
        to_schedule = generated_posts
    else:
        # Ask which post to schedule
        console.print(f"\n[dim]Enter post number (1-{len(generated_posts)}) or 'all':[/dim]")
        choice = typer.prompt("Schedule", default="1")

        if choice.lower() == "all":
            to_schedule = generated_posts
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(generated_posts):
                    to_schedule = [generated_posts[idx]]
                else:
                    console.print("[red]Invalid choice[/red]")
                    raise typer.Exit(1)
            except ValueError:
                console.print("[red]Invalid choice[/red]")
                raise typer.Exit(1)

    # Move posts to pending-post
    for post_path in to_schedule:
        generated_parent = post_path.parent  # e.g., posts/2025/12/generated
        pending_dir = generated_parent.parent / "pending-post"
        pending_dir.mkdir(parents=True, exist_ok=True)
        new_path = pending_dir / post_path.name

        try:
            shutil.move(str(post_path), str(new_path))
            console.print(f"[green]Scheduled:[/green] {post_path.name}")
        except Exception as e:
            console.print(f"[red]Failed to move {post_path.name}: {e}[/red]")

    console.print(f"\n[bold green]Scheduled {len(to_schedule)} post(s) for publishing.[/bold green]")
    console.print("\n[dim]To publish:[/dim]")
    console.print("  [cyan]python -m socials_automator.cli post <profile>[/cyan]")


@app.command()
def queue(
    profile: str = typer.Argument(..., help="Profile name"),
):
    """List all posts in the publishing queue.

    Shows posts from both 'generated' and 'pending-post' folders,
    sorted by timestamp (oldest first).

    Examples:
        socials queue ai.for.mortals
    """
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    posts_dir = profile_path / "posts"
    all_posts = []

    # Scan all year/month folders
    for year_dir in posts_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue
        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            # Check generated folder
            generated_dir = month_dir / "generated"
            if generated_dir.exists():
                for post_dir in generated_dir.iterdir():
                    if post_dir.is_dir() and (post_dir / "metadata.json").exists():
                        all_posts.append({
                            "path": post_dir,
                            "status": "generated",
                            "year": year_dir.name,
                            "month": month_dir.name,
                        })

            # Check pending-post folder
            pending_dir = month_dir / "pending-post"
            if pending_dir.exists():
                for post_dir in pending_dir.iterdir():
                    if post_dir.is_dir() and (post_dir / "metadata.json").exists():
                        all_posts.append({
                            "path": post_dir,
                            "status": "pending",
                            "year": year_dir.name,
                            "month": month_dir.name,
                        })

    if not all_posts:
        console.print("[yellow]No posts in queue.[/yellow]")
        console.print("\n[dim]Generate posts with:[/dim]")
        console.print("  [cyan]python -m socials_automator.cli generate <profile> --topic '...'[/cyan]")
        return

    # Sort by folder name (date-number-slug) for chronological order
    all_posts.sort(key=lambda p: (p["year"], p["month"], p["path"].name))

    # Build table
    table = Table(title=f"Publishing Queue for {profile}", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", width=10)
    table.add_column("Date", width=10)
    table.add_column("Post ID", width=40)
    table.add_column("Topic", width=40)

    for i, post in enumerate(all_posts, 1):
        # Load metadata for topic
        try:
            with open(post["path"] / "metadata.json", encoding="utf-8") as f:
                meta = json.load(f)
            topic = meta.get("post", {}).get("topic", "Unknown")[:38]
            post_date = meta.get("post", {}).get("date", f"{post['year']}-{post['month']}")
        except Exception:
            topic = "Unknown"
            post_date = f"{post['year']}-{post['month']}"

        status_color = "yellow" if post["status"] == "generated" else "cyan"
        table.add_row(
            str(i),
            f"[{status_color}]{post['status']}[/]",
            post_date,
            post["path"].name,
            topic,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(all_posts)} post(s) in queue[/dim]")
    console.print("\n[dim]To post all:[/dim]")
    console.print("  [cyan]python -m socials_automator.cli post <profile>[/cyan]")
    console.print("\n[dim]To post just one:[/dim]")
    console.print("  [cyan]python -m socials_automator.cli post <profile> --one[/cyan]")


@app.command()
def post(
    profile: str = typer.Argument(..., help="Profile name"),
    post_id: str = typer.Argument(None, help="Post ID to publish (if specified, posts only this one)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without posting"),
    one: bool = typer.Option(False, "--one", "-1", help="Post only the oldest pending (instead of all)"),
):
    """Post pending carousels to Instagram.

    By default, posts ALL pending posts in chronological order.
    Use --one to post only the oldest pending post.

    Workflow:
        1. generate: Creates posts in posts/YYYY/MM/generated/
        2. schedule: Moves to posts/YYYY/MM/pending-post/
        3. post: Publishes to Instagram, moves to posts/YYYY/MM/posted/ (this command)

    Examples:
        socials post ai.for.mortals                    # Post ALL pending posts
        socials post ai.for.mortals --one              # Post only oldest pending
        socials post ai.for.mortals 11-001            # Post specific (by prefix)
        socials post ai.for.mortals --dry-run         # Validate only
    """
    # Determine if posting all or just one
    all_posts = not one and post_id is None  # Post all unless --one or specific post_id given
    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Check for Instagram credentials
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from .instagram.models import InstagramConfig
        config = InstagramConfig.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        console.print("\n[yellow]To set up Instagram posting:[/]")
        console.print("  1. Create a Facebook App at https://developers.facebook.com")
        console.print("  2. Connect your Instagram Business/Creator account")
        console.print("  3. Generate an access token with instagram_content_publish permission")
        console.print("  4. Create a Cloudinary account at https://cloudinary.com")
        console.print("  5. Add credentials to your .env file")
        raise typer.Exit(1)

    # Try to auto-refresh token if needed
    if not dry_run:
        try:
            from .instagram import TokenManager
            token_manager = TokenManager.from_env()
            if token_manager.can_refresh:
                console.print("[dim]Checking token validity...[/dim]")
                new_token = asyncio.run(token_manager.ensure_valid_token())
                if new_token != config.access_token:
                    config.access_token = new_token
                    console.print("[green]Token refreshed successfully![/green]")
        except Exception as e:
            console.print(f"[yellow]Token auto-refresh unavailable: {e}[/yellow]")
            console.print("[dim]Continuing with current token...[/dim]")

    # Run async posting
    asyncio.run(_post_to_instagram(
        profile_path=profile_path,
        post_id=post_id,
        config=config,
        dry_run=dry_run,
        post_all=all_posts,  # True by default now (unless --one or specific post_id)
    ))


@app.command()
def token(
    check: bool = typer.Option(False, "--check", "-c", help="Check token validity"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Refresh the token"),
    exchange: bool = typer.Option(False, "--exchange", "-e", help="Exchange short-lived for long-lived token"),
):
    """Manage Instagram access token.

    Use this command to check, refresh, or exchange your Instagram token.

    Examples:
        socials token --check      # Check if token is valid
        socials token --refresh    # Refresh token (extends by 60 days)
        socials token --exchange   # Convert short-lived to long-lived token
    """
    from dotenv import load_dotenv
    load_dotenv()

    from .instagram import TokenManager

    try:
        manager = TokenManager.from_env()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not any([check, refresh, exchange]):
        # Default: check token
        check = True

    if check:
        console.print("\n[bold]Checking Instagram token...[/bold]\n")
        try:
            is_valid, message, days_left = asyncio.run(manager.check_token_validity())
            if is_valid:
                console.print(f"[green][OK] {message}[/green]")
                if days_left and days_left < 14:
                    console.print(f"\n[yellow]Tip: Run 'socials token --refresh' to extend token validity[/yellow]")
            else:
                console.print(f"[red][EXPIRED] {message}[/red]")
                if manager.can_refresh:
                    console.print(f"\n[yellow]Run 'socials token --refresh' to get a new token[/yellow]")
                else:
                    console.print(f"\n[yellow]Add FACEBOOK_APP_ID and FACEBOOK_APP_SECRET to .env for auto-refresh[/yellow]")
                    console.print("[dim]Then get a new token from Graph API Explorer[/dim]")
        except Exception as e:
            console.print(f"[red]Error checking token: {e}[/red]")
            raise typer.Exit(1)

    if refresh or exchange:
        if not manager.can_refresh:
            console.print("[red]Cannot refresh: FACEBOOK_APP_ID and FACEBOOK_APP_SECRET required[/red]")
            console.print("\n[yellow]Add these to your .env file:[/yellow]")
            console.print("  FACEBOOK_APP_ID=your_app_id")
            console.print("  FACEBOOK_APP_SECRET=your_app_secret")
            console.print("\n[dim]Get these from: https://developers.facebook.com/apps/ -> Your App -> Settings -> Basic[/dim]")
            raise typer.Exit(1)

        action = "Exchanging" if exchange else "Refreshing"
        console.print(f"\n[bold]{action} Instagram token...[/bold]\n")

        try:
            new_token = asyncio.run(manager.exchange_for_long_lived_token())
            console.print(f"[green][OK] Token {action.lower()} successful![/green]")

            # Save to .env
            if manager.update_env_file(new_token):
                console.print(f"[green][OK] Token saved to .env file[/green]")
            else:
                console.print(f"\n[yellow]New token (save this to your .env):[/yellow]")
                console.print(f"[dim]{new_token[:50]}...{new_token[-20:]}[/dim]")

            # Check new token validity
            manager.access_token = new_token
            is_valid, message, days_left = asyncio.run(manager.check_token_validity())
            console.print(f"\n[cyan]{message}[/cyan]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


async def _post_to_instagram(
    profile_path: Path,
    post_id: str | None,
    config,
    dry_run: bool,
    post_all: bool = False,
):
    """Async Instagram posting with progress display."""
    from .instagram import InstagramClient, CloudinaryUploader, InstagramProgress
    import shutil

    posts_dir = profile_path / "posts"

    # Step 1: Check for posts in 'generated' folders and auto-move to 'pending-post'
    generated_posts = []
    for year_dir in posts_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    generated_dir = month_dir / "generated"
                    if generated_dir.exists():
                        for post_dir in generated_dir.iterdir():
                            if post_dir.is_dir() and (post_dir / "metadata.json").exists():
                                generated_posts.append((post_dir, month_dir))

    # Auto-move generated posts to pending-post
    if generated_posts:
        console.print(f"\n[cyan]Found {len(generated_posts)} post(s) in 'generated' folder(s)[/cyan]")
        for post_dir, month_dir in generated_posts:
            pending_dir = month_dir / "pending-post"
            pending_dir.mkdir(parents=True, exist_ok=True)
            new_path = pending_dir / post_dir.name
            try:
                shutil.move(str(post_dir), str(new_path))
                console.print(f"  [green][OK][/green] Moved to pending: {post_dir.name}")
            except Exception as e:
                console.print(f"  [red][ERROR][/red] Failed to move {post_dir.name}: {e}")

    # Step 2: Find posts in pending-post folder (scan all year/month subfolders)
    pending_posts = []
    for year_dir in posts_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    pending_dir = month_dir / "pending-post"
                    if pending_dir.exists():
                        for post_dir in pending_dir.iterdir():
                            if post_dir.is_dir() and (post_dir / "metadata.json").exists():
                                pending_posts.append(post_dir)

    if not pending_posts:
        console.print("[yellow]No posts ready to publish.[/yellow]")
        console.print("\n[dim]Workflow:[/dim]")
        console.print("  1. Generate posts: [cyan]python -m socials_automator.cli generate <profile> --topic '...'[/cyan]")
        console.print("  2. Post to Instagram: [cyan]python -m socials_automator.cli post <profile>[/cyan]")
        console.print("\n[dim]Posts are auto-moved: generated -> pending-post -> posted[/dim]")
        raise typer.Exit(1)

    # Sort by folder name (date-number-slug) to get oldest first
    pending_posts.sort(key=lambda p: p.name)

    # Show queue summary
    if post_all or len(pending_posts) > 1:
        console.print(f"\n[bold cyan]Publishing Queue ({len(pending_posts)} posts):[/bold cyan]")
        for i, p in enumerate(pending_posts, 1):
            try:
                with open(p / "metadata.json", encoding="utf-8") as f:
                    meta = json.load(f)
                topic = meta.get("post", {}).get("topic", "Unknown")[:40]
            except Exception:
                topic = "Unknown"
            marker = "[yellow]>[/yellow]" if (post_all or i == 1) else " "
            console.print(f"  {marker} {i}. {p.name} - {topic}")
        console.print()

    # Determine which posts to publish
    if post_all:
        posts_to_publish = pending_posts
    elif post_id:
        # Find specific post
        post_path = None
        for p in pending_posts:
            if post_id in p.name or p.name.startswith(post_id):
                post_path = p
                break
        if not post_path:
            console.print(f"[red]Post not found in pending-post: {post_id}[/red]")
            console.print(f"[dim]Available: {[p.name for p in pending_posts]}[/dim]")
            raise typer.Exit(1)
        posts_to_publish = [post_path]
    else:
        # Use oldest pending post
        posts_to_publish = [pending_posts[0]]

    # Validate token once before posting any
    if not dry_run:
        from .instagram import InstagramClient, CloudinaryUploader, InstagramProgress
        import shutil

        # Create client for validation
        temp_client = InstagramClient(config=config)
        console.print("\n[dim]Validating Instagram access...[/dim]")
        try:
            account_info = await temp_client.validate_token()
            console.print(f"[green]Connected as @{account_info.get('username', 'unknown')}[/green]\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not validate token ({e})[/yellow]")
            console.print("[yellow]Proceeding with posting attempt...[/yellow]\n")

    # Publish each post
    published_count = 0
    failed_count = 0

    for post_idx, post_path in enumerate(posts_to_publish, 1):
        if post_all:
            console.print(f"\n{'='*60}")
            console.print(f"[bold cyan]Publishing {post_idx}/{len(posts_to_publish)}: {post_path.name}[/bold cyan]")
            console.print(f"{'='*60}")

        # Load post metadata
        metadata_path = post_path / "metadata.json"
        if not metadata_path.exists():
            console.print(f"[red]Post metadata not found: {metadata_path}[/red]")
            failed_count += 1
            continue

        with open(metadata_path, encoding="utf-8") as f:
            post_metadata = json.load(f)

        # Get slide images
        slide_paths = sorted(post_path.glob("slide_*.jpg"))
        if not slide_paths:
            console.print(f"[red]No slide images found in {post_path}[/red]")
            failed_count += 1
            continue

        # Load full caption with hashtags for Instagram posting
        # Prefer caption+hashtags.txt (full version), fallback to caption.txt + hashtags.txt
        full_caption_path = post_path / "caption+hashtags.txt"
        caption_path = post_path / "caption.txt"
        hashtags_path = post_path / "hashtags.txt"

        if full_caption_path.exists():
            # Use the full caption file directly (caption + hashtags combined)
            caption = full_caption_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            # Fallback: combine caption.txt + hashtags.txt
            caption = caption_path.read_text(encoding="utf-8")
            if hashtags_path.exists():
                hashtags = hashtags_path.read_text(encoding="utf-8").strip()
                if hashtags and not caption.endswith(hashtags):
                    caption = f"{caption}\n\n{hashtags}"
        else:
            caption = ""

        # Extract post info from metadata
        post_info = post_metadata.get("post", {})
        post_id_display = post_info.get("id", post_path.name)
        post_topic = post_info.get("topic", "Unknown topic")
        post_date = post_info.get("date", "Unknown")

        # Show post info
        console.print(Panel(
            f"[bold]Post ID:[/] {post_id_display}\n"
            f"[bold]Topic:[/] {post_topic}\n"
            f"[bold]Slides:[/] {len(slide_paths)}\n"
            f"[bold]Generated:[/] {post_date}\n"
            f"[bold]Location:[/] {post_path}",
            title="Instagram Posting",
        ))

        if dry_run:
            import re
            def safe_print_caption(text: str) -> str:
                """Remove characters that can't be displayed in Windows console."""
                try:
                    return text.encode('cp1252', errors='ignore').decode('cp1252')
                except Exception:
                    return re.sub(r'[^\x00-\x7F]+', '', text)

            console.print("\n[yellow]DRY RUN - Would upload these images:[/yellow]")
            for path in slide_paths:
                console.print(f"  - {path.name}")
            console.print(f"\n[yellow]Caption ({len(caption)} chars):[/yellow]")
            display_caption = caption[:500] + "..." if len(caption) > 500 else caption
            console.print(safe_print_caption(display_caption))
            published_count += 1
            continue

        # Import for actual posting
        from .instagram import InstagramClient, CloudinaryUploader, InstagramProgress
        import shutil

        # Initialize progress display
        display = InstagramPostingDisplay()

        async def progress_callback(progress: InstagramProgress):
            display.update(progress)

        # Create uploader and client
        uploader = CloudinaryUploader(
            config=config,
            progress_callback=lambda step, cur, tot: progress_callback(
                InstagramProgress(
                    status=InstagramProgress().status,
                    current_step=step,
                    images_uploaded=cur,
                    total_images=tot,
                )
            ),
        )

        client = InstagramClient(
            config=config,
            progress_callback=progress_callback,
        )

        result = None
        try:
            # Simple print-based progress (no Live display)
            print(f"\n  [Upload] Uploading {len(slide_paths)} images to Cloudinary...")

            folder = f"socials-automator/{profile_path.name}/{post_id_display}"
            image_urls = await uploader.upload_batch(slide_paths, folder=folder)

            print(f"  [OK] {len(image_urls)} images uploaded")
            print(f"  [Container] Creating {len(image_urls)} Instagram containers...")

            # Publish to Instagram
            result = await client.publish_carousel(
                image_urls=image_urls,
                caption=caption,
            )

            # Cleanup Cloudinary uploads
            if result.success:
                print(f"  [Cleanup] Removing temporary Cloudinary images...")
                await uploader.cleanup_async()

        except Exception as e:
            console.print(f"[red]Error publishing: {e}[/red]")
            failed_count += 1
            if not post_all:
                raise typer.Exit(1)
            continue

        # Show result
        if result and result.success:
            console.print(Panel(
                f"[bold green]Published successfully![/bold green]\n\n"
                f"[bold]Media ID:[/] {result.media_id}\n"
                f"[bold]URL:[/] {result.permalink or 'N/A'}",
                title="Success",
                border_style="green",
            ))

            # Update post metadata with Instagram info
            post_metadata["instagram"] = result.to_dict()
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(post_metadata, f, indent=2)

            # Move from pending-post to posted folder
            pending_parent = post_path.parent
            posted_dir = pending_parent.parent / "posted"
            posted_dir.mkdir(parents=True, exist_ok=True)
            new_post_path = posted_dir / post_path.name

            try:
                # If destination already exists, remove it first to avoid nested folders
                if new_post_path.exists():
                    shutil.rmtree(str(new_post_path))
                shutil.move(str(post_path), str(new_post_path))
                console.print(f"\n[green]Post moved to:[/green] {new_post_path}")
            except Exception as e:
                console.print(f"\n[yellow]Warning: Could not move post folder: {e}[/yellow]")

            published_count += 1
        else:
            error_msg = result.error_message if result else "Unknown error"
            console.print(Panel(
                f"[bold red]Publishing failed[/bold red]\n\n"
                f"[bold]Error:[/] {error_msg}",
                title="Error",
                border_style="red",
            ))
            failed_count += 1
            if not post_all:
                raise typer.Exit(1)

    # Show summary for batch posting
    if post_all or dry_run:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Publishing Summary[/bold]")
        console.print(f"{'='*60}")
        console.print(f"  [green]Published:[/green] {published_count}")
        if failed_count > 0:
            console.print(f"  [red]Failed:[/red] {failed_count}")
        console.print(f"  [dim]Total:[/dim] {len(posts_to_publish)}")

        if failed_count > 0:
            raise typer.Exit(1)


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
