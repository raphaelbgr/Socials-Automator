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
def generate_post(
    profile: str = typer.Argument(..., help="Profile name to generate for"),
    topic: str = typer.Option(None, "--topic", "-t", help="Topic for the post"),
    pillar: str = typer.Option(None, "--pillar", "-p", help="Content pillar"),
    count: int = typer.Option(1, "--count", "-n", help="Number of posts to generate"),
    slides: int = typer.Option(None, "--slides", "-s", help="Number of slides (default: AI decides)"),
    min_slides: int = typer.Option(3, "--min-slides", help="Minimum slides when AI decides"),
    max_slides: int = typer.Option(10, "--max-slides", help="Maximum slides when AI decides"),
    upload_after: bool = typer.Option(False, "--upload", help="Upload to Instagram after generating"),
    auto_retry: bool = typer.Option(False, "--auto-retry", help="Retry indefinitely until valid content"),
    text_ai: str = typer.Option(None, "--text-ai", help="Text AI provider (zai, groq, gemini, openai, lmstudio, ollama)"),
    image_ai: str = typer.Option(None, "--image-ai", help="Image AI provider (dalle, fal_flux, comfy)"),
    loop_each: str = typer.Option(None, "--loop-each", help="Loop interval (e.g., 5m, 1h, 30s)"),
    ai_tools: bool = typer.Option(False, "--ai-tools", help="Enable AI tool calling (AI decides when to search)"),
):
    """Generate carousel posts for a profile.

    By default, the AI decides the optimal number of slides (3-10) based on
    the topic content. Use --slides to force a specific count.

    Use --upload to automatically upload to Instagram after generation.
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
        # Loop mode - NEVER stops on errors, uses exponential backoff
        import time
        iteration = 0
        consecutive_errors = 0
        max_backoff = 3600  # Max 1 hour backoff

        try:
            while True:
                iteration += 1
                console.print(f"\n[bold cyan]--- Iteration {iteration} ---[/bold cyan]")

                try:
                    asyncio.run(_generate_posts(
                        profile_path=profile_path,
                        config=config,
                        topic=None,  # Always auto-generate topic in loop mode
                        pillar=pillar,
                        count=1,  # One post per iteration in loop mode
                        slides=slides,
                        min_slides=min_slides,
                        max_slides=max_slides,
                        upload_after=upload_after,
                        auto_retry=auto_retry,
                        text_ai=text_ai,
                        image_ai=image_ai,
                        ai_tools=ai_tools,
                    ))
                    # Success - reset error counter
                    consecutive_errors = 0
                    console.print(f"\n[dim]Waiting {loop_each} until next post... (Ctrl+C to stop)[/dim]")
                    time.sleep(loop_seconds)

                except KeyboardInterrupt:
                    raise  # Re-raise to exit loop
                except Exception as e:
                    # NEVER stop on errors - use exponential backoff
                    consecutive_errors += 1
                    backoff_time = min(loop_seconds * (2 ** consecutive_errors), max_backoff)

                    console.print(f"\n[red]Error in iteration {iteration}:[/red]")
                    console.print(f"  {str(e)[:200]}")
                    console.print(f"\n[yellow]Consecutive errors: {consecutive_errors}[/yellow]")
                    console.print(f"[dim]Loop will continue - backing off for {backoff_time:.0f}s...[/dim]")
                    console.print(f"[dim]Failed posts stay in pending-post/ and will be retried next iteration.[/dim]")

                    # Log error for debugging
                    import logging
                    logging.getLogger("cli").error(f"Loop iteration {iteration} failed: {e}", exc_info=True)

                    time.sleep(backoff_time)
                    console.print(f"\n[cyan]Resuming loop after backoff...[/cyan]")

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
            upload_after=upload_after,
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
    upload_after: bool = False,
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

    if upload_after and post_path_to_use:
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
            "id": name,  # Keep dots (e.g., ai.for.mortals)
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
        console.print("  [cyan]python -m socials_automator.cli generate-post <profile> --topic '...'[/cyan]")
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
    console.print("  [cyan]python -m socials_automator.cli upload-post <profile>[/cyan]")


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
        console.print("  [cyan]python -m socials_automator.cli generate-post <profile> --topic '...'[/cyan]")
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
    console.print("  [cyan]python -m socials_automator.cli upload-post <profile>[/cyan]")
    console.print("\n[dim]To post just one:[/dim]")
    console.print("  [cyan]python -m socials_automator.cli upload-post <profile> --one[/cyan]")


@app.command()
def upload_post(
    profile: str = typer.Argument(..., help="Profile name"),
    post_id: str = typer.Argument(None, help="Post ID to upload (if specified, uploads only this one)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without uploading"),
    one: bool = typer.Option(False, "--one", "-1", help="Upload only the oldest pending (instead of all)"),
):
    """Upload pending carousels to Instagram.

    By default, uploads ALL pending posts in chronological order.
    Use --one to upload only the oldest pending post.

    Workflow:
        1. generate-post: Creates posts in posts/YYYY/MM/generated/
        2. schedule: Moves to posts/YYYY/MM/pending-post/
        3. upload-post: Uploads to Instagram, moves to posts/YYYY/MM/posted/ (this command)

    Examples:
        socials upload-post ai.for.mortals                    # Upload ALL pending posts
        socials upload-post ai.for.mortals --one              # Upload only oldest pending
        socials upload-post ai.for.mortals 11-001            # Upload specific (by prefix)
        socials upload-post ai.for.mortals --dry-run         # Validate only
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


async def _cleanup_orphaned_cloudinary(posts_dir: Path, config) -> int:
    """Clean up orphaned Cloudinary files from all posted folders.

    Scans all posted folders for metadata with _upload_state containing
    cloudinary_urls that were never cleaned up (e.g., after a failed post).

    Returns:
        Number of files cleaned up.
    """
    from .instagram import CloudinaryUploader

    console.print(f"\n[dim]Checking for orphaned Cloudinary files...[/dim]")

    orphaned_cleaned = 0
    for year_dir in posts_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    # Check both posted and pending-post folders
                    for folder_name in ["posted", "pending-post"]:
                        check_dir = month_dir / folder_name
                        if check_dir.exists():
                            for post_folder in check_dir.iterdir():
                                if post_folder.is_dir():
                                    meta_path = post_folder / "metadata.json"
                                    if meta_path.exists():
                                        try:
                                            with open(meta_path, encoding="utf-8") as f:
                                                meta = json.load(f)

                                            # Check for orphaned upload state
                                            upload_state = meta.get("_upload_state", {})
                                            cloudinary_urls = upload_state.get("cloudinary_urls", [])

                                            # Only cleanup if post is already published (has instagram.media_id)
                                            # OR if the upload is old (more than 1 hour)
                                            should_cleanup = False
                                            if meta.get("instagram", {}).get("media_id"):
                                                should_cleanup = True
                                            elif upload_state.get("uploaded_at"):
                                                from datetime import datetime
                                                try:
                                                    uploaded_at = datetime.fromisoformat(upload_state["uploaded_at"])
                                                    age_hours = (datetime.now() - uploaded_at).total_seconds() / 3600
                                                    if age_hours > 24:  # Orphaned for more than 24 hours
                                                        should_cleanup = True
                                                        console.print(f"  [yellow]Found stale upload ({age_hours:.1f}h old): {post_folder.name}[/yellow]")
                                                except Exception:
                                                    pass

                                            if should_cleanup and cloudinary_urls:
                                                cleanup_uploader = CloudinaryUploader(config=config)

                                                for url in cloudinary_urls:
                                                    if "cloudinary.com" in url:
                                                        parts = url.split("/upload/")
                                                        if len(parts) > 1:
                                                            public_id = parts[1].rsplit(".", 1)[0]
                                                            cleanup_uploader._uploaded_public_ids.append(public_id)

                                                deleted = cleanup_uploader.cleanup()
                                                orphaned_cleaned += deleted

                                                # Remove upload state from metadata
                                                del meta["_upload_state"]
                                                with open(meta_path, "w", encoding="utf-8") as f:
                                                    json.dump(meta, f, indent=2)

                                                console.print(f"  [green]Cleaned {deleted} files from {post_folder.name}[/green]")

                                        except Exception as e:
                                            pass  # Ignore errors during cleanup scan

    if orphaned_cleaned > 0:
        console.print(f"  [green]Total: Cleaned {orphaned_cleaned} orphaned Cloudinary files[/green]")
    else:
        console.print(f"  [dim]No orphaned files found[/dim]")

    return orphaned_cleaned


async def _post_to_instagram(
    profile_path: Path,
    post_id: str | None,
    config,
    dry_run: bool,
    post_all: bool = False,
):
    """Async Instagram posting with progress display."""
    from datetime import datetime
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

        # Still run cleanup for orphaned Cloudinary files even if no posts to publish
        if not dry_run:
            await _cleanup_orphaned_cloudinary(posts_dir, config)

        # Return gracefully instead of raising - allows loop mode to continue
        return

    # Sort chronologically: oldest first
    # Path format: posts/YYYY/MM/pending-post/DD-NNN-slug
    pending_posts.sort(key=lambda p: (
        p.parent.parent.parent.name,  # Year (YYYY)
        p.parent.parent.name,         # Month (MM)
        p.name,                       # Folder name (DD-NNN-slug)
    ))

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
    skipped_count = 0

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

        # Phase 0: Check for duplicates
        print(f"\n  [Phase 0] Checking for duplicates...")
        existing_instagram = post_metadata.get("instagram", {})
        if existing_instagram.get("media_id"):
            existing_permalink = existing_instagram.get("permalink", "N/A")
            console.print(f"  [yellow][SKIP] Already posted to Instagram![/yellow]")
            console.print(f"  [dim]Media ID: {existing_instagram['media_id']}[/dim]")
            console.print(f"  [dim]URL: {existing_permalink}[/dim]")

            # Move to posted folder if still in pending
            if "pending-post" in str(post_path):
                pending_parent = post_path.parent
                posted_dir = pending_parent.parent / "posted"
                posted_dir.mkdir(parents=True, exist_ok=True)
                new_post_path = posted_dir / post_path.name
                try:
                    if new_post_path.exists():
                        shutil.rmtree(str(new_post_path))
                    shutil.move(str(post_path), str(new_post_path))
                    console.print(f"  [green]Moved duplicate to posted folder[/green]")
                except Exception as e:
                    console.print(f"  [yellow]Could not move: {e}[/yellow]")

            skipped_count += 1
            continue

        print(f"  [OK] No duplicate found")

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
        image_urls = []
        try:
            # Check for resume - if we already uploaded to Cloudinary
            upload_state = post_metadata.get("_upload_state", {})
            existing_urls = upload_state.get("cloudinary_urls", [])

            # Check if upload state is too old (stale URLs might cause issues)
            upload_is_stale = False
            if existing_urls and upload_state.get("uploaded_at"):
                try:
                    uploaded_at = datetime.fromisoformat(upload_state["uploaded_at"])
                    age_hours = (datetime.now() - uploaded_at).total_seconds() / 3600
                    if age_hours > 24:
                        upload_is_stale = True
                        print(f"\n  [Warning] Cloudinary upload is {age_hours:.1f}h old - uploading fresh")
                except Exception:
                    pass

            if existing_urls and len(existing_urls) == len(slide_paths) and not upload_is_stale:
                # Resume: use existing Cloudinary URLs
                print(f"\n  [Resume] Found {len(existing_urls)} existing Cloudinary uploads")

                # NOTE: We previously had ghost detection here that checked for posts made
                # after uploaded_at timestamp, but this caused false positives when a
                # DIFFERENT post was made between retries. Ghost detection now only happens
                # during the actual publish step (in the Instagram client).

                image_urls = existing_urls
                # Track these URLs for cleanup
                for url in existing_urls:
                    # Extract public_id from URL for cleanup tracking
                    # URL format: https://res.cloudinary.com/<cloud>/image/upload/<path>/<public_id>.<ext>
                    if "cloudinary.com" in url:
                        parts = url.split("/upload/")
                        if len(parts) > 1:
                            public_id = parts[1].rsplit(".", 1)[0]  # Remove extension
                            uploader._uploaded_public_ids.append(public_id)
            else:
                # Fresh upload to Cloudinary
                print(f"\n  [Upload] Uploading {len(slide_paths)} images to Cloudinary...")

                folder = f"socials-automator/{profile_path.name}/{post_id_display}"
                image_urls = await uploader.upload_batch(slide_paths, folder=folder)

                print(f"  [OK] {len(image_urls)} images uploaded")

                # Save upload state for resume capability
                post_metadata["_upload_state"] = {
                    "cloudinary_urls": image_urls,
                    "uploaded_at": datetime.now().isoformat(),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(post_metadata, f, indent=2)

            print(f"  [Container] Creating {len(image_urls)} Instagram containers...")

            # Publish to Instagram
            result = await client.publish_carousel(
                image_urls=image_urls,
                caption=caption,
            )

            # Cleanup Cloudinary uploads on success
            if result.success:
                deleted_count = await uploader.cleanup_async()
                print(f"  [Cleanup] Removed {deleted_count} temporary Cloudinary images")

                # Remove upload state from metadata (no longer needed)
                if "_upload_state" in post_metadata:
                    del post_metadata["_upload_state"]
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(post_metadata, f, indent=2)

        except Exception as e:
            console.print(f"[red]Error publishing: {e}[/red]")
            # Inform user about resume capability
            if image_urls:
                console.print(f"\n[yellow]Cloudinary uploads saved - run the command again to resume.[/yellow]")
                console.print(f"[dim]The script will skip re-uploading and retry Instagram posting.[/dim]")
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

            # Check for DAILY POSTING LIMIT (special case - not retryable, needs explanation)
            is_daily_limit = "[DAILY_POSTING_LIMIT]" in error_msg

            # Check if error is retryable (from our error codes)
            is_retryable = (
                not is_daily_limit and (
                    "[MEDIA_UPLOAD_FAILED]" in error_msg or
                    "[RATE_LIMIT]" in error_msg or
                    "[APP_RATE_LIMIT]" in error_msg or
                    "[USER_RATE_LIMIT]" in error_msg or
                    "ERROR_9" in error_msg or
                    "error code 9" in error_msg.lower()
                )
            )

            if is_daily_limit:
                # Special handling for daily posting limit
                console.print(Panel(
                    "[bold red]DAILY POSTING LIMIT REACHED[/bold red]\n\n"
                    "You've hit Instagram's Content Publishing API daily limit.\n\n"
                    "[bold]What this means:[/bold]\n"
                    "  - The API allows ~25 posts per day per account\n"
                    "  - Each carousel uses multiple API calls (1 per image + carousel + publish)\n"
                    "  - Failed retries also count towards the limit\n\n"
                    "[bold]What to do:[/bold]\n"
                    "  - Wait until midnight UTC for the limit to reset\n"
                    "  - Or try again tomorrow\n\n"
                    "[dim]Your Cloudinary uploads are saved - they'll be reused when you retry.[/dim]",
                    title="[bold red]Instagram Daily Limit[/bold red]",
                    border_style="red",
                ))
                # DON'T cleanup Cloudinary - keep uploads for retry tomorrow
            else:
                error_panel = (
                    f"[bold red]Publishing failed[/bold red]\n\n"
                    f"[bold]Error:[/] {error_msg}"
                )

                if is_retryable:
                    error_panel += (
                        f"\n\n[yellow]This error is usually temporary.[/yellow]\n"
                        f"[dim]Run the command again to retry - Cloudinary uploads are saved.[/dim]"
                    )
                else:
                    # Non-retryable error - cleanup Cloudinary to avoid orphaned files
                    if uploader._uploaded_public_ids:
                        deleted_count = await uploader.cleanup_async()
                        error_panel += f"\n\n[dim]Cleaned up {deleted_count} Cloudinary files.[/dim]"
                        # Also remove upload state
                        if "_upload_state" in post_metadata:
                            del post_metadata["_upload_state"]
                            with open(metadata_path, "w", encoding="utf-8") as f:
                                json.dump(post_metadata, f, indent=2)

                console.print(Panel(
                    error_panel,
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
        if skipped_count > 0:
            console.print(f"  [yellow]Skipped (duplicates):[/yellow] {skipped_count}")
        if failed_count > 0:
            console.print(f"  [red]Failed:[/red] {failed_count}")
        console.print(f"  [dim]Total:[/dim] {len(posts_to_publish)}")

    # Final Phase: Clean up orphaned Cloudinary files from all posted folders
    if not dry_run:
        await _cleanup_orphaned_cloudinary(posts_dir, config)

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


# Voice presets for reel command
VOICE_CHOICES = [
    "rvc_adam",  # THE viral TikTok voice - FREE, runs locally!
    "rvc_adam_excited",  # Same voice with faster rate and higher pitch
    "adam_excited",  # Alias for rvc_adam_excited
    "tiktok-adam",  # Alias for rvc_adam
    "adam",  # Short alias for rvc_adam
    "professional_female",
    "professional_male",
    "friendly_female",
    "friendly_male",
    "energetic",
    "british_female",
    "british_male",
]

# Video matcher sources
VIDEO_MATCHER_CHOICES = ["pexels"]


@app.command()
def generate_reel(
    profile: str = typer.Argument(..., help="Profile name to generate for"),
    topic: str = typer.Option(None, "--topic", "-t", help="Topic for the video (auto-generated if not provided)"),
    text_ai: str = typer.Option(None, "--text-ai", help="Text AI provider (zai, groq, gemini, openai, lmstudio, ollama)"),
    video_matcher: str = typer.Option("pexels", "--video-matcher", "-m", help="Video source (pexels)"),
    voice: str = typer.Option("rvc_adam", "--voice", "-v", help=f"Voice preset: {', '.join(VOICE_CHOICES)}"),
    voice_rate: str = typer.Option("+0%", "--voice-rate", help="Speech rate adjustment (e.g., '+12%' for excited, '-10%' for calm)"),
    voice_pitch: str = typer.Option("+0Hz", "--voice-pitch", help="Pitch adjustment (e.g., '+3Hz' for excited, '-2Hz' for calm)"),
    subtitle_size: int = typer.Option(80, "--subtitle-size", "-s", help="Subtitle font size in pixels (default: 80)"),
    font: str = typer.Option("Montserrat-Bold.ttf", "--font", help="Subtitle font from /fonts folder (default: Montserrat-Bold.ttf)"),
    length: str = typer.Option("1m", "--length", "-l", help="Target video length (e.g., 30s, 1m, 90s). Default: 1m"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory (default: temp)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only run first few steps without full video generation"),
    loop: bool = typer.Option(False, "--loop", "-L", help="Loop continuously until stopped (Ctrl+C)"),
    loop_count: int = typer.Option(None, "--loop-count", "-n", help="Generate exactly N videos then stop (implies --loop)"),
    gpu_accelerate: bool = typer.Option(False, "--gpu-accelerate", "-g", help="Enable GPU acceleration with NVENC (requires NVIDIA GPU)"),
    gpu: int = typer.Option(None, "--gpu", help="GPU index to use (0, 1, etc.). Auto-selects if not specified."),
):
    """Generate a video reel for Instagram/TikTok.

    This command matches stock footage to your content - it does NOT generate
    video with AI. The AI is used for topic selection and script planning.

    The narration audio is the source of truth for video length - video clips
    are trimmed to match the narration duration.

    Pipeline:
        1. [AI] Select topic from profile content pillars
        2. Research topic via web search
        3. [AI] Plan video script (targeting --length duration)
        4. Generate voiceover (determines actual video duration)
        5. Search video matcher for stock footage
        6. Download video clips
        7. Assemble into 9:16 vertical video (matches narration length)
        8. Add karaoke-style subtitles
        9. Output final.mp4

    Examples:
        socials generate-reel ai.for.mortals
        socials generate-reel ai.for.mortals --text-ai lmstudio --length 1m
        socials generate-reel ai.for.mortals --topic "5 AI productivity tips"
        socials generate-reel ai.for.mortals --voice british_female --length 90s
        socials generate-reel ai.for.mortals --video-matcher pexels
        socials generate-reel ai.for.mortals --subtitle-size 90 --font Poppins-Bold.ttf
        socials generate-reel ai.for.mortals --loop  # Generate videos indefinitely
        socials generate-reel ai.for.mortals -n 10  # Generate 10 videos then stop
        socials generate-reel ai.for.mortals --voice adam_excited  # Use excited preset
        socials generate-reel ai.for.mortals --voice-rate "+12%" --voice-pitch "+3Hz"  # Custom excitement
        socials generate-reel ai.for.mortals --gpu-accelerate  # Use GPU for faster rendering
        socials generate-reel ai.for.mortals -g --gpu 0  # Use specific GPU
    """
    from dotenv import load_dotenv
    load_dotenv()

    profile_path = get_profile_path(profile)

    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    # Parse length string (e.g., "1m", "30s", "90s") to seconds
    def parse_length(length_str: str) -> float:
        length_str = length_str.strip().lower()
        if length_str.endswith("m"):
            return float(length_str[:-1]) * 60
        elif length_str.endswith("s"):
            return float(length_str[:-1])
        else:
            # Assume seconds if no suffix
            return float(length_str)

    try:
        target_duration = parse_length(length)
        if target_duration < 15 or target_duration > 180:
            console.print(f"[red]Invalid length: {length}. Must be between 15s and 3m.[/red]")
            raise typer.Exit(1)
    except ValueError:
        console.print(f"[red]Invalid length format: {length}. Use formats like 30s, 1m, 90s.[/red]")
        raise typer.Exit(1)

    # Validate voice choice
    if voice not in VOICE_CHOICES:
        console.print(f"[red]Invalid voice: {voice}[/red]")
        console.print(f"[yellow]Available voices: {', '.join(VOICE_CHOICES)}[/yellow]")
        raise typer.Exit(1)

    # Validate video matcher choice
    if video_matcher not in VIDEO_MATCHER_CHOICES:
        console.print(f"[red]Invalid video matcher: {video_matcher}[/red]")
        console.print(f"[yellow]Available matchers: {', '.join(VIDEO_MATCHER_CHOICES)}[/yellow]")
        raise typer.Exit(1)

    # Check for Pexels API key (only if using pexels matcher)
    import os
    if video_matcher == "pexels" and not os.environ.get("PEXELS_API_KEY"):
        console.print("[red]PEXELS_API_KEY not found in environment[/red]")
        console.print("[yellow]Add PEXELS_API_KEY to your .env file[/yellow]")
        console.print("[dim]Get free API key at: https://www.pexels.com/api/[/dim]")
        raise typer.Exit(1)

    # Format duration for display
    if target_duration >= 60:
        length_display = f"{int(target_duration // 60)}m{int(target_duration % 60)}s" if target_duration % 60 else f"{int(target_duration // 60)}m"
    else:
        length_display = f"{int(target_duration)}s"

    # Format voice info
    voice_info = voice
    if voice_rate != "+0%" or voice_pitch != "+0Hz":
        voice_info += f" (rate={voice_rate}, pitch={voice_pitch})"

    # Format GPU info
    gpu_info = "Disabled"
    if gpu_accelerate:
        if gpu is not None:
            gpu_info = f"Enabled (GPU {gpu})"
        else:
            gpu_info = "Enabled (auto-select)"

    console.print(Panel(
        f"Generating video reel for [cyan]{profile}[/cyan]\n"
        f"Text AI: [yellow]{text_ai or 'default'}[/yellow]\n"
        f"Video Matcher: [yellow]{video_matcher}[/yellow]\n"
        f"Voice: [yellow]{voice_info}[/yellow]\n"
        f"Subtitle Size: [yellow]{subtitle_size}px[/yellow]\n"
        f"Font: [yellow]{font}[/yellow]\n"
        f"Target Length: [yellow]{length_display}[/yellow]\n"
        f"GPU Acceleration: [{'green' if gpu_accelerate else 'dim'}]{gpu_info}[/{'green' if gpu_accelerate else 'dim'}]\n"
        f"Topic: [green]{topic or 'Auto-generated'}[/green]",
        title="Video Reel Generation",
    ))

    # Run video generation
    asyncio.run(_generate_reel(
        profile_path=profile_path,
        topic=topic,
        text_ai=text_ai,
        video_matcher=video_matcher,
        voice=voice,
        voice_rate=voice_rate,
        voice_pitch=voice_pitch,
        subtitle_size=subtitle_size,
        font=font,
        target_duration=target_duration,
        output_dir=Path(output_dir) if output_dir else None,
        dry_run=dry_run,
        loop=loop,
        loop_count=loop_count,
        gpu_accelerate=gpu_accelerate,
        gpu_index=gpu,
    ))


async def _generate_reel(
    profile_path: Path,
    topic: str | None,
    text_ai: str | None,
    video_matcher: str,
    voice: str,
    voice_rate: str,
    voice_pitch: str,
    subtitle_size: int,
    font: str,
    target_duration: float,
    output_dir: Path | None,
    dry_run: bool,
    loop: bool = False,
    loop_count: int | None = None,
    gpu_accelerate: bool = False,
    gpu_index: int | None = None,
):
    """Async video reel generation."""
    from .video.pipeline import VideoPipeline, setup_logging

    # Setup logging for video pipeline
    setup_logging()

    def progress_callback(stage: str, progress: float, message: str):
        """Display progress updates."""
        pct = int(progress * 100)
        console.print(f"  [{pct:3d}%] {stage}: {message}")

    # Create pipeline with options
    pipeline = VideoPipeline(
        voice=voice,
        voice_rate=voice_rate,
        voice_pitch=voice_pitch,
        text_ai=text_ai,
        video_matcher=video_matcher,
        subtitle_size=subtitle_size,
        subtitle_font=font,
        target_duration=target_duration,
        progress_callback=progress_callback,
        gpu_accelerate=gpu_accelerate,
        gpu_index=gpu_index,
    )

    if dry_run:
        # Dry run: just test the first few steps using the pipeline's configured steps
        console.print("\n[yellow]DRY RUN - Testing pipeline steps...[/yellow]\n")

        from .video.pipeline import (
            ProfileMetadata,
            PipelineContext,
        )
        import tempfile

        profile = ProfileMetadata.from_file(profile_path / "metadata.json")
        temp_dir = Path(tempfile.mkdtemp())

        context = PipelineContext(
            profile=profile,
            post_id="dry-run-test",
            output_dir=temp_dir / "output",
            temp_dir=temp_dir,
        )

        # Use pipeline's configured steps (with AI client if specified)
        # Step 1: Topic Selection
        console.print("[bold]Step 1: Topic Selection[/bold]")
        context = await pipeline.steps[0].execute(context)  # TopicSelector
        console.print(f"  Topic: [green]{context.topic.topic}[/green]")
        console.print(f"  Pillar: {context.topic.pillar_name}")

        # Step 2: Research
        console.print("\n[bold]Step 2: Topic Research[/bold]")
        context = await pipeline.steps[1].execute(context)  # TopicResearcher
        console.print(f"  Key points: {len(context.research.key_points)}")

        # Step 3: Script Planning
        console.print("\n[bold]Step 3: Script Planning[/bold]")
        context = await pipeline.steps[2].execute(context)  # ScriptPlanner
        console.print(f"  Title: {context.script.title}")
        console.print(f"  Segments: {len(context.script.segments)}")
        console.print(f"  Duration: {context.script.total_duration}s")

        console.print("\n[green]Dry run complete! Pipeline is working.[/green]")
        console.print("[dim]Run without --dry-run to generate full video[/dim]")
        return

    # Full generation
    from datetime import datetime
    import re
    import time

    video_count = 0
    loop_enabled = loop or loop_count is not None  # --loop or -n was provided
    loop_limit = loop_count  # None means infinite

    if loop_enabled:
        if loop_limit:
            console.print(f"\n[bold yellow]LOOP MODE[/bold yellow] - Will generate {loop_limit} video(s). Press Ctrl+C to stop.\n")
        else:
            console.print("\n[bold yellow]LOOP MODE[/bold yellow] - Will generate videos continuously. Press Ctrl+C to stop.\n")

    while True:
        video_count += 1

        if loop_enabled and video_count > 1:
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Starting video #{video_count}...[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        try:
            console.print("\n[bold cyan]Starting video generation pipeline...[/bold cyan]\n")

            # Generate output path in reels folder structure
            # reels/YYYY/MM/generated/DD-NNN-slug/
            now = datetime.now()

            # Generate post ID: YYYYMMDD-HHMMSS
            post_id = now.strftime("%Y%m%d-%H%M%S")

            # Calculate reel number for today
            reels_today_dir = profile_path / "reels" / now.strftime("%Y") / now.strftime("%m") / "generated"
            reels_today_dir.mkdir(parents=True, exist_ok=True)

            existing_reels = list(reels_today_dir.glob(f"{now.strftime('%d')}-*"))
            reel_number = len(existing_reels) + 1

            # Default slug will be updated after topic selection
            reel_slug = "reel"

            # Create output directory (will be updated after topic is selected)
            reel_output_dir = reels_today_dir / f"{now.strftime('%d')}-{reel_number:03d}-{reel_slug}"

            video_path = await pipeline.generate(
                profile_path=profile_path,
                output_dir=output_dir or reel_output_dir,
                post_id=post_id,
            )

            # Rename folder with actual slug from topic
            if output_dir is None and video_path and video_path.exists():
                # Get topic from metadata to create slug
                metadata_path = video_path.parent / "metadata.json"
                if metadata_path.exists():
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    topic_from_meta = metadata.get("topic", "reel")
                    # Create slug from topic
                    reel_slug = re.sub(r'[^a-z0-9]+', '-', topic_from_meta.lower())[:50].strip('-')

                    # Rename folder with proper slug
                    new_reel_dir = reels_today_dir / f"{now.strftime('%d')}-{reel_number:03d}-{reel_slug}"
                    if video_path.parent != new_reel_dir:
                        import shutil
                        shutil.move(str(video_path.parent), str(new_reel_dir))
                        video_path = new_reel_dir / video_path.name

            # Get actual video duration from metadata
            actual_duration = 60  # Default fallback
            if video_path and video_path.exists():
                meta_path = video_path.parent / "metadata.json"
                if meta_path.exists():
                    try:
                        import json
                        with open(meta_path) as f:
                            meta = json.load(f)
                        actual_duration = int(meta.get("duration_seconds", 60))
                    except Exception:
                        pass

            # Build title with progress info
            if loop_enabled:
                if loop_limit:
                    title = f"Complete (Video #{video_count}/{loop_limit})"
                else:
                    title = f"Complete (Video #{video_count})"
            else:
                title = "Complete"

            console.print(Panel(
                f"[bold green]Video generated successfully![/bold green]\n\n"
                f"[bold]Output:[/] {video_path}\n"
                f"[bold]Duration:[/] {actual_duration} seconds\n"
                f"[bold]Resolution:[/] 1080x1920 (9:16)",
                title=title,
                border_style="green",
            ))

            # If not looping, exit after first successful generation
            if not loop_enabled:
                break

            # If we've reached the loop limit, exit
            if loop_limit and video_count >= loop_limit:
                console.print(f"\n[bold green]Completed all {loop_limit} videos![/bold green]")
                break

            # Brief pause before next iteration
            remaining = f" ({loop_limit - video_count} remaining)" if loop_limit else ""
            console.print(f"\n[dim]Starting next video in 3 seconds...{remaining} (Ctrl+C to stop)[/dim]")
            time.sleep(3)

        except KeyboardInterrupt:
            console.print(f"\n\n[yellow]Loop stopped by user after {video_count} video(s).[/yellow]")
            break

        except Exception as e:
            console.print(f"\n[red]Video generation failed: {e}[/red]")
            console.print("[dim]Check logs for details[/dim]")

            if not loop_enabled:
                raise typer.Exit(1)

            # In loop mode, ask if user wants to continue after error
            console.print("\n[yellow]Error occurred. Retrying in 5 seconds... (Ctrl+C to stop)[/yellow]")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                console.print(f"\n[yellow]Loop stopped by user after {video_count} attempt(s).[/yellow]")
                break


@app.command()
def upload_reel(
    profile: str = typer.Argument(..., help="Profile name"),
    reel_id: str = typer.Argument(None, help="Reel ID to upload (if specified, uploads only this one)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without uploading"),
    one: bool = typer.Option(False, "--one", "-1", help="Upload only the oldest pending (instead of all)"),
    share_to_feed: bool = typer.Option(True, "--share-to-feed/--no-share-to-feed", help="Show reel on profile grid (default: True)"),
):
    """Upload pending reels to Instagram.

    By default, uploads ALL pending reels in chronological order.
    Use --one to upload only the oldest pending reel.

    Workflow:
        1. generate-reel: Generate videos in reels/YYYY/MM/generated/
        2. (manual) Move to reels/YYYY/MM/pending-post/
        3. upload-reel: Upload to Instagram, move to reels/YYYY/MM/posted/ (this command)

    Examples:
        socials upload-reel ai.for.mortals                    # Upload ALL pending reels
        socials upload-reel ai.for.mortals --one              # Upload only oldest pending
        socials upload-reel ai.for.mortals 15-001            # Upload specific (by prefix)
        socials upload-reel ai.for.mortals --dry-run         # Validate only
        socials upload-reel ai.for.mortals --no-share-to-feed # Don't show on profile grid
    """
    # Determine if posting all or just one
    all_reels = not one and reel_id is None
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
    asyncio.run(_post_reels_to_instagram(
        profile_path=profile_path,
        reel_id=reel_id,
        config=config,
        dry_run=dry_run,
        post_all=all_reels,
        share_to_feed=share_to_feed,
    ))


async def _post_reels_to_instagram(
    profile_path: Path,
    reel_id: str | None,
    config,
    dry_run: bool,
    post_all: bool = False,
    share_to_feed: bool = True,
):
    """Async Instagram Reels posting with progress display."""
    from datetime import datetime
    from .instagram import InstagramClient, CloudinaryUploader, InstagramProgress
    import shutil

    reels_dir = profile_path / "reels"

    # Step 1: Check for reels in 'generated' folders and auto-move to 'pending-post'
    generated_reels = []
    for year_dir in reels_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    generated_dir = month_dir / "generated"
                    if generated_dir.exists():
                        for reel_dir in generated_dir.iterdir():
                            if reel_dir.is_dir() and (reel_dir / "metadata.json").exists():
                                # Check for video file
                                video_file = reel_dir / "final.mp4"
                                if not video_file.exists():
                                    # Try other common names
                                    for alt_name in ["video.mp4", "reel.mp4", "output.mp4"]:
                                        alt_path = reel_dir / alt_name
                                        if alt_path.exists():
                                            video_file = alt_path
                                            break
                                if video_file.exists():
                                    generated_reels.append((reel_dir, month_dir))

    # Auto-move generated reels to pending-post
    if generated_reels:
        console.print(f"\n[cyan]Found {len(generated_reels)} reel(s) in 'generated' folder(s)[/cyan]")
        for reel_dir, month_dir in generated_reels:
            pending_dir = month_dir / "pending-post"
            pending_dir.mkdir(parents=True, exist_ok=True)
            new_path = pending_dir / reel_dir.name
            try:
                shutil.move(str(reel_dir), str(new_path))
                console.print(f"  [green][OK][/green] Moved to pending: {reel_dir.name}")
            except Exception as e:
                console.print(f"  [red][ERROR][/red] Failed to move {reel_dir.name}: {e}")

    # Step 2: Find reels in pending-post folder
    pending_reels = []
    for year_dir in reels_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    pending_dir = month_dir / "pending-post"
                    if pending_dir.exists():
                        for reel_dir in pending_dir.iterdir():
                            if reel_dir.is_dir() and (reel_dir / "metadata.json").exists():
                                pending_reels.append(reel_dir)

    if not pending_reels:
        console.print("[yellow]No reels ready to publish.[/yellow]")
        console.print("\n[dim]Generate reels first:[/dim]")
        console.print("  [cyan]python -m socials_automator.cli generate-reel <profile>[/cyan]")
        console.print("\n[dim]Then move to pending-post folder or use this command to auto-move generated reels.[/dim]")
        return

    # Sort chronologically: oldest first
    # Path format: reels/YYYY/MM/pending-post/DD-NNN-slug
    pending_reels.sort(key=lambda p: (
        p.parent.parent.parent.name,  # Year (YYYY)
        p.parent.parent.name,         # Month (MM)
        p.name,                       # Folder name (DD-NNN-slug)
    ))

    # Show queue summary
    if post_all or len(pending_reels) > 1:
        console.print(f"\n[bold cyan]Reels Queue ({len(pending_reels)} reels):[/bold cyan]")
        for i, p in enumerate(pending_reels, 1):
            try:
                with open(p / "metadata.json", encoding="utf-8") as f:
                    meta = json.load(f)
                topic = meta.get("topic", meta.get("post", {}).get("topic", "Unknown"))[:40]
            except Exception:
                topic = "Unknown"
            marker = "[yellow]>[/yellow]" if (post_all or i == 1) else " "
            console.print(f"  {marker} {i}. {p.name} - {topic}")
        console.print()

    # Determine which reels to publish
    if post_all:
        reels_to_publish = pending_reels
    elif reel_id:
        # Find specific reel
        reel_path = None
        for p in pending_reels:
            if reel_id in p.name or p.name.startswith(reel_id):
                reel_path = p
                break
        if not reel_path:
            console.print(f"[red]Reel not found in pending-post: {reel_id}[/red]")
            console.print(f"[dim]Available: {[p.name for p in pending_reels]}[/dim]")
            raise typer.Exit(1)
        reels_to_publish = [reel_path]
    else:
        # Use oldest pending reel
        reels_to_publish = [pending_reels[0]]

    # Validate token once before posting any
    if not dry_run:
        temp_client = InstagramClient(config=config)
        console.print("\n[dim]Validating Instagram access...[/dim]")
        try:
            account_info = await temp_client.validate_token()
            console.print(f"[green]Connected as @{account_info.get('username', 'unknown')}[/green]\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not validate token ({e})[/yellow]")
            console.print("[yellow]Proceeding with posting attempt...[/yellow]\n")

    # Publish each reel
    published_count = 0
    failed_count = 0
    skipped_count = 0

    for reel_idx, reel_path in enumerate(reels_to_publish, 1):
        if post_all:
            console.print(f"\n{'='*60}")
            console.print(f"[bold cyan]Publishing Reel {reel_idx}/{len(reels_to_publish)}: {reel_path.name}[/bold cyan]")
            console.print(f"{'='*60}")

        # Load reel metadata
        metadata_path = reel_path / "metadata.json"
        if not metadata_path.exists():
            console.print(f"[red]Reel metadata not found: {metadata_path}[/red]")
            failed_count += 1
            continue

        with open(metadata_path, encoding="utf-8") as f:
            reel_metadata = json.load(f)

        # Check for duplicates
        print(f"\n  [Phase 0] Checking for duplicates...")
        existing_instagram = reel_metadata.get("instagram", {})
        if existing_instagram.get("media_id"):
            existing_permalink = existing_instagram.get("permalink", "N/A")
            console.print(f"  [yellow][SKIP] Already posted to Instagram![/yellow]")
            console.print(f"  [dim]Media ID: {existing_instagram['media_id']}[/dim]")
            console.print(f"  [dim]URL: {existing_permalink}[/dim]")

            # Move to posted folder if still in pending
            if "pending-post" in str(reel_path):
                pending_parent = reel_path.parent
                posted_dir = pending_parent.parent / "posted"
                posted_dir.mkdir(parents=True, exist_ok=True)
                new_reel_path = posted_dir / reel_path.name
                try:
                    if new_reel_path.exists():
                        shutil.rmtree(str(new_reel_path))
                    shutil.move(str(reel_path), str(new_reel_path))
                    console.print(f"  [green]Moved duplicate to posted folder[/green]")
                except Exception as e:
                    console.print(f"  [yellow]Could not move: {e}[/yellow]")

            skipped_count += 1
            continue

        print(f"  [OK] No duplicate found")

        # Find video file
        video_path = reel_path / "final.mp4"
        if not video_path.exists():
            for alt_name in ["video.mp4", "reel.mp4", "output.mp4"]:
                alt_path = reel_path / alt_name
                if alt_path.exists():
                    video_path = alt_path
                    break

        if not video_path.exists():
            console.print(f"[red]No video file found in {reel_path}[/red]")
            console.print(f"[dim]Expected: final.mp4, video.mp4, reel.mp4, or output.mp4[/dim]")
            failed_count += 1
            continue

        # Load caption
        caption_path = reel_path / "caption.txt"
        hashtags_path = reel_path / "hashtags.txt"
        full_caption_path = reel_path / "caption+hashtags.txt"

        if full_caption_path.exists():
            caption = full_caption_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
            if hashtags_path.exists():
                hashtags = hashtags_path.read_text(encoding="utf-8").strip()
                if hashtags:
                    caption = f"{caption}\n\n{hashtags}"
        else:
            # Try metadata
            caption = reel_metadata.get("caption", "")

        # Extract reel info
        reel_topic = reel_metadata.get("topic", reel_metadata.get("post", {}).get("topic", "Unknown topic"))
        reel_id_display = reel_metadata.get("id", reel_path.name)

        # Show reel info
        console.print(Panel(
            f"[bold]Reel ID:[/] {reel_id_display}\n"
            f"[bold]Topic:[/] {reel_topic}\n"
            f"[bold]Video:[/] {video_path.name}\n"
            f"[bold]Location:[/] {reel_path}",
            title="Instagram Reel Posting",
        ))

        if dry_run:
            import re
            def safe_print_caption(text: str) -> str:
                """Remove characters that can't be displayed in Windows console."""
                try:
                    return text.encode('cp1252', errors='ignore').decode('cp1252')
                except Exception:
                    return re.sub(r'[^\x00-\x7F]+', '', text)

            console.print("\n[yellow]DRY RUN - Would upload this video:[/yellow]")
            console.print(f"  - {video_path.name}")
            console.print(f"\n[yellow]Caption ({len(caption)} chars):[/yellow]")
            display_caption = caption[:500] + "..." if len(caption) > 500 else caption
            console.print(safe_print_caption(display_caption))
            published_count += 1
            continue

        # Initialize progress display
        display = InstagramPostingDisplay()

        async def progress_callback(progress: InstagramProgress):
            display.update(progress)

        # Create uploader and client
        uploader = CloudinaryUploader(config=config)
        client = InstagramClient(config=config, progress_callback=progress_callback)

        result = None
        video_url = None

        try:
            # Check for resume - if we already uploaded to Cloudinary
            upload_state = reel_metadata.get("_upload_state", {})
            existing_url = upload_state.get("cloudinary_url")

            # Check if upload state is too old
            upload_is_stale = False
            if existing_url and upload_state.get("uploaded_at"):
                try:
                    uploaded_at = datetime.fromisoformat(upload_state["uploaded_at"])
                    age_hours = (datetime.now() - uploaded_at).total_seconds() / 3600
                    if age_hours > 24:
                        upload_is_stale = True
                        print(f"\n  [Warning] Cloudinary upload is {age_hours:.1f}h old - uploading fresh")
                except Exception:
                    pass

            if existing_url and not upload_is_stale:
                # Resume: use existing Cloudinary URL
                print(f"\n  [Resume] Found existing Cloudinary upload")
                video_url = existing_url
                # Track for cleanup
                if "cloudinary.com" in video_url:
                    parts = video_url.split("/upload/")
                    if len(parts) > 1:
                        public_id = parts[1].rsplit(".", 1)[0]
                        uploader._uploaded_public_ids.append((public_id, "video"))
            else:
                # Fresh upload to Cloudinary - get file info first
                file_size_mb = video_path.stat().st_size / (1024 * 1024)

                # Get video duration using moviepy
                try:
                    from moviepy import VideoFileClip
                    with VideoFileClip(str(video_path)) as clip:
                        duration_secs = clip.duration
                    duration_str = f"{int(duration_secs // 60)}:{int(duration_secs % 60):02d}"
                except Exception:
                    duration_str = "unknown"

                print(f"\n  [Upload] Uploading video to Cloudinary...")
                print(f"           Size: {file_size_mb:.1f} MB | Duration: {duration_str}")

                folder = f"socials-automator/{profile_path.name}/reels/{reel_id_display}"

                # Upload with spinner
                with console.status("[bold cyan]Uploading...[/bold cyan]", spinner="dots") as status:
                    video_url = await uploader.upload_video_async(video_path, folder=folder)

                print(f"  [OK] Video uploaded: {video_url[:60]}...")

                # Save upload state for resume capability
                reel_metadata["_upload_state"] = {
                    "cloudinary_url": video_url,
                    "uploaded_at": datetime.now().isoformat(),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(reel_metadata, f, indent=2)

            # Check for thumbnail and upload if exists
            cover_url = None
            thumbnail_path = reel_path / "thumbnail.jpg"
            if thumbnail_path.exists():
                thumb_size_kb = thumbnail_path.stat().st_size / 1024
                print(f"  [Upload] Uploading thumbnail ({thumb_size_kb:.0f} KB)...")
                try:
                    thumb_folder = f"socials-automator/{profile_path.name}/reels/{reel_id_display}/thumb"
                    cover_url = uploader.upload_image(thumbnail_path, folder=thumb_folder)
                    print(f"  [OK] Thumbnail uploaded")
                except Exception as e:
                    print(f"  [Warning] Thumbnail upload failed: {e} (using default)")

            print(f"  [Publish] Publishing to Instagram Reels...")

            # Publish to Instagram
            result = await client.publish_reel(
                video_url=video_url,
                caption=caption,
                cover_url=cover_url,
                share_to_feed=share_to_feed,
            )

            # Cleanup Cloudinary on success
            if result.success:
                deleted_count = await uploader.cleanup_async()
                print(f"  [Cleanup] Removed {deleted_count} temporary Cloudinary video(s)")

                # Remove upload state from metadata
                if "_upload_state" in reel_metadata:
                    del reel_metadata["_upload_state"]
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(reel_metadata, f, indent=2)

        except Exception as e:
            console.print(f"[red]Error publishing reel: {e}[/red]")
            if video_url:
                console.print(f"\n[yellow]Cloudinary upload saved - run the command again to resume.[/yellow]")
            failed_count += 1
            if not post_all:
                raise typer.Exit(1)
            continue

        # Show result
        if result and result.success:
            console.print(Panel(
                f"[bold green]Reel published successfully![/bold green]\n\n"
                f"[bold]Media ID:[/] {result.media_id}\n"
                f"[bold]URL:[/] {result.permalink or 'N/A'}",
                title="Success",
                border_style="green",
            ))

            # Update metadata with Instagram info
            reel_metadata["instagram"] = result.to_dict()
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(reel_metadata, f, indent=2)

            # Move from pending-post to posted folder
            pending_parent = reel_path.parent
            posted_dir = pending_parent.parent / "posted"
            posted_dir.mkdir(parents=True, exist_ok=True)
            new_reel_path = posted_dir / reel_path.name

            try:
                if new_reel_path.exists():
                    shutil.rmtree(str(new_reel_path))
                shutil.move(str(reel_path), str(new_reel_path))
                console.print(f"\n[green]Reel moved to:[/green] {new_reel_path}")
            except Exception as e:
                console.print(f"\n[yellow]Warning: Could not move reel folder: {e}[/yellow]")

            published_count += 1
        else:
            error_msg = result.error_message if result else "Unknown error"

            # Check for daily limit
            is_daily_limit = "[DAILY_POSTING_LIMIT]" in error_msg

            if is_daily_limit:
                console.print(Panel(
                    "[bold red]DAILY POSTING LIMIT REACHED[/bold red]\n\n"
                    "You've hit Instagram's Content Publishing API daily limit.\n\n"
                    "[bold]What to do:[/bold]\n"
                    "  - Wait until midnight UTC for the limit to reset\n"
                    "  - Or try again tomorrow\n\n"
                    "[dim]Your Cloudinary upload is saved - it'll be reused when you retry.[/dim]",
                    title="[bold red]Instagram Daily Limit[/bold red]",
                    border_style="red",
                ))
            else:
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
        console.print(f"[bold]Reels Publishing Summary[/bold]")
        console.print(f"{'='*60}")
        console.print(f"  [green]Published:[/green] {published_count}")
        if skipped_count > 0:
            console.print(f"  [yellow]Skipped (duplicates):[/yellow] {skipped_count}")
        if failed_count > 0:
            console.print(f"  [red]Failed:[/red] {failed_count}")
        console.print(f"  [dim]Total:[/dim] {len(reels_to_publish)}")

    if failed_count > 0:
        raise typer.Exit(1)


@app.command()
def fix_thumbnails(
    profile: str = typer.Argument(..., help="Profile name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
    force: bool = typer.Option(False, "--force", help="Regenerate ALL thumbnails, not just missing ones"),
    font_size: int = typer.Option(54, "--font-size", "-s", help="Font size in pixels (default: 54)"),
):
    """Generate missing thumbnails for existing reels.

    Scans all reel folders (generated, pending-post, posted) and generates
    thumbnails for any reel that doesn't have one.

    Use --force to regenerate ALL thumbnails (useful to fix incorrectly generated ones).
    Use --font-size to customize text size (default: 54px).

    Note: Instagram doesn't support updating cover images for already-posted
    reels. For posted reels, thumbnails are generated locally for reference only.

    Examples:
        socials fix-thumbnails ai.for.mortals              # Generate missing thumbnails
        socials fix-thumbnails ai.for.mortals --force      # Regenerate ALL thumbnails
        socials fix-thumbnails ai.for.mortals --font-size 80  # Larger text
        socials fix-thumbnails ai.for.mortals --dry-run    # Preview what would be done
    """
    import json
    from pathlib import Path

    from .video.pipeline.thumbnail_generator import ThumbnailGenerator

    profile_path = get_profile_path(profile)
    reels_dir = profile_path / "reels"

    if not reels_dir.exists():
        console.print("[yellow]No reels directory found.[/yellow]")
        return

    # Find all reel folders
    all_reels = []
    for year_dir in reels_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    for status_dir in ["generated", "pending-post", "posted"]:
                        status_path = month_dir / status_dir
                        if status_path.exists():
                            for reel_dir in status_path.iterdir():
                                if reel_dir.is_dir():
                                    # Check for video file
                                    video_file = reel_dir / "final.mp4"
                                    if video_file.exists():
                                        all_reels.append({
                                            "path": reel_dir,
                                            "status": status_dir,
                                            "has_thumbnail": (reel_dir / "thumbnail.jpg").exists(),
                                            "video_path": video_file,
                                        })

    if not all_reels:
        console.print("[yellow]No reels found.[/yellow]")
        return

    # Count reels by status
    missing_thumbnails = [r for r in all_reels if not r["has_thumbnail"]]
    has_thumbnails = [r for r in all_reels if r["has_thumbnail"]]

    console.print(f"\n[bold]Thumbnail Status[/bold]")
    console.print(f"  [green]Have thumbnails:[/green] {len(has_thumbnails)}")
    console.print(f"  [yellow]Missing thumbnails:[/yellow] {len(missing_thumbnails)}")

    # Determine which reels to process
    if force:
        reels_to_process = all_reels
        console.print(f"\n[bold cyan]--force: Will regenerate ALL {len(all_reels)} thumbnail(s)[/bold cyan]")
    else:
        reels_to_process = missing_thumbnails
        if not reels_to_process:
            console.print("\n[green]All reels have thumbnails![/green]")
            console.print("[dim]Use --force to regenerate all thumbnails[/dim]")
            return

    # Group by status
    by_status = {}
    for reel in reels_to_process:
        status = reel["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(reel)

    label = "Thumbnails to regenerate" if force else "Missing thumbnails"
    console.print(f"\n[bold]{label} by status:[/bold]")
    for status, reels in by_status.items():
        console.print(f"  {status}: {len(reels)}")

    if dry_run:
        action = "regenerate" if force else "generate"
        console.print(f"\n[yellow]DRY RUN - Would {action} {len(reels_to_process)} thumbnail(s):[/yellow]")
        for reel in reels_to_process:
            console.print(f"  - {reel['path'].name} ({reel['status']})")
        return

    # Generate thumbnails
    action = "Regenerating" if force else "Generating"
    console.print(f"\n[bold cyan]{action} {len(reels_to_process)} thumbnail(s)...[/bold cyan]")
    console.print(f"  Font size: {font_size}px")

    # Initialize thumbnail generator
    thumb_gen = ThumbnailGenerator(font_size=font_size)

    generated_count = 0
    failed_count = 0
    posted_count = 0

    # Get pexels cache directory
    pexels_cache_dir = Path(__file__).parent.parent.parent / "pexels" / "cache"

    for reel in reels_to_process:
        reel_path = reel["path"]
        status = reel["status"]

        # Get hook text and first segment's pexels_id from metadata
        metadata_path = reel_path / "metadata.json"
        hook_text = None
        first_pexels_id = None
        metadata = None

        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                # Try different places where hook might be stored
                hook_text = metadata.get("hook")
                if not hook_text and "script" in metadata:
                    hook_text = metadata["script"].get("hook")
                if not hook_text and "video" in metadata:
                    hook_text = metadata["video"].get("hook")

                # Get first segment's pexels_id for raw video (no subtitles)
                segments = metadata.get("segments", [])
                if segments:
                    first_pexels_id = segments[0].get("pexels_id")
            except Exception as e:
                console.print(f"  [dim]Warning: Could not read metadata for {reel_path.name}: {e}[/dim]")

        # Use folder name as fallback hook text
        if not hook_text:
            # Extract topic from folder name (format: DD-NNN-topic-slug)
            folder_name = reel_path.name
            parts = folder_name.split("-", 2)
            if len(parts) >= 3:
                hook_text = parts[2].replace("-", " ").title()
            else:
                hook_text = folder_name.replace("-", " ").title()

        console.print(f"  [{status}] {reel_path.name}...", end=" ")

        try:
            # Use raw cached video (no subtitles) if available, fallback to final.mp4
            video_path = None
            using_raw_cache = False
            if first_pexels_id and pexels_cache_dir.exists():
                cached_video = pexels_cache_dir / f"{first_pexels_id}.mp4"
                if cached_video.exists():
                    video_path = cached_video
                    using_raw_cache = True

            # Fallback to final.mp4 if cache not available
            if not video_path:
                video_path = reel_path / "final.mp4"
                console.print("[dim](using final.mp4)[/dim] ", end="")

            # Extract frame from video
            frame_image = thumb_gen._extract_frame(video_path, thumb_gen.frame_time)

            # If using raw cache, need to crop/resize to 1080x1920 (9:16)
            # Raw Pexels videos can be any aspect ratio
            if using_raw_cache:
                from PIL import Image
                TARGET_WIDTH = 1080
                TARGET_HEIGHT = 1920
                TARGET_RATIO = 9 / 16

                w, h = frame_image.size
                current_ratio = w / h

                # Crop to 9:16 aspect ratio (center crop)
                if abs(current_ratio - TARGET_RATIO) > 0.01:
                    if current_ratio > TARGET_RATIO:
                        # Image is too wide, crop width
                        new_width = int(h * TARGET_RATIO)
                        left = (w - new_width) // 2
                        frame_image = frame_image.crop((left, 0, left + new_width, h))
                    else:
                        # Image is too tall, crop height
                        new_height = int(w / TARGET_RATIO)
                        top = (h - new_height) // 2
                        frame_image = frame_image.crop((0, top, w, top + new_height))

                # Resize to exact target dimensions
                frame_image = frame_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

            # Render hook text on frame
            thumbnail = thumb_gen._render_text_on_image(frame_image, hook_text)

            # Save thumbnail
            thumbnail_path = reel_path / "thumbnail.jpg"
            thumbnail.save(thumbnail_path, "JPEG", quality=95)

            # Update artifact metadata
            if metadata_path.exists():
                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Initialize artifacts section if it doesn't exist
                    if "artifacts" not in metadata:
                        metadata["artifacts"] = {}

                    # Update thumbnail artifact
                    metadata["artifacts"]["thumbnail"] = {
                        "status": "ok",
                        "file": "thumbnail.jpg",
                    }

                    # Also populate other artifacts if missing (for existing reels)
                    if "video" not in metadata["artifacts"]:
                        if (reel_path / "final.mp4").exists():
                            metadata["artifacts"]["video"] = {"status": "ok", "file": "final.mp4"}
                    if "voiceover" not in metadata["artifacts"]:
                        if (reel_path / "voiceover.mp3").exists():
                            metadata["artifacts"]["voiceover"] = {"status": "ok", "file": "voiceover.mp3"}
                    if "subtitles" not in metadata["artifacts"]:
                        if (reel_path / "final.mp4").exists():
                            metadata["artifacts"]["subtitles"] = {"status": "ok", "file": "final.mp4"}
                    if "caption" not in metadata["artifacts"]:
                        if (reel_path / "caption.txt").exists():
                            metadata["artifacts"]["caption"] = {"status": "ok", "file": "caption.txt"}
                    if "hashtags" not in metadata["artifacts"]:
                        if (reel_path / "caption+hashtags.txt").exists():
                            metadata["artifacts"]["hashtags"] = {"status": "ok", "file": "caption+hashtags.txt"}

                    # Write updated metadata
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                except Exception as meta_err:
                    console.print(f"[dim]Warning: Could not update metadata: {meta_err}[/dim]")

            if status == "posted":
                console.print("[green][OK][/green] [dim](local only - can't update Instagram)[/dim]")
                posted_count += 1
            else:
                console.print("[green][OK][/green]")
            generated_count += 1

        except Exception as e:
            console.print(f"[red][FAILED][/red] {e}")
            # Update artifact metadata with failure
            if metadata_path.exists():
                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                    if "artifacts" not in metadata:
                        metadata["artifacts"] = {}
                    metadata["artifacts"]["thumbnail"] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            failed_count += 1

    # Summary
    console.print(f"\n{'='*60}")
    console.print(f"[bold]Thumbnail Generation Summary[/bold]")
    console.print(f"{'='*60}")
    console.print(f"  [green]Generated:[/green] {generated_count}")
    if posted_count > 0:
        console.print(f"  [dim]  (of which {posted_count} are posted - local only)[/dim]")
    if failed_count > 0:
        console.print(f"  [red]Failed:[/red] {failed_count}")

    if posted_count > 0:
        console.print(f"\n[yellow]Note:[/yellow] Instagram doesn't support updating cover images for")
        console.print(f"already-posted reels. The {posted_count} posted reel thumbnail(s) are saved")
        console.print(f"locally for reference only.")


@app.command()
def update_artifacts(
    profile: str = typer.Argument(..., help="Profile name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
):
    """Update artifact metadata for all existing reels.

    Scans all reel folders and populates the artifacts section in metadata.json
    based on what files exist in each folder.

    Artifact tracking includes:
    - video: final.mp4
    - voiceover: voiceover.mp3
    - subtitles: (burned into final.mp4)
    - thumbnail: thumbnail.jpg
    - caption: caption.txt
    - hashtags: caption+hashtags.txt

    Examples:
        socials update-artifacts ai.for.mortals              # Update all metadata
        socials update-artifacts ai.for.mortals --dry-run    # Preview what would be done
    """
    import json
    from pathlib import Path

    profile_path = get_profile_path(profile)
    reels_dir = profile_path / "reels"

    if not reels_dir.exists():
        console.print("[yellow]No reels directory found.[/yellow]")
        return

    # Find all reel folders
    all_reels = []
    for year_dir in reels_dir.glob("*"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    for status_dir in ["generated", "pending-post", "posted"]:
                        status_path = month_dir / status_dir
                        if status_path.exists():
                            for reel_dir in status_path.iterdir():
                                if reel_dir.is_dir():
                                    metadata_path = reel_dir / "metadata.json"
                                    if metadata_path.exists():
                                        all_reels.append({
                                            "path": reel_dir,
                                            "status": status_dir,
                                            "metadata_path": metadata_path,
                                        })

    if not all_reels:
        console.print("[yellow]No reels found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(all_reels)} reel(s)[/bold]")

    # Check which need artifact updates
    needs_update = []
    already_has = []

    for reel in all_reels:
        with open(reel["metadata_path"], encoding="utf-8") as f:
            metadata = json.load(f)
        if "artifacts" in metadata and len(metadata["artifacts"]) >= 6:
            already_has.append(reel)
        else:
            needs_update.append(reel)

    console.print(f"  [green]Have artifacts:[/green] {len(already_has)}")
    console.print(f"  [yellow]Missing/incomplete artifacts:[/yellow] {len(needs_update)}")

    if not needs_update:
        console.print("\n[green]All reels have complete artifact metadata![/green]")
        return

    if dry_run:
        console.print(f"\n[yellow]DRY RUN - Would update {len(needs_update)} metadata file(s):[/yellow]")
        for reel in needs_update:
            console.print(f"  - {reel['path'].name} ({reel['status']})")
        return

    # Update metadata
    console.print(f"\n[bold cyan]Updating {len(needs_update)} metadata file(s)...[/bold cyan]")

    updated_count = 0
    failed_count = 0

    for reel in needs_update:
        reel_path = reel["path"]
        metadata_path = reel["metadata_path"]

        console.print(f"  [{reel['status']}] {reel_path.name}...", end=" ")

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            # Build artifacts based on existing files
            artifacts = metadata.get("artifacts", {})

            # Video (final.mp4)
            if (reel_path / "final.mp4").exists():
                artifacts["video"] = {"status": "ok", "file": "final.mp4"}
            else:
                artifacts["video"] = {"status": "missing"}

            # Voiceover (voiceover.mp3)
            if (reel_path / "voiceover.mp3").exists():
                artifacts["voiceover"] = {"status": "ok", "file": "voiceover.mp3"}
            else:
                artifacts["voiceover"] = {"status": "missing"}

            # Subtitles (burned into final.mp4)
            if (reel_path / "final.mp4").exists():
                artifacts["subtitles"] = {"status": "ok", "file": "final.mp4"}
            else:
                artifacts["subtitles"] = {"status": "missing"}

            # Thumbnail
            if (reel_path / "thumbnail.jpg").exists():
                artifacts["thumbnail"] = {"status": "ok", "file": "thumbnail.jpg"}
            else:
                artifacts["thumbnail"] = {"status": "missing"}

            # Caption
            if (reel_path / "caption.txt").exists():
                artifacts["caption"] = {"status": "ok", "file": "caption.txt"}
            else:
                artifacts["caption"] = {"status": "missing"}

            # Hashtags
            if (reel_path / "caption+hashtags.txt").exists():
                artifacts["hashtags"] = {"status": "ok", "file": "caption+hashtags.txt"}
            else:
                artifacts["hashtags"] = {"status": "missing"}

            metadata["artifacts"] = artifacts

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Count ok/missing
            ok_count = sum(1 for a in artifacts.values() if a.get("status") == "ok")
            console.print(f"[green][OK][/green] ({ok_count}/6 artifacts)")
            updated_count += 1

        except Exception as e:
            console.print(f"[red][FAILED][/red] {e}")
            failed_count += 1

    # Summary
    console.print(f"\n{'='*60}")
    console.print(f"[bold]Artifact Update Summary[/bold]")
    console.print(f"{'='*60}")
    console.print(f"  [green]Updated:[/green] {updated_count}")
    if failed_count > 0:
        console.print(f"  [red]Failed:[/red] {failed_count}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
