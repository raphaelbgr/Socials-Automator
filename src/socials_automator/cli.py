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
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich import box

from .content import ContentGenerator
from .content.models import GenerationProgress

# Configure logging to suppress console output - all logs go to files only
# This prevents WARNING/INFO messages from polluting the Rich CLI display

# Remove any default console handlers from root logger
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(logging.CRITICAL)  # Suppress root logger output

# Suppress loggers that might print to console (except ai_calls which has file handler)
for logger_name in ["httpx", "httpcore", "urllib3", "asyncio"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

# Ensure ai_calls and instagram_api loggers don't propagate to console
for logger_name in ["ai_calls", "instagram_api"]:
    logger = logging.getLogger(logger_name)
    logger.propagate = False  # Don't propagate to root (keep file handlers)

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
            "text_model": None,
            "text_prompt_preview": None,
            "image_provider": None,
            "image_model": None,
            "image_prompt_preview": None,
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

        # Track providers (persist values - don't overwrite with None)
        if progress.text_provider:
            self.stats["text_provider"] = progress.text_provider
        if progress.text_model:
            self.stats["text_model"] = progress.text_model
        if progress.text_prompt_preview:
            self.stats["text_prompt_preview"] = progress.text_prompt_preview
        if progress.text_provider:
            self.stats["providers_used"].add(progress.text_provider)

        if progress.image_provider:
            self.stats["image_provider"] = progress.image_provider
            self.stats["providers_used"].add(progress.image_provider)
        if progress.image_model:
            self.stats["image_model"] = progress.image_model
        if progress.image_prompt_preview:
            self.stats["image_prompt_preview"] = progress.image_prompt_preview

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

        # Show current action prominently (validation attempts, etc.)
        if progress.current_action:
            action_color = "yellow" if "Retry" in progress.current_action else "cyan"
            lines.append(f"[bold {action_color}]Action:[/] {progress.current_action}")

        # Show validation progress if in validation
        if progress.validation_attempt > 0:
            if progress.validation_max_attempts > 0:
                lines.append(f"[bold]Attempt:[/] {progress.validation_attempt}/{progress.validation_max_attempts}")
            else:
                lines.append(f"[bold]Attempt:[/] {progress.validation_attempt} (auto-retry)")
            if progress.validation_error:
                # Truncate long error messages
                error_text = progress.validation_error[:60] + "..." if len(progress.validation_error) > 60 else progress.validation_error
                lines.append(f"[dim red]Last error: {error_text}[/]")

        lines.append(f"[bold]Progress:[/] {progress.completed_steps}/{progress.total_steps} ({progress.progress_percent:.0f}%)")
        lines.append("")

        # Current slide info
        if progress.total_slides > 0:
            lines.append(f"[bold cyan]Slide:[/] {progress.current_slide}/{progress.total_slides}")
            lines.append("")

        # Text AI Activity - ALWAYS show if we have persisted provider info
        text_provider = progress.text_provider or self.stats.get("text_provider")
        text_model = progress.text_model or self.stats.get("text_model")
        text_prompt = progress.text_prompt_preview or self.stats.get("text_prompt_preview")

        if text_provider:
            lines.append("[bold magenta]Text AI:[/]")
            lines.append(f"  Provider: [green]{text_provider}[/]")
            if text_model:
                lines.append(f"  Model: [blue]{text_model}[/]")
            if text_prompt:
                lines.append(f"  [dim]Last prompt:[/] {text_prompt[:70]}...")
            if progress.text_failed_providers:
                failed = " | ".join(progress.text_failed_providers)
                lines.append(f"  [dim]Failed providers: {failed}[/]")
            lines.append("")

        # Image AI Activity - show if we have provider info
        image_provider = progress.image_provider or self.stats.get("image_provider")
        image_model = progress.image_model or self.stats.get("image_model")
        image_prompt = progress.image_prompt_preview or self.stats.get("image_prompt_preview")

        if image_provider:
            lines.append("[bold magenta]Image AI:[/]")
            lines.append(f"  Provider: [green]{image_provider}[/]")
            if image_model:
                lines.append(f"  Model: [blue]{image_model}[/]")
            if image_prompt:
                lines.append(f"  [dim]Last prompt:[/] {image_prompt[:70]}...")
            if progress.image_failed_providers:
                failed = " | ".join(progress.image_failed_providers)
                lines.append(f"  [dim]Failed providers: {failed}[/]")
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
    post_after: bool = typer.Option(False, "--post", help="Publish to Instagram after generating"),
    auto_retry: bool = typer.Option(False, "--auto-retry", help="Retry indefinitely until valid content"),
    text_ai: str = typer.Option(None, "--text-ai", help="Text AI provider (zai, groq, gemini, openai, lmstudio, ollama)"),
    image_ai: str = typer.Option(None, "--image-ai", help="Image AI provider (dalle, fal_flux, comfy)"),
    loop_each: str = typer.Option(None, "--loop-each", help="Loop interval (e.g., 5m, 1h, 30s)"),
):
    """Generate carousel posts for a profile.

    By default, the AI decides the optimal number of slides (3-10) based on
    the topic content. Use --slides to force a specific count.

    Use --post to automatically publish to Instagram after generation.
    Use --auto-retry to keep retrying until valid content is generated.
    Use --text-ai to select text provider (zai, groq, gemini, openai, lmstudio, ollama).
    Use --image-ai to select image provider (dalle, fal_flux, comfy).
    Use --loop-each to continuously generate posts (e.g., --loop-each 5m).
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
        auto_retry=auto_retry,
        text_provider_override=text_ai,
        image_provider_override=image_ai,
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
                except RuntimeError as e:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass
                    # Show a clear error panel
                    error_lines = [
                        "[bold red]Content Generation Failed[/bold red]",
                        "",
                        f"[red]{e}[/red]",
                        "",
                        "[bold yellow]What happened:[/bold yellow]",
                        "  The AI repeatedly failed to generate content that matches the topic.",
                        "  For example, if the topic asks for '5 tips', the AI kept generating",
                        "  a different number of content slides.",
                        "",
                        "[bold cyan]Suggestions:[/bold cyan]",
                        "  1. Try again - AI responses can vary",
                        "  2. Use a simpler topic (e.g., '3 tips' instead of '7 tips')",
                        "  3. Check logs/ai_calls.log for detailed error information",
                        "  4. Consider switching text provider in config/providers.yaml",
                    ]
                    console.print(Panel(
                        "\n".join(error_lines),
                        title="[bold red]Generation Error[/bold red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                    return
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
                except RuntimeError as e:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass
                    # Show a clear error panel
                    error_lines = [
                        "[bold red]Content Generation Failed[/bold red]",
                        "",
                        f"[red]{e}[/red]",
                        "",
                        "[bold yellow]What happened:[/bold yellow]",
                        "  The AI repeatedly failed to generate content that matches the topic.",
                        "  For example, if the topic asks for '5 tips', the AI kept generating",
                        "  a different number of content slides.",
                        "",
                        "[bold cyan]Suggestions:[/bold cyan]",
                        "  1. Try again - AI responses can vary",
                        "  2. Use a simpler topic with --topic flag",
                        "  3. Check logs/ai_calls.log for detailed error information",
                        "  4. Consider switching text provider in config/providers.yaml",
                    ]
                    console.print(Panel(
                        "\n".join(error_lines),
                        title="[bold red]Generation Error[/bold red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                    return
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
        shutil.move(str(generated_path), str(pending_path))
        console.print(f"\n[dim]Moved to pending-post: {pending_path.name}[/dim]")

        # Now post to Instagram
        console.print("\n[bold cyan]Publishing to Instagram...[/bold cyan]")
        await _post_to_instagram(
            profile_path=profile_path,
            post_id=None,
            config=ig_config,
            dry_run=False,
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

    # Build full caption with hashtags
    caption_text = post.caption if post.caption else post.hook_text
    hashtags_str = " ".join(f"#{tag.lstrip('#')}" for tag in post.hashtags) if post.hashtags else ""
    full_text = f"{caption_text}\n\n{hashtags_str}" if hashtags_str else caption_text
    char_count = len(full_text)

    # Threads compatibility indicator
    threads_ok = char_count <= 500
    threads_status = "[green]OK for Threads[/green]" if threads_ok else "[red]Too long for Threads (>500)[/red]"

    # Instagram-ready content
    console.print(f"\n[bold yellow]Instagram + Threads Ready ({char_count} chars - {threads_status}):[/bold yellow]")
    console.print(f"[bold cyan]{'-' * 60}[/bold cyan]\n")

    # Caption (with safe encoding for console)
    console.print(safe_print(caption_text))

    # Hashtags
    if hashtags_str:
        console.print(f"\n{safe_print(hashtags_str)}")

    console.print(f"\n[bold cyan]{'-' * 60}[/bold cyan]")


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


class InstagramProgressDisplay:
    """Rich display for Instagram publishing progress."""

    def __init__(self):
        from .instagram.models import InstagramProgress
        self.progress = InstagramProgress()

    def update(self, progress):
        """Update the progress state."""
        self.progress = progress

    def render(self) -> Panel:
        """Render the progress display."""
        from .instagram.models import InstagramPostStatus

        progress = self.progress
        lines = []

        # Status with color
        status_color = {
            InstagramPostStatus.PENDING: "white",
            InstagramPostStatus.UPLOADING: "yellow",
            InstagramPostStatus.CREATING_CONTAINERS: "cyan",
            InstagramPostStatus.PUBLISHING: "blue",
            InstagramPostStatus.PUBLISHED: "green",
            InstagramPostStatus.FAILED: "red",
        }.get(progress.status, "white")

        lines.append(f"[bold {status_color}]Status:[/] {progress.status.value.upper()}")
        lines.append(f"[bold]Step:[/] {progress.current_step}")
        lines.append(f"[bold]Progress:[/] {progress.progress_percent:.0f}%")
        lines.append("")

        # Image upload progress
        if progress.total_images > 0:
            lines.append(f"[bold cyan]Images:[/] {progress.images_uploaded}/{progress.total_images} uploaded")
            lines.append(f"[bold cyan]Containers:[/] {progress.containers_created}/{progress.total_images} created")
            lines.append("")

        # Error display
        if progress.error:
            lines.append(f"[bold red]Error:[/] {progress.error}")

        return Panel(
            "\n".join(lines),
            title="[bold]Instagram Publishing[/]",
            border_style="blue",
            box=box.ROUNDED,
        )


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
        with open(post_path / "metadata.json") as f:
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
def post(
    profile: str = typer.Argument(..., help="Profile name"),
    post_id: str = typer.Argument(None, help="Post ID to publish (oldest pending if not specified)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without posting"),
):
    """Post a pending carousel to Instagram.

    Posts from the pending-post queue. After successful posting,
    the post is moved to the posted folder.

    Workflow:
        1. generate: Creates posts in posts/YYYY/MM/generated/
        2. schedule: Moves to posts/YYYY/MM/pending-post/
        3. post: Publishes to Instagram, moves to posts/YYYY/MM/posted/ (this command)

    Examples:
        socials post ai.for.mortals                    # Post oldest pending
        socials post ai.for.mortals 11-001            # Post specific (by prefix)
        socials post ai.for.mortals --dry-run         # Validate only
    """
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
):
    """Async Instagram posting with progress display."""
    from .instagram import InstagramClient, CloudinaryUploader, InstagramProgress
    import shutil

    # Find posts in pending-post folder (scan all year/month subfolders)
    posts_dir = profile_path / "posts"
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
        console.print("[yellow]No posts in pending-post folders.[/yellow]")
        console.print("\n[dim]Workflow:[/dim]")
        console.print("  1. Generate posts: [cyan]python -m socials_automator.cli generate <profile> --topic '...'[/cyan]")
        console.print("  2. Schedule posts: [cyan]python -m socials_automator.cli schedule <profile>[/cyan]")
        console.print("  3. Post to Instagram: [cyan]python -m socials_automator.cli post <profile>[/cyan]")
        raise typer.Exit(1)

    # Sort by folder name (date-number-slug) to get oldest first
    pending_posts.sort(key=lambda p: p.name)

    # Select post to publish
    if post_id:
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
    else:
        # Use oldest pending post
        post_path = pending_posts[0]

    # Load post metadata
    metadata_path = post_path / "metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]Post metadata not found: {metadata_path}[/red]")
        raise typer.Exit(1)

    with open(metadata_path) as f:
        post_metadata = json.load(f)

    # Get slide images
    slide_paths = sorted(post_path.glob("slide_*.jpg"))
    if not slide_paths:
        console.print(f"[red]No slide images found in {post_path}[/red]")
        raise typer.Exit(1)

    # Load caption (use UTF-8 for emoji support)
    caption_path = post_path / "caption.txt"
    caption = caption_path.read_text(encoding="utf-8") if caption_path.exists() else ""

    # Load hashtags
    hashtags_path = post_path / "hashtags.txt"
    if hashtags_path.exists():
        hashtags = hashtags_path.read_text(encoding="utf-8").strip()
        if hashtags and not caption.endswith(hashtags):
            caption = f"{caption}\n\n{hashtags}"

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
        console.print("\n[green]Dry run complete. No images were uploaded.[/green]")
        return

    # Initialize progress display
    display = InstagramProgressDisplay()

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

    # Validate token first (skip on error - validation endpoint can be flaky)
    console.print("\n[dim]Validating Instagram access...[/dim]")
    try:
        account_info = await client.validate_token()
        console.print(f"[green]Connected as @{account_info.get('username', 'unknown')}[/green]\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not validate token ({e})[/yellow]")
        console.print("[yellow]Proceeding with posting attempt...[/yellow]\n")

    # Run with live progress display
    with Live(console=console, refresh_per_second=4) as live:
        async def update_display():
            while True:
                live.update(display.render())
                await asyncio.sleep(0.25)

        display_task = asyncio.create_task(update_display())

        try:
            # Step 1: Upload images to Cloudinary
            display.update(InstagramProgress(
                current_step="Uploading images to Cloudinary...",
                total_images=len(slide_paths),
            ))

            folder = f"socials-automator/{profile_path.name}/{post_id_display}"
            image_urls = await uploader.upload_batch(slide_paths, folder=folder)

            # Update progress
            display.update(InstagramProgress(
                current_step="Images uploaded, creating Instagram containers...",
                images_uploaded=len(image_urls),
                total_images=len(slide_paths),
                image_urls=image_urls,
                progress_percent=30.0,
            ))

            # Step 2: Publish to Instagram
            result = await client.publish_carousel(
                image_urls=image_urls,
                caption=caption,
            )

            # Step 3: Cleanup Cloudinary uploads
            if result.success:
                await uploader.cleanup_async()

        finally:
            display_task.cancel()
            try:
                await display_task
            except asyncio.CancelledError:
                pass

    # Show result
    if result.success:
        console.print(Panel(
            f"[bold green]Published successfully![/bold green]\n\n"
            f"[bold]Media ID:[/] {result.media_id}\n"
            f"[bold]URL:[/] {result.permalink or 'N/A'}",
            title="Success",
            border_style="green",
        ))

        # Update post metadata with Instagram info
        post_metadata["instagram"] = result.to_dict()
        with open(metadata_path, "w") as f:
            json.dump(post_metadata, f, indent=2)

        # Move from pending-post to posted folder
        pending_parent = post_path.parent  # e.g., posts/2025/12/pending-post
        posted_dir = pending_parent.parent / "posted"
        posted_dir.mkdir(parents=True, exist_ok=True)
        new_post_path = posted_dir / post_path.name

        try:
            shutil.move(str(post_path), str(new_post_path))
            console.print(f"\n[green]Post moved to:[/green] {new_post_path}")
        except Exception as e:
            console.print(f"\n[yellow]Warning: Could not move post folder: {e}[/yellow]")
            console.print(f"[dim]Post remains at: {post_path}[/dim]")
    else:
        console.print(Panel(
            f"[bold red]Publishing failed[/bold red]\n\n"
            f"[bold]Error:[/] {result.error_message}",
            title="Error",
            border_style="red",
        ))
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
