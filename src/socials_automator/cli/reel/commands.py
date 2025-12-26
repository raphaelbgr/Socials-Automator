"""Reel CLI commands - thin wrappers orchestrating params, validation, display, and service."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer

from ..core.console import console
from ..core.types import Failure
from .display import (
    show_loop_complete,
    show_loop_progress,
    show_reel_config,
    show_reel_error,
    show_reel_result,
    show_upload_config,
    show_upload_result,
)
from .params import ReelGenerationParams, ReelUploadParams
from .service import ReelGeneratorService, ReelUploaderService
from .validators import validate_reel_generation_params, validate_reel_upload_params


def generate_reel(
    profile: str = typer.Argument(..., help="Profile name"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Specific topic"),
    text_ai: Optional[str] = typer.Option(None, "--text-ai", help="Text AI provider"),
    video_matcher: str = typer.Option("pexels", "--video-matcher", help="Video source"),
    voice: str = typer.Option("rvc_adam", "--voice", "-v", help="TTS voice"),
    voice_rate: str = typer.Option("+0%", "--voice-rate", help="Voice speed"),
    voice_pitch: str = typer.Option("+0Hz", "--voice-pitch", help="Voice pitch"),
    subtitle_size: int = typer.Option(80, "--subtitle-size", help="Subtitle font size"),
    font: str = typer.Option("Montserrat-Bold.ttf", "--font", help="Font file"),
    length: str = typer.Option("1m", "--length", "-l", help="Target duration (30s, 1m, 1m30s)"),
    hashtags: int = typer.Option(5, "--hashtags", "-H", help="Max hashtags to generate (default: 5, Instagram limit)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without full generation"),
    upload: bool = typer.Option(False, "--upload", help="Upload to Instagram after generating"),
    loop: bool = typer.Option(False, "--loop", help="Loop continuously"),
    loop_count: Optional[int] = typer.Option(None, "--loop-count", "-n", help="Number of videos to generate"),
    loop_each: Optional[str] = typer.Option(None, "--loop-each", help="Interval between loops (e.g., 5m, 1h, 30s)"),
    gpu_accelerate: bool = typer.Option(False, "--gpu-accelerate", "-g", help="Use GPU acceleration"),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="Specific GPU index"),
    # News-specific options
    news: bool = typer.Option(False, "--news", help="Force news mode (auto-detected for news profiles)"),
    edition: Optional[str] = typer.Option(None, "--edition", "-e", help="News edition: morning, midday, evening, night"),
    story_count: Optional[str] = typer.Option("auto", "--stories", "-s", help="Number of news stories (auto = AI decides)"),
    news_max_age: int = typer.Option(24, "--news-age", help="Max news age in hours"),
    # Image overlay feature
    overlay_images: bool = typer.Option(False, "--overlay-images", help="Add contextual images that illustrate narration"),
    image_provider: str = typer.Option("websearch", "--image-provider", help="Image provider for overlays (websearch, pexels, pixabay)"),
    use_tor: bool = typer.Option(False, "--use-tor", help="Route websearch through Tor for anonymity"),
    blur: Optional[str] = typer.Option(None, "--blur", help="Dim background during overlays: light, medium, heavy"),
    smart_pick: bool = typer.Option(False, "--smart-pick", help="Use AI vision to select best matching images"),
    smart_pick_count: int = typer.Option(10, "--smart-pick-count", help="Number of candidates to compare (default: 10)"),
    # Dense overlay mode
    overlay_image_ttl: Optional[str] = typer.Option(None, "--overlay-image-ttl", help="Fixed display time per image (e.g., '3s'). Enables dense mode."),
    overlay_image_minimum: Optional[int] = typer.Option(None, "--overlay-image-minimum", help="Target number of images (default: auto-calculated from TTL)."),
) -> None:
    """Generate a video reel for a profile.

    Use --upload to automatically upload to Instagram after generation.

    For news profiles (auto-detected via 'news_sources' in metadata.json):
    - Uses RSS feeds and web search to aggregate news
    - AI curates and ranks stories
    - Generates news briefing video

    News options:
    - --edition: Force specific edition (morning/midday/evening/night)
    - --stories: Number of stories per video (default 4)
    - --news-age: Max article age in hours (default 24)
    """
    # Build immutable params from CLI args
    params = ReelGenerationParams.from_cli(
        profile=profile,
        topic=topic,
        text_ai=text_ai,
        video_matcher=video_matcher,
        voice=voice,
        voice_rate=voice_rate,
        voice_pitch=voice_pitch,
        subtitle_size=subtitle_size,
        font=font,
        length=length,
        hashtags=hashtags,
        output_dir=output_dir,
        dry_run=dry_run,
        upload=upload,
        loop=loop,
        loop_count=loop_count,
        loop_each=loop_each,
        gpu_accelerate=gpu_accelerate,
        gpu=gpu,
        # News options
        news=news,
        edition=edition,
        story_count=story_count,
        news_max_age=news_max_age,
        # Image overlay
        overlay_images=overlay_images,
        image_provider=image_provider,
        use_tor=use_tor,
        blur=blur,
        smart_pick=smart_pick,
        smart_pick_count=smart_pick_count,
        # Dense overlay mode
        overlay_image_ttl=overlay_image_ttl,
        overlay_image_minimum=overlay_image_minimum,
    )

    # Validate params
    validation = validate_reel_generation_params(params)
    if isinstance(validation, Failure):
        show_reel_error(console, validation.error, validation.details)
        raise typer.Exit(1)

    # Handle dry run
    if params.dry_run:
        show_reel_config(console, params)
        _run_dry_run(params)
        return

    # Handle loop mode (shows config at start of each iteration)
    if params.loop:
        _run_loop_mode(params)
        return

    # Display configuration (single generation only)
    show_reel_config(console, params)

    # Single generation
    reel_path = _run_single_generation(params)

    # Upload if requested and generation succeeded
    if reel_path is not None and params.upload_after:
        _upload_generated_reel(params, reel_path)


def _run_dry_run(params: ReelGenerationParams) -> None:
    """Execute dry run mode."""
    service = ReelGeneratorService()
    result = asyncio.run(service.dry_run(params))

    if isinstance(result, Failure):
        show_reel_error(console, result.error, result.details)
        raise typer.Exit(1)

    console.print("\n[bold]Dry Run Results:[/bold]")
    for step, data in result.value.items():
        console.print(f"  [cyan]{step}:[/cyan] {data}")


def _run_single_generation(
    params: ReelGenerationParams,
    video_count: Optional[int] = None,
    loop_limit: Optional[int] = None,
) -> Optional[Path]:
    """Execute single video generation. Returns output path on success, None on failure."""
    service = ReelGeneratorService()
    result = asyncio.run(service.generate(params))

    if isinstance(result, Failure):
        show_reel_error(console, result.error, result.details)
        return None

    show_reel_result(
        console,
        result.value.output_path,
        int(result.value.duration_seconds),
        video_count=video_count,
        loop_limit=loop_limit,
    )
    return result.value.output_path


def _upload_generated_reel(params: ReelGenerationParams, reel_path: Path) -> bool:
    """Upload a generated reel using the same flow as upload-reel command.

    Uses upload_all() with reel_id filter for consistent behavior:
    - Targeted preflight (validation, repair)
    - Upload
    - Postflight (verify and fix posted folders)

    Returns True on success.
    """
    console.print("\n[bold cyan]Uploading to Instagram...[/bold cyan]")

    # Get the reel folder - handle both file and folder paths
    reel_folder = reel_path.parent if reel_path.is_file() else reel_path
    reel_id = reel_folder.name

    # Build upload params with reel_id filter
    upload_params = ReelUploadParams.from_cli(
        profile=params.profile,
        reel_id=reel_id,  # This triggers targeted preflight mode
        one=True,
        dry_run=False,
    )

    # Execute upload - uses same flow as upload-reel command
    service = ReelUploaderService()
    result = asyncio.run(service.upload_all(upload_params))

    if isinstance(result, Failure):
        show_reel_error(console, result.error, result.details)
        return False

    # Show result
    results = result.value if isinstance(result.value, list) else [result.value]
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count
    show_upload_result(console, success_count, failed_count, results)
    return success_count > 0


def _run_loop_mode(params: ReelGenerationParams) -> None:
    """Execute loop mode for continuous generation."""
    video_count = 0
    loop_limit = params.loop_count

    try:
        while True:
            video_count += 1

            # Check if we've reached the limit
            if loop_limit and video_count > loop_limit:
                show_loop_complete(console, loop_limit)
                break

            # Show config panel at start of each iteration
            console.print()  # Blank line before panel
            show_reel_config(console, params)

            # Generate video
            reel_path = _run_single_generation(
                params,
                video_count=video_count,
                loop_limit=loop_limit,
            )

            if reel_path is None:
                console.print("[yellow]Generation failed, continuing loop...[/yellow]")
            elif params.upload_after:
                # Upload immediately after generation in loop mode
                _upload_generated_reel(params, reel_path)

            # Check if we've completed all requested videos
            if loop_limit and video_count >= loop_limit:
                show_loop_complete(console, loop_limit)
                break

            # Show progress and wait
            wait_seconds = params.loop_each if params.loop_each else 3
            show_loop_progress(console, video_count, loop_limit, wait_seconds)
            time.sleep(wait_seconds)

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Loop stopped. Generated {video_count} video(s).[/yellow]")


def upload_tiktok_browser(
    source: str = typer.Option(..., "--source", "-s", help="Source Instagram profile to get reels from (required)"),
    reel_id: Optional[str] = typer.Argument(None, help="Reel ID to upload (uploads only this one if specified)"),
    port: int = typer.Option(9333, "--port", "-P", help="Chrome remote debugging port"),
    interval: Optional[str] = typer.Option(None, "--interval", "-i", help="Interval between uploads (e.g., 2m, 30s, 1h)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate upload"),
    one: bool = typer.Option(False, "--one", "-1", help="Upload only the oldest pending"),
    posted_only: bool = typer.Option(False, "--posted-only", "-p", help="Only look in posted/ folder (for cross-posting)"),
    # Rate limiting options
    sort: str = typer.Option("most-recent", "--sort", help="Sort order: most-recent, oldest"),
    daily_limit: int = typer.Option(3, "--daily-limit", "-L", help="Max uploads per day (default: 3)"),
    timing: str = typer.Option("random", "--timing", help="Timing mode: random, spaced, immediate"),
    time_window: str = typer.Option("8-23", "--time-window", "-w", help="Hours to post (e.g., 8-23 for 8am-11pm)"),
    min_gap: str = typer.Option("2h", "--min-gap", "-g", help="Minimum time between uploads (e.g., 2h, 90m)"),
) -> None:
    """Upload reel(s) to TikTok using Chrome with remote debugging.

    This command uses Selenium to connect to a Chrome browser running in
    debug mode. Your TikTok login session persists across uploads.

    RATE LIMITING (to avoid TikTok shadowban):
    - Default: 3 videos/day, random timing within 8am-11pm window
    - Safe posting: TikTok recommends 1-3 videos/day max
    - Spacing: Minimum 2 hours between uploads

    TIMING MODES:
    - random: Post at random times within time window (recommended)
    - spaced: Post at evenly spaced intervals within time window
    - immediate: Post now, ignore time window (for testing)

    FIRST TIME SETUP:
    1. pip install selenium
    2. Close ALL Chrome windows
    3. Start Chrome with remote debugging
    4. Log in to TikTok in that Chrome window
    5. Run this command

    Examples:
        # Upload with rate limiting (default: 3/day, random timing)
        upload-tiktok-browser --source ai.for.mortals --posted-only

        # Upload 5 videos/day with custom time window
        upload-tiktok-browser --source ai.for.mortals --daily-limit 5 --time-window "9-21"

        # Upload oldest videos first
        upload-tiktok-browser --source ai.for.mortals --sort oldest

        # Upload immediately (bypass rate limiting)
        upload-tiktok-browser --source ai.for.mortals --timing immediate

        # Cross-post with 4-hour minimum gap
        upload-tiktok-browser --source ai.for.mortals --posted-only --min-gap 4h
    """
    import json
    import random
    from datetime import datetime, timedelta

    from ..core.paths import get_profile_path
    from ..core.parsers import parse_interval
    from socials_automator.tiktok.browser_uploader import (
        TikTokBrowserUploader,
        get_chrome_profile_dir,
        get_chrome_launch_command,
        print_chrome_instructions,
        TIKTOK_STUDIO_URL,
    )
    # Import display helpers early for timestamps
    from socials_automator.tiktok.display import (
        get_local_timestamp,
        get_video_info,
        get_reel_metadata,
        count_hashtags,
        format_duration,
        format_size,
        get_caption_preview,
        calculate_eta,
        log_json,
    )

    profile_path = get_profile_path(source)
    chrome_profile_dir = get_chrome_profile_dir(source)
    chrome_commands = get_chrome_launch_command(source, port)

    # Parse interval (default 3 seconds if not specified)
    interval_seconds = parse_interval(interval) if interval else 3

    # Parse min_gap
    min_gap_seconds = parse_interval(min_gap) if min_gap else 7200  # 2h default

    # Parse time window (e.g., "8-23" -> 8am to 11pm)
    try:
        window_start, window_end = map(int, time_window.split("-"))
        if not (0 <= window_start <= 23 and 0 <= window_end <= 23):
            raise ValueError
    except ValueError:
        console.print(f"[red][X] Invalid time window: {time_window}. Use format like '8-23'[/red]")
        raise typer.Exit(1)

    # Validate sort option
    if sort not in ["most-recent", "oldest"]:
        console.print(f"[red][X] Invalid sort option: {sort}. Use 'most-recent' or 'oldest'[/red]")
        raise typer.Exit(1)

    # Validate timing option
    if timing not in ["random", "spaced", "immediate"]:
        console.print(f"[red][X] Invalid timing option: {timing}. Use 'random', 'spaced', or 'immediate'[/red]")
        raise typer.Exit(1)

    ts = get_local_timestamp

    # Daily tracking file
    tracking_file = profile_path / "tiktok" / "daily_uploads.json"
    tracking_file.parent.mkdir(parents=True, exist_ok=True)

    # Load or initialize daily tracking
    today = datetime.now().strftime("%Y-%m-%d")
    tracking_data = {"date": today, "uploads": [], "count": 0}
    if tracking_file.exists():
        try:
            with open(tracking_file, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("date") == today:
                tracking_data = saved
            # else: new day, reset tracking
        except Exception:
            pass

    uploads_today = tracking_data["count"]
    remaining_today = max(0, daily_limit - uploads_today)
    last_upload_time = None
    if tracking_data["uploads"]:
        last_upload_time = datetime.fromisoformat(tracking_data["uploads"][-1])

    # Calculate time since last upload
    time_since_last = None
    if last_upload_time:
        time_since_last = (datetime.now() - last_upload_time).total_seconds()

    # Check current hour for time window
    current_hour = datetime.now().hour
    in_time_window = window_start <= current_hour < window_end

    # Calculate next available upload time
    next_upload_time = None
    wait_reason = None

    if timing != "immediate":
        # Check daily limit
        if uploads_today >= daily_limit:
            tomorrow = datetime.now().replace(hour=window_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
            next_upload_time = tomorrow
            wait_reason = f"Daily limit reached ({daily_limit}/day)"

        # Check min gap
        elif last_upload_time and time_since_last < min_gap_seconds:
            next_upload_time = last_upload_time + timedelta(seconds=min_gap_seconds)
            wait_reason = f"Min gap not met ({min_gap})"

        # Check time window
        elif not in_time_window:
            if current_hour < window_start:
                next_upload_time = datetime.now().replace(hour=window_start, minute=0, second=0, microsecond=0)
            else:  # current_hour >= window_end
                next_upload_time = datetime.now().replace(hour=window_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_reason = f"Outside time window ({window_start}:00-{window_end}:00)"

    # Display settings panel
    console.print()
    console.print("=" * 70)
    console.print("[bold cyan]>>> TikTok Browser Upload[/bold cyan]")
    console.print("=" * 70)
    console.print()

    # Source & Chrome settings
    console.print("[bold]CONNECTION[/bold]")
    console.print(f"  Source profile:     [cyan]{source}[/cyan]")
    console.print(f"  Chrome port:        [dim]{port}[/dim]")
    console.print(f"  Chrome profile:     [dim]profiles/{source}/tiktok/browser/[/dim]")
    console.print()

    # Rate limiting settings
    console.print("[bold]RATE LIMITING[/bold]")
    console.print(f"  Daily limit:        [yellow]{daily_limit}[/yellow] videos/day")
    console.print(f"  Min gap:            [yellow]{min_gap}[/yellow] ({min_gap_seconds // 60}m) between uploads")
    console.print(f"  Time window:        [yellow]{window_start}:00 - {window_end}:00[/yellow] ({window_end - window_start}h window)")
    console.print(f"  Timing mode:        [yellow]{timing}[/yellow]", end="")
    if timing == "random":
        console.print(" [dim](random times within window)[/dim]")
    elif timing == "spaced":
        console.print(" [dim](evenly distributed)[/dim]")
    else:
        console.print(" [dim](bypass rate limiting)[/dim]")
    console.print()

    # Current status
    console.print("[bold]TODAY'S STATUS[/bold]")
    console.print(f"  Date:               [dim]{today}[/dim]")
    console.print(f"  Uploaded today:     [{'green' if uploads_today < daily_limit else 'red'}]{uploads_today}/{daily_limit}[/{'green' if uploads_today < daily_limit else 'red'}]")
    console.print(f"  Remaining:          [{'green' if remaining_today > 0 else 'red'}]{remaining_today}[/{'green' if remaining_today > 0 else 'red'}] videos")
    console.print(f"  Current time:       [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")
    console.print(f"  In time window:     [{'green' if in_time_window else 'yellow'}]{'Yes' if in_time_window else 'No'}[/{'green' if in_time_window else 'yellow'}]", end="")
    if not in_time_window:
        console.print(f" [dim](window: {window_start}:00-{window_end}:00)[/dim]")
    else:
        console.print()
    if last_upload_time:
        mins_ago = int(time_since_last // 60) if time_since_last else 0
        console.print(f"  Last upload:        [dim]{mins_ago}m ago ({last_upload_time.strftime('%H:%M:%S')})[/dim]")
    else:
        console.print(f"  Last upload:        [dim]None today[/dim]")
    console.print()

    # Upload settings
    console.print("[bold]UPLOAD SETTINGS[/bold]")
    console.print(f"  Sort order:         [yellow]{sort}[/yellow]", end="")
    if sort == "most-recent":
        console.print(" [dim](newest first)[/dim]")
    else:
        console.print(" [dim](oldest first)[/dim]")
    console.print(f"  Posted only:        [{'green' if posted_only else 'dim'}]{'Yes (cross-posting)' if posted_only else 'No'}[/{'green' if posted_only else 'dim'}]")
    console.print(f"  Single reel:        [{'green' if one else 'dim'}]{'Yes' if one else 'No'}[/{'green' if one else 'dim'}]")
    if reel_id:
        console.print(f"  Specific reel:      [cyan]{reel_id}[/cyan]")
    console.print()

    # Dry run warning
    if dry_run:
        console.print("[bold yellow]MODE: DRY RUN[/bold yellow] - No actual uploads will be performed")
        console.print()

    # Wait/block notice
    if next_upload_time and timing != "immediate":
        wait_seconds = (next_upload_time - datetime.now()).total_seconds()
        wait_mins = int(wait_seconds // 60)
        wait_hours = int(wait_mins // 60)
        wait_mins_rem = wait_mins % 60

        console.print("[bold red]UPLOAD BLOCKED[/bold red]")
        console.print(f"  Reason:             {wait_reason}")
        console.print(f"  Next upload at:     [yellow]{next_upload_time.strftime('%Y-%m-%d %H:%M')}[/yellow]")
        if wait_hours > 0:
            console.print(f"  Wait time:          [yellow]{wait_hours}h {wait_mins_rem}m[/yellow]")
        else:
            console.print(f"  Wait time:          [yellow]{wait_mins}m[/yellow]")
        console.print()

        if not dry_run:
            console.print("[dim]Use --timing immediate to bypass rate limiting (not recommended)[/dim]")
            raise typer.Exit(0)

    console.print("=" * 70)

    # Find reels to upload
    console.print()
    console.print(f"[dim][{ts()}][/dim] Scanning for pending reels...")
    pending_reels = _find_tiktok_pending_reels(profile_path, reel_id, posted_only=posted_only)

    if not pending_reels:
        console.print(f"[dim][{ts()}][/dim] [yellow]No pending reels found for TikTok upload.[/yellow]")
        raise typer.Exit(0)

    total_pending = len(pending_reels)

    # Sort reels based on --sort option
    if sort == "most-recent":
        # Sort by folder name descending (DD-NNN format means higher = newer)
        pending_reels = sorted(pending_reels, key=lambda p: p.name, reverse=True)
    else:  # oldest
        # Sort by folder name ascending
        pending_reels = sorted(pending_reels, key=lambda p: p.name, reverse=False)

    # Limit to remaining daily quota (unless timing=immediate)
    if timing != "immediate" and remaining_today < len(pending_reels):
        console.print(f"[dim][{ts()}][/dim]   Total pending: {total_pending}")
        console.print(f"[dim][{ts()}][/dim]   [yellow]Limiting to {remaining_today} (daily quota)[/yellow]")
        pending_reels = pending_reels[:remaining_today]

    # Limit to one if requested
    if one:
        pending_reels = pending_reels[:1]

    console.print(f"[dim][{ts()}][/dim]   Found: [green]{len(pending_reels)}[/green] reel(s) to upload")
    console.print(f"[dim][{ts()}][/dim]   Sort:  [dim]{sort}[/dim]")

    if dry_run:
        for reel_path in pending_reels:
            console.print(f"[dim][{ts()}][/dim]     - {reel_path.name}")
        return

    # Check if Selenium is installed
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        console.print(f"[dim][{ts()}][/dim] [bold red][X] Selenium not installed![/bold red]")
        console.print(f"[dim][{ts()}][/dim]   Run: [cyan]pip install selenium[/cyan]")
        raise typer.Exit(1)

    # Show Chrome setup instructions first
    console.print()
    console.print("=" * 70)
    console.print(f"[bold]CHROME SETUP FOR TIKTOK ({source})[/bold]")
    console.print("=" * 70)
    console.print()
    console.print("1. Close ALL Chrome windows completely")
    console.print()
    console.print("2. Open a [bold]separate terminal[/bold] and run ONE of these commands:")
    console.print()
    console.print("-" * 70)
    console.print()
    console.print("[bold][Windows PowerShell / CMD]:[/bold]")
    console.print(f"  [cyan]{chrome_commands['windows']}[/cyan]")
    console.print()
    console.print("[bold][macOS Terminal]:[/bold]")
    console.print(f"  [cyan]{chrome_commands['macos']}[/cyan]")
    console.print()
    console.print("[bold][Linux Terminal]:[/bold]")
    console.print(f"  [cyan]{chrome_commands['linux']}[/cyan]")
    console.print()
    console.print("-" * 70)
    console.print()
    console.print(f"3. Chrome profile will be stored at:")
    console.print(f"   [dim]profiles/{source}/tiktok/browser/[/dim]")
    console.print()
    console.print(f"4. Navigate to: [link]{TIKTOK_STUDIO_URL}[/link]")
    console.print()
    console.print("5. Log in to TikTok (session will persist for future uploads)")
    console.print()
    console.print("=" * 70)
    console.print()
    console.print("[bold yellow]Press Enter when Chrome is running and you're logged in to TikTok...[/bold yellow]")
    input()

    # Try to connect to Chrome
    console.print()
    console.print(f"[dim][{ts()}][/dim] Connecting to Chrome...")

    uploader = TikTokBrowserUploader(profile_name=source, port=port)

    try:
        connected = uploader.connect()
    except Exception as e:
        console.print(f"[dim][{ts()}][/dim] [red][X] Connection error: {e}[/red]")
        connected = False

    if not connected:
        console.print(f"[dim][{ts()}][/dim] [bold red][X] Could not connect to Chrome![/bold red]")
        console.print(f"[dim][{ts()}][/dim]   Make sure:")
        console.print(f"[dim][{ts()}][/dim]     1. Chrome is running with --remote-debugging-port={port}")
        console.print(f"[dim][{ts()}][/dim]     2. You used the correct user-data-dir path")
        console.print(f"[dim][{ts()}][/dim]     3. No other Chrome instance is using port {port}")
        console.print(f"[dim][{ts()}][/dim]   Try closing ALL Chrome windows and running the command from step 2 again.")
        raise typer.Exit(1)

    console.print(f"[dim][{ts()}][/dim] [green][OK] Connected to Chrome[/green]")

    # Check if logged in
    console.print(f"[dim][{ts()}][/dim] Checking TikTok login status...")

    try:
        logged_in = uploader.ensure_logged_in()
    except Exception as e:
        console.print(f"[dim][{ts()}][/dim] [red][X] Error checking login: {e}[/red]")
        logged_in = False

    if not logged_in:
        console.print(f"[dim][{ts()}][/dim] [yellow]Not logged in yet. Please log in to TikTok in the Chrome window.[/yellow]")
        console.print(f"[dim][{ts()}][/dim]   Navigate to: [link]{TIKTOK_STUDIO_URL}[/link]")
        console.print()
        console.print("[bold yellow]Press Enter when logged in...[/bold yellow]")
        input()

        # Re-check
        console.print(f"[dim][{ts()}][/dim] Re-checking login status...")
        try:
            logged_in = uploader.ensure_logged_in()
        except Exception as e:
            console.print(f"[dim][{ts()}][/dim] [red][X] Error: {e}[/red]")
            logged_in = False

        if not logged_in:
            console.print(f"[dim][{ts()}][/dim] [red][X] Still not logged in. Please log in and try again.[/red]")
            raise typer.Exit(1)

    console.print(f"[dim][{ts()}][/dim] [green][OK] Logged in to TikTok[/green]")
    console.print()

    # Upload each reel
    success_count = 0
    failed_count = 0
    upload_start_time = time.time()

    for i, reel_path in enumerate(pending_reels, 1):
        reel_start_time = time.time()
        reel_id = reel_path.name

        # Calculate progress and ETA
        elapsed_total = time.time() - upload_start_time
        pct_complete = int((i - 1) / len(pending_reels) * 100)
        eta = calculate_eta(i - 1, len(pending_reels), elapsed_total) if i > 1 else "calculating..."

        console.print()
        console.print("-" * 65)
        console.print(f"[dim][{ts()}][/dim] [bold white on blue] {i}/{len(pending_reels)} [/bold white on blue] ({pct_complete}% complete, {eta} remaining)")
        console.print(f"[dim][{ts()}][/dim] [bold]Reel:[/bold] {reel_id}")
        console.print("-" * 65)

        # Get video and caption
        video_path = reel_path / "final.mp4"
        caption_path = reel_path / "caption+hashtags.txt"

        if not video_path.exists():
            console.print(f"[dim][{ts()}][/dim]   [red][X] Video not found: final.mp4[/red]")
            log_json("upload_error", reel_id, error="Video file not found")
            failed_count += 1
            continue

        # Get video info
        video_size = video_path.stat().st_size
        video_info = get_video_info(video_path)

        # Load caption
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            caption_alt = reel_path / "caption.txt"
            if caption_alt.exists():
                caption = caption_alt.read_text(encoding="utf-8").strip()
            else:
                caption = ""

        # Count hashtags
        hashtag_count, hashtags = count_hashtags(caption)
        hashtags_kept = hashtags[:2]  # TikTok limit
        hashtags_removed = hashtags[2:] if len(hashtags) > 2 else []

        # Find thumbnail
        thumbnail_path = None
        thumb_size = 0
        for ext in ["jpg", "jpeg", "png"]:
            thumb = reel_path / f"thumbnail.{ext}"
            if thumb.exists():
                thumbnail_path = thumb
                thumb_size = thumb.stat().st_size
                break

        # Get metadata
        meta = get_reel_metadata(reel_path)

        # Display detailed info (Option B style)
        console.print(f"[dim][{ts()}][/dim]   [bold]Video:[/bold] {format_size(video_size)} | {format_duration(video_info['duration_s'])} | {video_info['width']}x{video_info['height']} | {video_info['codec'].upper()}")

        # Caption info
        console.print(f"[dim][{ts()}][/dim]   [bold]Caption:[/bold] {len(caption)} chars")
        console.print(f"[dim][{ts()}][/dim]     Preview: \"{get_caption_preview(caption, 50)}\"")
        if hashtag_count > 0:
            kept_str = ", ".join(hashtags_kept) if hashtags_kept else "none"
            hashtag_msg = f"Hashtags: {hashtag_count} -> 2 ({kept_str} kept"
            if hashtags_removed:
                removed_str = ", ".join(hashtags_removed[:3])
                if len(hashtags_removed) > 3:
                    removed_str += f" +{len(hashtags_removed) - 3} more"
                hashtag_msg += f", {removed_str} removed"
            hashtag_msg += ")"
            console.print(f"[dim][{ts()}][/dim]     {hashtag_msg}")

        # Thumbnail info
        if thumbnail_path:
            console.print(f"[dim][{ts()}][/dim]   [bold]Thumbnail:[/bold] {thumbnail_path.name} ({format_size(thumb_size)})")
        else:
            console.print(f"[dim][{ts()}][/dim]   [bold]Thumbnail:[/bold] [yellow]not found[/yellow]")

        # Instagram info
        if meta["instagram_posted_at"]:
            ig_info = f"Posted {meta['instagram_posted_at']}"
            if meta["instagram_media_id"]:
                ig_info += f" | Media ID: {meta['instagram_media_id'][:15]}..."
            console.print(f"[dim][{ts()}][/dim]   [bold]Instagram:[/bold] {ig_info}")

        # Previous attempts
        if meta["tiktok_attempts"] > 0:
            console.print(f"[dim][{ts()}][/dim]   [bold]Previous TikTok attempts:[/bold] [yellow]{meta['tiktok_attempts']}[/yellow]")

        # Log upload start
        log_json("upload_start", reel_id,
            size_mb=round(video_size / (1024*1024), 2),
            duration_s=video_info["duration_s"],
            resolution=f"{video_info['width']}x{video_info['height']}",
            hashtags_total=hashtag_count,
            hashtags_kept=2 if hashtag_count >= 2 else hashtag_count,
            has_thumbnail=thumbnail_path is not None
        )

        # Progress callback with timestamps
        def progress(msg: str, pct: float) -> None:
            timestamp = get_local_timestamp()
            # Progress bar
            filled = int(pct / 5)  # 20 chars total
            bar = ">" * filled + " " * (20 - filled)
            console.print(f"[dim][{timestamp}][/dim]   [{bar}] {int(pct)}% [cyan]{msg}[/cyan]")

        # Upload
        console.print(f"[dim][{ts()}][/dim]   [yellow]Starting upload...[/yellow]")

        result = uploader.upload(
            video_path=video_path,
            description=caption,
            thumbnail_path=thumbnail_path,
            video_name=reel_path.name,
            progress_callback=progress,
        )

        # Calculate upload duration
        reel_elapsed = time.time() - reel_start_time

        if result.success:
            console.print(f"[dim][{ts()}][/dim]   [>>>>>>>>>>>>>>>>>>>>] 100% [bold green]Done![/bold green]")
            console.print(f"[dim][{ts()}][/dim]   [bold green][OK] Uploaded to TikTok! (Total: {int(reel_elapsed)}s)[/bold green]")

            # Show video URL if captured
            if result.video_url:
                console.print(f"[dim][{ts()}][/dim]   [bold]URL:[/bold] {result.video_url}")

            success_count += 1

            # Update daily tracking
            tracking_data["uploads"].append(datetime.now().isoformat())
            tracking_data["count"] = len(tracking_data["uploads"])
            with open(tracking_file, "w", encoding="utf-8") as f:
                json.dump(tracking_data, f, indent=2)
            console.print(f"[dim][{ts()}][/dim]   Daily count: {tracking_data['count']}/{daily_limit}")

            # Log success
            log_json("upload_complete", reel_id,
                status="success",
                elapsed_s=int(reel_elapsed),
                retries=0,
                video_url=result.video_url,
                video_id=result.video_id,
            )

            # Update metadata with video URL
            _update_tiktok_status(
                reel_path,
                success=True,
                video_url=result.video_url,
                video_id=result.video_id,
            )

            # Move to posted if not already there
            if "posted" not in str(reel_path):
                new_path = _move_reel_to_posted(reel_path, profile_path)
                if new_path:
                    console.print(f"[dim][{ts()}][/dim]   Moved to posted/")
        else:
            console.print(f"[dim][{ts()}][/dim]   [bold red][X] Upload failed: {result.error}[/bold red]")

            # Log failure
            log_json("upload_error", reel_id,
                status="failed",
                error=result.error,
                elapsed_s=int(reel_elapsed)
            )

            # Check if it's a rate limit error - retry with exponential backoff
            is_rate_limit = result.error and (
                "rate limit" in result.error.lower() or
                "muitas tentativas" in result.error.lower() or
                "too many" in result.error.lower() or
                "carregamento" in result.error.lower()
            )

            if is_rate_limit:
                max_retries = 3
                retry_success = False

                for retry in range(1, max_retries + 1):
                    # Exponential backoff: interval * 2^retry
                    backoff_seconds = interval_seconds * (2 ** retry)
                    console.print()
                    console.print(f"[dim][{ts()}][/dim]   [yellow]Rate limit detected. Retry {retry}/{max_retries}[/yellow]")
                    console.print(f"[dim][{ts()}][/dim]   [yellow]Waiting {backoff_seconds}s ({backoff_seconds // 60}m {backoff_seconds % 60}s)...[/yellow]")

                    # Log retry attempt
                    log_json("retry_wait", reel_id,
                        retry=retry,
                        wait_seconds=backoff_seconds
                    )

                    time.sleep(backoff_seconds)

                    # Refresh the page before retry
                    console.print(f"[dim][{ts()}][/dim]   Refreshing upload page...")
                    try:
                        uploader.driver.get(TIKTOK_STUDIO_URL)
                        time.sleep(3)
                    except Exception as e:
                        console.print(f"[dim][{ts()}][/dim]   [red]Failed to refresh page: {e}[/red]")

                    # Retry upload
                    console.print(f"[dim][{ts()}][/dim]   [cyan]Retrying upload...[/cyan]")
                    retry_start = time.time()
                    result = uploader.upload(
                        video_path=video_path,
                        description=caption,
                        thumbnail_path=thumbnail_path,
                        video_name=reel_path.name,
                        progress_callback=progress,
                    )
                    retry_elapsed = time.time() - retry_start

                    if result.success:
                        console.print(f"[dim][{ts()}][/dim]   [bold green][OK] Retry {retry} successful! ({int(retry_elapsed)}s)[/bold green]")

                        # Show video URL if captured
                        if result.video_url:
                            console.print(f"[dim][{ts()}][/dim]   [bold]URL:[/bold] {result.video_url}")

                        success_count += 1

                        # Update daily tracking
                        tracking_data["uploads"].append(datetime.now().isoformat())
                        tracking_data["count"] = len(tracking_data["uploads"])
                        with open(tracking_file, "w", encoding="utf-8") as f:
                            json.dump(tracking_data, f, indent=2)
                        console.print(f"[dim][{ts()}][/dim]   Daily count: {tracking_data['count']}/{daily_limit}")

                        # Log retry success
                        log_json("upload_complete", reel_id,
                            status="success",
                            elapsed_s=int(retry_elapsed),
                            retries=retry,
                            video_url=result.video_url,
                            video_id=result.video_id,
                        )

                        _update_tiktok_status(
                            reel_path,
                            success=True,
                            video_url=result.video_url,
                            video_id=result.video_id,
                        )
                        if "posted" not in str(reel_path):
                            new_path = _move_reel_to_posted(reel_path, profile_path)
                            if new_path:
                                console.print(f"[dim][{ts()}][/dim]   Moved to posted/")
                        retry_success = True
                        break
                    else:
                        console.print(f"[dim][{ts()}][/dim]   [red]Retry {retry} failed: {result.error}[/red]")
                        log_json("retry_failed", reel_id,
                            retry=retry,
                            error=result.error
                        )

                if not retry_success:
                    console.print(f"[dim][{ts()}][/dim]   [bold red][X] All {max_retries} retries failed for {reel_path.name}[/bold red]")
                    failed_count += 1

                    # Log final failure
                    log_json("upload_failed", reel_id,
                        status="failed_all_retries",
                        retries=max_retries,
                        error=f"Rate limit - all {max_retries} retries failed"
                    )

                    _update_tiktok_status(reel_path, success=False, error=f"Rate limit - all {max_retries} retries failed")
            else:
                # Non-rate-limit error
                failed_count += 1
                _update_tiktok_status(reel_path, success=False, error=result.error)

        # Delay between uploads (based on timing mode)
        if i < len(pending_reels):
            if timing == "immediate":
                # Minimal delay for testing
                wait_seconds = interval_seconds if interval else 3
            elif timing == "random":
                # Random delay between min_gap and min_gap * 2
                wait_seconds = random.randint(min_gap_seconds, min_gap_seconds * 2)
            else:  # spaced
                # Calculate evenly spaced intervals within remaining time window
                now = datetime.now()
                window_end_time = now.replace(hour=window_end, minute=0, second=0)
                if now.hour >= window_end:
                    window_end_time += timedelta(days=1)
                remaining_window = (window_end_time - now).total_seconds()
                remaining_uploads = len(pending_reels) - i
                wait_seconds = max(min_gap_seconds, int(remaining_window / (remaining_uploads + 1)))

            if wait_seconds > 60:
                wait_hours = wait_seconds // 3600
                wait_mins = (wait_seconds % 3600) // 60
                wait_secs = wait_seconds % 60
                if wait_hours > 0:
                    wait_display = f"{int(wait_hours)}h {int(wait_mins)}m"
                else:
                    wait_display = f"{int(wait_mins)}m {int(wait_secs)}s"
                console.print(f"[dim][{ts()}][/dim]   [yellow]Next upload in {wait_display} (timing: {timing})[/yellow]")

                # For long waits, show countdown every minute
                if wait_seconds > 300:  # > 5 minutes
                    next_time = datetime.now() + timedelta(seconds=wait_seconds)
                    console.print(f"[dim][{ts()}][/dim]   Next upload at: {next_time.strftime('%H:%M:%S')}")

            time.sleep(wait_seconds)

    # Clean up
    uploader.close()

    # Calculate total elapsed time
    total_elapsed = time.time() - upload_start_time
    total_mins = int(total_elapsed // 60)
    total_secs = int(total_elapsed % 60)
    elapsed_display = f"{total_mins}m {total_secs}s" if total_mins > 0 else f"{total_secs}s"

    # Summary
    console.print()
    console.print("=" * 65)
    console.print(f"[dim][{ts()}][/dim] [bold]>>> Upload Complete[/bold]")
    console.print("=" * 65)
    console.print(f"[dim][{ts()}][/dim]   Total: [bold]{len(pending_reels)}[/bold] reels | Elapsed: [bold]{elapsed_display}[/bold]")
    console.print(f"[dim][{ts()}][/dim]   Success: [green]{success_count}[/green]")
    if failed_count:
        console.print(f"[dim][{ts()}][/dim]   Failed: [red]{failed_count}[/red]")

    # Log session summary
    log_json("session_complete", "all",
        total_reels=len(pending_reels),
        success=success_count,
        failed=failed_count,
        elapsed_s=int(total_elapsed)
    )

    if failed_count > 0:
        raise typer.Exit(1)


def _find_tiktok_pending_reels(
    profile_path: Path,
    reel_id: Optional[str] = None,
    posted_only: bool = False,
) -> list[Path]:
    """Find reels pending TikTok upload.

    Args:
        profile_path: Path to the profile directory.
        reel_id: Optional reel ID to filter by.
        posted_only: If True, only look in posted/ folder (for cross-posting).

    Returns:
        List of reel directories pending TikTok upload.
    """
    import json

    pending = []

    # When posted_only is True, only look in "posted" folder (for cross-posting)
    statuses = ["posted"] if posted_only else ["generated", "pending-post", "posted"]

    for status in statuses:
        base_dir = profile_path / "reels"
        if not base_dir.exists():
            continue

        for year_dir in sorted(base_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                status_dir = month_dir / status
                if not status_dir.exists():
                    continue
                for reel_dir in sorted(status_dir.iterdir()):
                    if not reel_dir.is_dir():
                        continue
                    if not (reel_dir / "final.mp4").exists():
                        continue

                    # Filter by reel_id if specified
                    if reel_id and reel_id not in reel_dir.name:
                        continue

                    # Check if already uploaded to TikTok
                    metadata_path = reel_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, encoding="utf-8") as f:
                                metadata = json.load(f)
                            tiktok_status = metadata.get("platform_status", {}).get("tiktok", {})
                            if tiktok_status.get("uploaded"):
                                continue  # Already uploaded
                        except Exception:
                            pass

                    pending.append(reel_dir)

    return pending


def _update_tiktok_status(
    reel_path: Path,
    success: bool,
    error: Optional[str] = None,
    video_url: Optional[str] = None,
    video_id: Optional[str] = None,
) -> None:
    """Update TikTok upload status in metadata."""
    from .service import update_platform_status

    metadata_path = reel_path / "metadata.json"
    update_platform_status(
        metadata_path=metadata_path,
        platform="tiktok",
        success=success,
        media_id=video_id,
        permalink=video_url,
        error=error,
    )


def _move_reel_to_posted(reel_path: Path, profile_path: Path) -> Optional[Path]:
    """Move reel to posted folder."""
    import shutil

    if "posted" in str(reel_path):
        return None

    parts = list(reel_path.parts)
    for i, part in enumerate(parts):
        if part in ["generated", "pending-post"]:
            parts[i] = "posted"
            break

    new_path = Path(*parts)
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if reel_path.exists() and not new_path.exists():
        shutil.move(str(reel_path), str(new_path))
        return new_path

    return None


def upload_reel(
    profile: str = typer.Argument(..., help="Profile name"),
    reel_id: Optional[str] = typer.Argument(None, help="Reel ID to upload (uploads only this one if specified)"),
    # Platform flags
    instagram: bool = typer.Option(False, "--instagram", "-I", help="Upload to Instagram"),
    tiktok: bool = typer.Option(False, "--tiktok", "-T", help="Upload to TikTok"),
    all_platforms: bool = typer.Option(False, "--all", "-a", help="Upload to all enabled platforms"),
    # Other options
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate upload"),
    one: bool = typer.Option(False, "--one", "-1", help="Upload only the oldest pending"),
) -> None:
    """Upload reel(s) to social platforms.

    By default, uploads to Instagram only (backwards compatible).
    Use --tiktok to upload to TikTok, --all for all enabled platforms.
    Use --one to upload only the oldest pending reel.
    """
    # Build immutable params
    params = ReelUploadParams.from_cli(
        profile=profile,
        reel_id=reel_id,
        instagram=instagram,
        tiktok=tiktok,
        all_platforms=all_platforms,
        one=one,
        dry_run=dry_run,
    )

    # Validate params
    validation = validate_reel_upload_params(params)
    if isinstance(validation, Failure):
        show_reel_error(console, validation.error, validation.details)
        raise typer.Exit(1)

    # Display configuration
    show_upload_config(console, params)

    # Execute upload
    service = ReelUploaderService()

    if params.reel_id:
        result = asyncio.run(service.upload_single(params))
    else:
        result = asyncio.run(service.upload_all(params))

    if isinstance(result, Failure):
        show_reel_error(console, result.error, result.details)
        raise typer.Exit(1)

    # Count results
    results = result.value if isinstance(result.value, list) else [result.value]
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count

    show_upload_result(console, success_count, failed_count, results)

    if failed_count > 0:
        raise typer.Exit(1)
