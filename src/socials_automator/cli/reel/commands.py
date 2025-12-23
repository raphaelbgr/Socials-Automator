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
