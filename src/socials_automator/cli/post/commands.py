"""Post CLI commands - thin wrappers orchestrating params, validation, display, and service."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import typer

from ..core.console import console
from ..core.types import Failure
from .display import (
    show_loop_progress,
    show_loop_stopped,
    show_post_config,
    show_post_error,
    show_post_result,
    show_upload_config,
    show_upload_result,
)
from .params import PostGenerationParams, PostUploadParams
from .service import PostGeneratorService, PostUploaderService
from .validators import validate_post_generation_params, validate_post_upload_params


def generate_post(
    profile: str = typer.Argument(..., help="Profile name to generate for"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic for the post"),
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Content pillar"),
    count: int = typer.Option(1, "--count", "-n", help="Number of posts to generate"),
    slides: Optional[int] = typer.Option(None, "--slides", "-s", help="Number of slides (default: AI decides)"),
    min_slides: int = typer.Option(3, "--min-slides", help="Minimum slides when AI decides"),
    max_slides: int = typer.Option(10, "--max-slides", help="Maximum slides when AI decides"),
    upload_after: bool = typer.Option(False, "--upload", help="Upload to Instagram after generating"),
    auto_retry: bool = typer.Option(False, "--auto-retry", help="Retry indefinitely until valid content"),
    text_ai: Optional[str] = typer.Option(None, "--text-ai", help="Text AI provider"),
    image_ai: Optional[str] = typer.Option(None, "--image-ai", help="Image AI provider"),
    loop_each: Optional[str] = typer.Option(None, "--loop-each", help="Loop interval (e.g., 5m, 1h, 30s)"),
    ai_tools: bool = typer.Option(False, "--ai-tools", help="Enable AI tool calling"),
) -> None:
    """Generate carousel posts for a profile.

    By default, the AI decides the optimal number of slides (3-10) based on
    the topic content. Use --slides to force a specific count.
    """
    # Build immutable params from CLI args
    params = PostGenerationParams.from_cli(
        profile=profile,
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
        loop_each=loop_each,
        ai_tools=ai_tools,
    )

    # Validate params
    validation = validate_post_generation_params(params)
    if isinstance(validation, Failure):
        show_post_error(console, validation.error, validation.details)
        raise typer.Exit(1)

    # Display configuration
    show_post_config(console, params)

    # Handle loop mode
    if params.loop_seconds:
        _run_loop_mode(params)
        return

    # Generate requested number of posts
    for i in range(params.count):
        post_number = i + 1 if params.count > 1 else None
        total_posts = params.count if params.count > 1 else None

        success = _run_single_generation(
            params,
            post_number=post_number,
            total_posts=total_posts,
        )

        if not success and not params.auto_retry:
            raise typer.Exit(1)


def _run_single_generation(
    params: PostGenerationParams,
    post_number: Optional[int] = None,
    total_posts: Optional[int] = None,
) -> bool:
    """Execute single post generation. Returns True on success."""
    service = PostGeneratorService()

    # Retry loop for auto_retry mode
    max_retries = 100 if params.auto_retry else 1
    retry_count = 0

    while retry_count < max_retries:
        retry_count += 1

        result = asyncio.run(service.generate(
            profile_path=params.profile_path,
            topic=params.topic,
            pillar=params.pillar,
            slides=params.slides,
            min_slides=params.min_slides,
            max_slides=params.max_slides,
            text_ai=params.text_ai,
            image_ai=params.image_ai,
            ai_tools=params.ai_tools,
            auto_retry=params.auto_retry,
        ))

        if isinstance(result, Failure):
            if params.auto_retry and retry_count < max_retries:
                console.print(f"[yellow]Retry {retry_count}/{max_retries}: {result.error}[/yellow]")
                time.sleep(2)
                continue
            show_post_error(console, result.error, result.details)
            return False

        # Success
        metadata = result.value.metadata or {}
        show_post_result(
            console,
            result.value.output_path,
            metadata.get("slide_count", 0),
            metadata.get("topic", "Unknown"),
            post_number=post_number,
            total_posts=total_posts,
        )

        # Handle upload after generation
        if params.upload_after:
            _upload_generated_post(result.value.output_path, params)

        return True

    return False


def _upload_generated_post(output_path, params: PostGenerationParams) -> None:
    """Upload a just-generated post to Instagram."""
    upload_params = PostUploadParams(
        profile=params.profile,
        profile_path=params.profile_path,
        post_id=output_path.name,
        post_one=True,
        dry_run=False,
    )

    service = PostUploaderService()
    result = asyncio.run(service.upload_single(
        profile_path=params.profile_path,
        post_id=output_path.name,
        dry_run=False,
    ))

    if isinstance(result, Failure):
        show_post_error(console, f"Upload failed: {result.error}")


def _run_loop_mode(params: PostGenerationParams) -> None:
    """Execute loop mode for continuous generation."""
    post_count = 0
    consecutive_errors = 0

    try:
        while True:
            post_count += 1

            success = _run_single_generation(params, post_number=post_count)

            if success:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                # Exponential backoff on errors (max 1 hour)
                backoff = min(params.loop_seconds * (2 ** consecutive_errors), 3600)
                console.print(f"[yellow]Error #{consecutive_errors}, waiting {backoff}s before retry[/yellow]")
                time.sleep(backoff)
                continue

            # Wait for next iteration
            show_loop_progress(console, post_count, params.loop_seconds)
            time.sleep(params.loop_seconds)

    except KeyboardInterrupt:
        show_loop_stopped(console, post_count)


def upload_post(
    profile: str = typer.Argument(..., help="Profile name"),
    post_id: Optional[str] = typer.Argument(None, help="Post ID to upload"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without uploading"),
    one: bool = typer.Option(False, "--one", "-1", help="Upload only the oldest pending"),
) -> None:
    """Upload pending carousels to Instagram.

    By default, uploads ALL pending posts in chronological order.
    Use --one to upload only the oldest pending post.
    """
    # Build immutable params
    params = PostUploadParams.from_cli(
        profile=profile,
        post_id=post_id,
        one=one,
        dry_run=dry_run,
    )

    # Validate params
    validation = validate_post_upload_params(params)
    if isinstance(validation, Failure):
        show_post_error(console, validation.error, validation.details)
        raise typer.Exit(1)

    # Display configuration
    show_upload_config(console, params)

    # Execute upload
    service = PostUploaderService()

    if params.post_id:
        result = asyncio.run(service.upload_single(
            profile_path=params.profile_path,
            post_id=params.post_id,
            dry_run=params.dry_run,
        ))
        results = [result.value] if not isinstance(result, Failure) else []
        if isinstance(result, Failure):
            show_post_error(console, result.error, result.details)
            raise typer.Exit(1)
    else:
        result = asyncio.run(service.upload_all(
            profile_path=params.profile_path,
            dry_run=params.dry_run,
            post_one=params.post_one,
        ))

        if isinstance(result, Failure):
            show_post_error(console, result.error, result.details)
            raise typer.Exit(1)

        results = result.value

    # Count results
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count

    show_upload_result(console, success_count, failed_count, results)

    if failed_count > 0:
        raise typer.Exit(1)
