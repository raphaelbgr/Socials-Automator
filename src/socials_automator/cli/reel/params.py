"""Immutable parameter dataclasses for reel commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ReelGenerationParams:
    """Immutable parameters for reel generation."""

    profile: str
    profile_path: Path
    topic: Optional[str]
    text_ai: Optional[str]
    video_matcher: str
    voice: str
    voice_rate: str
    voice_pitch: str
    subtitle_size: int
    font: str
    target_duration: float
    output_dir: Optional[Path]
    dry_run: bool
    upload_after: bool
    loop: bool
    loop_count: Optional[int]
    loop_each: Optional[int]  # Interval in seconds between loops
    gpu_accelerate: bool
    gpu_index: Optional[int]
    # Hashtag limit (Instagram max is 5 as of Dec 2025)
    max_hashtags: int = 5
    # News-specific parameters
    is_news_profile: bool = False
    news_edition: Optional[str] = None  # morning, midday, evening, night
    news_story_count: Optional[int] = None  # None = auto (AI decides)
    news_max_age_hours: int = 24
    # Image overlay feature
    overlay_images: bool = False
    image_provider: str = "websearch"  # websearch, pexels, pixabay
    use_tor: bool = False  # Route websearch through Tor
    blur: Optional[str] = None  # None=disabled, light/medium/heavy
    smart_pick: bool = False  # Use AI vision to select best image
    smart_pick_count: int = 10  # Number of candidates to compare

    @classmethod
    def from_cli(
        cls,
        profile: str,
        topic: Optional[str] = None,
        text_ai: Optional[str] = None,
        video_matcher: str = "pexels",
        voice: str = "rvc_adam",
        voice_rate: str = "+0%",
        voice_pitch: str = "+0Hz",
        subtitle_size: int = 80,
        font: str = "Montserrat-Bold.ttf",
        length: str = "1m",
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        upload: bool = False,
        loop: bool = False,
        loop_count: Optional[int] = None,
        loop_each: Optional[str] = None,
        gpu_accelerate: bool = False,
        gpu: Optional[int] = None,
        # Hashtag limit
        hashtags: int = 5,
        # News-specific options
        news: bool = False,
        edition: Optional[str] = None,
        story_count: Optional[str] = "auto",  # "auto" or a number
        news_max_age: int = 24,
        # Image overlay feature
        overlay_images: bool = False,
        image_provider: str = "websearch",  # websearch, pexels, pixabay
        use_tor: bool = False,  # Route websearch through Tor
        blur: Optional[str] = None,  # None=disabled, light/medium/heavy
        smart_pick: bool = False,  # Use AI vision to select best image
        smart_pick_count: int = 10,  # Number of candidates to compare
        **kwargs,  # Ignore extra kwargs
    ) -> "ReelGenerationParams":
        """Create from CLI arguments with parsing and defaults.

        This is a factory method that handles all parsing logic.
        """
        from ..core.parsers import parse_interval, parse_length, parse_voice_preset
        from ..core.paths import get_profile_path

        profile_path = get_profile_path(profile)

        # Parse length string to seconds
        target_duration = parse_length(length)

        # Resolve voice preset (e.g., 'adam_excited' -> ('rvc_adam', '+12%', '+3Hz'))
        resolved_voice, resolved_rate, resolved_pitch = parse_voice_preset(
            voice, voice_rate, voice_pitch
        )

        # Detect if this is a news profile (auto-detect or explicit)
        is_news = news or _is_news_profile(profile_path)

        # Parse loop interval if provided
        loop_each_seconds = parse_interval(loop_each) if loop_each else None

        # Parse story count ("auto" -> None, number string -> int)
        parsed_story_count: Optional[int] = None
        if story_count and story_count.lower() != "auto":
            try:
                parsed_story_count = int(story_count)
            except ValueError:
                parsed_story_count = None  # Invalid value -> auto

        # Validate and clamp hashtag limit (1-30, default 5)
        from socials_automator.hashtag import INSTAGRAM_MAX_HASHTAGS
        max_hashtags = max(1, min(30, hashtags)) if hashtags else INSTAGRAM_MAX_HASHTAGS

        # Normalize blur value: empty string or invalid -> "medium"
        blur_normalized: Optional[str] = None
        if blur is not None:
            blur_lower = blur.lower().strip() if blur else "medium"
            if blur_lower in ("light", "medium", "heavy"):
                blur_normalized = blur_lower
            else:
                blur_normalized = "medium"  # Default for invalid/empty values

        return cls(
            profile=profile,
            profile_path=profile_path,
            topic=topic,
            text_ai=text_ai,
            video_matcher=video_matcher,
            voice=resolved_voice,
            voice_rate=resolved_rate,
            voice_pitch=resolved_pitch,
            subtitle_size=subtitle_size,
            font=font,
            target_duration=target_duration,
            output_dir=Path(output_dir) if output_dir else None,
            dry_run=dry_run,
            upload_after=upload,
            loop=loop or loop_count is not None or loop_each_seconds is not None,
            loop_count=loop_count,
            loop_each=loop_each_seconds,
            gpu_accelerate=gpu_accelerate,
            gpu_index=gpu,
            max_hashtags=max_hashtags,
            # News params
            is_news_profile=is_news,
            news_edition=edition,
            news_story_count=parsed_story_count,
            news_max_age_hours=news_max_age,
            # Image overlay
            overlay_images=overlay_images,
            image_provider=image_provider,
            use_tor=use_tor,
            blur=blur_normalized,
            smart_pick=smart_pick,
            smart_pick_count=smart_pick_count,
        )


def _is_news_profile(profile_path: Path) -> bool:
    """Check if a profile is configured for news content.

    Looks for 'news_sources' key in metadata.json.
    """
    import json

    metadata_path = profile_path / "metadata.json"
    if not metadata_path.exists():
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "news_sources" in data
    except Exception:
        return False


@dataclass(frozen=True)
class ReelUploadParams:
    """Immutable parameters for reel upload."""

    profile: str
    profile_path: Path
    reel_id: Optional[str]
    platforms: tuple[str, ...]  # Immutable tuple of platform names
    post_one: bool
    dry_run: bool

    @classmethod
    def from_cli(
        cls,
        profile: str,
        reel_id: Optional[str] = None,
        instagram: bool = False,
        tiktok: bool = False,
        all_platforms: bool = False,
        one: bool = False,
        dry_run: bool = False,
        **kwargs,
    ) -> "ReelUploadParams":
        """Create from CLI arguments.

        Platform selection logic:
        - No flags: Instagram only (default, backwards compatible)
        - --instagram: Instagram only
        - --tiktok: TikTok only
        - --instagram --tiktok: Both platforms
        - --all: All enabled platforms from profile config
        """
        from ..core.paths import get_profile_path

        profile_path = get_profile_path(profile)

        # Resolve platforms
        platforms = resolve_platforms(
            profile_path=profile_path,
            instagram=instagram,
            tiktok=tiktok,
            all_platforms=all_platforms,
        )

        return cls(
            profile=profile,
            profile_path=profile_path,
            reel_id=reel_id,
            platforms=tuple(platforms),
            post_one=one,
            dry_run=dry_run,
        )


def resolve_platforms(
    profile_path: Path,
    instagram: bool = False,
    tiktok: bool = False,
    all_platforms: bool = False,
) -> list[str]:
    """Resolve which platforms to upload to based on CLI flags.

    Args:
        profile_path: Path to the profile directory.
        instagram: --instagram flag.
        tiktok: --tiktok flag.
        all_platforms: --all flag.

    Returns:
        List of platform names to upload to.
    """
    if all_platforms:
        # Get all enabled platforms from profile
        from socials_automator.platforms import PlatformRegistry
        return PlatformRegistry.get_enabled_platforms(profile_path)

    if instagram or tiktok:
        # Explicit platform selection
        platforms = []
        if instagram:
            platforms.append("instagram")
        if tiktok:
            platforms.append("tiktok")
        return platforms

    # Default: Instagram only (backwards compatible)
    return ["instagram"]
