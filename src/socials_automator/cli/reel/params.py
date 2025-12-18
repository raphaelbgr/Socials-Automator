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
    gpu_accelerate: bool
    gpu_index: Optional[int]
    # News-specific parameters
    is_news_profile: bool = False
    news_edition: Optional[str] = None  # morning, midday, evening, night
    news_story_count: int = 4
    news_max_age_hours: int = 24

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
        gpu_accelerate: bool = False,
        gpu: Optional[int] = None,
        # News-specific options
        news: bool = False,
        edition: Optional[str] = None,
        story_count: int = 4,
        news_max_age: int = 24,
        **kwargs,  # Ignore extra kwargs
    ) -> "ReelGenerationParams":
        """Create from CLI arguments with parsing and defaults.

        This is a factory method that handles all parsing logic.
        """
        from ..core.parsers import parse_length, parse_voice_preset
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
            loop=loop or loop_count is not None,
            loop_count=loop_count,
            gpu_accelerate=gpu_accelerate,
            gpu_index=gpu,
            # News params
            is_news_profile=is_news,
            news_edition=edition,
            news_story_count=story_count,
            news_max_age_hours=news_max_age,
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
