"""Immutable parameter dataclasses for reel commands."""

from __future__ import annotations

from dataclasses import dataclass
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
        **kwargs,  # Ignore extra kwargs
    ) -> "ReelGenerationParams":
        """Create from CLI arguments with parsing and defaults.

        This is a factory method that handles all parsing logic.
        """
        from ..core.parsers import parse_length, parse_voice_preset
        from ..core.paths import get_profile_path

        # Parse length string to seconds
        target_duration = parse_length(length)

        # Resolve voice preset (e.g., 'adam_excited' -> ('rvc_adam', '+12%', '+3Hz'))
        resolved_voice, resolved_rate, resolved_pitch = parse_voice_preset(
            voice, voice_rate, voice_pitch
        )

        return cls(
            profile=profile,
            profile_path=get_profile_path(profile),
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
        )


@dataclass(frozen=True)
class ReelUploadParams:
    """Immutable parameters for reel upload."""

    profile: str
    profile_path: Path
    reel_id: Optional[str]
    post_one: bool
    dry_run: bool

    @classmethod
    def from_cli(
        cls,
        profile: str,
        reel_id: Optional[str] = None,
        one: bool = False,
        dry_run: bool = False,
        **kwargs,
    ) -> "ReelUploadParams":
        """Create from CLI arguments."""
        from ..core.paths import get_profile_path

        return cls(
            profile=profile,
            profile_path=get_profile_path(profile),
            reel_id=reel_id,
            post_one=one,
            dry_run=dry_run,
        )
