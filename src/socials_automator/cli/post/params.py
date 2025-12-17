"""Immutable parameter dataclasses for post commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PostGenerationParams:
    """Immutable parameters for carousel post generation."""

    profile: str
    profile_path: Path
    topic: Optional[str]
    pillar: Optional[str]
    count: int
    slides: Optional[int]
    min_slides: int
    max_slides: int
    upload_after: bool
    auto_retry: bool
    text_ai: Optional[str]
    image_ai: Optional[str]
    loop_seconds: Optional[int]
    ai_tools: bool

    @classmethod
    def from_cli(
        cls,
        profile: str,
        topic: Optional[str] = None,
        pillar: Optional[str] = None,
        count: int = 1,
        slides: Optional[int] = None,
        min_slides: int = 3,
        max_slides: int = 10,
        upload_after: bool = False,
        auto_retry: bool = False,
        text_ai: Optional[str] = None,
        image_ai: Optional[str] = None,
        loop_each: Optional[str] = None,
        ai_tools: bool = False,
        **kwargs,
    ) -> "PostGenerationParams":
        """Create from CLI arguments with parsing and defaults."""
        from ..core.parsers import parse_interval
        from ..core.paths import get_profile_path

        # Parse loop interval if provided
        loop_seconds = None
        if loop_each:
            loop_seconds = parse_interval(loop_each)

        return cls(
            profile=profile,
            profile_path=get_profile_path(profile),
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
            loop_seconds=loop_seconds,
            ai_tools=ai_tools,
        )


@dataclass(frozen=True)
class PostUploadParams:
    """Immutable parameters for carousel post upload."""

    profile: str
    profile_path: Path
    post_id: Optional[str]
    post_one: bool
    dry_run: bool

    @classmethod
    def from_cli(
        cls,
        profile: str,
        post_id: Optional[str] = None,
        one: bool = False,
        dry_run: bool = False,
        **kwargs,
    ) -> "PostUploadParams":
        """Create from CLI arguments."""
        from ..core.paths import get_profile_path

        return cls(
            profile=profile,
            profile_path=get_profile_path(profile),
            post_id=post_id,
            post_one=one,
            dry_run=dry_run,
        )
