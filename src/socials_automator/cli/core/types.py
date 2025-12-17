"""Core types for CLI - immutable data structures and Result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic, Union, Any

T = TypeVar("T")


@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result containing a value."""

    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False


@dataclass(frozen=True)
class Failure:
    """Failed result containing an error message."""

    error: str
    details: dict[str, Any] | None = None

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True


# Result type - either Success[T] or Failure
Result = Union[Success[T], Failure]


@dataclass(frozen=True)
class ProfileConfig:
    """Immutable profile configuration."""

    name: str
    path: Path
    handle: str
    niche: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_path(cls, profile_path: Path) -> "ProfileConfig":
        """Load profile config from path."""
        import json

        metadata_file = profile_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Profile metadata not found: {metadata_file}")

        with open(metadata_file, encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(
            name=metadata.get("name", profile_path.name),
            path=profile_path,
            handle=metadata.get("instagram_handle", ""),
            niche=metadata.get("niche", ""),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GenerationResult:
    """Result of content generation."""

    success: bool
    output_path: Path | None = None
    duration_seconds: float | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class UploadResult:
    """Result of Instagram upload."""

    success: bool
    post_id: str | None = None
    permalink: str | None = None
    error: str | None = None
    was_ghost_published: bool = False
