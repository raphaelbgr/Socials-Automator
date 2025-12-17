"""Pure validation functions for CLI arguments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .types import Result, Success, Failure


def validate_profile(profile: str, profiles_dir: Path | None = None) -> Result[Path]:
    """Validate that a profile exists and has required files.

    Pure function - only reads filesystem, no side effects.

    Args:
        profile: Profile name
        profiles_dir: Optional profiles directory (defaults to cwd/profiles)

    Returns:
        Result containing profile path or failure
    """
    if profiles_dir is None:
        profiles_dir = Path.cwd() / "profiles"

    profile_path = profiles_dir / profile

    if not profile_path.exists():
        return Failure(
            f"Profile not found: {profile}",
            {"path": str(profile_path), "available": _list_profiles(profiles_dir)},
        )

    metadata_file = profile_path / "metadata.json"
    if not metadata_file.exists():
        return Failure(
            f"Profile missing metadata.json: {profile}",
            {"path": str(profile_path)},
        )

    return Success(profile_path)


def validate_voice(voice: str, valid_voices: List[str]) -> Result[str]:
    """Validate voice choice against allowed voices.

    Pure function - no side effects.

    Args:
        voice: Voice name to validate
        valid_voices: List of valid voice names

    Returns:
        Result containing voice name or failure
    """
    if voice not in valid_voices:
        return Failure(
            f"Invalid voice: {voice}",
            {"valid_voices": valid_voices},
        )
    return Success(voice)


def validate_length(length: float, min_length: float = 15.0, max_length: float = 180.0) -> Result[float]:
    """Validate video length is within bounds.

    Pure function - no side effects.

    Args:
        length: Length in seconds
        min_length: Minimum allowed length (default 15s)
        max_length: Maximum allowed length (default 180s/3min)

    Returns:
        Result containing length or failure
    """
    if length < min_length:
        return Failure(
            f"Length too short: {length}s (minimum {min_length}s)",
            {"min": min_length, "max": max_length},
        )
    if length > max_length:
        return Failure(
            f"Length too long: {length}s (maximum {max_length}s)",
            {"min": min_length, "max": max_length},
        )
    return Success(length)


def validate_api_key(key_name: str, required: bool = True) -> Result[str | None]:
    """Validate that an API key environment variable is set.

    Pure function - only reads environment, no side effects.

    Args:
        key_name: Environment variable name
        required: Whether the key is required

    Returns:
        Result containing key value or failure
    """
    value = os.environ.get(key_name)

    if value:
        return Success(value)

    if required:
        return Failure(
            f"Missing required API key: {key_name}",
            {"env_var": key_name},
        )

    return Success(None)


def validate_video_matcher(matcher: str, valid_matchers: List[str]) -> Result[str]:
    """Validate video matcher choice.

    Pure function - no side effects.

    Args:
        matcher: Video matcher name
        valid_matchers: List of valid matcher names

    Returns:
        Result containing matcher name or failure
    """
    if matcher not in valid_matchers:
        return Failure(
            f"Invalid video matcher: {matcher}",
            {"valid_matchers": valid_matchers},
        )
    return Success(matcher)


def validate_text_ai(provider: str | None, valid_providers: List[str]) -> Result[str | None]:
    """Validate text AI provider choice.

    Pure function - no side effects.

    Args:
        provider: Provider name (or None for default)
        valid_providers: List of valid provider names

    Returns:
        Result containing provider name or failure
    """
    if provider is None:
        return Success(None)

    if provider not in valid_providers:
        return Failure(
            f"Invalid text AI provider: {provider}",
            {"valid_providers": valid_providers},
        )
    return Success(provider)


def validate_image_ai(provider: str | None, valid_providers: List[str]) -> Result[str | None]:
    """Validate image AI provider choice.

    Pure function - no side effects.

    Args:
        provider: Provider name (or None for default)
        valid_providers: List of valid provider names

    Returns:
        Result containing provider name or failure
    """
    if provider is None:
        return Success(None)

    if provider not in valid_providers:
        return Failure(
            f"Invalid image AI provider: {provider}",
            {"valid_providers": valid_providers},
        )
    return Success(provider)


def _list_profiles(profiles_dir: Path) -> List[str]:
    """List available profile names.

    Helper function for error messages.
    """
    if not profiles_dir.exists():
        return []

    return [
        d.name
        for d in profiles_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]
