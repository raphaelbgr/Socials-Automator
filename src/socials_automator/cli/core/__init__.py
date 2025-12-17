"""Core utilities for CLI - pure functions and shared types."""

from .types import Result, Success, Failure, ProfileConfig
from .parsers import parse_interval, parse_length, parse_voice_preset
from .validators import validate_profile, validate_voice, validate_length, validate_api_key
from .paths import get_profiles_dir, get_profile_path, get_output_dir
from .console import console

__all__ = [
    # Types
    "Result",
    "Success",
    "Failure",
    "ProfileConfig",
    # Parsers
    "parse_interval",
    "parse_length",
    "parse_voice_preset",
    # Validators
    "validate_profile",
    "validate_voice",
    "validate_length",
    "validate_api_key",
    # Paths
    "get_profiles_dir",
    "get_profile_path",
    "get_output_dir",
    # Console
    "console",
]
