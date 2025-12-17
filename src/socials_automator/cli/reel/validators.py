"""Reel-specific validators."""

from __future__ import annotations

from typing import List

from ..core.types import Result, Success, Failure
from ..core.validators import validate_profile, validate_length, validate_api_key
from .params import ReelGenerationParams, ReelUploadParams

# Valid voice choices
VALID_VOICES: List[str] = [
    "rvc_adam",
    "rvc_adam_excited",
    "adam_excited",
    "adam",
    "tiktok-adam",
    "professional_female",
    "professional_male",
    "friendly_female",
    "friendly_male",
    "energetic",
    "british_female",
    "british_male",
]

# Valid video matchers
VALID_VIDEO_MATCHERS: List[str] = ["pexels"]

# Valid text AI providers
VALID_TEXT_AI_PROVIDERS: List[str] = [
    "zai",
    "groq",
    "gemini",
    "openai",
    "lmstudio",
    "ollama",
]


def validate_reel_generation_params(params: ReelGenerationParams) -> Result[ReelGenerationParams]:
    """Validate all reel generation parameters.

    Returns Result with params if valid, or Failure with error.
    """
    # Validate profile exists
    profile_result = validate_profile(params.profile)
    if isinstance(profile_result, Failure):
        return profile_result

    # Validate length
    length_result = validate_length(params.target_duration)
    if isinstance(length_result, Failure):
        return length_result

    # Validate voice (after preset resolution, so check base voice)
    base_voice = params.voice
    # Map resolved voices back for validation
    if base_voice not in ["rvc_adam", "professional_female", "professional_male",
                          "friendly_female", "friendly_male", "energetic",
                          "british_female", "british_male"]:
        return Failure(
            f"Invalid voice: {base_voice}",
            {"valid_voices": VALID_VOICES},
        )

    # Validate video matcher
    if params.video_matcher not in VALID_VIDEO_MATCHERS:
        return Failure(
            f"Invalid video matcher: {params.video_matcher}",
            {"valid_matchers": VALID_VIDEO_MATCHERS},
        )

    # Validate text AI provider if specified
    if params.text_ai and params.text_ai not in VALID_TEXT_AI_PROVIDERS:
        return Failure(
            f"Invalid text AI provider: {params.text_ai}",
            {"valid_providers": VALID_TEXT_AI_PROVIDERS},
        )

    # Validate Pexels API key if using pexels matcher
    if params.video_matcher == "pexels":
        pexels_result = validate_api_key("PEXELS_API_KEY", required=True)
        if isinstance(pexels_result, Failure):
            return Failure(
                "PEXELS_API_KEY not found in environment",
                {"hint": "Get free API key at: https://www.pexels.com/api/"},
            )

    return Success(params)


def validate_reel_upload_params(params: ReelUploadParams) -> Result[ReelUploadParams]:
    """Validate all reel upload parameters.

    Returns Result with params if valid, or Failure with error.
    """
    # Validate profile exists
    profile_result = validate_profile(params.profile)
    if isinstance(profile_result, Failure):
        return profile_result

    # Validate Instagram credentials
    instagram_result = validate_api_key("INSTAGRAM_ACCESS_TOKEN", required=True)
    if isinstance(instagram_result, Failure):
        return Failure(
            "Instagram credentials not configured",
            {"hint": "See README for Instagram API setup"},
        )

    # Validate Cloudinary credentials
    cloudinary_result = validate_api_key("CLOUDINARY_CLOUD_NAME", required=True)
    if isinstance(cloudinary_result, Failure):
        return Failure(
            "Cloudinary credentials not configured",
            {"hint": "See README for Cloudinary setup"},
        )

    return Success(params)
