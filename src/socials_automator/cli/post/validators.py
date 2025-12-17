"""Post-specific validators."""

from __future__ import annotations

from typing import List

from ..core.types import Result, Success, Failure
from ..core.validators import validate_profile, validate_api_key
from .params import PostGenerationParams, PostUploadParams

# Valid text AI providers
VALID_TEXT_AI_PROVIDERS: List[str] = [
    "zai",
    "groq",
    "gemini",
    "openai",
    "lmstudio",
    "ollama",
]

# Valid image AI providers
VALID_IMAGE_AI_PROVIDERS: List[str] = [
    "dalle",
    "fal_flux",
    "comfy",
]


def validate_post_generation_params(params: PostGenerationParams) -> Result[PostGenerationParams]:
    """Validate all post generation parameters.

    Returns Result with params if valid, or Failure with error.
    """
    # Validate profile exists
    profile_result = validate_profile(params.profile)
    if isinstance(profile_result, Failure):
        return profile_result

    # Validate slide count
    if params.slides is not None:
        if params.slides < 2 or params.slides > 10:
            return Failure(
                f"Invalid slide count: {params.slides}",
                {"hint": "Slide count must be between 2 and 10"},
            )

    # Validate min/max slides
    if params.min_slides < 2:
        return Failure(
            f"Invalid min_slides: {params.min_slides}",
            {"hint": "Minimum slides must be at least 2"},
        )

    if params.max_slides > 10:
        return Failure(
            f"Invalid max_slides: {params.max_slides}",
            {"hint": "Maximum slides cannot exceed 10"},
        )

    if params.min_slides > params.max_slides:
        return Failure(
            f"min_slides ({params.min_slides}) > max_slides ({params.max_slides})",
            {"hint": "min_slides must be <= max_slides"},
        )

    # Validate text AI provider if specified
    if params.text_ai and params.text_ai not in VALID_TEXT_AI_PROVIDERS:
        return Failure(
            f"Invalid text AI provider: {params.text_ai}",
            {"valid_providers": VALID_TEXT_AI_PROVIDERS},
        )

    # Validate image AI provider if specified
    if params.image_ai and params.image_ai not in VALID_IMAGE_AI_PROVIDERS:
        return Failure(
            f"Invalid image AI provider: {params.image_ai}",
            {"valid_providers": VALID_IMAGE_AI_PROVIDERS},
        )

    # Validate count
    if params.count < 1:
        return Failure(
            f"Invalid count: {params.count}",
            {"hint": "Count must be at least 1"},
        )

    return Success(params)


def validate_post_upload_params(params: PostUploadParams) -> Result[PostUploadParams]:
    """Validate all post upload parameters.

    Returns Result with params if valid, or Failure with error.
    """
    # Validate profile exists
    profile_result = validate_profile(params.profile)
    if isinstance(profile_result, Failure):
        return profile_result

    # Skip credential validation for dry run
    if params.dry_run:
        return Success(params)

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
