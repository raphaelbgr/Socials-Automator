"""Caption Service - Reusable caption and hashtag generation.

This service provides caption and hashtag generation that can be used by:
- Pipeline's CaptionGenerator step
- Artifact regeneration in upload flow
- Any other component needing caption generation

Uses LLMFallbackManager for automatic retry and provider switching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .llm_fallback import LLMFallbackManager, FallbackConfig, FallbackResult
from ..providers.config import load_provider_config

_logger = logging.getLogger("caption_service")


def sanitize_json_string(json_str: str) -> str:
    """Sanitize JSON string by escaping literal newlines inside quoted strings.

    LMStudio/local LLMs sometimes output literal newlines inside JSON string values,
    which is invalid JSON. This function fixes that by escaping them.

    Args:
        json_str: Raw JSON string that may have literal newlines in values.

    Returns:
        Sanitized JSON string with escaped newlines inside quoted values.
    """
    # If it doesn't look like JSON, return as-is
    if not json_str.strip().startswith("{"):
        return json_str

    # Replace literal newlines inside strings with escaped \n
    # This regex matches content between double quotes and escapes any literal newlines
    def escape_newlines_in_string(match: re.Match) -> str:
        content = match.group(1)
        # Escape literal newlines (but not already-escaped ones)
        # Replace literal newline with \n escape sequence
        content = content.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        return f'"{content}"'

    # Pattern matches strings: "..." but not escaped quotes
    # This is a simplified approach that works for most LLM outputs
    result = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines_in_string, json_str)
    return result


@dataclass
class CaptionResult:
    """Result of caption generation."""

    success: bool
    caption: str = ""
    error: Optional[str] = None
    provider_used: Optional[str] = None
    attempts: int = 0


@dataclass
class HashtagResult:
    """Result of hashtag generation."""

    success: bool
    hashtags: list[str] = None
    hashtag_string: str = ""
    error: Optional[str] = None
    provider_used: Optional[str] = None
    attempts: int = 0

    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []


class CaptionService:
    """Service for generating captions and hashtags.

    Encapsulates all caption generation logic in one place.
    Uses LLMFallbackManager for reliable generation with retries.

    Example:
        service = CaptionService()
        caption_result = await service.generate_caption(
            topic="5 AI Tools for Productivity",
            narration="Here are 5 AI tools...",
            profile_handle="@ai.for.mortals"
        )
        if caption_result.success:
            print(caption_result.caption)
    """

    def __init__(
        self,
        fallback_manager: Optional[LLMFallbackManager] = None,
        preferred_provider: Optional[str] = None,
    ):
        """Initialize caption service.

        Args:
            fallback_manager: Optional pre-configured LLMFallbackManager.
            preferred_provider: Preferred LLM provider (e.g., 'lmstudio').
        """
        self._fallback_manager = fallback_manager
        self._preferred_provider = preferred_provider

    def _get_fallback_manager(self) -> LLMFallbackManager:
        """Get or create the LLMFallbackManager."""
        if self._fallback_manager:
            return self._fallback_manager

        provider_config = load_provider_config()
        config = FallbackConfig.from_provider_config(provider_config)

        self._fallback_manager = LLMFallbackManager(
            preferred_provider=self._preferred_provider,
            config=config,
            provider_config=provider_config,
        )
        return self._fallback_manager

    async def generate_caption(
        self,
        topic: str,
        narration: str = "",
        profile_handle: str = "",
        profile_hashtag: str = "",
    ) -> CaptionResult:
        """Generate an Instagram caption for a reel.

        Args:
            topic: The video topic/title.
            narration: Optional narration text for context.
            profile_handle: Instagram handle (e.g., "@ai.for.mortals").
            profile_hashtag: Profile hashtag (e.g., "#AiForMortals").

        Returns:
            CaptionResult with generated caption or error.
        """
        if not topic and not narration:
            return CaptionResult(
                success=False,
                error="Need at least topic or narration to generate caption",
            )

        prompt = self._build_caption_prompt(topic, narration, profile_handle, profile_hashtag)

        try:
            manager = self._get_fallback_manager()
            result = await manager.generate(prompt, task="caption")

            if not result.success:
                return CaptionResult(
                    success=False,
                    error=result.error or "Caption generation failed",
                    attempts=result.total_attempts,
                )

            # Parse the response
            caption = self._parse_caption_response(result.result)

            # Add profile handle CTA if not present
            if profile_handle and profile_handle not in caption:
                caption = caption.rstrip()
                if not caption.endswith("!"):
                    caption += "."
                caption += f"\n\nFollow {profile_handle} for more!"

            # Add profile hashtag if not present
            if profile_hashtag and profile_hashtag not in caption:
                caption += f" {profile_hashtag}"

            return CaptionResult(
                success=True,
                caption=caption.strip(),
                provider_used=result.provider_used,
                attempts=result.total_attempts,
            )

        except Exception as e:
            _logger.error(f"Caption generation error: {e}")
            return CaptionResult(
                success=False,
                error=str(e),
            )

    async def generate_hashtags(
        self,
        topic: str,
        caption: str = "",
        count: int = 5,  # Instagram limit as of Dec 2025
    ) -> HashtagResult:
        """Generate relevant hashtags for a reel.

        Args:
            topic: The video topic.
            caption: Optional caption for more context.
            count: Number of hashtags to generate (default 12).

        Returns:
            HashtagResult with generated hashtags or error.
        """
        if not topic:
            return HashtagResult(
                success=False,
                error="Topic required for hashtag generation",
            )

        prompt = self._build_hashtag_prompt(topic, caption, count)

        try:
            manager = self._get_fallback_manager()
            result = await manager.generate(prompt, task="hashtags")

            if not result.success:
                return HashtagResult(
                    success=False,
                    error=result.error or "Hashtag generation failed",
                    attempts=result.total_attempts,
                )

            # Parse hashtags from response
            hashtags = self._parse_hashtag_response(result.result)

            # Format as string
            hashtag_string = " ".join(f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags)

            return HashtagResult(
                success=True,
                hashtags=hashtags,
                hashtag_string=hashtag_string,
                provider_used=result.provider_used,
                attempts=result.total_attempts,
            )

        except Exception as e:
            _logger.error(f"Hashtag generation error: {e}")
            return HashtagResult(
                success=False,
                error=str(e),
            )

    async def generate_caption_with_hashtags(
        self,
        topic: str,
        narration: str = "",
        profile_handle: str = "",
        profile_hashtag: str = "",
        hashtag_count: int = 5,  # Instagram limit as of Dec 2025
    ) -> tuple[CaptionResult, HashtagResult]:
        """Generate both caption and hashtags.

        Convenience method that generates both in sequence.

        Args:
            topic: The video topic.
            narration: Optional narration text.
            profile_handle: Instagram handle.
            profile_hashtag: Profile hashtag.
            hashtag_count: Number of hashtags to generate.

        Returns:
            Tuple of (CaptionResult, HashtagResult).
        """
        caption_result = await self.generate_caption(
            topic=topic,
            narration=narration,
            profile_handle=profile_handle,
            profile_hashtag=profile_hashtag,
        )

        if not caption_result.success:
            return caption_result, HashtagResult(success=False, error="Caption failed, skipping hashtags")

        hashtag_result = await self.generate_hashtags(
            topic=topic,
            caption=caption_result.caption,
            count=hashtag_count,
        )

        return caption_result, hashtag_result

    def _build_caption_prompt(
        self,
        topic: str,
        narration: str,
        profile_handle: str,
        profile_hashtag: str,
    ) -> str:
        """Build the caption generation prompt."""
        context = f"Topic: {topic}"
        if narration:
            # Truncate narration if too long
            narration_preview = narration[:800] if len(narration) > 800 else narration
            context += f"\n\nVideo narration/script:\n{narration_preview}"

        cta = f"Follow {profile_handle} for more!" if profile_handle else "Follow for more!"
        hashtag_note = f"Include {profile_hashtag} in the caption." if profile_hashtag else ""

        return f"""Generate a short, engaging Instagram Reels caption.

{context}

Requirements:
- 1-3 sentences maximum (under 150 words)
- Engaging hook that makes people want to watch
- Use simple, conversational language
- Include bullet points if listing multiple items (use - not *)
- NO hashtags (they will be added separately)
- NO emojis unless absolutely necessary
- End with call-to-action: "{cta}"
{hashtag_note}

Return ONLY the caption text, nothing else. No JSON, no markdown formatting."""

    def _build_hashtag_prompt(self, topic: str, caption: str, count: int) -> str:
        """Build the hashtag generation prompt."""
        context = f"Topic: {topic}"
        if caption:
            context += f"\nCaption: {caption[:200]}"

        return f"""Generate {count} relevant Instagram hashtags for this video.

{context}

Requirements:
- Mix of popular hashtags (100K+ posts) and niche ones
- Relevant to the topic
- NO # symbol, just the words
- One hashtag per line
- No spaces in hashtags

Return ONLY the hashtags, one per line, nothing else."""

    def _parse_caption_response(self, response: str) -> str:
        """Parse caption from AI response."""
        caption = response.strip()

        # Remove markdown code blocks if present
        if caption.startswith("```"):
            lines = caption.split("\n")
            caption = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Remove JSON wrapper if present
        if caption.startswith("{") and "caption" in caption:
            import json
            try:
                # Sanitize JSON to handle literal newlines from local LLMs
                sanitized = sanitize_json_string(caption)
                data = json.loads(sanitized)
                caption = data.get("caption", caption)
            except json.JSONDecodeError:
                # If still fails, try extracting caption value manually
                # Pattern: "caption": "..." or "caption":"..."
                import re
                match = re.search(r'"caption"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', caption, re.DOTALL)
                if match:
                    caption = match.group(1).replace('\\"', '"').replace('\\n', '\n')

        return caption.strip()

    def _parse_hashtag_response(self, response: str) -> list[str]:
        """Parse hashtags from AI response."""
        lines = response.strip().split("\n")
        hashtags = []

        for line in lines:
            # Clean up the line
            tag = line.strip().lstrip("#").lstrip("-").lstrip("*").strip()

            # Skip empty or invalid
            if not tag or len(tag) < 2:
                continue

            # Remove spaces (hashtags can't have spaces)
            tag = tag.replace(" ", "")

            # Skip if it looks like a sentence
            if len(tag) > 30:
                continue

            hashtags.append(tag)

        # Limit to Instagram max (5 as of Dec 2025)
        from socials_automator.hashtag import INSTAGRAM_MAX_HASHTAGS
        return hashtags[:INSTAGRAM_MAX_HASHTAGS]


# Convenience function for simple usage
async def generate_caption_simple(
    topic: str,
    narration: str = "",
    profile_handle: str = "",
) -> str:
    """Simple function to generate a caption.

    Args:
        topic: The video topic.
        narration: Optional narration text.
        profile_handle: Instagram handle.

    Returns:
        Generated caption string, or empty string on failure.
    """
    service = CaptionService()
    result = await service.generate_caption(
        topic=topic,
        narration=narration,
        profile_handle=profile_handle,
    )
    return result.caption if result.success else ""


async def generate_hashtags_simple(topic: str, caption: str = "") -> list[str]:
    """Simple function to generate hashtags.

    Args:
        topic: The video topic.
        caption: Optional caption for context.

    Returns:
        List of hashtag strings, or empty list on failure.
    """
    service = CaptionService()
    result = await service.generate_hashtags(topic=topic, caption=caption)
    return result.hashtags if result.success else []
