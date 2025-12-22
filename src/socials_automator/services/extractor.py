"""Structured data extraction service using Agno framework.

Provides reliable JSON extraction from AI responses with:
- Automatic schema enforcement via Pydantic models
- Provider fallback on failures
- Conversation history support for multi-turn extraction
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable, TypeVar

from pydantic import BaseModel

from ..providers.config import ProviderConfig, load_provider_config
from ..providers.text import TextProvider

_logger = logging.getLogger("ai_calls")

# Generic type for response models
T = TypeVar("T", bound=BaseModel)

# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


def _format_history_as_context(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """Format message history as system prompt and user prompt.

    Converts a multi-turn conversation into a single-turn prompt with context.

    Args:
        messages: List of message dicts with 'role' and 'content'.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_content = None
    context_parts = []
    last_user_prompt = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_content = content
        elif role == "assistant":
            context_parts.append(f"[Previous Response]\n{content}")
        elif role == "user":
            if context_parts:
                # This is a follow-up user message, add previous as context
                context_parts.append(f"[Previous Request]\n{last_user_prompt}")
            last_user_prompt = content

    # Build final prompt with context
    if context_parts:
        context = "\n\n".join(context_parts)
        final_prompt = f"<conversation_context>\n{context}\n</conversation_context>\n\n{last_user_prompt}"
    else:
        final_prompt = last_user_prompt

    return system_content, final_prompt


class StructuredExtractor:
    """Service for extracting structured data from AI responses.

    Uses Agno framework for reliable JSON extraction with:
    - Automatic schema enforcement via Pydantic models
    - Provider fallback on failures
    - Multi-turn conversation support via context formatting

    Usage:
        extractor = StructuredExtractor()

        # Single extraction
        plan = await extractor.extract(
            prompt="Analyze this topic...",
            response_model=PlanningResponse,
            system="You are a content strategist.",
        )

        # Multi-turn conversation
        result1, history = await extractor.extract_with_history(
            prompt="Phase 1: Plan the content",
            response_model=Phase1Response,
            system="You are a content planner.",
        )
        result2, history = await extractor.extract_with_history(
            prompt="Phase 2: Create structure",
            response_model=Phase2Response,
            history=history,
        )
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        event_callback: AIEventCallback = None,
        provider_override: str | None = None,
    ):
        """Initialize the structured extractor.

        Args:
            config: Provider configuration. Loads from default if None.
            event_callback: Optional callback for AI events.
            provider_override: Force specific provider (e.g., 'lmstudio').
        """
        self.config = config or load_provider_config()
        self._event_callback = event_callback
        self._provider_override = provider_override

        # Create underlying TextProvider
        self._text_provider = TextProvider(
            config=self.config,
            event_callback=event_callback,
            provider_override=provider_override,
        )

        # Stats tracking
        self._total_calls = 0
        self._total_cost = 0.0

    @property
    def current_provider(self) -> str | None:
        """Get the name of the last used provider."""
        return self._text_provider.current_provider

    @property
    def current_model(self) -> str | None:
        """Get the model of the last used provider."""
        return self._text_provider.current_model

    @property
    def total_calls(self) -> int:
        """Get total number of extraction calls."""
        return self._total_calls

    @property
    def total_cost(self) -> float:
        """Get total estimated cost."""
        return self._total_cost

    async def extract(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        task: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> T:
        """Extract structured data matching a Pydantic model.

        Args:
            prompt: The user prompt to send.
            response_model: Pydantic model class for the response.
            system: Optional system prompt.
            task: Optional task name for tracking.
            max_retries: Retry attempts on validation failure.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Instance of response_model with extracted data.

        Raises:
            Exception: If all providers fail or validation fails after retries.
        """
        self._total_calls += 1

        result = await self._text_provider.generate_structured(
            prompt=prompt,
            response_model=response_model,
            system=system,
            task=task,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._total_cost = self._text_provider._total_cost
        return result

    async def extract_with_history(
        self,
        prompt: str,
        response_model: type[T],
        history: list[dict[str, str]] | None = None,
        system: str | None = None,
        task: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> tuple[T, list[dict[str, str]]]:
        """Extract structured data with conversation history.

        Useful for multi-phase generation where context builds up.
        History is formatted as context in the prompt for the AI.

        Args:
            prompt: The user prompt for this turn.
            response_model: Pydantic model class for the response.
            history: Previous conversation messages (or None to start fresh).
            system: System prompt (only used if history is None).
            task: Optional task name for tracking.
            max_retries: Retry attempts on validation failure.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (extracted_result, updated_history).
        """
        if history is None:
            history = []
            if system:
                history.append({"role": "system", "content": system})

        # Add new user message
        history.append({"role": "user", "content": prompt})

        # Format history as context for single-turn call
        system_prompt, formatted_prompt = _format_history_as_context(history)

        self._total_calls += 1

        result = await self._text_provider.generate_structured(
            prompt=formatted_prompt,
            response_model=response_model,
            system=system_prompt,
            task=task,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._total_cost = self._text_provider._total_cost

        # Add assistant response to history
        history.append({"role": "assistant", "content": result.model_dump_json()})

        return result, history


# Module-level singleton for convenience
_default_extractor: StructuredExtractor | None = None


def get_extractor(
    config: ProviderConfig | None = None,
    provider_override: str | None = None,
) -> StructuredExtractor:
    """Get the default structured extractor instance."""
    global _default_extractor
    if _default_extractor is None or config is not None:
        _default_extractor = StructuredExtractor(
            config=config, provider_override=provider_override
        )
    return _default_extractor
