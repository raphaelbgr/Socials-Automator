"""LLM Fallback Manager - Handles retry logic and provider switching.

This service provides automatic fallback between LLM providers with configurable
retry counts for local vs external providers. It abstracts away all retry logic
from the rest of the application.

Usage:
    manager = LLMFallbackManager()
    result = await manager.generate(prompt, task="caption")
    # Internally handles: lmstudio(10x) -> zai(5x) -> gemini(5x) -> groq(5x)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ..providers.text import TextProvider
from ..providers.config import load_provider_config, ProviderConfig, LLMFallbackConfig


_logger = logging.getLogger("llm_fallback")

T = TypeVar("T", bound=BaseModel)


@dataclass
class ProviderAttempt:
    """Record of a single provider attempt."""
    provider: str
    model: str
    attempt: int
    max_attempts: int
    success: bool
    error: str | None = None
    duration_ms: int = 0


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    success: bool
    result: Any = None
    provider_used: str | None = None
    model_used: str | None = None
    attempts: list[ProviderAttempt] = field(default_factory=list)
    total_attempts: int = 0
    error: str | None = None


@dataclass
class FallbackConfig:
    """Configuration for LLM fallback behavior.

    Note: These values can be loaded from providers.yaml via load_from_provider_config().
    """
    # Retry counts
    local_max_retries: int = 10
    external_max_retries: int = 5

    # Provider classification
    local_providers: list[str] = field(default_factory=lambda: ["lmstudio", "ollama"])

    # Provider priority order (will use config if not specified)
    provider_priority: list[str] | None = None

    # Logging
    verbose: bool = True
    log_to_console: bool = True

    @classmethod
    def from_provider_config(cls, provider_config: ProviderConfig) -> "FallbackConfig":
        """Create FallbackConfig from ProviderConfig (loaded from providers.yaml)."""
        llm_config = provider_config.llm_fallback
        return cls(
            local_max_retries=llm_config.local_max_retries,
            external_max_retries=llm_config.external_max_retries,
            local_providers=llm_config.local_providers,
            provider_priority=llm_config.fallback_priority,
        )


class LLMFallbackManager:
    """Manages LLM calls with automatic retry and provider fallback.

    Features:
    - Automatic retry with configurable limits per provider type
    - Seamless fallback between providers
    - Clean console logging showing progress
    - Debug logging for detailed troubleshooting
    - Works with both generate() and extract() operations

    Example:
        manager = LLMFallbackManager(
            preferred_provider="lmstudio",
            config=FallbackConfig(
                local_max_retries=10,
                external_max_retries=5
            )
        )

        # Simple generation
        result = await manager.generate("Write a caption", task="caption")

        # Structured extraction
        result = await manager.extract(
            prompt="Generate caption JSON",
            response_model=CaptionResponse,
            task="caption"
        )
    """

    def __init__(
        self,
        preferred_provider: str | None = None,
        config: FallbackConfig | None = None,
        provider_config: ProviderConfig | None = None,
        display: Any = None,
    ):
        """Initialize the fallback manager.

        Args:
            preferred_provider: Preferred provider to try first (e.g., 'lmstudio').
            config: Fallback configuration.
            provider_config: Provider configuration (loads from file if None).
            display: Optional CLI display for logging.
        """
        self.preferred_provider = preferred_provider
        self.config = config or FallbackConfig()
        self.provider_config = provider_config or load_provider_config()
        self._display = display

        # Build provider chain
        self._provider_chain = self._build_provider_chain()

        # Track statistics
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "provider_usage": {},
        }

    def _build_provider_chain(self) -> list[tuple[str, int]]:
        """Build ordered list of (provider_name, max_retries) tuples.

        Returns:
            List of (provider, max_retries) in priority order.
        """
        chain = []

        # If preferred provider is set, add it first with local retry count
        if self.preferred_provider:
            is_local = self.preferred_provider.lower() in self.config.local_providers
            max_retries = self.config.local_max_retries if is_local else self.config.external_max_retries
            chain.append((self.preferred_provider.lower(), max_retries))

        # Use configured priority or default
        if self.config.provider_priority:
            priority_list = self.config.provider_priority
        else:
            # Get from provider config
            enabled = list(self.provider_config.get_enabled_text_providers())
            priority_list = [name for name, _ in enabled]

        # Add remaining providers
        for provider in priority_list:
            provider_lower = provider.lower()
            # Skip if already added as preferred
            if provider_lower == (self.preferred_provider or "").lower():
                continue

            is_local = provider_lower in self.config.local_providers
            max_retries = self.config.local_max_retries if is_local else self.config.external_max_retries
            chain.append((provider_lower, max_retries))

        return chain

    def _log_console(self, message: str, level: str = "info") -> None:
        """Log message to console if enabled."""
        if not self.config.log_to_console:
            return

        if self._display:
            if level == "error":
                self._display.error(message, "LLM")
            elif level == "warning":
                self._display.warning(message, "LLM")
            elif level == "success":
                self._display.success(message, "LLM")
            else:
                self._display.detail(message, "LLM")
        else:
            # Fallback to print with formatting
            prefix_map = {
                "info": "  [>]",
                "error": "  [X]",
                "warning": "  [!]",
                "success": "  [OK]",
            }
            prefix = prefix_map.get(level, "  [>]")
            print(f"{prefix} {message}")

    def _log_attempt_start(
        self,
        provider: str,
        model: str,
        attempt: int,
        max_attempts: int,
        task: str | None = None,
    ) -> None:
        """Log the start of an attempt."""
        task_str = f" ({task})" if task else ""
        self._log_console(f"{provider}/{model} [{attempt}/{max_attempts}]{task_str}...")
        _logger.debug(f"Attempt {attempt}/{max_attempts} with {provider}/{model}{task_str}")

    def _log_attempt_result(
        self,
        provider: str,
        success: bool,
        error: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Log the result of an attempt."""
        if success:
            self._log_console(f"{provider}: OK ({duration_ms}ms)", "success")
            _logger.info(f"{provider} succeeded in {duration_ms}ms")
        else:
            short_error = (error[:80] + "...") if error and len(error) > 80 else error
            self._log_console(f"{provider}: {short_error}", "error")
            _logger.warning(f"{provider} failed: {error}")

    def _log_provider_switch(self, from_provider: str, to_provider: str, reason: str) -> None:
        """Log switching to a different provider."""
        self._log_console(f"Switching: {from_provider} -> {to_provider} ({reason})", "warning")
        _logger.info(f"Switching from {from_provider} to {to_provider}: {reason}")

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        **kwargs: Any,
    ) -> FallbackResult:
        """Generate text with automatic fallback.

        Args:
            prompt: The prompt to send.
            system: Optional system prompt.
            task: Task name for logging.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            validator: Optional function(result) -> (is_valid, error_message).
            **kwargs: Additional arguments.

        Returns:
            FallbackResult with success status and result.
        """
        self._stats["total_calls"] += 1

        attempts: list[ProviderAttempt] = []
        total_attempts = 0
        last_error: str | None = None

        task_display = task or "generation"
        self._log_console(f"Starting {task_display}...", "info")

        for provider_name, max_retries in self._provider_chain:
            # Create provider instance for this provider
            try:
                text_provider = TextProvider(
                    config=self.provider_config,
                    provider_override=provider_name,
                )
            except Exception as e:
                _logger.warning(f"Could not create provider {provider_name}: {e}")
                continue

            # Get model name for logging
            model_name = self._get_model_name(provider_name)

            for attempt in range(1, max_retries + 1):
                total_attempts += 1
                self._log_attempt_start(provider_name, model_name, attempt, max_retries, task)

                start_time = time.time()
                try:
                    result = await text_provider.generate(
                        prompt=prompt,
                        system=system,
                        task=task,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    duration_ms = int((time.time() - start_time) * 1000)

                    # Validate if validator provided
                    if validator:
                        is_valid, validation_error = validator(result)
                        if not is_valid:
                            raise ValueError(f"Validation failed: {validation_error}")

                    # Success!
                    self._log_attempt_result(provider_name, True, duration_ms=duration_ms)

                    attempts.append(ProviderAttempt(
                        provider=provider_name,
                        model=model_name,
                        attempt=attempt,
                        max_attempts=max_retries,
                        success=True,
                        duration_ms=duration_ms,
                    ))

                    self._stats["successful_calls"] += 1
                    self._stats["provider_usage"][provider_name] = \
                        self._stats["provider_usage"].get(provider_name, 0) + 1

                    return FallbackResult(
                        success=True,
                        result=result,
                        provider_used=provider_name,
                        model_used=model_name,
                        attempts=attempts,
                        total_attempts=total_attempts,
                    )

                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    error_msg = str(e)
                    last_error = error_msg

                    self._log_attempt_result(provider_name, False, error_msg, duration_ms)

                    attempts.append(ProviderAttempt(
                        provider=provider_name,
                        model=model_name,
                        attempt=attempt,
                        max_attempts=max_retries,
                        success=False,
                        error=error_msg,
                        duration_ms=duration_ms,
                    ))

            # Provider exhausted, log switch
            next_provider = self._get_next_provider(provider_name)
            if next_provider:
                self._log_provider_switch(
                    provider_name,
                    next_provider,
                    f"{provider_name} exhausted after {max_retries} attempts"
                )

        # All providers exhausted
        self._stats["failed_calls"] += 1
        self._log_console(f"All providers exhausted after {total_attempts} total attempts", "error")

        return FallbackResult(
            success=False,
            attempts=attempts,
            total_attempts=total_attempts,
            error=f"All providers exhausted. Last error: {last_error}",
        )

    async def extract(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        validator: Callable[[T], tuple[bool, str | None]] | None = None,
        **kwargs: Any,
    ) -> FallbackResult:
        """Extract structured data with automatic fallback.

        Args:
            prompt: The prompt to send.
            response_model: Pydantic model for the response.
            system: Optional system prompt.
            task: Task name for logging.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            validator: Optional function(result) -> (is_valid, error_message).
            **kwargs: Additional arguments.

        Returns:
            FallbackResult with success status and result.
        """
        self._stats["total_calls"] += 1

        attempts: list[ProviderAttempt] = []
        total_attempts = 0
        last_error: str | None = None

        task_display = task or "extraction"
        self._log_console(f"Starting {task_display} (structured)...", "info")

        for provider_name, max_retries in self._provider_chain:
            # Create provider instance for this provider
            try:
                text_provider = TextProvider(
                    config=self.provider_config,
                    provider_override=provider_name,
                )
            except Exception as e:
                _logger.warning(f"Could not create provider {provider_name}: {e}")
                continue

            # Get model name for logging
            model_name = self._get_model_name(provider_name)

            for attempt in range(1, max_retries + 1):
                total_attempts += 1
                self._log_attempt_start(provider_name, model_name, attempt, max_retries, task)

                start_time = time.time()
                try:
                    result = await text_provider.generate_structured(
                        prompt=prompt,
                        response_model=response_model,
                        system=system,
                        task=task,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    duration_ms = int((time.time() - start_time) * 1000)

                    # Validate if validator provided
                    if validator:
                        is_valid, validation_error = validator(result)
                        if not is_valid:
                            raise ValueError(f"Validation failed: {validation_error}")

                    # Success!
                    self._log_attempt_result(provider_name, True, duration_ms=duration_ms)

                    attempts.append(ProviderAttempt(
                        provider=provider_name,
                        model=model_name,
                        attempt=attempt,
                        max_attempts=max_retries,
                        success=True,
                        duration_ms=duration_ms,
                    ))

                    self._stats["successful_calls"] += 1
                    self._stats["provider_usage"][provider_name] = \
                        self._stats["provider_usage"].get(provider_name, 0) + 1

                    return FallbackResult(
                        success=True,
                        result=result,
                        provider_used=provider_name,
                        model_used=model_name,
                        attempts=attempts,
                        total_attempts=total_attempts,
                    )

                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    error_msg = str(e)
                    last_error = error_msg

                    self._log_attempt_result(provider_name, False, error_msg, duration_ms)

                    attempts.append(ProviderAttempt(
                        provider=provider_name,
                        model=model_name,
                        attempt=attempt,
                        max_attempts=max_retries,
                        success=False,
                        error=error_msg,
                        duration_ms=duration_ms,
                    ))

            # Provider exhausted, log switch
            next_provider = self._get_next_provider(provider_name)
            if next_provider:
                self._log_provider_switch(
                    provider_name,
                    next_provider,
                    f"{provider_name} exhausted after {max_retries} attempts"
                )

        # All providers exhausted
        self._stats["failed_calls"] += 1
        self._log_console(f"All providers exhausted after {total_attempts} total attempts", "error")

        return FallbackResult(
            success=False,
            attempts=attempts,
            total_attempts=total_attempts,
            error=f"All providers exhausted. Last error: {last_error}",
        )

    def _get_model_name(self, provider_name: str) -> str:
        """Get model name for a provider."""
        if provider_name in self.provider_config.text_providers:
            config = self.provider_config.text_providers[provider_name]
            model = config.litellm_model
            if "/" in model:
                return model.split("/", 1)[1]
            return model

        # Default model names
        defaults = {
            "lmstudio": "local-model",
            "ollama": "llama3.2",
            "zai": "GLM-4.5-Air",
            "groq": "llama-3.3-70b",
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o",
        }
        return defaults.get(provider_name, "unknown")

    def _get_next_provider(self, current: str) -> str | None:
        """Get the next provider in the chain after current."""
        found_current = False
        for provider, _ in self._provider_chain:
            if found_current:
                return provider
            if provider == current:
                found_current = True
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "provider_usage": {},
        }


def create_fallback_manager(
    preferred_provider: str | None = None,
    local_max_retries: int = 10,
    external_max_retries: int = 5,
    provider_priority: list[str] | None = None,
    display: Any = None,
) -> LLMFallbackManager:
    """Factory function to create a configured LLMFallbackManager.

    Args:
        preferred_provider: Provider to try first (e.g., 'lmstudio').
        local_max_retries: Max retries for local providers.
        external_max_retries: Max retries for external providers.
        provider_priority: Custom provider priority order.
        display: CLI display for logging.

    Returns:
        Configured LLMFallbackManager instance.
    """
    config = FallbackConfig(
        local_max_retries=local_max_retries,
        external_max_retries=external_max_retries,
        provider_priority=provider_priority,
    )

    return LLMFallbackManager(
        preferred_provider=preferred_provider,
        config=config,
        display=display,
    )
