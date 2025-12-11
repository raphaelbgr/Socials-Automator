"""Text generation provider using direct API calls."""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Callable, Awaitable

import httpx
from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, APIStatusError
from pydantic import BaseModel

from .config import ProviderConfig, TextProviderConfig, load_provider_config


# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


class TextProvider:
    """Unified text generation provider using direct APIs.

    Supports:
    - OpenAI (gpt-4o, gpt-4o-mini)
    - Groq (via OpenAI-compatible API)
    - Gemini (via OpenAI-compatible API)
    - Z.AI (via OpenAI-compatible API)
    - Local providers (LM Studio, Ollama via OpenAI-compatible API)

    Usage:
        provider = TextProvider()
        response = await provider.generate("Write a haiku about AI")
    """

    def __init__(self, config: ProviderConfig | None = None, event_callback: AIEventCallback = None):
        """Initialize the text provider.

        Args:
            config: Provider configuration. If None, loads from default config file.
            event_callback: Optional callback for AI events (for progress tracking).
        """
        self.config = config or load_provider_config()
        self._current_provider: str | None = None
        self._current_model: str | None = None
        self._event_callback = event_callback
        self._total_calls = 0
        self._total_cost = 0.0
        self._clients: dict[str, AsyncOpenAI] = {}

    # Providers that are NOT OpenAI-compatible (need their own SDK)
    _INCOMPATIBLE_PROVIDERS = {"anthropic"}

    def _get_client(self, provider_name: str, provider_config: TextProviderConfig) -> AsyncOpenAI | None:
        """Get or create an OpenAI client for a provider.

        Returns None if provider is incompatible or has no API key.
        """
        # Skip incompatible providers
        if provider_name in self._INCOMPATIBLE_PROVIDERS:
            return None

        # Skip providers without API keys (except local ones)
        api_key = provider_config.get_api_key()
        base_url = provider_config.get_base_url()

        # Local providers don't need real API keys
        is_local = provider_name in ("lmstudio", "ollama") or (base_url and "localhost" in base_url)

        if not api_key and not is_local:
            return None

        if provider_name not in self._clients:
            # Map provider names to their base URLs if not specified
            if base_url is None:
                base_url_map = {
                    "groq": "https://api.groq.com/openai/v1",
                    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                }
                base_url = base_url_map.get(provider_name)

            # Use shorter timeout for local providers
            timeout = 5 if is_local else provider_config.timeout

            self._clients[provider_name] = AsyncOpenAI(
                api_key=api_key or "lm-studio",  # Local providers use dummy key
                base_url=base_url,
                timeout=timeout,
            )

        return self._clients[provider_name]

    def _get_model_name(self, provider_config: TextProviderConfig) -> str:
        """Extract model name from litellm_model format."""
        # litellm_model is like "openai/gpt-4o" or "groq/llama-3.3-70b-versatile"
        model = provider_config.litellm_model
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an AI event if callback is set."""
        if self._event_callback:
            await self._event_callback(event)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system prompt for context.
            task: Optional task name for provider/model override.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            Generated text response.

        Raises:
            Exception: If all providers fail.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Try providers in priority order with fallback
        providers = self.config.get_enabled_text_providers()
        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                client = self._get_client(provider_name, provider_config)

                # Skip if client couldn't be created (incompatible or no API key)
                if client is None:
                    continue

                model = self._get_model_name(provider_config)

                self._current_provider = provider_name
                self._current_model = model

                # Emit "calling" event with failed providers info
                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model,
                    "prompt_preview": prompt[:200],
                    "task": task,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                # Build request kwargs
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Disable thinking mode for GLM-4.5 models (reasoning models)
                # This gives direct responses instead of putting content in reasoning_content
                if "glm-4.5" in model.lower() or "glm-4" in model.lower():
                    request_kwargs["extra_body"] = {
                        "thinking": {"type": "disabled"}
                    }

                response = await client.chat.completions.create(**request_kwargs)

                duration = time.time() - start_time

                # Handle response - check for reasoning_content if content is empty
                result = response.choices[0].message.content or ""

                # For GLM-4.5 models, if content is empty, check reasoning_content
                if not result and hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning = getattr(response.choices[0].message, "reasoning_content", "")
                    if reasoning:
                        # Try to extract any JSON or useful content from reasoning
                        result = reasoning
                self._total_calls += 1

                # Estimate cost
                cost = self._estimate_cost(provider_name, len(prompt) + len(result))
                self._total_cost += cost

                # Emit "response" event
                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": model,
                    "response_preview": result[:200],
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "total_calls": self._total_calls,
                    "total_cost": self._total_cost,
                    "failed_providers": failed_providers.copy(),
                })

                return result

            except (APIConnectionError, AuthenticationError, httpx.ConnectError, httpx.TimeoutException, ConnectionError, OSError) as e:
                # Connection or auth errors - immediately try next provider
                last_error = e
                failed_providers.append(f"{provider_name}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": f"{type(e).__name__}",
                    "failed_providers": failed_providers.copy(),
                })
                continue  # Always fallback on connection/auth errors

            except APIStatusError as e:
                # API returned an error status
                last_error = e
                failed_providers.append(f"{provider_name}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": f"{e.status_code}",
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

            except Exception as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                # Emit error event
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": str(e)[:50],
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No providers available")

    def _estimate_cost(self, provider: str, total_chars: int) -> float:
        """Estimate cost based on provider and character count."""
        cost_per_1k = {
            "openai": 0.01,
            "anthropic": 0.015,
            "groq": 0.0001,
            "gemini": 0.0001,
            "zai": 0.001,
            "lmstudio": 0.0,
            "ollama": 0.0,
        }
        tokens = total_chars / 4
        rate = cost_per_1k.get(provider, 0.001)
        return (tokens / 1000) * rate

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output matching a Pydantic model."""
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
```json
{schema_str}
```

Return ONLY the JSON, no other text."""

        response_text = await self.generate(
            prompt=structured_prompt,
            system=system,
            task=task,
            **kwargs,
        )

        # Parse JSON from response
        try:
            json_str = response_text.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                json_str = "\n".join(json_lines)

            data = json.loads(json_str)
            return response_model.model_validate(data)

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured response: {e}\nResponse: {response_text}")

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate text with streaming response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        providers = self.config.get_enabled_text_providers()
        provider_name, provider_config = providers[0]

        client = self._get_client(provider_name, provider_config)
        model = self._get_model_name(provider_config)

        self._current_provider = provider_name
        self._current_model = model

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def current_provider(self) -> str | None:
        """Get the name of the last used provider."""
        return self._current_provider

    @property
    def current_model(self) -> str | None:
        """Get the model of the last used provider."""
        return self._current_model


# Module-level singleton for convenience
_default_provider: TextProvider | None = None


def get_text_provider(config: ProviderConfig | None = None) -> TextProvider:
    """Get the default text provider instance."""
    global _default_provider
    if _default_provider is None or config is not None:
        _default_provider = TextProvider(config)
    return _default_provider
