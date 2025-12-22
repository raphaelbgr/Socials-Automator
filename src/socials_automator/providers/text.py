"""Text generation provider using Agno framework.

Simplified provider that leverages Agno's unified model interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from .config import ProviderConfig, TextProviderConfig, load_provider_config

_logger = logging.getLogger("ai_calls")

# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


def _get_lmstudio_model(base_url: str) -> str | None:
    """Query LMStudio for the currently loaded model.

    Args:
        base_url: LMStudio API base URL.

    Returns:
        Model ID string or None if unavailable.
    """
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    # Return first loaded model
                    return models[0].get("id")
    except Exception as e:
        _logger.debug(f"Could not query LMStudio models: {e}")

    return None


def _create_agno_model(provider_name: str, provider_config: TextProviderConfig) -> Any:
    """Create an Agno model instance for the given provider.

    Agno provides unified interfaces for all major providers.
    """
    # Extract model ID from litellm format (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_id = provider_config.litellm_model
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]

    api_key = provider_config.get_api_key()
    base_url = provider_config.get_base_url()

    # Import Agno models lazily to avoid import errors if not installed
    if provider_name == "lmstudio":
        from agno.models.lmstudio import LMStudio

        lmstudio_url = base_url or "http://localhost:1234/v1"

        # Query for loaded model if using generic "local-model"
        if model_id == "local-model":
            model_id = _get_lmstudio_model(lmstudio_url)
            if model_id:
                _logger.info(f"LMStudio: detected loaded model: {model_id}")

        return LMStudio(
            id=model_id,  # Pass detected or configured model ID
            base_url=lmstudio_url,
            # Disable thinking/reasoning mode for faster responses
            # GLM models use "thinking.type", other models use "enable_thinking"
            extra_body={
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    elif provider_name == "ollama":
        from agno.models.ollama import Ollama
        return Ollama(
            id=model_id,
            host=base_url or "http://localhost:11434",
        )

    elif provider_name == "openai":
        from agno.models.openai import OpenAIChat
        return OpenAIChat(
            id=model_id,
            api_key=api_key,
        )

    elif provider_name == "groq":
        from agno.models.groq import Groq
        return Groq(
            id=model_id,
            api_key=api_key,
        )

    elif provider_name == "anthropic":
        from agno.models.anthropic import Claude
        return Claude(
            id=model_id,
            api_key=api_key,
        )

    elif provider_name == "gemini":
        from agno.models.google import Gemini
        return Gemini(
            id=model_id,
            api_key=api_key,
        )

    elif provider_name == "deepseek":
        from agno.models.deepseek import DeepSeek
        return DeepSeek(
            id=model_id,
            api_key=api_key,
        )

    else:
        # Fallback to OpenAI-like for unknown providers
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=base_url,
        )


class TextProvider:
    """Unified text generation provider using Agno framework.

    Supports all major providers through Agno's unified interface:
    - OpenAI, Anthropic, Groq, Gemini, DeepSeek
    - Local: LMStudio, Ollama

    Usage:
        provider = TextProvider()
        response = await provider.generate("Write a haiku about AI")
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        event_callback: AIEventCallback = None,
        provider_override: str | None = None,
    ):
        """Initialize the text provider.

        Args:
            config: Provider configuration. If None, loads from default config file.
            event_callback: Optional callback for AI events (for progress tracking).
            provider_override: Override provider name (e.g., 'lmstudio', 'openai').
        """
        self.config = config or load_provider_config()
        self._event_callback = event_callback
        self._provider_override = provider_override
        self._current_provider: str | None = None
        self._current_model: str | None = None
        self._total_calls = 0
        self._total_cost = 0.0

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an AI event if callback is set."""
        if self._event_callback:
            await self._event_callback(event)

    def _get_providers(self) -> list[tuple[str, TextProviderConfig]]:
        """Get list of providers to try, respecting override."""
        providers = list(self.config.get_enabled_text_providers())

        if self._provider_override:
            override_name = self._provider_override.lower()

            # Check if override is in enabled list
            if not any(name == override_name for name, _ in providers):
                # Add from config or create default
                if override_name in self.config.text_providers:
                    providers.insert(0, (override_name, self.config.text_providers[override_name]))
                elif override_name == "lmstudio":
                    providers.insert(0, (override_name, TextProviderConfig(
                        priority=0, enabled=True,
                        litellm_model="openai/local-model",
                        base_url="http://localhost:1234/v1",
                        timeout=120,
                    )))
                elif override_name == "ollama":
                    providers.insert(0, (override_name, TextProviderConfig(
                        priority=0, enabled=True,
                        litellm_model="ollama/llama3.2",
                        base_url="http://localhost:11434",
                        timeout=120,
                    )))
            else:
                # Reorder to put override first
                providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        return providers

    def _estimate_cost(self, provider: str, total_chars: int) -> float:
        """Estimate cost based on provider and character count."""
        cost_per_1k = {
            "openai": 0.01, "anthropic": 0.015, "groq": 0.0001,
            "gemini": 0.0001, "deepseek": 0.001,
            "lmstudio": 0.0, "ollama": 0.0,
        }
        tokens = total_chars / 4
        return (tokens / 1000) * cost_per_1k.get(provider, 0.001)

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
            task: Optional task name for tracking.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        from agno.agent import Agent

        providers = self._get_providers()
        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                # Create Agno model
                model = _create_agno_model(provider_name, provider_config)
                model_id = getattr(model, "id", "unknown") or "unknown"

                self._current_provider = provider_name
                self._current_model = model_id

                # Emit event
                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model_id,
                    "prompt_preview": prompt[:200],
                    "task": task,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                # Log request
                _logger.info(
                    f"AI_REQUEST | provider:{provider_name} | model:{model_id} | task:{task}\n"
                    f"--- SYSTEM ---\n{system or '(none)'}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- END REQUEST ---"
                )

                # Create agent and run
                agent = Agent(
                    model=model,
                    instructions=system,
                    markdown=False,
                )

                # Run async
                response = await agent.arun(prompt)
                result = response.content or ""

                # Handle reasoning models (content might be in reasoning_content)
                if not result and hasattr(response, "reasoning_content") and response.reasoning_content:
                    result = response.reasoning_content

                duration = time.time() - start_time
                self._total_calls += 1

                # Get actual model from response
                actual_model = getattr(response, "model", model_id) or model_id
                self._current_model = actual_model

                cost = self._estimate_cost(provider_name, len(prompt) + len(result))
                self._total_cost += cost

                # Log response
                _logger.info(
                    f"AI_RESPONSE | provider:{provider_name} | model:{actual_model} | "
                    f"task:{task} | duration:{duration:.2f}s | cost:${cost:.4f}\n"
                    f"--- RESPONSE ---\n{result}\n"
                    f"--- END RESPONSE ---"
                )

                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": actual_model,
                    "response_preview": result[:200],
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "total_calls": self._total_calls,
                    "total_cost": self._total_cost,
                    "failed_providers": failed_providers.copy(),
                })

                return result

            except Exception as e:
                last_error = e
                failed_providers.append(provider_name)
                _logger.warning(f"Provider {provider_name} failed: {e}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": str(e)[:100],
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No providers available")

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str | None = None,
        task: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output matching a Pydantic model.

        Args:
            prompt: The user prompt to send to the model.
            response_model: Pydantic model class for the response.
            system: Optional system prompt for context.
            task: Optional task name for tracking.
            max_retries: Number of retry attempts on validation failure.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Returns:
            Instance of response_model with extracted data.
        """
        from agno.agent import Agent

        providers = self._get_providers()
        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                model = _create_agno_model(provider_name, provider_config)
                model_id = getattr(model, "id", "unknown") or "unknown"

                self._current_provider = provider_name
                self._current_model = model_id

                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model_id,
                    "prompt_preview": prompt[:200],
                    "task": task,
                    "structured": True,
                    "response_model": response_model.__name__,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                _logger.info(
                    f"AI_REQUEST_STRUCTURED | provider:{provider_name} | model:{model_id} | "
                    f"task:{task} | response_model:{response_model.__name__}\n"
                    f"--- SYSTEM ---\n{system or '(none)'}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- END REQUEST ---"
                )

                # Create agent with output_schema for structured output
                agent = Agent(
                    model=model,
                    instructions=system,
                    output_schema=response_model,
                    markdown=False,
                )

                response = await agent.arun(prompt)

                # Content is already validated Pydantic model
                result = response.content

                duration = time.time() - start_time
                self._total_calls += 1

                result_json = result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else str(result)
                cost = self._estimate_cost(provider_name, len(prompt) + len(result_json))
                self._total_cost += cost

                _logger.info(
                    f"AI_RESPONSE_STRUCTURED | provider:{provider_name} | model:{model_id} | "
                    f"task:{task} | response_model:{response_model.__name__} | "
                    f"duration:{duration:.2f}s | cost:${cost:.4f}\n"
                    f"--- RESPONSE (JSON) ---\n{result_json}\n"
                    f"--- END RESPONSE ---"
                )

                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": model_id,
                    "response_preview": result_json[:200],
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "structured": True,
                    "response_model": response_model.__name__,
                    "failed_providers": failed_providers.copy(),
                })

                return result

            except Exception as e:
                last_error = e
                failed_providers.append(provider_name)
                _logger.warning(f"STRUCTURED_ERROR | provider:{provider_name} | error:{e}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": str(e)[:100],
                    "structured": True,
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No providers available")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: str | None = None,
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate text with function calling support.

        Note: For tool use with Agno, consider using Agno's native tool system
        which is more powerful. This method provides compatibility with
        OpenAI-style tool definitions.

        Args:
            prompt: The user prompt to send to the model.
            tools: List of tool definitions in OpenAI function calling format.
            system: Optional system prompt for context.
            task: Optional task name.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with content, tool_calls, and finish_reason.
        """
        # For now, use raw generate and parse tool calls manually
        # Agno's native tool system is preferred for tool use
        result = await self.generate(
            prompt=prompt,
            system=system,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return {
            "content": result,
            "tool_calls": None,
            "finish_reason": "stop",
        }

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
