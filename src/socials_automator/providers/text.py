"""Text generation provider using direct API calls."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator, Callable, Awaitable

import httpx
import instructor
from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, APIStatusError
from pydantic import BaseModel

from .config import ProviderConfig, TextProviderConfig, load_provider_config


_logger = logging.getLogger("ai_calls")


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
        self._current_provider: str | None = None
        self._current_model: str | None = None
        self._event_callback = event_callback
        self._total_calls = 0
        self._total_cost = 0.0
        self._clients: dict[str, AsyncOpenAI] = {}
        self._instructor_clients: dict[str, instructor.AsyncInstructor] = {}

        # Provider override
        self._provider_override = provider_override

    # Providers that return JSON in markdown code blocks (need MD_JSON mode)
    _MD_JSON_PROVIDERS = {"lmstudio", "ollama", "deepseek"}

    # Providers with native tool/function calling support (use TOOLS mode)
    _TOOLS_MODE_PROVIDERS = {"openai", "groq", "gemini", "zai"}

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

            # Use provider config timeout (local models may need longer)
            timeout = provider_config.timeout or 120

            self._clients[provider_name] = AsyncOpenAI(
                api_key=api_key or "lm-studio",  # Local providers use dummy key
                base_url=base_url,
                timeout=timeout,
            )

        return self._clients[provider_name]

    def _get_instructor_client(
        self, provider_name: str, provider_config: TextProviderConfig
    ) -> instructor.AsyncInstructor | None:
        """Get or create an Instructor-wrapped client for a provider.

        Uses appropriate mode based on provider capabilities:
        - TOOLS mode for providers with native tool calling (OpenAI, Groq, Gemini)
        - MD_JSON mode for providers returning JSON in markdown blocks (LMStudio, Ollama, DeepSeek)
        - JSON mode as fallback for others

        Returns None if base client couldn't be created.
        """
        client = self._get_client(provider_name, provider_config)
        if client is None:
            return None

        cache_key = provider_name
        if cache_key not in self._instructor_clients:
            # Select mode based on provider capabilities
            if provider_name.lower() in self._TOOLS_MODE_PROVIDERS:
                mode = instructor.Mode.TOOLS
            elif provider_name.lower() in self._MD_JSON_PROVIDERS:
                mode = instructor.Mode.MD_JSON
            else:
                # Default to JSON mode for unknown providers
                mode = instructor.Mode.JSON

            _logger.debug(f"Creating instructor client for {provider_name} with mode {mode}")

            self._instructor_clients[cache_key] = instructor.from_openai(
                client, mode=mode
            )

        return self._instructor_clients[cache_key]

    def _get_model_name(self, provider_name: str, provider_config: TextProviderConfig) -> str:
        """Extract model name from litellm_model format.

        For local providers (lmstudio, ollama), auto-detects available models
        if the configured model is a placeholder like 'local-model'.
        """
        # litellm_model is like "openai/gpt-4o" or "groq/llama-3.3-70b-versatile"
        model = provider_config.litellm_model
        if "/" in model:
            model = model.split("/", 1)[1]

        # For local providers, detect actual model if placeholder is used
        if model in ("local-model", "local") and provider_name in ("lmstudio", "ollama"):
            detected = self._detect_local_model(provider_config)
            if detected:
                return detected

        return model

    def _detect_local_model(self, provider_config: TextProviderConfig) -> str | None:
        """Detect available model from local provider (LMStudio/Ollama)."""
        import httpx

        base_url = provider_config.get_base_url() or "http://localhost:1234/v1"

        try:
            # Query models endpoint synchronously (called before async operation)
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{base_url}/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    # Filter out embedding models, prefer chat models
                    chat_models = [
                        m["id"] for m in models
                        if "embed" not in m["id"].lower()
                    ]
                    if chat_models:
                        return chat_models[0]
                    elif models:
                        return models[0]["id"]
        except Exception:
            pass  # Fall back to configured model

        return None

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
        providers = list(self.config.get_enabled_text_providers())

        # If override is set, force that provider to be first (even if disabled)
        if self._provider_override:
            override_name = self._provider_override.lower()

            # Check if override provider is already in enabled list
            override_in_list = any(name == override_name for name, _ in providers)

            if not override_in_list:
                # Check if provider exists in config but is disabled
                if override_name in self.config.text_providers:
                    # Add disabled provider to front of list
                    override_config = self.config.text_providers[override_name]
                    providers.insert(0, (override_name, override_config))
                else:
                    # Create default config for known local providers
                    default_configs = {
                        "lmstudio": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="openai/local-model",
                            base_url="http://localhost:1234/v1",
                            api_key="lm-studio",
                            timeout=120,
                        ),
                        "ollama": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="ollama/llama3.2",
                            base_url="http://localhost:11434",
                            timeout=120,
                        ),
                    }
                    if override_name in default_configs:
                        providers.insert(0, (override_name, default_configs[override_name]))
            else:
                # Reorder to put override first
                providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                client = self._get_client(provider_name, provider_config)

                # Skip if client couldn't be created (incompatible or no API key)
                if client is None:
                    continue

                model = self._get_model_name(provider_name, provider_config)

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

                # Log FULL request
                _logger.info(
                    f"AI_REQUEST | provider:{provider_name} | model:{model} | task:{task}\n"
                    f"--- SYSTEM ---\n{system or '(none)'}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- END REQUEST ---"
                )

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

                # Use actual model from response if available (especially for LM Studio)
                actual_model = getattr(response, "model", model) or model
                self._current_model = actual_model

                # Estimate cost
                cost = self._estimate_cost(provider_name, len(prompt) + len(result))
                self._total_cost += cost

                # Log FULL response
                _logger.info(
                    f"AI_RESPONSE | provider:{provider_name} | model:{actual_model} | "
                    f"task:{task} | duration:{duration:.2f}s | cost:${cost:.4f}\n"
                    f"--- RESPONSE ---\n{result}\n"
                    f"--- END RESPONSE ---"
                )

                # Emit "response" event
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
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output matching a Pydantic model using Instructor.

        Uses Instructor library for reliable JSON extraction with:
        - Automatic schema enforcement
        - Validation retries with error context
        - Mode selection based on provider (TOOLS, MD_JSON, or JSON)

        Args:
            prompt: The user prompt to send to the model.
            response_model: Pydantic model class for the response.
            system: Optional system prompt for context.
            task: Optional task name for tracking.
            max_retries: Number of retry attempts on validation failure (default 3).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            Instance of response_model with extracted data.

        Raises:
            Exception: If all providers fail or validation fails after retries.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Try providers in priority order with fallback
        providers = list(self.config.get_enabled_text_providers())

        # Apply provider override (same logic as generate())
        if self._provider_override:
            override_name = self._provider_override.lower()
            override_in_list = any(name == override_name for name, _ in providers)

            if not override_in_list:
                if override_name in self.config.text_providers:
                    override_config = self.config.text_providers[override_name]
                    providers.insert(0, (override_name, override_config))
                else:
                    default_configs = {
                        "lmstudio": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="openai/local-model",
                            base_url="http://localhost:1234/v1",
                            api_key="lm-studio",
                            timeout=120,
                        ),
                        "ollama": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="ollama/llama3.2",
                            base_url="http://localhost:11434",
                            timeout=120,
                        ),
                    }
                    if override_name in default_configs:
                        providers.insert(0, (override_name, default_configs[override_name]))
            else:
                providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                instructor_client = self._get_instructor_client(provider_name, provider_config)

                if instructor_client is None:
                    continue

                model = self._get_model_name(provider_name, provider_config)

                self._current_provider = provider_name
                self._current_model = model

                # Emit "calling" event
                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model,
                    "prompt_preview": prompt[:200],
                    "task": task,
                    "structured": True,
                    "response_model": response_model.__name__,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                # Log FULL request for structured extraction
                _logger.info(
                    f"AI_REQUEST_STRUCTURED | provider:{provider_name} | model:{model} | "
                    f"task:{task} | response_model:{response_model.__name__}\n"
                    f"--- SYSTEM ---\n{system or '(none)'}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- END REQUEST ---"
                )

                # Use Instructor for structured extraction with retries
                result = await instructor_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                duration = time.time() - start_time
                self._total_calls += 1

                # Estimate cost (use model JSON for rough size estimate)
                result_json = result.model_dump_json(indent=2)
                cost = self._estimate_cost(provider_name, len(prompt) + len(result_json))
                self._total_cost += cost

                # Log FULL structured response
                _logger.info(
                    f"AI_RESPONSE_STRUCTURED | provider:{provider_name} | model:{model} | "
                    f"task:{task} | response_model:{response_model.__name__} | "
                    f"duration:{duration:.2f}s | cost:${cost:.4f}\n"
                    f"--- RESPONSE (JSON) ---\n{result_json}\n"
                    f"--- END RESPONSE ---"
                )

                # Emit "response" event
                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": model,
                    "response_preview": result_json[:200],
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "structured": True,
                    "response_model": response_model.__name__,
                    "failed_providers": failed_providers.copy(),
                })

                return result

            except (APIConnectionError, AuthenticationError, httpx.ConnectError, httpx.TimeoutException, ConnectionError, OSError) as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": f"{type(e).__name__}",
                    "structured": True,
                    "failed_providers": failed_providers.copy(),
                })
                continue

            except APIStatusError as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": f"{e.status_code}",
                    "structured": True,
                    "failed_providers": failed_providers.copy(),
                })
                if self.config.provider_settings.fallback_on_error:
                    continue
                raise

            except Exception as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                error_msg = str(e)[:100]
                _logger.warning(
                    f"INSTRUCTOR_ERROR | provider:{provider_name} | error:{error_msg}"
                )
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": error_msg,
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

        Args:
            prompt: The user prompt to send to the model.
            tools: List of tool definitions in OpenAI function calling format.
            system: Optional system prompt for context.
            task: Optional task name for provider/model override.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            Dict with:
            - "content": str | None - The text response (if any)
            - "tool_calls": list[dict] | None - List of tool calls (if any)
            - "finish_reason": str - Why generation stopped

        Raises:
            Exception: If all providers fail.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Try providers in priority order with fallback
        providers = list(self.config.get_enabled_text_providers())

        # Apply provider override (same logic as generate())
        if self._provider_override:
            override_name = self._provider_override.lower()
            override_in_list = any(name == override_name for name, _ in providers)

            if not override_in_list:
                if override_name in self.config.text_providers:
                    override_config = self.config.text_providers[override_name]
                    providers.insert(0, (override_name, override_config))
                else:
                    default_configs = {
                        "lmstudio": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="openai/local-model",
                            base_url="http://localhost:1234/v1",
                            api_key="lm-studio",
                            timeout=120,
                        ),
                        "ollama": TextProviderConfig(
                            priority=0,
                            enabled=True,
                            litellm_model="ollama/llama3.2",
                            base_url="http://localhost:11434",
                            timeout=120,
                        ),
                    }
                    if override_name in default_configs:
                        providers.insert(0, (override_name, default_configs[override_name]))
            else:
                providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        last_error: Exception | None = None
        failed_providers: list[str] = []

        for provider_name, provider_config in providers:
            try:
                client = self._get_client(provider_name, provider_config)

                if client is None:
                    continue

                model = self._get_model_name(provider_name, provider_config)

                self._current_provider = provider_name
                self._current_model = model

                # Emit "calling" event
                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model,
                    "prompt_preview": prompt[:200],
                    "task": task,
                    "has_tools": True,
                    "tool_count": len(tools),
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                # Log FULL request with tools
                tool_names = [t.get("function", {}).get("name", "?") for t in tools]
                _logger.info(
                    f"AI_REQUEST_TOOLS | provider:{provider_name} | model:{model} | "
                    f"task:{task} | tools:{tool_names}\n"
                    f"--- SYSTEM ---\n{system or '(none)'}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- END REQUEST ---"
                )

                # Build request with tools
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tools,
                    "tool_choice": "auto",  # Let AI decide when to use tools
                }

                # Handle GLM models
                if "glm-4" in model.lower():
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

                response = await client.chat.completions.create(**request_kwargs)

                duration = time.time() - start_time
                self._total_calls += 1

                # Extract response
                message = response.choices[0].message
                content = message.content or ""
                finish_reason = response.choices[0].finish_reason

                # Extract tool calls if present
                tool_calls = None
                if message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]

                actual_model = getattr(response, "model", model) or model
                self._current_model = actual_model

                cost = self._estimate_cost(provider_name, len(prompt) + len(content))
                self._total_cost += cost

                # Log FULL response with tool calls
                tool_calls_str = ""
                if tool_calls:
                    tool_calls_str = "\n--- TOOL CALLS ---\n" + json.dumps(tool_calls, indent=2)
                _logger.info(
                    f"AI_RESPONSE_TOOLS | provider:{provider_name} | model:{actual_model} | "
                    f"task:{task} | duration:{duration:.2f}s | "
                    f"tool_calls:{len(tool_calls) if tool_calls else 0} | finish:{finish_reason}\n"
                    f"--- CONTENT ---\n{content or '(none)'}{tool_calls_str}\n"
                    f"--- END RESPONSE ---"
                )

                # Emit "response" event
                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": actual_model,
                    "response_preview": content[:200] if content else f"[{len(tool_calls or [])} tool calls]",
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "tool_calls_count": len(tool_calls) if tool_calls else 0,
                    "finish_reason": finish_reason,
                    "failed_providers": failed_providers.copy(),
                })

                return {
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                }

            except (APIConnectionError, AuthenticationError, httpx.ConnectError, httpx.TimeoutException, ConnectionError, OSError) as e:
                last_error = e
                failed_providers.append(f"{provider_name}")
                await self._emit_event({
                    "type": "text_error",
                    "provider": provider_name,
                    "error": f"{type(e).__name__}",
                    "failed_providers": failed_providers.copy(),
                })
                continue

            except APIStatusError as e:
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

    async def continue_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Continue conversation after tool execution.

        Args:
            messages: Full conversation history including tool results.
            tools: List of tool definitions.
            task: Optional task name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Same format as generate_with_tools().
        """
        providers = list(self.config.get_enabled_text_providers())

        # Apply provider override
        if self._provider_override:
            override_name = self._provider_override.lower()
            override_in_list = any(name == override_name for name, _ in providers)
            if override_in_list:
                providers = sorted(providers, key=lambda x: 0 if x[0] == override_name else 1)

        last_error: Exception | None = None

        for provider_name, provider_config in providers:
            try:
                client = self._get_client(provider_name, provider_config)
                if client is None:
                    continue

                model = self._get_model_name(provider_name, provider_config)
                self._current_provider = provider_name
                self._current_model = model

                start_time = time.time()

                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tools,
                    "tool_choice": "auto",
                }

                if "glm-4" in model.lower():
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

                response = await client.chat.completions.create(**request_kwargs)

                duration = time.time() - start_time
                self._total_calls += 1

                message = response.choices[0].message
                content = message.content or ""
                finish_reason = response.choices[0].finish_reason

                tool_calls = None
                if message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]

                cost = self._estimate_cost(provider_name, len(str(messages)) + len(content))
                self._total_cost += cost

                await self._emit_event({
                    "type": "text_response",
                    "provider": provider_name,
                    "model": model,
                    "response_preview": content[:200] if content else f"[{len(tool_calls or [])} tool calls]",
                    "duration_seconds": duration,
                    "cost_usd": cost,
                    "tool_calls_count": len(tool_calls) if tool_calls else 0,
                    "finish_reason": finish_reason,
                })

                return {
                    "content": content if content else None,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                }

            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        raise RuntimeError("No providers available")

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
        model = self._get_model_name(provider_name, provider_config)

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
