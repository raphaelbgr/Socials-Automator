"""Structured data extraction service using Instructor.

Provides reliable JSON extraction from AI responses with:
- Automatic schema enforcement via Pydantic models
- Native JSON schema support for LMStudio (no retries needed)
- Validation retries with error context for other providers
- Mode selection based on provider capabilities
- Conversation history support for multi-turn extraction
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Awaitable, TypeVar

import httpx
import instructor
from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, APIStatusError
from pydantic import BaseModel

from ..providers.config import ProviderConfig, TextProviderConfig, load_provider_config


_logger = logging.getLogger("ai_calls")

# Generic type for response models
T = TypeVar("T", bound=BaseModel)

# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


class StructuredExtractor:
    """Service for extracting structured data from AI responses.

    Uses Instructor library for reliable JSON extraction with:
    - Automatic schema enforcement via Pydantic models
    - Validation retries with error context sent back to AI
    - Mode selection based on provider (TOOLS, MD_JSON, JSON)
    - Provider fallback on failures

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

    # Providers with native JSON schema support (use response_format directly)
    # These guarantee valid JSON output without retries
    _NATIVE_JSON_SCHEMA_PROVIDERS = {"lmstudio"}

    # Providers that return JSON in markdown code blocks (need MD_JSON mode)
    _MD_JSON_PROVIDERS = {"ollama", "deepseek"}

    # Providers with native tool/function calling support (use TOOLS mode)
    _TOOLS_MODE_PROVIDERS = {"openai", "groq", "gemini", "zai"}

    # Providers that are NOT OpenAI-compatible
    _INCOMPATIBLE_PROVIDERS = {"anthropic"}

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

        # Client caches
        self._clients: dict[str, AsyncOpenAI] = {}
        self._instructor_clients: dict[str, instructor.AsyncInstructor] = {}

        # Stats tracking
        self._total_calls = 0
        self._total_cost = 0.0

        # Current provider info (for external access)
        self._current_provider: str | None = None
        self._current_model: str | None = None

    @property
    def current_provider(self) -> str | None:
        """Get the name of the last used provider."""
        return self._current_provider

    @property
    def current_model(self) -> str | None:
        """Get the model of the last used provider."""
        return self._current_model

    @property
    def total_calls(self) -> int:
        """Get total number of extraction calls."""
        return self._total_calls

    @property
    def total_cost(self) -> float:
        """Get total estimated cost."""
        return self._total_cost

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an AI event if callback is set."""
        if self._event_callback:
            await self._event_callback(event)

    def _get_client(
        self, provider_name: str, provider_config: TextProviderConfig
    ) -> AsyncOpenAI | None:
        """Get or create an OpenAI client for a provider."""
        if provider_name in self._INCOMPATIBLE_PROVIDERS:
            return None

        api_key = provider_config.get_api_key()
        base_url = provider_config.get_base_url()
        is_local = provider_name in ("lmstudio", "ollama") or (
            base_url and "localhost" in base_url
        )

        if not api_key and not is_local:
            return None

        if provider_name not in self._clients:
            if base_url is None:
                base_url_map = {
                    "groq": "https://api.groq.com/openai/v1",
                    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
                }
                base_url = base_url_map.get(provider_name)

            timeout = provider_config.timeout or 120

            self._clients[provider_name] = AsyncOpenAI(
                api_key=api_key or "lm-studio",
                base_url=base_url,
                timeout=timeout,
            )

        return self._clients[provider_name]

    def _get_instructor_client(
        self, provider_name: str, provider_config: TextProviderConfig
    ) -> instructor.AsyncInstructor | None:
        """Get or create an Instructor-wrapped client for a provider."""
        client = self._get_client(provider_name, provider_config)
        if client is None:
            return None

        if provider_name not in self._instructor_clients:
            # Select mode based on provider capabilities
            if provider_name.lower() in self._TOOLS_MODE_PROVIDERS:
                mode = instructor.Mode.TOOLS
            elif provider_name.lower() in self._MD_JSON_PROVIDERS:
                mode = instructor.Mode.MD_JSON
            else:
                mode = instructor.Mode.JSON

            _logger.debug(f"Creating instructor client: {provider_name} mode={mode}")

            self._instructor_clients[provider_name] = instructor.from_openai(
                client, mode=mode
            )

        return self._instructor_clients[provider_name]

    def _build_json_schema_response_format(
        self, response_model: type[BaseModel]
    ) -> dict[str, Any]:
        """Build LMStudio-compatible response_format with JSON schema.

        Args:
            response_model: Pydantic model class.

        Returns:
            response_format dict for LMStudio API.
        """
        schema = response_model.model_json_schema()

        # Remove unsupported fields that LMStudio might not handle
        def clean_schema(obj: Any) -> Any:
            if isinstance(obj, dict):
                # Remove $defs, definitions, and other meta fields
                cleaned = {
                    k: clean_schema(v)
                    for k, v in obj.items()
                    if k not in ("$defs", "definitions", "title", "description")
                }
                return cleaned
            elif isinstance(obj, list):
                return [clean_schema(item) for item in obj]
            return obj

        cleaned_schema = clean_schema(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": cleaned_schema,
            },
        }

    async def _extract_native_json_schema(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> T:
        """Extract using native JSON schema (for LMStudio).

        Uses LMStudio's native response_format with json_schema type,
        which guarantees valid JSON output without needing retries.

        Args:
            client: AsyncOpenAI client.
            model: Model name.
            messages: Conversation messages.
            response_model: Pydantic model class.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Instance of response_model.
        """
        response_format = self._build_json_schema_response_format(response_model)

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        return response_model.model_validate(data)

    def _get_model_name(
        self, provider_name: str, provider_config: TextProviderConfig
    ) -> str:
        """Extract model name from config."""
        model = provider_config.litellm_model
        if "/" in model:
            model = model.split("/", 1)[1]

        # Auto-detect for local providers
        if model in ("local-model", "local") and provider_name in ("lmstudio", "ollama"):
            detected = self._detect_local_model(provider_config)
            if detected:
                return detected

        return model

    def _detect_local_model(self, provider_config: TextProviderConfig) -> str | None:
        """Detect available model from local provider."""
        base_url = provider_config.get_base_url() or "http://localhost:1234/v1"

        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{base_url}/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    chat_models = [
                        m["id"] for m in models if "embed" not in m["id"].lower()
                    ]
                    if chat_models:
                        return chat_models[0]
                    elif models:
                        return models[0]["id"]
        except Exception:
            pass

        return None

    def _estimate_cost(self, provider: str, total_chars: int) -> float:
        """Estimate cost based on provider and character count."""
        cost_per_1k = {
            "openai": 0.01,
            "groq": 0.0001,
            "gemini": 0.0001,
            "zai": 0.001,
            "lmstudio": 0.0,
            "ollama": 0.0,
        }
        tokens = total_chars / 4
        rate = cost_per_1k.get(provider, 0.001)
        return (tokens / 1000) * rate

    def _get_providers(self) -> list[tuple[str, TextProviderConfig]]:
        """Get providers list with override applied."""
        providers = list(self.config.get_enabled_text_providers())

        if self._provider_override:
            override_name = self._provider_override.lower()
            override_in_list = any(name == override_name for name, _ in providers)

            if not override_in_list:
                if override_name in self.config.text_providers:
                    override_config = self.config.text_providers[override_name]
                    providers.insert(0, (override_name, override_config))
                else:
                    # Default configs for known local providers
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
                providers = sorted(
                    providers, key=lambda x: 0 if x[0] == override_name else 1
                )

        return providers

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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self._extract_from_messages(
            messages=messages,
            response_model=response_model,
            task=task,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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

        result = await self._extract_from_messages(
            messages=history,
            response_model=response_model,
            task=task,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Add assistant response to history
        history.append({"role": "assistant", "content": result.model_dump_json()})

        return result, history

    async def _extract_from_messages(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        task: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> T:
        """Internal extraction method that works with message list."""
        providers = self._get_providers()
        last_error: Exception | None = None
        failed_providers: list[str] = []

        # Get prompt preview from last user message
        prompt_preview = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                prompt_preview = msg["content"][:200]
                break

        for provider_name, provider_config in providers:
            try:
                # Check if provider supports native JSON schema
                use_native_schema = provider_name.lower() in self._NATIVE_JSON_SCHEMA_PROVIDERS

                if use_native_schema:
                    # Use native JSON schema (no retries needed - guaranteed valid JSON)
                    client = self._get_client(provider_name, provider_config)
                    if client is None:
                        continue
                else:
                    # Use Instructor for other providers
                    instructor_client = self._get_instructor_client(
                        provider_name, provider_config
                    )
                    if instructor_client is None:
                        continue

                model = self._get_model_name(provider_name, provider_config)
                self._current_provider = provider_name
                self._current_model = model

                # Emit "calling" event
                extraction_mode = "native_json_schema" if use_native_schema else "instructor"
                await self._emit_event({
                    "type": "text_call",
                    "provider": provider_name,
                    "model": model,
                    "prompt_preview": prompt_preview,
                    "task": task,
                    "structured": True,
                    "response_model": response_model.__name__,
                    "extraction_mode": extraction_mode,
                    "failed_providers": failed_providers.copy(),
                })

                start_time = time.time()

                # Log FULL request (messages)
                messages_str = "\n".join([
                    f"[{m['role'].upper()}] {m['content']}"
                    for m in messages
                ])
                _logger.info(
                    f"AI_REQUEST_EXTRACT | provider:{provider_name} | model:{model} | "
                    f"task:{task} | response_model:{response_model.__name__} | "
                    f"mode:{'native_json_schema' if use_native_schema else 'instructor'}\n"
                    f"--- MESSAGES ---\n{messages_str}\n"
                    f"--- END REQUEST ---"
                )

                if use_native_schema:
                    # Use native JSON schema extraction (LMStudio)
                    result = await self._extract_native_json_schema(
                        client=client,
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    # Use Instructor for extraction
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

                result_json = result.model_dump_json(indent=2)
                cost = self._estimate_cost(provider_name, len(prompt_preview) + len(result_json))
                self._total_cost += cost

                # Log FULL response
                _logger.info(
                    f"AI_RESPONSE_EXTRACT | provider:{provider_name} | model:{model} | "
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

            except (
                APIConnectionError,
                AuthenticationError,
                httpx.ConnectError,
                httpx.TimeoutException,
                ConnectionError,
                OSError,
            ) as e:
                last_error = e
                failed_providers.append(provider_name)
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
                failed_providers.append(provider_name)
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
                failed_providers.append(provider_name)
                error_msg = str(e)[:100]
                _logger.warning(
                    f"EXTRACT_ERROR | provider:{provider_name} | model:{response_model.__name__} | error:{error_msg}"
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
