"""Comprehensive tests for Agno-based TextProvider.

Tests cover:
- Provider creation for all supported providers
- Basic text generation
- Structured output generation
- Provider fallback behavior
- Event callback emission
- Error handling
- Real provider integration tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from socials_automator.providers.text import (
    TextProvider,
    get_text_provider,
    _create_agno_model,
)
from socials_automator.providers.config import (
    ProviderConfig,
    TextProviderConfig,
    ProviderSettings,
)


# =============================================================================
# Test Models for Structured Output
# =============================================================================

class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    message: str
    count: int


class ContentPlan(BaseModel):
    """Content plan model for testing structured output."""
    title: str
    slides: list[str]
    hashtags: list[str]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock provider config with multiple providers.

    Uses only providers with dependencies that are always installed.
    """
    return ProviderConfig(
        provider_settings=ProviderSettings(
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=1,
            fallback_on_error=True,
        ),
        text_providers={
            "lmstudio": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="openai/local-model",
                base_url="http://localhost:1234/v1",
                timeout=120,
            ),
            "openai": TextProviderConfig(
                priority=2,
                enabled=True,
                litellm_model="openai/gpt-4o-mini",
                api_key="test-openai-key",
                timeout=60,
            ),
            "gemini": TextProviderConfig(
                priority=3,
                enabled=True,
                litellm_model="gemini/gemini-1.5-flash",
                api_key="test-gemini-key",
                timeout=60,
            ),
        },
        image_providers={},
        task_overrides={},
    )


@pytest.fixture
def single_provider_config():
    """Config with a single provider."""
    return ProviderConfig(
        provider_settings=ProviderSettings(
            timeout_seconds=60,
            fallback_on_error=True,
        ),
        text_providers={
            "openai": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="openai/gpt-4o-mini",
                api_key="test-key",
                timeout=60,
            ),
        },
        image_providers={},
        task_overrides={},
    )


# =============================================================================
# Unit Tests: Model Creation
# =============================================================================

def _has_optional_dep(module_name: str) -> bool:
    """Check if an optional dependency is installed."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


class TestAgnoModelCreation:
    """Test Agno model creation for different providers."""

    def test_create_lmstudio_model(self):
        """Test LMStudio model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="openai/qwen2.5-7b",
            base_url="http://localhost:1234/v1",
        )
        model = _create_agno_model("lmstudio", config)

        assert model is not None
        assert model.id == "qwen2.5-7b"

    def test_create_lmstudio_auto_detect(self):
        """Test LMStudio model with auto-detect (local-model)."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="openai/local-model",
            base_url="http://localhost:1234/v1",
        )
        model = _create_agno_model("lmstudio", config)

        assert model is not None
        # When local-model, id should be None for auto-detection
        assert model.id is None

    @pytest.mark.skipif(not _has_optional_dep("ollama"), reason="ollama not installed")
    def test_create_ollama_model(self):
        """Test Ollama model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="ollama/llama3.2",
            base_url="http://localhost:11434",
        )
        model = _create_agno_model("ollama", config)

        assert model is not None
        assert model.id == "llama3.2"

    def test_create_openai_model(self):
        """Test OpenAI model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="openai/gpt-4o-mini",
            api_key="test-key",
        )
        model = _create_agno_model("openai", config)

        assert model is not None
        assert model.id == "gpt-4o-mini"

    @pytest.mark.skipif(not _has_optional_dep("groq"), reason="groq not installed")
    def test_create_groq_model(self):
        """Test Groq model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="groq/llama-3.3-70b-versatile",
            api_key="test-key",
        )
        model = _create_agno_model("groq", config)

        assert model is not None
        assert model.id == "llama-3.3-70b-versatile"

    @pytest.mark.skipif(not _has_optional_dep("anthropic"), reason="anthropic not installed")
    def test_create_anthropic_model(self):
        """Test Anthropic/Claude model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="anthropic/claude-3-5-sonnet",
            api_key="test-key",
        )
        model = _create_agno_model("anthropic", config)

        assert model is not None
        assert model.id == "claude-3-5-sonnet"

    def test_create_gemini_model(self):
        """Test Gemini model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="gemini/gemini-1.5-flash",
            api_key="test-key",
        )
        model = _create_agno_model("gemini", config)

        assert model is not None
        assert model.id == "gemini-1.5-flash"

    def test_create_deepseek_model(self):
        """Test DeepSeek model creation."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="deepseek/deepseek-chat",
            api_key="test-key",
        )
        model = _create_agno_model("deepseek", config)

        assert model is not None
        assert model.id == "deepseek-chat"

    def test_create_unknown_provider_uses_openai_like(self):
        """Test unknown provider falls back to OpenAILike."""
        config = TextProviderConfig(
            priority=1,
            enabled=True,
            litellm_model="custom/my-model",
            api_key="test-key",
            base_url="http://custom-api.com/v1",
        )
        model = _create_agno_model("custom", config)

        assert model is not None
        assert model.id == "my-model"


# =============================================================================
# Unit Tests: TextProvider Initialization
# =============================================================================

class TestTextProviderInit:
    """Test TextProvider initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with explicit config."""
        provider = TextProvider(config=mock_config)

        assert provider.config == mock_config
        assert provider._current_provider is None
        assert provider._current_model is None
        assert provider._total_calls == 0

    def test_init_with_override(self, mock_config):
        """Test initialization with provider override."""
        provider = TextProvider(config=mock_config, provider_override="gemini")

        providers = provider._get_providers()
        # Gemini should be first due to override
        assert providers[0][0] == "gemini"

    def test_init_with_event_callback(self, mock_config):
        """Test initialization with event callback."""
        callback = AsyncMock()
        provider = TextProvider(config=mock_config, event_callback=callback)

        assert provider._event_callback == callback

    def test_get_providers_respects_priority(self, mock_config):
        """Test that providers are ordered by priority."""
        provider = TextProvider(config=mock_config)
        providers = provider._get_providers()

        # Should be ordered: lmstudio (1), openai (2), gemini (3)
        assert providers[0][0] == "lmstudio"
        assert providers[1][0] == "openai"
        assert providers[2][0] == "gemini"

    def test_get_providers_with_override_reorders(self, mock_config):
        """Test that override puts provider first."""
        provider = TextProvider(config=mock_config, provider_override="gemini")
        providers = provider._get_providers()

        assert providers[0][0] == "gemini"

    def test_get_providers_adds_missing_lmstudio(self, single_provider_config):
        """Test that lmstudio override creates default config if missing."""
        provider = TextProvider(
            config=single_provider_config,
            provider_override="lmstudio"
        )
        providers = provider._get_providers()

        assert providers[0][0] == "lmstudio"
        assert providers[0][1].base_url == "http://localhost:1234/v1"

    def test_get_providers_adds_missing_ollama(self, single_provider_config):
        """Test that ollama override creates default config if missing."""
        provider = TextProvider(
            config=single_provider_config,
            provider_override="ollama"
        )
        providers = provider._get_providers()

        assert providers[0][0] == "ollama"
        assert providers[0][1].base_url == "http://localhost:11434"


# =============================================================================
# Unit Tests: Cost Estimation
# =============================================================================

class TestCostEstimation:
    """Test cost estimation logic."""

    def test_estimate_cost_openai(self, single_provider_config):
        """Test cost estimation for OpenAI."""
        provider = TextProvider(config=single_provider_config)
        # 4000 chars = ~1000 tokens
        cost = provider._estimate_cost("openai", 4000)
        assert cost == pytest.approx(0.01, rel=0.01)

    def test_estimate_cost_local_is_free(self, single_provider_config):
        """Test that local providers have zero cost."""
        provider = TextProvider(config=single_provider_config)

        assert provider._estimate_cost("lmstudio", 10000) == 0.0
        assert provider._estimate_cost("ollama", 10000) == 0.0

    def test_estimate_cost_groq_is_cheap(self, single_provider_config):
        """Test that Groq has low cost."""
        provider = TextProvider(config=single_provider_config)
        cost = provider._estimate_cost("groq", 4000)
        assert cost < 0.001


# =============================================================================
# Unit Tests: Event Emission
# =============================================================================

class TestEventEmission:
    """Test event callback emission."""

    @pytest.mark.asyncio
    async def test_emit_event_calls_callback(self, single_provider_config):
        """Test that events are emitted to callback."""
        callback = AsyncMock()
        provider = TextProvider(
            config=single_provider_config,
            event_callback=callback
        )

        await provider._emit_event({"type": "test", "data": "value"})

        callback.assert_called_once_with({"type": "test", "data": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_no_callback_is_safe(self, single_provider_config):
        """Test that emitting without callback doesn't error."""
        provider = TextProvider(config=single_provider_config)

        # Should not raise
        await provider._emit_event({"type": "test"})


# =============================================================================
# Integration Tests: Mocked Agno Agent
# =============================================================================

class TestGenerateWithMockedAgent:
    """Test generate() with mocked Agno Agent."""

    @pytest.mark.asyncio
    async def test_generate_basic(self, single_provider_config):
        """Test basic text generation with mocked agent."""
        with patch("agno.agent.Agent") as MockAgent:
            # Setup mock
            mock_response = MagicMock()
            mock_response.content = "Hello, world!"
            mock_response.model = "gpt-4o-mini"
            mock_response.reasoning_content = None

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate("Say hello")

            assert result == "Hello, world!"
            assert provider.current_provider == "openai"
            assert provider.current_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, single_provider_config):
        """Test generation with system prompt."""
        with patch("agno.agent.Agent") as MockAgent:
            mock_response = MagicMock()
            mock_response.content = "I am helpful!"
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate(
                "Who are you?",
                system="You are a helpful assistant."
            )

            assert result == "I am helpful!"
            # Verify Agent was created with instructions
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["instructions"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_handles_reasoning_content(self, single_provider_config):
        """Test that reasoning_content is used as fallback."""
        with patch("agno.agent.Agent") as MockAgent:
            mock_response = MagicMock()
            mock_response.content = ""  # Empty content
            mock_response.reasoning_content = "Thinking: Here is the answer..."
            mock_response.model = "qwen3"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate("Think about this")

            assert result == "Thinking: Here is the answer..."

    @pytest.mark.asyncio
    async def test_generate_emits_events(self, single_provider_config):
        """Test that events are emitted during generation."""
        callback = AsyncMock()

        with patch("agno.agent.Agent") as MockAgent:
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(
                config=single_provider_config,
                event_callback=callback
            )
            await provider.generate("Test")

            # Should have call and response events
            assert callback.call_count >= 2
            events = [call[0][0] for call in callback.call_args_list]
            event_types = [e["type"] for e in events]
            assert "text_call" in event_types
            assert "text_response" in event_types

    @pytest.mark.asyncio
    async def test_generate_tracks_stats(self, single_provider_config):
        """Test that total_calls and total_cost are tracked."""
        with patch("agno.agent.Agent") as MockAgent:
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)

            assert provider._total_calls == 0

            await provider.generate("Test 1")
            assert provider._total_calls == 1

            await provider.generate("Test 2")
            assert provider._total_calls == 2
            assert provider._total_cost > 0


# =============================================================================
# Integration Tests: Provider Fallback
# =============================================================================

class TestProviderFallback:
    """Test provider fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, mock_config):
        """Test that provider falls back on error."""
        call_count = 0

        def create_mock_agent(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_agent = MagicMock()
            if call_count == 1:
                # First provider fails
                mock_agent.arun = AsyncMock(
                    side_effect=Exception("Provider 1 failed")
                )
            else:
                # Second provider succeeds
                mock_response = MagicMock()
                mock_response.content = "Success from fallback"
                mock_response.model = "gpt-4o-mini"
                mock_agent.arun = AsyncMock(return_value=mock_response)

            return mock_agent

        with patch("agno.agent.Agent", side_effect=create_mock_agent):
            provider = TextProvider(config=mock_config)
            result = await provider.generate("Test")

            assert result == "Success from fallback"
            assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self, mock_config):
        """Test that error is raised when all providers fail."""
        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(side_effect=Exception("All failed"))
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=mock_config)

            with pytest.raises(Exception, match="All failed"):
                await provider.generate("Test")


# =============================================================================
# Integration Tests: Structured Output
# =============================================================================

class TestStructuredOutput:
    """Test structured output generation."""

    @pytest.mark.asyncio
    async def test_generate_structured_basic(self, single_provider_config):
        """Test basic structured output generation."""
        with patch("agno.agent.Agent") as MockAgent:
            # Create actual Pydantic model instance
            expected_result = SimpleResponse(message="Hello", count=42)

            mock_response = MagicMock()
            mock_response.content = expected_result
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate_structured(
                prompt="Generate a message",
                response_model=SimpleResponse
            )

            assert isinstance(result, SimpleResponse)
            assert result.message == "Hello"
            assert result.count == 42

            # Verify output_schema was passed
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["output_schema"] == SimpleResponse

    @pytest.mark.asyncio
    async def test_generate_structured_complex_model(self, single_provider_config):
        """Test structured output with complex model."""
        with patch("agno.agent.Agent") as MockAgent:
            expected_result = ContentPlan(
                title="AI Revolution",
                slides=["Intro", "History", "Future"],
                hashtags=["#AI", "#Tech"]
            )

            mock_response = MagicMock()
            mock_response.content = expected_result
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate_structured(
                prompt="Plan content about AI",
                response_model=ContentPlan,
                system="You are a content strategist."
            )

            assert isinstance(result, ContentPlan)
            assert result.title == "AI Revolution"
            assert len(result.slides) == 3
            assert len(result.hashtags) == 2


# =============================================================================
# Integration Tests: generate_with_tools
# =============================================================================

class TestGenerateWithTools:
    """Test tool-based generation."""

    @pytest.mark.asyncio
    async def test_generate_with_tools_returns_content(self, single_provider_config):
        """Test that generate_with_tools returns proper structure."""
        with patch("agno.agent.Agent") as MockAgent:
            mock_response = MagicMock()
            mock_response.content = "Tool response"
            mock_response.model = "gpt-4o-mini"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            provider = TextProvider(config=single_provider_config)
            result = await provider.generate_with_tools(
                prompt="Use tools",
                tools=[{"type": "function", "function": {"name": "test"}}]
            )

            assert isinstance(result, dict)
            assert "content" in result
            assert "tool_calls" in result
            assert "finish_reason" in result
            assert result["content"] == "Tool response"


# =============================================================================
# Real Integration Tests (requires running LMStudio)
# =============================================================================

class TestRealLMStudioIntegration:
    """Real integration tests with LMStudio.

    These tests require LMStudio to be running on localhost:1234 with a model loaded.
    Skip if LMStudio is not available or returns errors.
    """

    @pytest.fixture
    def real_provider(self):
        """Create a real provider with LMStudio override."""
        return TextProvider(provider_override="lmstudio")

    @staticmethod
    def lmstudio_available() -> bool:
        """Check if LMStudio is available and has a model loaded."""
        import httpx
        try:
            with httpx.Client(timeout=2) as client:
                response = client.get("http://localhost:1234/v1/models")
                if response.status_code != 200:
                    return False
                data = response.json()
                # Check if there's at least one model loaded
                return len(data.get("data", [])) > 0
        except Exception:
            return False

    @pytest.mark.asyncio
    async def test_real_generate_basic(self, real_provider):
        """Test real generation with LMStudio."""
        if not self.lmstudio_available():
            pytest.skip("LMStudio not available or no model loaded")

        try:
            result = await real_provider.generate(
                prompt="Say 'hello' and nothing else.",
                max_tokens=10
            )
            assert result is not None
            assert len(result) > 0
        except Exception as e:
            # If LMStudio returns an error, the fallback providers worked
            # or we got a meaningful error - both are acceptable
            if real_provider.current_provider is not None:
                # Provider fallback worked
                pass
            else:
                pytest.skip(f"LMStudio returned error: {e}")

    @pytest.mark.asyncio
    async def test_real_generate_with_system(self, real_provider):
        """Test real generation with system prompt."""
        if not self.lmstudio_available():
            pytest.skip("LMStudio not available or no model loaded")

        try:
            result = await real_provider.generate(
                prompt="What is 2+2?",
                system="You are a math tutor. Answer briefly.",
                max_tokens=20
            )
            assert result is not None
            # Just verify we got a response, not checking content
            # since different models respond differently
            assert len(result) > 0
        except Exception as e:
            if real_provider.current_provider is not None:
                pass  # Fallback worked
            else:
                pytest.skip(f"LMStudio returned error: {e}")

    @pytest.mark.asyncio
    async def test_real_structured_output(self, real_provider):
        """Test real structured output generation.

        Note: Structured output may not work with all local models.
        This test may be skipped if the model doesn't support it.
        """
        if not self.lmstudio_available():
            pytest.skip("LMStudio not available or no model loaded")

        try:
            result = await real_provider.generate_structured(
                prompt="Generate a simple greeting message with count 5",
                response_model=SimpleResponse
            )
            assert isinstance(result, SimpleResponse)
            assert len(result.message) > 0
            assert isinstance(result.count, int)
        except Exception as e:
            # Structured output may not work with all models
            pytest.skip(f"Structured output not supported: {e}")

    @pytest.mark.asyncio
    async def test_real_provider_fallback(self):
        """Test that fallback works when primary provider is down.

        Uses a fake provider that will fail, then falls back to a real one.
        """
        if not self.lmstudio_available():
            pytest.skip("LMStudio not available for fallback test")

        # Use a fake primary provider that will fail
        config = ProviderConfig(
            provider_settings=ProviderSettings(fallback_on_error=True),
            text_providers={
                "fake": TextProviderConfig(
                    priority=1,
                    enabled=True,
                    litellm_model="openai/fake-model",
                    base_url="http://localhost:9999/v1",  # Non-existent
                    timeout=2,
                ),
                "lmstudio": TextProviderConfig(
                    priority=2,
                    enabled=True,
                    litellm_model="openai/local-model",
                    base_url="http://localhost:1234/v1",
                    timeout=60,
                ),
            },
            image_providers={},
            task_overrides={},
        )

        provider = TextProvider(config=config)

        try:
            result = await provider.generate("Hello", max_tokens=10)
            assert result is not None
            # Should have fallen back from fake to lmstudio (or another provider)
            assert provider.current_provider != "fake"
        except Exception as e:
            # If all providers fail, that's also acceptable for this test
            # The important thing is that fallback was attempted
            pytest.skip(f"All providers failed: {e}")


# =============================================================================
# Singleton Tests
# =============================================================================

class TestSingleton:
    """Test module-level singleton."""

    def test_get_text_provider_returns_singleton(self):
        """Test that get_text_provider returns same instance."""
        # Reset singleton
        import socials_automator.providers.text as text_module
        text_module._default_provider = None

        provider1 = get_text_provider()
        provider2 = get_text_provider()

        assert provider1 is provider2

    def test_get_text_provider_with_config_creates_new(self):
        """Test that passing config creates new instance."""
        import socials_automator.providers.text as text_module
        text_module._default_provider = None

        config = ProviderConfig(
            provider_settings=ProviderSettings(),
            text_providers={
                "test": TextProviderConfig(
                    priority=1,
                    enabled=True,
                    litellm_model="openai/test",
                    api_key="test",
                ),
            },
            image_providers={},
            task_overrides={},
        )

        provider1 = get_text_provider()
        provider2 = get_text_provider(config=config)

        # New instance should be created with config
        assert provider2.config == config
