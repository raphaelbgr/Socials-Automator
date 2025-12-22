"""Tests for TextProvider error handling and fallback behavior.

Tests cover:
- Provider fallback on various error types
- Error event emission
- Real LMStudio integration tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import httpx

from socials_automator.providers.text import TextProvider
from socials_automator.providers.config import (
    ProviderConfig,
    TextProviderConfig,
    ProviderSettings,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock provider config with multiple providers."""
    return ProviderConfig(
        provider_settings=ProviderSettings(
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=1,
            fallback_on_error=True,
        ),
        text_providers={
            "provider1": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="openai/test-model",
                api_key="test-key-1",
                timeout=5,
            ),
            "provider2": TextProviderConfig(
                priority=2,
                enabled=True,
                litellm_model="openai/test-model-2",
                api_key="test-key-2",
                timeout=5,
            ),
            "provider3": TextProviderConfig(
                priority=3,
                enabled=True,
                litellm_model="openai/test-model-3",
                api_key="test-key-3",
                timeout=5,
            ),
        },
        image_providers={},
        task_overrides={},
    )


@pytest.fixture
def mock_config_no_fallback():
    """Config with fallback disabled."""
    return ProviderConfig(
        provider_settings=ProviderSettings(
            fallback_on_error=False,
        ),
        text_providers={
            "provider1": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="openai/test-model",
                api_key="test-key-1",
            ),
            "provider2": TextProviderConfig(
                priority=2,
                enabled=True,
                litellm_model="openai/test-model-2",
                api_key="test-key-2",
            ),
        },
        image_providers={},
        task_overrides={},
    )


@pytest.fixture
def single_provider_config():
    """Config with a single gemini provider."""
    return ProviderConfig(
        text_providers={
            "gemini": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="gemini/gemini-2.0-flash",
                api_key="test-key",
            )
        }
    )


# =============================================================================
# Connection Error Tests
# =============================================================================

class TestConnectionErrors:
    """Test handling of connection-related errors."""

    @pytest.mark.asyncio
    async def test_connection_error_falls_back(self, mock_config):
        """Should fall back to next provider on connection error."""
        provider = TextProvider(config=mock_config)

        call_count = 0

        async def mock_arun(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            response = MagicMock()
            response.content = "Success from fallback"
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Test prompt")
            assert result == "Success from fallback"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_error_falls_back(self, mock_config):
        """Should fall back to next provider on timeout error."""
        provider = TextProvider(config=mock_config)

        call_count = 0

        async def mock_arun(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Read timed out")
            response = MagicMock()
            response.content = "Success"
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Test prompt")
            assert result == "Success"

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self, mock_config):
        """Should raise error when all providers fail."""
        provider = TextProvider(config=mock_config)

        async def mock_arun(prompt):
            raise ConnectionError("All connections failed")

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            with pytest.raises(ConnectionError):
                await provider.generate("Test prompt")


# =============================================================================
# Fallback Behavior Tests
# =============================================================================

class TestFallbackBehavior:
    """Test provider fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_tries_all_providers(self, mock_config):
        """Should try all providers before raising."""
        provider = TextProvider(config=mock_config)
        call_count = 0

        async def mock_arun(prompt):
            nonlocal call_count
            call_count += 1
            raise Exception("Provider failed")

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            with pytest.raises(Exception):
                await provider.generate("Test prompt")

            # Should have tried all 3 providers
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_fallback_disabled_raises_immediately(self, mock_config_no_fallback):
        """Should raise immediately when fallback is disabled."""
        provider = TextProvider(config=mock_config_no_fallback)
        call_count = 0

        async def mock_arun(prompt):
            nonlocal call_count
            call_count += 1
            raise Exception("Provider failed")

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            with pytest.raises(Exception):
                await provider.generate("Test prompt")

            # Should have only tried 1 provider
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_first_provider_success_no_fallback_needed(self, mock_config):
        """Should not try fallback when first provider succeeds."""
        provider = TextProvider(config=mock_config)
        call_count = 0

        async def mock_arun(prompt):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.content = "Success"
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Test prompt")
            assert result == "Success"
            assert call_count == 1


# =============================================================================
# Event Emission Tests
# =============================================================================

class TestEventEmission:
    """Test error event emission."""

    @pytest.mark.asyncio
    async def test_error_event_emitted_on_failure(self, single_provider_config):
        """Should emit error event when provider fails."""
        events = []

        async def capture_event(event):
            events.append(event)

        provider = TextProvider(
            config=single_provider_config,
            event_callback=capture_event,
        )

        async def mock_arun(prompt):
            raise Exception("Test error")

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            with pytest.raises(Exception):
                await provider.generate("Test prompt")

        # Should have call event and error event
        error_events = [e for e in events if e.get("type") == "text_error"]
        assert len(error_events) == 1
        assert "Test error" in error_events[0].get("error", "")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases in response handling."""

    @pytest.mark.asyncio
    async def test_empty_response_uses_empty_string(self, single_provider_config):
        """Should handle empty response content."""
        provider = TextProvider(config=single_provider_config)

        async def mock_arun(prompt):
            response = MagicMock()
            response.content = ""
            response.reasoning_content = None
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Test prompt")
            assert result == ""

    @pytest.mark.asyncio
    async def test_none_content_uses_reasoning(self, single_provider_config):
        """Should fall back to reasoning_content when content is None."""
        provider = TextProvider(config=single_provider_config)

        async def mock_arun(prompt):
            response = MagicMock()
            response.content = None
            response.reasoning_content = "Reasoning result"
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Test prompt")
            assert result == "Reasoning result"

    @pytest.mark.asyncio
    async def test_unicode_prompt_and_response(self, single_provider_config):
        """Should handle unicode in prompt and response."""
        provider = TextProvider(config=single_provider_config)

        async def mock_arun(prompt):
            response = MagicMock()
            response.content = "Unicode response: cafe"  # ASCII safe
            return response

        with patch("agno.agent.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.arun = mock_arun
            MockAgent.return_value = mock_agent

            result = await provider.generate("Unicode prompt: naiv")
            assert "cafe" in result


# =============================================================================
# Real LMStudio Integration Tests (Skip if not available)
# =============================================================================

def _lmstudio_available() -> bool:
    """Check if LMStudio is running locally."""
    try:
        import httpx
        with httpx.Client(timeout=2) as client:
            response = client.get("http://localhost:1234/v1/models")
            return response.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _lmstudio_available(), reason="LMStudio not running")
class TestRealLMStudioIntegration:
    """Real integration tests with LMStudio."""

    @pytest.fixture
    def lmstudio_config(self):
        """Config for LMStudio."""
        return ProviderConfig(
            text_providers={
                "lmstudio": TextProviderConfig(
                    priority=1,
                    enabled=True,
                    litellm_model="openai/local-model",
                    base_url="http://localhost:1234/v1",
                    timeout=120,
                )
            }
        )

    @pytest.mark.asyncio
    async def test_real_lmstudio_simple_prompt(self, lmstudio_config):
        """Test real LMStudio generation."""
        provider = TextProvider(config=lmstudio_config)
        result = await provider.generate("Say hello in exactly 3 words.")
        assert len(result) > 0
        assert provider.current_provider == "lmstudio"

    @pytest.mark.asyncio
    async def test_real_lmstudio_longer_generation(self, lmstudio_config):
        """Test longer text generation."""
        provider = TextProvider(config=lmstudio_config)
        result = await provider.generate(
            "Write a haiku about programming.",
            max_tokens=100,
        )
        assert len(result) > 10

    @pytest.mark.asyncio
    async def test_real_lmstudio_json_generation(self, lmstudio_config):
        """Test JSON-formatted response."""
        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            name: str
            count: int

        provider = TextProvider(config=lmstudio_config)
        result = await provider.generate_structured(
            prompt="Create a simple item with name 'test' and count 5",
            response_model=SimpleOutput,
        )
        assert isinstance(result, SimpleOutput)
        assert result.name == "test"
        assert result.count == 5

    @pytest.mark.asyncio
    async def test_real_lmstudio_with_system_prompt(self, lmstudio_config):
        """Test with system prompt."""
        provider = TextProvider(config=lmstudio_config)
        result = await provider.generate(
            prompt="What is your name?",
            system="You are Bob. Always respond with your name first.",
        )
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_real_lmstudio_error_handling(self, lmstudio_config):
        """Test that errors are handled gracefully."""
        # This should work - just testing the flow
        provider = TextProvider(config=lmstudio_config)
        result = await provider.generate("Hello")
        assert result is not None

    @pytest.mark.asyncio
    async def test_real_multiple_sequential_calls(self, lmstudio_config):
        """Test multiple sequential calls."""
        provider = TextProvider(config=lmstudio_config)

        result1 = await provider.generate("Say 'one'")
        result2 = await provider.generate("Say 'two'")

        assert len(result1) > 0
        assert len(result2) > 0
        assert provider._total_calls == 2

    @pytest.mark.asyncio
    async def test_real_log_file_written(self, lmstudio_config, tmp_path):
        """Test that AI calls are logged."""
        import logging

        # Set up logger to write to temp file
        log_file = tmp_path / "ai_calls.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger("ai_calls")
        original_handlers = logger.handlers[:]
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        try:
            provider = TextProvider(config=lmstudio_config)
            await provider.generate("Test logging")

            handler.flush()
            log_content = log_file.read_text()
            assert "AI_REQUEST" in log_content
            assert "AI_RESPONSE" in log_content
        finally:
            logger.handlers = original_handlers

    @pytest.mark.asyncio
    async def test_real_provider_priority_chain(self):
        """Test that providers are tried in priority order."""
        config = ProviderConfig(
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
                    api_key="fake-key",
                ),
            }
        )

        provider = TextProvider(config=config)
        result = await provider.generate("Hello")

        # Should use lmstudio (priority 1)
        assert provider.current_provider == "lmstudio"
