"""Tests for the StructuredExtractor service using Agno framework."""

from __future__ import annotations

from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from pydantic import BaseModel

from socials_automator.services.extractor import (
    StructuredExtractor,
    get_extractor,
    _format_history_as_context,
)
from socials_automator.providers.config import ProviderConfig, TextProviderConfig


# Test response models
class SimpleResponse(BaseModel):
    """Simple test response model."""
    answer: str


class PlanResponse(BaseModel):
    """Test response model for planning."""
    title: str
    steps: list[str]


class ContentResponse(BaseModel):
    """Test response model for content."""
    hook: str
    body: str


# --- Fixtures ---


@pytest.fixture
def single_provider_config() -> ProviderConfig:
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


@pytest.fixture
def multi_provider_config() -> ProviderConfig:
    """Config with multiple providers for fallback testing."""
    return ProviderConfig(
        text_providers={
            "lmstudio": TextProviderConfig(
                priority=1,
                enabled=True,
                litellm_model="openai/local-model",
                base_url="http://localhost:1234/v1",
            ),
            "openai": TextProviderConfig(
                priority=2,
                enabled=True,
                litellm_model="openai/gpt-4o-mini",
                api_key="test-openai-key",
            ),
        }
    )


# --- Tests for _format_history_as_context ---


class TestFormatHistoryAsContext:
    """Tests for the history formatting function."""

    def test_single_user_message_no_context(self):
        """Single user message should return as-is."""
        messages = [{"role": "user", "content": "Hello"}]
        system, prompt = _format_history_as_context(messages)
        assert system is None
        assert prompt == "Hello"

    def test_system_and_user_message(self):
        """System + user should extract system prompt."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        system, prompt = _format_history_as_context(messages)
        assert system == "You are helpful."
        assert prompt == "What is 2+2?"

    def test_multi_turn_conversation(self):
        """Multi-turn should format as context."""
        messages = [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Phase 1: Plan"},
            {"role": "assistant", "content": '{"step": 1}'},
            {"role": "user", "content": "Phase 2: Execute"},
        ]
        system, prompt = _format_history_as_context(messages)

        assert system == "You are a planner."
        assert "<conversation_context>" in prompt
        assert "[Previous Response]" in prompt
        assert '{"step": 1}' in prompt
        assert "Phase 2: Execute" in prompt

    def test_three_turn_conversation(self):
        """Three turn conversation should have all context."""
        messages = [
            {"role": "system", "content": "Expert assistant"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
            {"role": "user", "content": "Third question"},
        ]
        system, prompt = _format_history_as_context(messages)

        assert system == "Expert assistant"
        assert "First answer" in prompt
        assert "Second answer" in prompt
        assert "Third question" in prompt


# --- Tests for StructuredExtractor initialization ---


class TestExtractorInit:
    """Tests for StructuredExtractor initialization."""

    def test_init_with_config(self, single_provider_config):
        """Should initialize with provided config."""
        extractor = StructuredExtractor(config=single_provider_config)
        assert extractor.config == single_provider_config
        assert extractor._text_provider is not None

    def test_init_with_override(self, single_provider_config):
        """Should pass override to TextProvider."""
        extractor = StructuredExtractor(
            config=single_provider_config,
            provider_override="lmstudio",
        )
        assert extractor._provider_override == "lmstudio"

    def test_properties_delegate_to_text_provider(self, single_provider_config):
        """Properties should delegate to TextProvider."""
        extractor = StructuredExtractor(config=single_provider_config)
        extractor._text_provider._current_provider = "gemini"
        extractor._text_provider._current_model = "gemini-2.0-flash"

        assert extractor.current_provider == "gemini"
        assert extractor.current_model == "gemini-2.0-flash"


# --- Tests for extract() with mocked TextProvider ---


class TestExtractWithMock:
    """Tests for extract() with mocked TextProvider."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, single_provider_config):
        """Should delegate to TextProvider.generate_structured."""
        extractor = StructuredExtractor(config=single_provider_config)

        mock_result = SimpleResponse(answer="42")
        extractor._text_provider.generate_structured = AsyncMock(return_value=mock_result)

        result = await extractor.extract(
            prompt="What is the answer?",
            response_model=SimpleResponse,
            system="You are a calculator.",
            task="test",
        )

        assert result.answer == "42"
        assert extractor.total_calls == 1

        # Verify the call
        extractor._text_provider.generate_structured.assert_called_once()
        call_kwargs = extractor._text_provider.generate_structured.call_args.kwargs
        assert call_kwargs["prompt"] == "What is the answer?"
        assert call_kwargs["response_model"] == SimpleResponse
        assert call_kwargs["system"] == "You are a calculator."

    @pytest.mark.asyncio
    async def test_extract_passes_all_params(self, single_provider_config):
        """Should pass all parameters to TextProvider."""
        extractor = StructuredExtractor(config=single_provider_config)

        mock_result = SimpleResponse(answer="result")
        extractor._text_provider.generate_structured = AsyncMock(return_value=mock_result)

        await extractor.extract(
            prompt="test prompt",
            response_model=SimpleResponse,
            system="system prompt",
            task="my_task",
            max_retries=5,
            temperature=0.3,
            max_tokens=500,
        )

        call_kwargs = extractor._text_provider.generate_structured.call_args.kwargs
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 500


# --- Tests for extract_with_history() ---


class TestExtractWithHistory:
    """Tests for extract_with_history()."""

    @pytest.mark.asyncio
    async def test_extract_with_history_first_call(self, single_provider_config):
        """First call should initialize history."""
        extractor = StructuredExtractor(config=single_provider_config)

        mock_result = PlanResponse(title="Plan", steps=["Step 1"])
        extractor._text_provider.generate_structured = AsyncMock(return_value=mock_result)

        result, history = await extractor.extract_with_history(
            prompt="Create a plan",
            response_model=PlanResponse,
            system="You are a planner.",
        )

        assert result.title == "Plan"
        assert len(history) == 3  # system + user + assistant
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_extract_with_history_continuation(self, single_provider_config):
        """Continuation should include previous context."""
        extractor = StructuredExtractor(config=single_provider_config)

        # First call
        first_result = PlanResponse(title="Plan", steps=["Step 1"])
        extractor._text_provider.generate_structured = AsyncMock(return_value=first_result)

        _, history = await extractor.extract_with_history(
            prompt="Phase 1",
            response_model=PlanResponse,
            system="Be helpful.",
        )

        # Second call with history
        second_result = ContentResponse(hook="Hook", body="Body")
        extractor._text_provider.generate_structured = AsyncMock(return_value=second_result)

        result, updated_history = await extractor.extract_with_history(
            prompt="Phase 2",
            response_model=ContentResponse,
            history=history,
        )

        assert result.hook == "Hook"
        assert len(updated_history) == 5  # system + user1 + assistant1 + user2 + assistant2

        # The formatted prompt should contain context
        call_kwargs = extractor._text_provider.generate_structured.call_args.kwargs
        prompt = call_kwargs["prompt"]
        assert "<conversation_context>" in prompt

    @pytest.mark.asyncio
    async def test_extract_with_history_tracks_calls(self, single_provider_config):
        """Should track call count across multiple extractions."""
        extractor = StructuredExtractor(config=single_provider_config)

        mock_result = SimpleResponse(answer="ok")
        extractor._text_provider.generate_structured = AsyncMock(return_value=mock_result)

        await extractor.extract_with_history(
            prompt="First",
            response_model=SimpleResponse,
            system="sys",
        )
        await extractor.extract_with_history(
            prompt="Second",
            response_model=SimpleResponse,
        )

        assert extractor.total_calls == 2


# --- Tests for get_extractor singleton ---


class TestGetExtractor:
    """Tests for the get_extractor singleton function."""

    def test_get_extractor_returns_singleton(self):
        """Should return the same instance."""
        # Reset singleton
        import socials_automator.services.extractor as ext_module
        ext_module._default_extractor = None

        ext1 = get_extractor()
        ext2 = get_extractor()
        assert ext1 is ext2

    def test_get_extractor_with_config_creates_new(self, single_provider_config):
        """Passing config should create new instance."""
        import socials_automator.services.extractor as ext_module
        ext_module._default_extractor = None

        ext1 = get_extractor()
        ext2 = get_extractor(config=single_provider_config)
        assert ext1 is not ext2


# --- Integration tests with mocked Agno Agent ---


class TestIntegrationWithMockedAgent:
    """Integration tests with mocked Agno Agent."""

    @pytest.mark.asyncio
    async def test_full_extraction_flow(self, single_provider_config):
        """Test full extraction flow with mocked Agent."""
        with patch("agno.agent.Agent") as MockAgent:
            # Setup mock
            mock_response = MagicMock()
            mock_response.content = PlanResponse(title="My Plan", steps=["Do thing"])
            mock_response.model = "gemini-2.0-flash"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(return_value=mock_response)
            MockAgent.return_value = mock_agent

            extractor = StructuredExtractor(config=single_provider_config)
            result = await extractor.extract(
                prompt="Create a plan",
                response_model=PlanResponse,
                system="You are a planner.",
            )

            assert result.title == "My Plan"
            assert result.steps == ["Do thing"]

            # Verify Agent was created with output_schema
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs["output_schema"] == PlanResponse

    @pytest.mark.asyncio
    async def test_multi_phase_extraction(self, single_provider_config):
        """Test multi-phase extraction simulating content planning."""
        with patch("agno.agent.Agent") as MockAgent:
            # Phase 1 response
            phase1_response = MagicMock()
            phase1_response.content = PlanResponse(title="Content Plan", steps=["Research", "Write"])
            phase1_response.model = "gemini-2.0-flash"

            # Phase 2 response
            phase2_response = MagicMock()
            phase2_response.content = ContentResponse(hook="Attention!", body="Full content here")
            phase2_response.model = "gemini-2.0-flash"

            mock_agent = MagicMock()
            mock_agent.arun = AsyncMock(side_effect=[phase1_response, phase2_response])
            MockAgent.return_value = mock_agent

            extractor = StructuredExtractor(config=single_provider_config)

            # Phase 1
            plan, history = await extractor.extract_with_history(
                prompt="Plan the content",
                response_model=PlanResponse,
                system="You are a content strategist.",
            )

            assert plan.title == "Content Plan"

            # Phase 2
            content, final_history = await extractor.extract_with_history(
                prompt="Now write the content based on the plan",
                response_model=ContentResponse,
                history=history,
            )

            assert content.hook == "Attention!"
            assert extractor.total_calls == 2
