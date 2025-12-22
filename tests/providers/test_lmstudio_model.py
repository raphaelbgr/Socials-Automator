"""Tests for LMStudio model detection."""

import pytest
from unittest.mock import patch, MagicMock

from socials_automator.providers.text import _get_lmstudio_model, _create_agno_model
from socials_automator.providers.config import TextProviderConfig


class TestLMStudioModelDetection:
    """Tests for LMStudio model detection functionality."""

    def test_get_lmstudio_model_success(self):
        """Test successful model detection from LMStudio."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "zai-org/glm-4.6v-flash"},
                {"id": "mistralai/devstral-small"},
            ]
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            model = _get_lmstudio_model("http://localhost:1234/v1")
            assert model == "zai-org/glm-4.6v-flash"

    def test_get_lmstudio_model_no_models(self):
        """Test when LMStudio has no models loaded."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            model = _get_lmstudio_model("http://localhost:1234/v1")
            assert model is None

    def test_get_lmstudio_model_connection_error(self):
        """Test when LMStudio is not running."""
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Connection refused")

            model = _get_lmstudio_model("http://localhost:1234/v1")
            assert model is None

    def test_create_agno_model_lmstudio_with_local_model(self):
        """Test that local-model config queries LMStudio for actual model."""
        config = TextProviderConfig(
            priority=0,
            enabled=True,
            litellm_model="openai/local-model",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout=120,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "zai-org/glm-4.6v-flash"}]
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            model = _create_agno_model("lmstudio", config)

            # Should have queried LMStudio
            mock_client.return_value.__enter__.return_value.get.assert_called_with(
                "http://localhost:1234/v1/models"
            )

            # Model should have detected ID
            assert model.id == "zai-org/glm-4.6v-flash"

    def test_create_agno_model_lmstudio_with_explicit_model(self):
        """Test that explicit model config skips detection."""
        config = TextProviderConfig(
            priority=0,
            enabled=True,
            litellm_model="openai/qwen2.5-coder",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout=120,
        )

        with patch("httpx.Client") as mock_client:
            model = _create_agno_model("lmstudio", config)

            # Should NOT query LMStudio (model is not "local-model")
            mock_client.return_value.__enter__.return_value.get.assert_not_called()

            # Model should use configured ID
            assert model.id == "qwen2.5-coder"

    def test_create_agno_model_lmstudio_fallback_to_none(self):
        """Test fallback when LMStudio detection fails."""
        config = TextProviderConfig(
            priority=0,
            enabled=True,
            litellm_model="openai/local-model",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout=120,
        )

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Connection refused")

            model = _create_agno_model("lmstudio", config)

            # Model ID should be None (LMStudio will use default)
            assert model.id is None
